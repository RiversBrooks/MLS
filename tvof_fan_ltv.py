"""
Austin FC TVOF | Cohort retention and fan lifetime value (v1, 2026-09-23)
-------------------------------------------------------------
Individual/household paying accounts (tvof_fan_value.py definition: never more than
GROUP_MAX_ITEMS items for one product), grouped by their FIRST paying season and by
what they bought that season:

    Season-ticket member   bought a plan (not the waitlist)
    Waitlist member        paid only the waitlist fee
    Single-ticket buyer    everything else (single matches, events, parking)

Seasons: plans take the season in their product name ("2024 Full Season Membership"
is sold in 2023 but belongs to 2024). Other purchases take the transaction year,
with December counted toward the next season. Only complete seasons (2021-2025)
are fitted; 2026 is still selling single tickets and 2027 renewals are in progress.

Method (chain ladder, as in actuarial loss development):
    v0          revenue per cohort member in the first season, pooled across cohorts
    f(k)        development factor = SUM rev(age k) / SUM rev(age k-1), over the cohorts
                observed at both ages
    v(k)        v(k-1) * f(k); the same for share active
    tail        beyond the last observed age: share active decays by the average of its
                last two factors (capped at 1.0); revenue per ACTIVE member is held at the
                last observed level (no price growth)
    LTV         SUM v(k) / (1 + r)^k, k = 0..H-1   (gross club revenue, nominal prices)

Usage:
    python tvof_fan_ltv.py --data-dir "<your folder of client CSVs>"

Output: fan_ltv_<date>.xlsx in the data folder (aggregate only).
"""
import argparse, datetime as dt, os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tvof_data_audit as A   # noqa: E402

GROUP_MAX_ITEMS = 8
FIRST_SEASON, LAST_COMPLETE = 2021, 2025
BASE_RATE, BASE_HORIZON = 0.08, 10
RATES, HORIZONS = [0.06, 0.08, 0.10, 0.12], [5, 10, 15]
SEGMENTS = ["Season-ticket member", "Waitlist member", "Single-ticket buyer"]
RECENT_FROM = 2023   # "recent cohorts" scenario: newer buyers return less often than 2021's


def season_revenue(s):
    """Club revenue per account per season, counted once (same rule as tvof_fan_value.py)."""
    it = s.item_type.astype(object).str.lower().to_numpy()
    sub = s.subscription_instance_id
    sub_na = sub.isna().to_numpy()
    pay = s.total_payment.fillna(0).to_numpy()
    plan_ids = set(sub[it == "subscription"].dropna().to_numpy())
    orphan = (it == "ticket") & ~sub_na & ~sub.isin(plan_ids).to_numpy()
    club = np.where((it == "subscription") | ((it == "ticket") & sub_na) | orphan, pay, 0.0)
    desc = pd.Series(s.product_description.astype(object).fillna("").to_numpy())
    t = s.transaction_date
    tyr = (t.dt.year + (t.dt.month == 12)).to_numpy()
    dyr = desc.str.extract(r"^(20\d\d)")[0].astype(float).to_numpy()
    isplan = (it == "subscription") | orphan
    season = np.maximum(np.where(isplan & ~np.isnan(dyr), dyr, tyr), FIRST_SEASON)
    wait = desc.str.contains("waitlist", case=False).to_numpy()
    R = pd.DataFrame({"a": s.internal_account_id.array, "club": club, "season": season.astype(int),
                      "plan": np.where(isplan & ~wait, club, 0.0), "wait": np.where(wait, club, 0.0),
                      "prod": s.product_id.astype(object).to_numpy()})
    R = R.dropna(subset=["a"])
    return R[R.club != 0]


def chain_ladder(tri, sizes, horizon):
    """Cohort x age triangle of totals -> per-member curve for ages 0..horizon-1, factors, observed flag."""
    ages = sorted(tri.columns)
    v0 = tri[0].sum() / sizes.sum()
    f = {}
    for k in ages[1:]:
        both = tri[[k - 1, k]].dropna().index
        f[k] = tri.loc[both, k].sum() / tri.loc[both, k - 1].sum() if len(both) else np.nan
    # the drop after the first season is structural, so the tail uses factors from age 2 on
    late = [f[k] for k in ages[1:] if k >= 2]
    last = (late or [f[k] for k in ages[1:]])[-2:]
    tail = min(1.0, float(np.mean(last))) if last else 1.0
    curve, fac, obs = [v0], [np.nan], [True]
    for k in range(1, horizon):
        fk = f.get(k, tail)
        curve.append(curve[-1] * fk)
        fac.append(fk)
        obs.append(k in f)
    return np.array(curve), np.array(fac), np.array(obs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=A.DEFAULT_DIR)
    ap.add_argument("--chunksize", type=int, default=A.CHUNK)
    a = ap.parse_args()
    files = A.find_files(a.data_dir)
    if "sales" not in files:
        sys.exit(f"sales file not found in {a.data_dir}")

    A.log("Loading sales ...")
    s = A.load("sales", files["sales"], a.chunksize)
    R = season_revenue(s)
    del s
    mx = R.groupby(["a", "prod"]).size().groupby(level=0).max()
    R = R[R.a.isin(set(mx[mx <= GROUP_MAX_ITEMS].index))]

    AS = R.groupby(["a", "season"]).agg(rev=("club", "sum"), plan=("plan", "sum"), wait=("wait", "sum"))
    AS = AS[AS.rev > 0].reset_index()
    AS = AS.join(AS.groupby("a").season.min().rename("cohort"), on="a")
    AS["age"] = AS.season - AS.cohort
    f0 = AS[AS.age == 0].set_index("a")
    seg = pd.Series(np.select([f0.plan > 0, f0.rev - f0.wait > 0], SEGMENTS[::2], SEGMENTS[1]), index=f0.index)
    AS["segment"] = AS.a.map(seg)
    fit = AS[(AS.season <= LAST_COMPLETE)]
    A.log(f"  {AS.a.nunique():,} individual/household paying accounts; fitting seasons {FIRST_SEASON}-{LAST_COMPLETE}")

    cohorts, curves, ltv, factors = [], [], [], []
    runs = [(sg, sg, 0) for sg in SEGMENTS + ["All individual buyers"]] + \
           [(f"{sg} ({RECENT_FROM}+ cohorts)", sg, RECENT_FROM) for sg in SEGMENTS[1:]]
    for sg, base_sg, cmin in runs:
        x = fit if base_sg.startswith("All") else fit[fit.segment == base_sg]
        x = x[x.cohort >= cmin]
        n = x[x.age == 0].groupby("cohort").size()
        act = x.groupby(["cohort", "age"]).size().unstack()
        rev = x.groupby(["cohort", "age"]).rev.sum().unstack()
        for c in n.index:
            for k in act.columns:
                if c + k <= LAST_COMPLETE:
                    cohorts.append({"segment": sg, "cohort": c, "age": k, "cohort_size": int(n[c]),
                                    "share_active": act.loc[c, k] / n[c] if pd.notna(act.loc[c, k]) else 0.0,
                                    "revenue_per_member": rev.loc[c, k] / n[c] if pd.notna(rev.loc[c, k]) else 0.0})
        tri_rev = rev.copy()
        tri_act = act.copy()
        for c in tri_rev.index:          # blank out seasons not yet observed (keep 0 for observed but inactive)
            for k in tri_rev.columns:
                if c + k > LAST_COMPLETE:
                    tri_rev.loc[c, k] = tri_act.loc[c, k] = np.nan
                elif pd.isna(tri_rev.loc[c, k]):
                    tri_rev.loc[c, k] = tri_act.loc[c, k] = 0.0
        H = max(HORIZONS)
        v, fv, obs = chain_ladder(tri_rev, n, H)
        r, fr, _ = chain_ladder(tri_act, n, H)
        # tail: share active keeps its own decay; revenue per ACTIVE member stays at the last
        # observed level. (Extending the revenue factor alone would imply the few survivors
        # spend more every season.)
        L_ = int(np.flatnonzero(obs)[-1])
        for k in range(L_ + 1, H):
            v[k] = r[k] * v[L_] / r[L_]
            fv[k] = v[k] / v[k - 1]
        for k in range(H):
            curves.append({"segment": sg, "age": k, "season_offset": f"season {k + 1}",
                           "share_active": r[k], "revenue_per_starting_member": v[k],
                           "revenue_per_active_member": v[k] / r[k] if r[k] else np.nan,
                           "revenue_factor": fv[k], "observed": bool(obs[k])})
            factors.append({"segment": sg, "age": k, "revenue_factor": fv[k], "active_factor": fr[k], "observed": bool(obs[k])})
        for h in HORIZONS:
            for rate in RATES:
                disc = (1 + rate) ** -np.arange(h)
                ltv.append({"segment": sg, "horizon_seasons": h, "discount_rate": rate,
                            "ltv": float((v[:h] * disc).sum()), "undiscounted": float(v[:h].sum()),
                            "first_season_value": float(v[0]), "starting_members": int(n.sum())})

    C = pd.DataFrame(cohorts)
    V = pd.DataFrame(curves)
    L = pd.DataFrame(ltv)
    base = L[(L.horizon_seasons == BASE_HORIZON) & (L.discount_rate == BASE_RATE)].set_index("segment")

    # blended LTV for a NEW buyer today: segment mix of the last three complete cohorts
    mix = AS[(AS.age == 0) & AS.cohort.between(LAST_COMPLETE - 2, LAST_COMPLETE)].segment.value_counts(normalize=True)
    blend = []
    for h in HORIZONS:
        for rate in RATES:
            x = L[(L.horizon_seasons == h) & (L.discount_rate == rate)].set_index("segment").ltv
            blend.append({"horizon_seasons": h, "discount_rate": rate,
                          "blended_ltv_new_buyer": float(sum(mix.get(sg, 0) * x[sg] for sg in SEGMENTS)),
                          "blended_ltv_new_buyer_recent_cohorts": float(
                              mix.get(SEGMENTS[0], 0) * x[SEGMENTS[0]] +
                              sum(mix.get(sg, 0) * x[f"{sg} ({RECENT_FROM}+ cohorts)"] for sg in SEGMENTS[1:]))})
    Bl = pd.DataFrame(blend)
    M = mix.rename("share_of_new_buyers_2023_2025").reset_index().rename(columns={"index": "segment"})

    notes = pd.DataFrame({"note": [
        "Population: individual/household paying accounts (never more than 8 items for one product). Group/corporate/broker accounts are excluded.",
        "Revenue = club revenue counted once: plans (plan row), single tickets not in a plan, orphan plan game rows. Resale excluded.",
        f"Seasons fitted: {FIRST_SEASON}-{LAST_COMPLETE}. 2026 is excluded because single tickets are still selling; 2027 because renewals are in progress.",
        "Chain ladder: each factor compares the same cohorts at consecutive ages, so a change in cohort mix does not bias it.",
        "Tail beyond the observed ages: share active decays by the average of its last two factors (capped at 1.0); revenue per active member stays at the last observed level. Nominal prices: no price growth assumed.",
        "'All individual buyers' uses the HISTORICAL mix (the 2021 cohort had many season-ticket members). For a buyer acquired today use LTV_Blended_New_Buyer (2023-2025 mix: about 97% single-ticket buyers).",
        "LTV is GROSS REVENUE, not profit. Multiply by the contribution margin for a profit-based LTV (e.g. to compare with acquisition cost).",
        f"Recent-cohort scenario refits waitlist and single-ticket curves on {RECENT_FROM}+ cohorts only (newer single-ticket buyers return far less often); treat it as the downside case.",
        "Season-ticket member cohorts after 2021 are small (most new members join from the waitlist), so that curve leans on the 2021 cohort.",
        "Base case: 10 seasons, 8% discount rate. See LTV_Sensitivity for 5/10/15 seasons and 6-12%."]})
    out = os.path.join(a.data_dir, f"fan_ltv_{dt.date.today():%Y%m%d}.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        base.reset_index().to_excel(xw, sheet_name="LTV_Base_Case", index=False)
        Bl.to_excel(xw, sheet_name="LTV_Blended_New_Buyer", index=False)
        M.to_excel(xw, sheet_name="New_Buyer_Mix", index=False)
        L.to_excel(xw, sheet_name="LTV_Sensitivity", index=False)
        V.to_excel(xw, sheet_name="Projected_Curves", index=False)
        C.to_excel(xw, sheet_name="Cohort_Triangles", index=False)
        pd.DataFrame(factors).to_excel(xw, sheet_name="Development_Factors", index=False)
        notes.to_excel(xw, sheet_name="Notes", index=False)
    A.autofilter(out)
    A.log(f"Wrote {out}  (aggregate only)")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    print("\nBASE CASE (10 seasons, 8%):")
    print(base[["starting_members", "first_season_value", "undiscounted", "ltv"]].round(0).to_string())
    print("\nNew-buyer mix 2023-2025:", mix.round(3).to_dict())
    print(Bl.pivot(index="horizon_seasons", columns="discount_rate", values="blended_ltv_new_buyer").round(0).to_string())
    print("recent-cohort basis:")
    print(Bl.pivot(index="horizon_seasons", columns="discount_rate", values="blended_ltv_new_buyer_recent_cohorts").round(0).to_string())
    print("\nshare active, recent vs all cohorts:")
    print(V[V.age < 6].pivot(index="age", columns="segment", values="share_active").round(3).to_string())
    for sg in SEGMENTS:
        print(f"\n{sg}: LTV by horizon x rate")
        print(L[L.segment == sg].pivot(index="horizon_seasons", columns="discount_rate", values="ltv").round(0).to_string())
    print("\nProjected curves (first 10 seasons):")
    pv = V[V.age < 10].pivot(index="age", columns="segment", values=["share_active", "revenue_per_starting_member"])
    print(pv.round(3).to_string())
    print(V[V.age < 10].pivot(index="age", columns="segment", values="observed").to_string())


if __name__ == "__main__":
    main()
