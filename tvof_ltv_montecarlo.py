"""
Austin FC TVOF | Monte Carlo on fan LTV and customer equity (v1, 2026-09-26)
-------------------------------------------------------------
Puts a range around the chain-ladder point estimates in fan_ltv_<date>.xlsx and the customer
equity in value_statements_<date>.xlsx.

What is random (drawn once per simulation, per segment):
  retention factor at each age     Beta(active at k + 0.5, lapsed at k + 0.5) pooled over the cohorts observed
                                   at both ages (sampling error), times a lognormal cohort-heterogeneity draw
                                   whose sd is the size-weighted sd of log cohort factors at that age
                                   (ages seen in one cohort only borrow the segment's mean sd); capped at 1.0
  spend per active member factor   pooled ratio of revenue per active member at k vs k-1, times the same kind
                                   of lognormal heterogeneity draw
  first-season spend               pooled revenue per member at age 0, times a lognormal draw from cohort dispersion
  tail                             beyond the last observed age the retention factor is the mean of the last two
                                   simulated factors (capped at 1.0) and spend per active member is flat,
                                   the same rule as the deterministic model
What is fixed: discount rate, horizon, segment mix of new buyers, active accounts by segment and age,
nominal prices (no growth), no reactivation of lapsed accounts.

Usage:
    python tvof_ltv_montecarlo.py [--data-dir D] [--sims 20000] [--seed 20260926] [--json-out p]

Output: ltv_montecarlo_<date>.xlsx in the data folder (aggregate only).
"""
import argparse, datetime as dt, glob, json, os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tvof_data_audit as A   # noqa: E402
from tvof_fan_ltv import SEGMENTS, LAST_COMPLETE, BASE_RATE, BASE_HORIZON, RATES, HORIZONS, RECENT_FROM   # noqa: E402

PCTS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
RECENT = {SEGMENTS[1]: f"{SEGMENTS[1]} ({RECENT_FROM}+ cohorts)", SEGMENTS[2]: f"{SEGMENTS[2]} ({RECENT_FROM}+ cohorts)"}


def newest(data_dir, prefix):
    c = sorted(f for f in glob.glob(os.path.join(data_dir, prefix + "_*.xlsx")) if "LOCAL_ONLY" not in f)
    if not c:
        sys.exit(f"{prefix}_<date>.xlsx not found in {data_dir}")
    return c[-1]


def wstat(x, w):
    x, w = np.asarray(x, dtype=float), np.asarray(w, dtype=float)
    mu = np.average(x, weights=w)
    return mu, float(np.sqrt(np.average((x - mu) ** 2, weights=w)))


def fit_inputs(tri):
    """Observed cohort x age cells for one segment -> pooled factors, Beta counts and heterogeneity sds."""
    t = tri.copy()
    t["active"] = (t.share_active * t.cohort_size).round()
    t["revenue"] = t.revenue_per_member * t.cohort_size
    t["m"] = t.revenue / t.active.replace(0, np.nan)
    L = int(t.age.max())
    pa = t.pivot(index="cohort", columns="age", values="active")
    pr = t.pivot(index="cohort", columns="age", values="revenue")
    pm = t.pivot(index="cohort", columns="age", values="m")
    f, g = {}, {}
    for k in range(1, L + 1):
        both = pa[[k - 1, k]].dropna()
        both = both[both[k - 1] > 0]
        a0, a1 = float(both[k - 1].sum()), float(both[k].sum())
        fc = (both[k] / both[k - 1]).clip(lower=1e-6)
        _, sd_f = wstat(np.log(fc), both[k - 1]) if len(both) > 1 else (np.nan, np.nan)
        idx = both.index
        m0, m1 = pm.loc[idx, k - 1], pm.loc[idx, k]
        ok = m0.notna() & m1.notna() & (pa.loc[idx, k] > 0)
        g_pool = float(pr.loc[idx[ok], k].sum() / pa.loc[idx[ok], k].sum() / (pr.loc[idx[ok], k - 1].sum() / pa.loc[idx[ok], k - 1].sum())) if ok.any() else 1.0
        gc = (m1[ok] / m0[ok]).clip(lower=1e-6)
        _, sd_g = wstat(np.log(gc), pa.loc[idx[ok], k]) if ok.sum() > 1 else (np.nan, np.nan)
        f[k] = {"a0": a0, "a1": a1, "n_cohorts": int(len(both)), "pooled": a1 / a0 if a0 else np.nan, "sd_log": sd_f}
        g[k] = {"pooled": g_pool, "n_cohorts": int(ok.sum()), "sd_log": sd_g}
    fill_f = np.nanmean([v["sd_log"] for v in f.values()]) if any(not np.isnan(v["sd_log"]) for v in f.values()) else 0.05
    fill_g = np.nanmean([v["sd_log"] for v in g.values()]) if any(not np.isnan(v["sd_log"]) for v in g.values()) else 0.05
    for k in f:
        if np.isnan(f[k]["sd_log"]):
            f[k]["sd_log"], f[k]["sd_source"] = float(fill_f), "segment mean (one cohort at this age)"
        else:
            f[k]["sd_source"] = "cohorts at this age"
        if np.isnan(g[k]["sd_log"]):
            g[k]["sd_log"] = float(fill_g)
    a0 = t[t.age == 0]
    m0 = float(a0.revenue.sum() / a0.cohort_size.sum())
    _, m0_sd = wstat(np.log(a0.revenue_per_member.clip(lower=1e-6)), a0.cohort_size) if len(a0) > 1 else (np.nan, 0.0)
    return {"L": L, "f": f, "g": g, "m0": m0, "m0_sd_log": float(m0_sd), "cohorts": int(t.cohort.nunique()), "members": int(a0.cohort_size.sum())}


def simulate(inp, n, K, rng):
    """-> share active r[n, K] and revenue per starting member v[n, K] by age 0..K-1."""
    L = inp["L"]
    r = np.ones((n, K)); m = np.empty((n, K)); fs = np.full((n, K), np.nan)
    m[:, 0] = inp["m0"] * np.exp(rng.normal(0, inp["m0_sd_log"], n))
    for k in range(1, K):
        if k <= L:
            p = inp["f"][k]
            base = rng.beta(p["a1"] + 0.5, max(p["a0"] - p["a1"], 0) + 0.5, n)
            fk = np.minimum(1.0, base * np.exp(rng.normal(0, p["sd_log"], n)))
            gk = inp["g"][k]["pooled"] * np.exp(rng.normal(0, inp["g"][k]["sd_log"], n))
        else:
            lo = max(2, L - 1) if L >= 2 else 1
            fk = np.minimum(1.0, np.nanmean(fs[:, lo:L + 1], axis=1))
            gk = 1.0
        fs[:, k] = fk
        r[:, k] = r[:, k - 1] * fk
        m[:, k] = m[:, k - 1] * gk
    return r, r * m


def pct_row(x, name, point=None, unit="$"):
    x = np.asarray(x, dtype=float)
    q = np.quantile(x, PCTS)
    d = {"metric": name, "unit": unit, "point_estimate": point, "mean": float(x.mean()), "sd": float(x.std()),
         **{f"p{int(p * 100)}": float(v) for p, v in zip(PCTS, q)}}
    d["p95_over_p5"] = float(q[-1] / q[0]) if q[0] else np.nan
    if point is not None:
        d["share_of_sims_below_point"] = float((x < point).mean())
    return d


def hist(x, bins=24):
    c, e = np.histogram(np.asarray(x, dtype=float), bins=bins)
    return [{"lo": float(e[i]), "hi": float(e[i + 1]), "count": int(c[i]), "share": float(c[i] / c.sum())} for i in range(len(c))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=A.DEFAULT_DIR)
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()
    ltv_path, st_path = newest(a.data_dir, "fan_ltv"), newest(a.data_dir, "value_statements")
    tri = pd.read_excel(ltv_path, sheet_name="Cohort_Triangles")
    base_ltv = pd.read_excel(ltv_path, sheet_name="LTV_Base_Case").set_index("segment")
    blend = pd.read_excel(ltv_path, sheet_name="LTV_Blended_New_Buyer")
    mix = pd.read_excel(ltv_path, sheet_name="New_Buyer_Mix").set_index("segment").share_of_new_buyers_2023_2025
    eqd = pd.read_excel(st_path, sheet_name="5_Equity_by_Segment_Age")
    eqs = pd.read_excel(st_path, sheet_name="5_Customer_Equity")
    eq_point = float(eqs[eqs.segment.str.startswith("All")].remaining_total.iloc[0])
    bl = blend[(blend.horizon_seasons == BASE_HORIZON) & (blend.discount_rate == BASE_RATE)].iloc[0]
    rng = np.random.default_rng(a.seed)
    n, H = a.sims, BASE_HORIZON
    K = int(eqd.age.max()) + H + 2
    disc = (1 + BASE_RATE) ** -np.arange(H)
    A.log(f"Monte Carlo: {n:,} simulations, seed {a.seed}, horizon {H} seasons at {BASE_RATE:.0%}; inputs {os.path.basename(ltv_path)}, {os.path.basename(st_path)}")

    inputs, curves = {}, {}
    for sg in SEGMENTS + list(RECENT.values()):
        inputs[sg] = fit_inputs(tri[tri.segment.eq(sg)])
        curves[sg] = simulate(inputs[sg], n, K, rng)
    ltv = {sg: (curves[sg][1][:, :H] * disc).sum(axis=1) for sg in curves}
    new_base = sum(mix.get(sg, 0) * ltv[sg] for sg in SEGMENTS)
    new_recent = mix.get(SEGMENTS[0], 0) * ltv[SEGMENTS[0]] + sum(mix.get(sg, 0) * ltv[RECENT[sg]] for sg in RECENT)

    def equity(curve_of):
        tot = np.zeros(n); by_seg = {}
        for _, row in eqd.iterrows():
            r, v = curves[curve_of(row.segment)]
            age = int(row.age); js = np.arange(1, H + 1)
            rem = (v[:, age + js] / r[:, [age]] / (1 + BASE_RATE) ** js).sum(axis=1)
            by_seg[row.segment] = by_seg.get(row.segment, 0) + row.active_accounts * rem
            tot += row.active_accounts * rem
        return tot, by_seg
    eq_base, eq_base_seg = equity(lambda sg: sg)
    eq_recent, _ = equity(lambda sg: RECENT.get(sg, sg))

    rows = [pct_row(ltv[sg], f"LTV per new {sg.lower()}, {H} seasons at {BASE_RATE:.0%}", float(base_ltv.loc[sg, "ltv"])) for sg in SEGMENTS]
    rows += [pct_row(ltv[RECENT[sg]], f"LTV per new {sg.lower()}, {RECENT_FROM}+ cohorts", float(base_ltv.loc[RECENT[sg], "ltv"])) for sg in RECENT]
    rows.append(pct_row(new_base, "LTV of a new buyer today, 2023-2025 mix", float(bl.blended_ltv_new_buyer)))
    rows.append(pct_row(new_recent, "LTV of a new buyer today, recent-cohort basis", float(bl.blended_ltv_new_buyer_recent_cohorts)))
    for sg in SEGMENTS:
        pt = float(eqs[eqs.segment.eq(sg)].remaining_total.iloc[0]) if eqs.segment.eq(sg).any() else None
        rows.append(pct_row(eq_base_seg[sg], f"Customer equity, {sg.lower()}s active in {LAST_COMPLETE}", pt))
    rows.append(pct_row(eq_base, f"Customer equity, all individual accounts active in {LAST_COMPLETE}", eq_point))
    rows.append(pct_row(eq_recent, f"Customer equity, recent-cohort curves for waitlist and single-ticket", None))
    Sm = pd.DataFrame(rows)
    probs = pd.DataFrame([
        {"event": "New-buyer LTV below $500", "probability": float((new_base < 500).mean())},
        {"event": "New-buyer LTV below $600", "probability": float((new_base < 600).mean())},
        {"event": f"Customer equity below $100M", "probability": float((eq_base < 100e6).mean())},
        {"event": f"Customer equity below $120M", "probability": float((eq_base < 120e6).mean())},
        {"event": f"Customer equity above $150M", "probability": float((eq_base > 150e6).mean())},
        {"event": "Season-ticket member LTV below $20,000", "probability": float((ltv[SEGMENTS[0]] < 20000).mean())},
    ])
    # share still active by age, percentile bands (base segments)
    bands = []
    for sg in SEGMENTS:
        r, v = curves[sg]
        for k in range(min(K, 10)):
            qr = np.quantile(r[:, k], [0.05, 0.5, 0.95]); qv = np.quantile(v[:, k], [0.05, 0.5, 0.95])
            bands.append({"segment": sg, "season": k + 1, "observed": k <= inputs[sg]["L"], "active_p5": qr[0], "active_p50": qr[1], "active_p95": qr[2],
                          "rev_per_member_p5": qv[0], "rev_per_member_p50": qv[1], "rev_per_member_p95": qv[2]})
    Bd = pd.DataFrame(bands)
    inp_rows = []
    for sg, ip in inputs.items():
        for k in sorted(ip["f"]):
            inp_rows.append({"segment": sg, "age": k, "retention_factor_pooled": ip["f"][k]["pooled"], "active_prev": ip["f"][k]["a0"], "active_now": ip["f"][k]["a1"],
                             "cohorts": ip["f"][k]["n_cohorts"], "retention_sd_log": ip["f"][k]["sd_log"], "retention_sd_source": ip["f"][k]["sd_source"],
                             "spend_factor_pooled": ip["g"][k]["pooled"], "spend_sd_log": ip["g"][k]["sd_log"]})
        inp_rows.append({"segment": sg, "age": 0, "retention_factor_pooled": 1.0, "cohorts": ip["cohorts"], "spend_factor_pooled": ip["m0"], "spend_sd_log": ip["m0_sd_log"],
                         "retention_sd_source": f"first-season spend per member and its cohort dispersion; {ip['members']:,} members"})
    In = pd.DataFrame(inp_rows).sort_values(["segment", "age"])
    H1 = pd.DataFrame(hist(new_base)).assign(metric="new_buyer_ltv")
    H2 = pd.DataFrame(hist(eq_base)).assign(metric="customer_equity")
    notes = pd.DataFrame({"note": [
        f"{n:,} simulations, seed {a.seed}. Horizon {H} seasons, discount rate {BASE_RATE:.0%}, nominal prices, no reactivation: all fixed.",
        "Random per simulation and segment: retention factor at each age (Beta on pooled active counts, times lognormal cohort heterogeneity, capped at 1), spend per active member factor (pooled ratio times lognormal heterogeneity), first-season spend (lognormal around the pooled mean).",
        "Heterogeneity sds are size-weighted standard deviations of log cohort-level factors at that age; ages observed in a single cohort (the 2021 cohort at age 4) borrow the segment's mean sd.",
        "Tail beyond the last observed age: retention factor = mean of the last two simulated factors (capped at 1), spend per active member flat, as in tvof_fan_ltv.py.",
        "Customer equity: remaining value per active account at its age (conditional on being active) times the active accounts by segment and age from value_statements 5_Equity_by_Segment_Age.",
        "The recent-cohort scenario uses the 2023+ cohort triangles for waitlist and single-ticket buyers; season-ticket members keep all cohorts (their post-2021 cohorts are small).",
        "share_of_sims_below_point compares the simulated distribution with the deterministic chain-ladder value; a share near 0.5 means the point estimate sits at the median.",
        "Gross ticketing revenue, not profit. Widening the range further would need price uncertainty and macro scenarios, which the data does not inform.",
    ]})
    out = os.path.join(a.data_dir, f"ltv_montecarlo_{dt.date.today():%Y%m%d}.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        Sm.to_excel(xw, sheet_name="Summary", index=False)
        probs.to_excel(xw, sheet_name="Probabilities", index=False)
        Bd.to_excel(xw, sheet_name="Curve_Bands", index=False)
        pd.concat([H1, H2], ignore_index=True).to_excel(xw, sheet_name="Histograms", index=False)
        In.to_excel(xw, sheet_name="Inputs", index=False)
        notes.to_excel(xw, sheet_name="Notes", index=False)
    A.autofilter(out)
    A.log(f"Wrote {out}  (aggregate only)")
    if a.json_out:
        json.dump({"summary": json.loads(Sm.to_json(orient="records")), "probabilities": json.loads(probs.to_json(orient="records")),
                   "bands": json.loads(Bd.to_json(orient="records")), "hist_new_buyer": hist(new_base), "hist_equity": hist(eq_base),
                   "inputs": json.loads(In.to_json(orient="records")), "notes": notes.note.tolist(), "sims": n, "seed": a.seed,
                   "horizon": H, "rate": BASE_RATE, "source": os.path.basename(out)}, open(a.json_out, "w"), indent=1)
        A.log(f"Wrote {a.json_out}")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
    show = Sm[["metric", "point_estimate", "mean", "p5", "p25", "p50", "p75", "p95", "share_of_sims_below_point"]].copy()
    for c_ in show.columns[1:-1]:
        show[c_] = show[c_].round(0)
    show["share_of_sims_below_point"] = show.share_of_sims_below_point.round(3)
    print(show.to_string(index=False))
    print(probs.to_string(index=False))


if __name__ == "__main__":
    main()
