"""
Austin FC TVOF | LTV model validation: rolling backtest, tornado and variance decomposition (v1, 2026-09-27)
-------------------------------------------------------------
Reads fan_ltv_<date>.xlsx (Cohort_Triangles, LTV_Base_Case, LTV_Blended_New_Buyer, New_Buyer_Mix) and
value_statements_<date>.xlsx (5_Equity_by_Segment_Age, 5_Customer_Equity). No client CSV is read; it runs in seconds.

1  Backtest (rolling origin). For each cutoff T in 2022, 2023, 2024 the chain ladder is refitted on the cells
   observed by the end of season T (cohort + age <= T) and projects every cohort from its last observed age with
   the pooled factors, the way tvof_fan_ltv.py does. The held-out cells (seasons T+1 .. 2025) are compared with
   what happened: revenue per starting member, share active, and the total revenue of those cohorts by season.
   The Monte Carlo is refitted on the same cells; the share of held-out cells inside its 90% interval is the
   calibration check. "Carry the last observed season forward" is the naive benchmark. LTV as it would have been
   estimated at each cutoff shows how the estimate moved as seasons accrued.
2  Tornado. One input at a time from its low to its high setting, everything else at base: retention factor at
   age 1, retention factors at ages 2+ (they set the tail), spend per active member factors, first-season spend
   (low/high = p5/p95 of the Monte Carlo draws for that input), discount rate 6%/12%, horizon 5/15 seasons,
   new-buyer mix, price growth 0/+3% a season, cohort basis all/2023+ cohorts.
   Outputs: LTV of a new buyer (2023-2025 mix, 10 seasons at 8%) and customer equity of the 2025 active base.
3  Variance decomposition. Sobol first-order and total-effect indices by pick-freeze (Saltelli 2010 first-order,
   Jansen 1999 total-effect estimators) on the Monte Carlo inputs, by input type, by segment and by both.

Usage:
    python tvof_ltv_validation.py [--data-dir D] [--sims 20000] [--sobol-sims 50000] [--seed 20260927] [--json-out p]

Output: ltv_validation_<date>.xlsx in the data folder (aggregate only).
"""
import argparse, datetime as dt, json, os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tvof_data_audit as A   # noqa: E402
from tvof_fan_ltv import SEGMENTS, LAST_COMPLETE, BASE_RATE, BASE_HORIZON, HORIZONS, RECENT_FROM, chain_ladder   # noqa: E402
from tvof_ltv_montecarlo import fit_inputs, newest, RECENT   # noqa: E402

ALL = "All individual buyers"
CUTOFFS = [2022, 2023, 2024]
TYPES = ["first-season spend", "retention age 1", "retention ages 2+", "spend per active factors"]
TRI_COLS = ["segment", "cohort", "age", "cohort_size", "share_active", "revenue_per_member"]


# ----------------------------------------------------------------------------------------------------- model
def totals(tri):
    t = tri.copy()
    t["revenue"] = t.revenue_per_member * t.cohort_size
    t["active"] = t.share_active * t.cohort_size
    return t


def det_project(tri_T, H):
    """Chain ladder exactly as tvof_fan_ltv.main: revenue per starting member v and share active r, ages 0..H-1."""
    t = totals(tri_T)
    n = t[t.age == 0].set_index("cohort").cohort_size.astype(float)
    rev = t.pivot(index="cohort", columns="age", values="revenue")
    act = t.pivot(index="cohort", columns="age", values="active")
    v, fv, obs = chain_ladder(rev, n, H)
    r, fr, _ = chain_ladder(act, n, H)
    L_ = int(np.flatnonzero(obs)[-1])
    for k in range(L_ + 1, H):
        v[k] = r[k] * v[L_] / r[L_] if r[L_] else 0.0
    return v, r, obs


def draw(inp, n, rng):
    """The Monte Carlo's random inputs, drawn up front: m0 multiplier (n,), retention factors f and spend multipliers g (n, L+1)."""
    L = inp["L"]
    d = {"m0": np.exp(rng.normal(0, inp["m0_sd_log"], n)), "f": np.ones((n, L + 1)), "g": np.ones((n, L + 1))}
    for k in range(1, L + 1):
        p = inp["f"][k]
        d["f"][:, k] = rng.beta(p["a1"] + 0.5, max(p["a0"] - p["a1"], 0) + 0.5, n) * np.exp(rng.normal(0, p["sd_log"], n))
        d["g"][:, k] = np.exp(rng.normal(0, inp["g"][k]["sd_log"], n))
    return d


def base_draw(inp):
    """Every input at its pooled value: reproduces the deterministic chain ladder."""
    L = inp["L"]
    f = np.ones((1, L + 1))
    for k in range(1, L + 1):
        f[0, k] = inp["f"][k]["pooled"]
    return {"m0": np.ones(1), "f": f, "g": np.ones((1, L + 1))}


def curve(inp, d, K, growth=0.0):
    """Replica of tvof_ltv_montecarlo.simulate from pre-drawn inputs -> share active r[n, K], revenue per starting member v[n, K]."""
    L = inp["L"]
    n = len(d["m0"])
    r = np.ones((n, K)); m = np.empty((n, K)); fs = np.full((n, K), np.nan)
    m[:, 0] = inp["m0"] * d["m0"]
    for k in range(1, K):
        if k <= L:
            fk = np.minimum(1.0, d["f"][:, k])
            gk = inp["g"][k]["pooled"] * d["g"][:, k]
        else:
            lo = max(2, L - 1) if L >= 2 else 1
            fk = np.minimum(1.0, np.nanmean(fs[:, lo:L + 1], axis=1))
            gk = 1.0
        fs[:, k] = fk
        r[:, k] = r[:, k - 1] * fk
        m[:, k] = m[:, k - 1] * gk * (1 + growth)
    return r, r * m


def ltv_of(v, H, rate):
    return (v[:, :H] * (1 + rate) ** -np.arange(H)).sum(axis=1)


def equity_of(curves, eqd, H, rate):
    tot = 0.0
    for row in eqd.itertuples():
        r, v = curves[row.segment]
        age = int(row.age); js = np.arange(1, H + 1)
        tot = tot + row.active_accounts * (v[:, age + js] / r[:, [age]] / (1 + rate) ** js).sum(axis=1)
    return tot


def outputs(inps, draws, K, H, rate, mix, eqd, growth=0.0, recent=False):
    """-> new-buyer LTV (n,), customer equity (n,), LTV by segment. recent=True swaps in the 2023+ cohort curves for waitlist and single-ticket."""
    curves = {sg: curve(inps[sg], draws[sg], K, growth) for sg in SEGMENTS}
    if recent:
        for sg in RECENT:
            curves[sg] = curve(inps[RECENT[sg]], draws[RECENT[sg]], K, growth)
    ltv = {sg: ltv_of(curves[sg][1], H, rate) for sg in SEGMENTS}
    new = sum(mix.get(sg, 0) * ltv[sg] for sg in SEGMENTS)
    return new, equity_of(curves, eqd, H, rate), ltv


# -------------------------------------------------------------------------------------------------- backtest
def backtest(tri, T, H, n, rng):
    cells, by_season, summ, ltv_T = [], [], [], {}
    for sg in SEGMENTS + [ALL]:
        t_all = totals(tri[tri.segment == sg])
        t_T = t_all[(t_all.cohort + t_all.age) <= T]
        if t_T.age.max() < 1:
            continue
        v, r, obs = det_project(t_T, H)
        inp = fit_inputs(t_T[TRI_COLS])
        rs, vs = curve(inp, draw(inp, n, rng), H)
        ltv_T[sg] = float((v[:BASE_HORIZON] * (1 + BASE_RATE) ** -np.arange(BASE_HORIZON)).sum())
        held = t_all[(t_all.cohort + t_all.age > T) & (t_all.cohort <= T)]
        agg = {}
        w_ape, w_sum, inside, n_cells = 0.0, 0.0, 0, 0
        for row in held.itertuples():
            c, k = int(row.cohort), int(row.age); k0 = T - c
            o = t_all[(t_all.cohort == c) & (t_all.age == k0)].iloc[0]
            pv = o.revenue_per_member * v[k] / v[k0] if v[k0] else 0.0
            pr = o.share_active * r[k] / r[k0] if r[k0] else 0.0
            sim = o.revenue_per_member * vs[:, k] / vs[:, k0]
            q5, q50, q95 = np.quantile(sim, [0.05, 0.5, 0.95])
            ins = bool(q5 <= row.revenue_per_member <= q95)
            inside += ins; n_cells += 1
            if row.revenue_per_member > 0:
                w_ape += row.revenue * abs(pv / row.revenue_per_member - 1)
                w_sum += row.revenue
            cells.append({"cutoff": T, "segment": sg, "cohort": c, "age": k, "season": c + k, "cohort_size": int(row.cohort_size),
                          "seasons_ahead": k - k0, "actual_rev_per_member": row.revenue_per_member, "predicted_rev_per_member": pv,
                          "naive_rev_per_member": o.revenue_per_member, "error_pct": pv / row.revenue_per_member - 1 if row.revenue_per_member else np.nan,
                          "mc_p5": q5, "mc_p50": q50, "mc_p95": q95, "inside_90pct_interval": ins,
                          "actual_share_active": row.share_active, "predicted_share_active": pr})
            s = c + k
            a = agg.setdefault(s, {"actual": 0.0, "pred": 0.0, "naive": 0.0, "sim": np.zeros(n), "act_actual": 0.0, "act_pred": 0.0, "members": 0})
            a["actual"] += row.revenue; a["pred"] += row.cohort_size * pv; a["naive"] += row.cohort_size * o.revenue_per_member
            a["sim"] += row.cohort_size * sim; a["act_actual"] += row.active; a["act_pred"] += row.cohort_size * pr; a["members"] += int(row.cohort_size)
        tot = {"actual": 0.0, "pred": 0.0, "naive": 0.0, "sim": np.zeros(n), "act_actual": 0.0, "act_pred": 0.0}
        for s in sorted(agg):
            a = agg[s]; q5, q95 = np.quantile(a["sim"], [0.05, 0.95])
            by_season.append({"cutoff": T, "segment": sg, "season": s, "seasons_ahead": s - T, "cohort_members": a["members"],
                              "actual_revenue": a["actual"], "predicted_revenue": a["pred"], "error_pct": a["pred"] / a["actual"] - 1 if a["actual"] else np.nan,
                              "naive_revenue": a["naive"], "naive_error_pct": a["naive"] / a["actual"] - 1 if a["actual"] else np.nan,
                              "mc_p5": q5, "mc_p95": q95, "inside_90pct_interval": bool(q5 <= a["actual"] <= q95),
                              "actual_active": a["act_actual"], "predicted_active": a["act_pred"],
                              "active_error_pct": a["act_pred"] / a["act_actual"] - 1 if a["act_actual"] else np.nan})
            for key in tot:
                tot[key] = tot[key] + a[key]
        q5, q95 = np.quantile(tot["sim"], [0.05, 0.95])
        summ.append({"cutoff": T, "segment": sg, "fitted_seasons": f"{int(tri.cohort.min())}-{T}", "held_out_seasons": f"{T + 1}-{LAST_COMPLETE}",
                     "held_out_cells": n_cells, "actual_revenue": tot["actual"], "predicted_revenue": tot["pred"],
                     "error_pct": tot["pred"] / tot["actual"] - 1 if tot["actual"] else np.nan,
                     "naive_error_pct": tot["naive"] / tot["actual"] - 1 if tot["actual"] else np.nan,
                     "weighted_mape_cells": w_ape / w_sum if w_sum else np.nan, "cells_inside_90pct": inside / n_cells if n_cells else np.nan,
                     "mc_p5": q5, "mc_p95": q95, "total_inside_90pct": bool(q5 <= tot["actual"] <= q95),
                     "active_error_pct": tot["act_pred"] / tot["act_actual"] - 1 if tot["act_actual"] else np.nan,
                     "observed_ages_at_cutoff": int(t_T.age.max()), "ltv_10y_8pct_at_cutoff": ltv_T[sg]})
    return cells, by_season, summ, ltv_T


# ---------------------------------------------------------------------------------------------------- tornado
def setting(inp, d, kind, q):
    """Base draw with one input group at the q-quantile of its Monte Carlo draws (ages in the group move together)."""
    b = base_draw(inp)
    if kind == TYPES[0]:
        b["m0"][:] = np.quantile(d["m0"], q)
    elif kind == TYPES[1]:
        b["f"][0, 1] = np.quantile(np.minimum(1.0, d["f"][:, 1]), q)
    elif kind == TYPES[2]:
        for k in range(2, inp["L"] + 1):
            b["f"][0, k] = np.quantile(np.minimum(1.0, d["f"][:, k]), q)
    elif kind == TYPES[3]:
        for k in range(1, inp["L"] + 1):
            b["g"][0, k] = np.quantile(d["g"][:, k], q)
    return b


def tornado(inps, draws, K, mix, eqd, base_new, base_eq):
    bd = {sg: base_draw(inps[sg]) for sg in inps}

    def run(kind, q):
        dr = {**bd, **{sg: setting(inps[sg], draws[sg], kind, q) for sg in SEGMENTS}}
        return outputs(inps, dr, K, BASE_HORIZON, BASE_RATE, mix, eqd)

    labels = {TYPES[0]: "First-season spend per member", TYPES[1]: "Retention factor, age 1 (season 1 to 2)",
              TYPES[2]: "Retention factors, ages 2+ (and the tail)", TYPES[3]: "Spend per active member, factors"}
    rows = []
    for kind in TYPES:
        lo, hi = run(kind, 0.05), run(kind, 0.95)
        rows.append({"input": labels[kind], "low_setting": "p5 of the Monte Carlo draws", "high_setting": "p95 of the Monte Carlo draws",
                     "new_low": float(lo[0][0]), "new_high": float(hi[0][0]), "eq_low": float(lo[1][0]), "eq_high": float(hi[1][0])})
    lo = outputs(inps, bd, K, BASE_HORIZON, 0.12, mix, eqd); hi = outputs(inps, bd, K, BASE_HORIZON, 0.06, mix, eqd)
    rows.append({"input": "Discount rate", "low_setting": "12%", "high_setting": "6%", "new_low": float(lo[0][0]), "new_high": float(hi[0][0]),
                 "eq_low": float(lo[1][0]), "eq_high": float(hi[1][0])})
    lo = outputs(inps, bd, K, 5, BASE_RATE, mix, eqd); hi = outputs(inps, bd, K, 15, BASE_RATE, mix, eqd)
    rows.append({"input": "Horizon", "low_setting": "5 seasons", "high_setting": "15 seasons", "new_low": float(lo[0][0]), "new_high": float(hi[0][0]),
                 "eq_low": float(lo[1][0]), "eq_high": float(hi[1][0])})
    hi = outputs(inps, bd, K, BASE_HORIZON, BASE_RATE, mix, eqd, growth=0.03)
    rows.append({"input": "Price growth per season", "low_setting": "0% (base, nominal prices flat)", "high_setting": "+3% a season",
                 "new_low": base_new, "new_high": float(hi[0][0]), "eq_low": base_eq, "eq_high": float(hi[1][0])})
    mix_lo = pd.Series({SEGMENTS[2]: 1.0})
    mix_hi = mix.copy(); mix_hi[SEGMENTS[0]] *= 2; mix_hi[SEGMENTS[1]] *= 2; mix_hi[SEGMENTS[2]] = 1 - mix_hi[SEGMENTS[0]] - mix_hi[SEGMENTS[1]]
    lo = outputs(inps, bd, K, BASE_HORIZON, BASE_RATE, mix_lo, eqd); hi = outputs(inps, bd, K, BASE_HORIZON, BASE_RATE, mix_hi, eqd)
    rows.append({"input": "New-buyer mix", "low_setting": "100% single-ticket buyers",
                 "high_setting": f"plan and waitlist shares doubled ({mix_hi[SEGMENTS[0]]:.2%} plan, {mix_hi[SEGMENTS[1]]:.2%} waitlist)",
                 "new_low": float(lo[0][0]), "new_high": float(hi[0][0]), "eq_low": np.nan, "eq_high": np.nan})
    lo = outputs(inps, bd, K, BASE_HORIZON, BASE_RATE, mix, eqd, recent=True)
    rows.append({"input": "Cohort basis", "low_setting": f"{RECENT_FROM}+ cohorts for waitlist and single-ticket", "high_setting": "all cohorts (base)",
                 "new_low": float(lo[0][0]), "new_high": base_new, "eq_low": float(lo[1][0]), "eq_high": base_eq})
    out = []
    for target, lo_c, hi_c, base in [("new_buyer_ltv", "new_low", "new_high", base_new), ("customer_equity", "eq_low", "eq_high", base_eq)]:
        for r_ in rows:
            if np.isnan(r_[lo_c]):
                continue
            out.append({"output": target, "input": r_["input"], "low_setting": r_["low_setting"], "high_setting": r_["high_setting"],
                        "base": base, "value_at_low": r_[lo_c], "value_at_high": r_[hi_c], "swing": abs(r_[hi_c] - r_[lo_c]),
                        "swing_pct_of_base": abs(r_[hi_c] - r_[lo_c]) / base})
    Tn = pd.DataFrame(out)
    return Tn.sort_values(["output", "swing"], ascending=[True, False]).reset_index(drop=True)


# ------------------------------------------------------------------------------------------------------ sobol
def sobol(inps, K, mix, eqd, n, rng):
    Ad = {sg: draw(inps[sg], n, rng) for sg in SEGMENTS}
    Bd = {sg: draw(inps[sg], n, rng) for sg in SEGMENTS}

    def Y(dr):
        new, eq, _ = outputs(inps, dr, K, BASE_HORIZON, BASE_RATE, mix, eqd)
        return np.column_stack([new, eq])

    def swap(sgs, kinds):
        dr = {sg: {k: v.copy() for k, v in Ad[sg].items()} for sg in SEGMENTS}
        for sg in sgs:
            for kind in kinds:
                if kind == TYPES[0]:
                    dr[sg]["m0"] = Bd[sg]["m0"].copy()
                elif kind == TYPES[1]:
                    dr[sg]["f"][:, 1] = Bd[sg]["f"][:, 1]
                elif kind == TYPES[2]:
                    dr[sg]["f"][:, 2:] = Bd[sg]["f"][:, 2:]
                elif kind == TYPES[3]:
                    dr[sg]["g"][:, 1:] = Bd[sg]["g"][:, 1:]
        return dr

    YA, YB = Y(Ad), Y(Bd)
    V = np.var(np.vstack([YA, YB]), axis=0)
    groups = [("input type", kind, SEGMENTS, [kind]) for kind in TYPES] + \
             [("segment", sg, [sg], TYPES) for sg in SEGMENTS] + \
             [("input type x segment", f"{kind}, {sg}", [sg], [kind]) for sg in SEGMENTS for kind in TYPES]
    rows = []
    for level, name, sgs, kinds in groups:
        YAB = Y(swap(sgs, kinds))
        S = (YB * (YAB - YA)).mean(axis=0) / V
        Tt = ((YA - YAB) ** 2).mean(axis=0) / (2 * V)
        rows.append({"level": level, "group": name, "first_order_new_buyer_ltv": float(S[0]), "total_effect_new_buyer_ltv": float(Tt[0]),
                     "first_order_customer_equity": float(S[1]), "total_effect_customer_equity": float(Tt[1])})
    return pd.DataFrame(rows), {"new_buyer_ltv_sd": float(np.sqrt(V[0])), "customer_equity_sd": float(np.sqrt(V[1])), "sims": n}


# ------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=A.DEFAULT_DIR)
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--sobol-sims", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=20260927)
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
    H = BASE_HORIZON
    K = int(eqd.age.max()) + max(HORIZONS) + 2
    A.log(f"LTV validation: inputs {os.path.basename(ltv_path)}, {os.path.basename(st_path)}; seed {a.seed}")

    # deterministic replica of the point estimates, so the tornado starts from the published numbers
    inps = {sg: fit_inputs(tri[tri.segment.eq(sg)]) for sg in SEGMENTS + list(RECENT.values())}
    bd = {sg: base_draw(inps[sg]) for sg in inps}
    new_det, eq_det, ltv_det = outputs(inps, bd, K, H, BASE_RATE, mix, eqd)
    rec = [{"metric": f"LTV per new {sg.lower()}, {H} seasons at {BASE_RATE:.0%}", "this_script": float(ltv_det[sg][0]),
            "published": float(base_ltv.loc[sg, "ltv"]), "source": "fan_ltv LTV_Base_Case"} for sg in SEGMENTS]
    rec.append({"metric": "LTV of a new buyer today, 2023-2025 mix", "this_script": float(new_det[0]), "published": float(bl.blended_ltv_new_buyer),
                "source": "fan_ltv LTV_Blended_New_Buyer"})
    rec.append({"metric": f"Customer equity, all individual accounts active in {LAST_COMPLETE}", "this_script": float(eq_det[0]), "published": eq_point,
                "source": "value_statements 5_Customer_Equity"})
    Rc = pd.DataFrame(rec); Rc["difference_pct"] = Rc.this_script / Rc.published - 1
    base_new, base_eq = float(new_det[0]), float(eq_det[0])

    # 1 backtest
    cells, by_season, summ, ltv_rows = [], [], [], []
    for T in CUTOFFS:
        c_, s_, m_, ltv_T = backtest(tri, T, H, a.sims, rng)
        cells += c_; by_season += s_; summ += m_
        n0 = tri[(tri.age == 0) & tri.cohort.between(T - 2, T) & tri.segment.isin(SEGMENTS)].groupby("segment").cohort_size.sum()
        mix_T = n0 / n0.sum()
        ltv_rows.append({"data_through": T, "observed_ages": T - int(tri.cohort.min()), **{sg: ltv_T.get(sg, np.nan) for sg in SEGMENTS + [ALL]},
                         "new_buyer_blended": float(sum(mix_T.get(sg, 0) * ltv_T.get(sg, np.nan) for sg in SEGMENTS)),
                         "new_buyer_mix_note": ", ".join(f"{sg} {mix_T.get(sg, 0):.1%}" for sg in SEGMENTS)})
    ltv_rows.append({"data_through": LAST_COMPLETE, "observed_ages": LAST_COMPLETE - int(tri.cohort.min()),
                     **{sg: float(base_ltv.loc[sg, "ltv"]) for sg in SEGMENTS + [ALL]}, "new_buyer_blended": float(bl.blended_ltv_new_buyer),
                     "new_buyer_mix_note": ", ".join(f"{sg} {mix.get(sg, 0):.1%}" for sg in SEGMENTS) + " (published)"})
    Bs, Bq, Bc, Bl = pd.DataFrame(summ), pd.DataFrame(by_season), pd.DataFrame(cells), pd.DataFrame(ltv_rows)

    # 2 tornado
    draws = {sg: draw(inps[sg], a.sims, rng) for sg in inps}
    Tn = tornado(inps, draws, K, mix, eqd, base_new, base_eq)

    # 3 sobol
    Sb, sb_meta = sobol(inps, K, mix, eqd, a.sobol_sims, rng)

    notes = pd.DataFrame({"note": [
        "Backtest: for each cutoff T the chain ladder and the Monte Carlo are refitted on the cohort x age cells observed by the end of season T "
        "(cohort + age <= T); each cohort is projected from its own last observed value with the pooled factors, and the held-out seasons T+1 to 2025 "
        "are compared with what happened.",
        "Naive benchmark: the cohort's last observed revenue per member carried forward unchanged.",
        "Cells inside 90%: share of held-out cohort x season cells whose actual revenue per member lies between the Monte Carlo p5 and p95 (refitted at "
        "the cutoff). Well-calibrated intervals cover about 90%.",
        "Weighted MAPE: absolute percentage error per cell weighted by the cell's actual revenue.",
        "With data through 2022 only one development factor exists per segment, so the tail rule (which skips the structural drop after season 1) cannot "
        "apply and the 2022 cutoff projects the age-1 drop forever: it shows what one season of history is worth, not a flaw in the later fits.",
        "LTV at cutoff: the 10-season, 8% LTV as it would have been estimated with data through that season; the new-buyer blend uses the mix of the "
        "last three cohorts observed at that cutoff.",
        "Tornado: one input group moved from its low to its high setting with everything else at base. Low/high for model inputs are the 5th and 95th "
        "percentiles of the Monte Carlo draws for that input (Beta sampling error times lognormal cohort heterogeneity); ages within a group move together.",
        "Sobol indices: first-order = share of output variance explained by that input group alone; total effect = share removed when the group is "
        "frozen, including interactions. Pick-freeze estimators (Saltelli et al. 2010 for first order, Jansen 1999 for total effect) on independent "
        "draw matrices; small negative values are estimation noise.",
        "Deterministic replica: the script rebuilds the chain ladder from the Monte Carlo's pooled inputs; the Reconciliation sheet shows how closely "
        "that reproduces fan_ltv and value_statements.",
        "Gross ticketing revenue at nominal prices, no reactivation of lapsed accounts: the same model as fan_ltv and ltv_montecarlo.",
    ]})
    out = os.path.join(a.data_dir, f"ltv_validation_{dt.date.today():%Y%m%d}.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        Bs.to_excel(xw, sheet_name="Backtest_Summary", index=False)
        Bq.to_excel(xw, sheet_name="Backtest_by_Season", index=False)
        Bl.to_excel(xw, sheet_name="Backtest_LTV_at_Cutoff", index=False)
        Bc.to_excel(xw, sheet_name="Backtest_Cells", index=False)
        Tn.to_excel(xw, sheet_name="Tornado", index=False)
        Sb.to_excel(xw, sheet_name="Sobol_Indices", index=False)
        Rc.to_excel(xw, sheet_name="Reconciliation", index=False)
        notes.to_excel(xw, sheet_name="Notes", index=False)
    A.autofilter(out)
    A.log(f"Wrote {out}  (aggregate only)")
    if a.json_out:
        js = lambda df: json.loads(df.to_json(orient="records"))
        json.dump({"backtest_summary": js(Bs), "backtest_by_season": js(Bq), "backtest_ltv": js(Bl), "backtest_cells": js(Bc),
                   "tornado": js(Tn), "sobol": js(Sb), "sobol_meta": sb_meta, "reconciliation": js(Rc), "notes": notes.note.tolist(),
                   "base": {"new_buyer_ltv": base_new, "customer_equity": base_eq, "horizon": H, "rate": BASE_RATE},
                   "sims": a.sims, "seed": a.seed, "source": os.path.basename(out)}, open(a.json_out, "w"), indent=1)
        A.log(f"Wrote {a.json_out}")

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 200)
    print("\nRECONCILIATION"); print(Rc.round(4).to_string(index=False))
    print("\nBACKTEST SUMMARY")
    print(Bs[["cutoff", "segment", "held_out_cells", "actual_revenue", "predicted_revenue", "error_pct", "naive_error_pct", "weighted_mape_cells",
              "cells_inside_90pct", "total_inside_90pct", "active_error_pct", "ltv_10y_8pct_at_cutoff"]].round(3).to_string(index=False))
    print("\nBACKTEST BY SEASON")
    print(Bq[["cutoff", "segment", "season", "seasons_ahead", "actual_revenue", "predicted_revenue", "error_pct", "naive_error_pct", "mc_p5", "mc_p95",
              "inside_90pct_interval", "active_error_pct"]].round(3).to_string(index=False))
    print("\nLTV AT CUTOFF"); print(Bl.round(0).to_string(index=False))
    print("\nTORNADO"); print(Tn.round(3).to_string(index=False))
    print("\nSOBOL"); print(Sb.round(3).to_string(index=False)); print(sb_meta)


if __name__ == "__main__":
    main()
