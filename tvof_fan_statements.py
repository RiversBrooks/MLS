"""
Austin FC TVOF | Five value statements (v1, 2026-09-26)
-------------------------------------------------------------
Built from Sales History (club revenue counted once, the same rule as tvof_fan_value.py and
tvof_fan_ltv.py), the audit's event crosswalk (match season = event year), the Attendance scans
and the cohort curves in fan_ltv_<date>.xlsx.

  1  Fan contribution statement   line items per first-season segment, totals and per account
  2  Cohort revenue statement     cohort x seasons-since-first-purchase triangle (individuals)
  3  Revenue bridge 2022 -> 2025  Austin FC match tickets by match season: volume, mix, price; resale memo
  4  Capacity utilization         per season and per match: capacity, sold, scanned, unused by path and type
  5  Customer equity              remaining LTV of the 2025 active base by segment and age

Seasons: plan products take the year in their name; match tickets take the event year from the
crosswalk (High/Medium), plan game rows without one take their plan row's season; everything
else takes the transaction year with December rolled forward. Seasons floor at 2021.

Usage:
    python tvof_fan_statements.py [--data-dir D] [--json-out statements.json]

Output: value_statements_<date>.xlsx in the data folder (aggregate only).
"""
import argparse, datetime as dt, glob, json, os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tvof_data_audit as A   # noqa: E402
from tvof_fan_ltv import FIRST_SEASON, LAST_COMPLETE, GROUP_MAX_ITEMS, SEGMENTS, BASE_RATE, BASE_HORIZON, RATES, HORIZONS   # noqa: E402

BRIDGE_FROM, BRIDGE_TO = 2022, 2025
GROUP_SEG = "Group / corporate"
TYPES = ["plan", "single", "group", "comp", "hospitality"]
LINES = [("plan", "Plan memberships (season, partial, COI)"), ("wait", "Waitlist fees"), ("match_single", "Single Austin FC match tickets"),
         ("parking", "Parking"), ("other_event", "Other stadium events"), ("rest", "Add-ons, Austin FC II and other")]


def ptg_class(v):
    v = str(v).lower()
    if "comp" in v:
        return "comp"
    if "hospitality" in v or "suite" in v or "loft" in v:
        return "hospitality"
    if "season" in v or "plan" in v or "member" in v or "coi" in v or "partial" in v or "half" in v:
        return "plan"
    if "group" in v:
        return "group"
    return "single"


def newest(data_dir, prefix):
    c = sorted(f for f in glob.glob(os.path.join(data_dir, prefix + "_*.xlsx")) if "LOCAL_ONLY" not in f)
    if not c:
        sys.exit(f"{prefix}_<date>.xlsx not found in {data_dir}")
    return c[-1]


def by_cat(series, fn, default=np.nan):
    """Apply fn to each category once and expand to rows (categorical column, no string copy per row)."""
    cats = series.cat.categories
    vals = np.array([fn(c) for c in cats], dtype=object)
    codes = series.cat.codes.to_numpy()
    out = np.full(len(series), default, dtype=object)
    ok = codes >= 0
    out[ok] = vals[codes[ok]]
    return out


def lookup(keys, table, default=None):
    """Exact lookup of hash keys in a Series index (no Series.map: it rounds UInt64 through float64)."""
    pos = pd.Index(table.index).get_indexer(pd.Index(keys))
    vals = table.to_numpy(dtype=object)
    return np.where(pos >= 0, vals[np.clip(pos, 0, None)], default)


def norm_cat(series):
    cats = pd.Series(series.cat.categories.astype(str))
    normed = A._seat_norm(cats).to_numpy(dtype=object)
    codes = series.cat.codes.to_numpy()
    out = np.full(len(series), None, dtype=object)
    ok = codes >= 0
    out[ok] = normed[codes[ok]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=A.DEFAULT_DIR)
    ap.add_argument("--chunksize", type=int, default=A.CHUNK)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()
    files = A.find_files(a.data_dir)
    for k in ("sales", "attendance"):
        if k not in files:
            sys.exit(f"{k} file not found in {a.data_dir}")
    audit_path, ltv_path = newest(a.data_dir, "TVOF_audit_results"), newest(a.data_dir, "fan_ltv")

    xw = pd.read_excel(audit_path, sheet_name="Event_Crosswalk")
    tr = xw[xw.confidence.isin(["High", "Medium", "Manual override"]) & xw.product_class.eq("Austin FC match")]
    p2e = dict(zip(tr.product_id.astype(str), tr.EventKey.astype(int).astype(str)))
    p2y = dict(zip(tr.product_id.astype(str), pd.to_datetime(tr.event_date).dt.year.astype(int)))
    evname = dict(zip(tr.EventKey.astype(int).astype(str), tr.attendance_MasterEventName.astype(str)))
    A.log(f"Crosswalk: {len(p2e)} match products mapped to {len(evname)} events ({os.path.basename(audit_path)})")

    # ------------------------------------------------------------------ sales, row level
    A.log("Loading sales ...")
    s = A.load("sales", files["sales"], a.chunksize)
    it = s.item_type.astype(object).str.lower().to_numpy()
    sub = s.subscription_instance_id
    sub_na = sub.isna().to_numpy()
    pay = s.total_payment.fillna(0).to_numpy(dtype=float)
    plan_ids = set(sub[it == "subscription"].dropna().to_numpy())
    orphan = (it == "ticket") & ~sub_na & ~sub.isin(plan_ids).to_numpy()
    isplan = (it == "subscription") | orphan
    game = (it == "ticket") & ~sub_na & ~orphan
    cls = by_cat(s.product_description, A.product_class, default="")
    dyr = by_cat(s.product_description, lambda d: float(str(d)[:4]) if str(d)[:2] == "20" and str(d)[:4].isdigit() else np.nan).astype(float)
    wait = by_cat(s.product_description, lambda d: "waitlist" in str(d).lower(), default=False).astype(bool)
    evyr = by_cat(s.product_id, lambda p: p2y.get(str(p), np.nan)).astype(float)
    evkey = by_cat(s.product_id, lambda p: p2e.get(str(p)), default=None)
    match = cls == "Austin FC match"
    t = s.transaction_date
    tyr = (t.dt.year + (t.dt.month == 12)).to_numpy(dtype=float)
    season = np.where(isplan & ~np.isnan(dyr), dyr, tyr)
    season = np.where(match & ~np.isnan(evyr), evyr, season)
    # plan game rows without a crosswalk year take their plan row's season
    ps = pd.Series(season[it == "subscription"], index=pd.Index(sub[it == "subscription"])).groupby(level=0).max()
    need = game & np.isnan(evyr)
    idx = pd.Index(ps.index).get_indexer(pd.Index(sub[need]))
    season[need] = np.where(idx >= 0, ps.to_numpy()[np.clip(idx, 0, None)], season[need])
    season = np.where(np.isnan(season), tyr, season)
    season = np.maximum(np.nan_to_num(season, nan=0), FIRST_SEASON).astype(int)

    club = np.where(isplan | ((it == "ticket") & sub_na), pay, 0.0)
    R = pd.DataFrame({"acct": s.internal_account_id.array, "season": season,
                      "plan": np.where(isplan & ~wait, pay, 0.0),
                      "match_single": np.where((it == "ticket") & sub_na & match, pay, 0.0),
                      "parking": np.where((it == "ticket") & sub_na & (cls == "parking"), pay, 0.0),
                      "other_event": np.where((it == "ticket") & sub_na & (cls == "other stadium event"), pay, 0.0),
                      "club": club, "secondary": np.where(it == "resale", pay, 0.0),
                      "wait": np.where(isplan & wait, pay, 0.0), "transfer": (it == "transfer").astype(int),
                      "prod": s.product_id.cat.codes.to_numpy()})
    R["rest"] = R.club - R.plan - R.wait - R.match_single - R.parking - R.other_event
    R = R[R.acct.notna()]
    A.log(f"  club revenue ${R.club.sum():,.0f}; secondary ${R.secondary.sum():,.0f}; rows with an account {len(R):,}")

    # account attributes: group flag, cohort, first-season segment
    mx = R[R.club != 0].groupby(["acct", "prod"]).size().groupby(level=0).max()
    group_accts = pd.Index(mx[mx > GROUP_MAX_ITEMS].index)
    AS = R.groupby(["acct", "season"]).agg(rev=("club", "sum"), plan=("plan", "sum"), match_single=("match_single", "sum"),
                                           parking=("parking", "sum"), other_event=("other_event", "sum"), rest=("rest", "sum"),
                                           secondary=("secondary", "sum"), wait=("wait", "sum"), transfers=("transfer", "sum")).reset_index()
    paying = AS[AS.rev > 0]
    cohort = paying.groupby("acct").season.min()
    f0 = paying[paying.season.to_numpy() == cohort.reindex(paying.acct).to_numpy()].set_index("acct")
    seg0 = pd.Series(np.select([f0.plan > 0, f0.rev - f0.wait > 0], [SEGMENTS[0], SEGMENTS[2]], SEGMENTS[1]), index=f0.index)
    seg = seg0.copy()
    seg[seg.index.isin(group_accts)] = GROUP_SEG
    AS["segment"] = lookup(AS.acct, seg)
    AS["cohort"] = pd.to_numeric(lookup(AS.acct, cohort), errors="coerce")
    AS["age"] = AS.season - AS.cohort
    A.log(f"  {len(cohort):,} paying accounts: " + ", ".join(f"{k} {v:,}" for k, v in seg.value_counts().items()))

    # ------------------------------------------------------------------ 1. fan contribution statement
    acc = AS.groupby("acct").agg(plan=("plan", "sum"), wait=("wait", "sum"), match_single=("match_single", "sum"), parking=("parking", "sum"),
                                 other_event=("other_event", "sum"), rest=("rest", "sum"), club=("rev", "sum"),
                                 secondary=("secondary", "sum"), transfers=("transfers", "sum"),
                                 paying_seasons=("rev", lambda x: int((x > 0).sum())))
    acc["segment"] = seg.reindex(acc.index)
    acc["per_season"] = acc.club / acc.paying_seasons.replace(0, np.nan)
    cols = SEGMENTS + [GROUP_SEG]
    rows = []

    def line(label, fn, kind="money", indent=False, total=False):
        d = {"line_item": ("  " if indent else "") + label, "kind": kind, "total_row": total}
        for c in cols + ["All accounts"]:
            x = acc[acc.segment.eq(c)] if c != "All accounts" else acc[acc.club > 0]
            d[c] = fn(x)
        rows.append(d)

    for key, label in LINES:
        line(label, lambda x, k=key: x[k].sum(), indent=True)
    line("Club revenue, counted once", lambda x: x.club.sum(), total=True)
    line("Secondary market spend, paid to sellers (memo)", lambda x: x.secondary.sum())
    line("Transfer rows received, $0 (memo)", lambda x: int(x.transfers.sum()), kind="count")
    line("Paying accounts", lambda x: int(len(x)), kind="count")
    line("Paying seasons per account, mean", lambda x: x.paying_seasons.mean(), kind="num")
    line("Club revenue per account, mean", lambda x: x.club.mean())
    line("Club revenue per account, median", lambda x: x.club.median())
    line("Club revenue per paying season, mean", lambda x: x.per_season.mean())
    line("Club revenue per paying season, median", lambda x: x.per_season.median())
    line("Secondary spend per account, mean", lambda x: x.secondary.mean())
    for key, label in LINES:
        line(label + ", per account", lambda x, k=key: x[k].mean(), indent=True)
    zero = int((acc.club <= 0).sum())
    contrib = pd.DataFrame(rows)
    season_rev = AS.groupby("season").agg(plan=("plan", "sum"), wait=("wait", "sum"), match_single=("match_single", "sum"), parking=("parking", "sum"),
                                          other_event=("other_event", "sum"), rest=("rest", "sum"), club=("rev", "sum"),
                                          secondary=("secondary", "sum"), paying_accounts=("rev", lambda x: int((x > 0).sum()))).reset_index()

    # ------------------------------------------------------------------ 2. cohort revenue statement
    ind = AS[~AS.segment.eq(GROUP_SEG)]
    def triangle(x):
        x = x[(x.age >= 0) & (x.cohort <= LAST_COMPLETE + 1)]   # seasons before the first payment carry no revenue; 2027 renewals are not a cohort yet
        n = x[(x.age == 0) & (x.rev > 0)].groupby("cohort").size()
        act = x[x.rev > 0].groupby(["cohort", "age"]).size().unstack()
        rev = x.groupby(["cohort", "age"]).rev.sum().unstack()
        out = []
        for c in n.index:
            for k in rev.columns:
                if c + k > LAST_COMPLETE + 1:
                    continue
                r = rev.loc[c, k] if pd.notna(rev.loc[c, k]) else 0.0
                out.append({"cohort": int(c), "age": int(k), "season": int(c + k), "cohort_size": int(n[c]),
                            "active": int(act.loc[c, k]) if (k in act.columns and pd.notna(act.loc[c, k])) else 0,
                            "share_active": (act.loc[c, k] / n[c]) if (k in act.columns and pd.notna(act.loc[c, k])) else 0.0,
                            "revenue": float(r), "revenue_per_member": float(r) / n[c],
                            "partial": bool(c + k > LAST_COMPLETE)})
        return pd.DataFrame(out)
    tri_all = triangle(ind)
    tri_seg = pd.concat([triangle(ind[ind.segment.eq(sg)]).assign(segment=sg) for sg in SEGMENTS], ignore_index=True)

    # ------------------------------------------------------------------ 3. revenue bridge, match tickets by season
    ptg = by_cat(s.price_type_group, ptg_class, default="single")
    stype = by_cat(s.sale_type, lambda v: str(v).lower(), default="")
    acct_arr = s.internal_account_id
    gpos = group_accts.get_indexer(pd.Index(acct_arr)) >= 0
    typ = np.where(~sub_na, "plan", np.where(gpos, "group", ptg))
    typ = np.where((typ == "single") & (pay == 0), "comp", typ)
    tk = (it == "ticket") & match & (season >= BRIDGE_FROM) & (season <= LAST_COMPLETE + 1)
    conf = tk & (stype == "reservation confirmation")
    A.log(f"  match ticket rows {tk.sum():,}, of which {conf.sum():,} reservation confirmations (plan holders' reserved cup and playoff seats, "
          f"${pay[conf].sum():,.0f}) are kept: they are plan game rows like any other")
    B = pd.DataFrame({"season": season[tk], "type": typ[tk], "pay": pay[tk]})
    det = B.groupby(["season", "type"]).agg(seats=("pay", "size"), revenue=("pay", "sum")).reset_index()
    det["price"] = det.revenue / det.seats
    resale = pd.DataFrame({"season": season[(it == "resale") & match], "pay": pay[(it == "resale") & match]}).groupby("season").pay.sum()
    transfers = pd.Series(season[(it == "transfer") & match]).value_counts()

    def pvm(y0, y1):
        d0 = det[det.season == y0].set_index("type"); d1 = det[det.season == y1].set_index("type")
        types = sorted(set(d0.index) | set(d1.index), key=lambda x: TYPES.index(x) if x in TYPES else 9)
        q0 = d0.seats.reindex(types).fillna(0); q1 = d1.seats.reindex(types).fillna(0)
        p0 = (d0.revenue / d0.seats).reindex(types).fillna(0); p1 = (d1.revenue / d1.seats).reindex(types).fillna(0)
        r0, r1 = float((q0 * p0).sum()), float((q1 * p1).sum())
        pavg0 = r0 / q0.sum()
        vol = (q1.sum() - q0.sum()) * pavg0
        mix = float((q1 * p0).sum()) - q1.sum() * pavg0
        price = float((q1 * (p1 - p0)).sum())
        mix_t = (q1 - q1.sum() * q0 / q0.sum()) * p0
        price_t = q1 * (p1 - p0)
        steps = [{"step": f"Match-ticket revenue, {y0} season", "amount": r0, "kind": "total"},
                 {"step": "Volume: seats sold", "amount": vol, "kind": "effect",
                  "note": f"{int(q0.sum()):,} to {int(q1.sum()):,} seats at the {y0} average of ${pavg0:,.2f}"},
                 {"step": "Mix: plan, single, group, comp, hospitality", "amount": mix, "kind": "effect",
                  "note": "; ".join(f"{t} {mix_t[t]:+,.0f}" for t in types)},
                 {"step": "Price: payment per seat within each type", "amount": price, "kind": "effect",
                  "note": "; ".join(f"{t} ${p0[t]:,.2f} to ${p1[t]:,.2f}" for t in types if q0[t] > 0 and q1[t] > 0 and p0[t] > 0)},
                 {"step": f"Match-ticket revenue, {y1} season", "amount": r1, "kind": "total"},
                 {"step": f"Change {y0} to {y1}", "amount": r1 - r0, "kind": "change", "note": f"{(r1 / r0 - 1) * 100:+.1f}%"},
                 {"step": "Memo: resale payments to sellers, not club revenue", "amount": float(resale.get(y1, 0) - resale.get(y0, 0)), "kind": "memo",
                  "note": f"${resale.get(y0, 0):,.0f} to ${resale.get(y1, 0):,.0f}"},
                 {"step": "Memo: transfer rows, $0", "amount": float(transfers.get(y1, 0) - transfers.get(y0, 0)), "kind": "memo",
                  "note": f"{int(transfers.get(y0, 0)):,} to {int(transfers.get(y1, 0)):,} rows"}]
        assert abs(vol + mix + price - (r1 - r0)) < 1.0
        return pd.DataFrame(steps)
    bridge = pvm(BRIDGE_FROM, BRIDGE_TO)
    det["resale_memo"] = det.season.map(resale)

    # ------------------------------------------------------------------ 4. capacity utilization
    A.log("Capacity: final holder per seat and event ...")
    hold = A._holder_mask(s).to_numpy() & pd.notna(evkey)
    H = pd.DataFrame({"EventKey": evkey[hold], "sec": norm_cat(s.section)[hold], "row": norm_cat(s.row)[hold], "seat": norm_cat(s.seat)[hold],
                      "td": t.to_numpy()[hold], "path": s.item_type.astype(object).to_numpy()[hold], "type": typ[hold], "pay": pay[hold]})
    H = H.dropna(subset=["sec", "row", "seat"]).sort_values("td").drop_duplicates(["EventKey", "sec", "row", "seat"], keep="last")
    H["season"] = H.EventKey.map(evname).str[:4].astype(int)
    del s
    att = pd.read_csv(files["attendance"], usecols=["EventKey", "SectionName", "RowName", "SeatName"], dtype="category")
    att["EventKey"] = att.EventKey.astype(str)
    att = att[att.EventKey.isin(set(evname))]
    for c_out, c_in in (("sec", "SectionName"), ("row", "RowName"), ("seat", "SeatName")):
        att[c_out] = norm_cat(att[c_in])
    AU = att.dropna(subset=["sec", "row", "seat"]).drop_duplicates(["EventKey", "sec", "row", "seat"])
    scans_by_event = AU.groupby("EventKey").size()
    H = H.merge(AU[["EventKey", "sec", "row", "seat"]].assign(scanned=True), on=["EventKey", "sec", "row", "seat"], how="left")
    H["scanned"] = H.scanned.fillna(False).astype(bool)
    H["path"] = H.path.str.capitalize()

    def cap_table(by, F):
        g = F.groupby(by)
        out = pd.DataFrame({"matches": g.EventKey.nunique(), "sold": g.size(), "scanned_of_sold": g.scanned.sum()})
        out["capacity"] = out.matches * A.Q2_CAPACITY
        for pth, lbl in (("Ticket", "unused_kept_by_buyer"), ("Transfer", "unused_transferred"), ("Resale", "unused_resold")):
            out[lbl] = F[~F.scanned & F.path.eq(pth)].groupby(by).size().reindex(out.index).fillna(0).astype(int)
        for tp in TYPES:
            out[f"unused_{tp}"] = F[~F.scanned & F.type.eq(tp)].groupby(by).size().reindex(out.index).fillna(0).astype(int)
        out["unused"] = out.sold - out.scanned_of_sold
        return out
    no_scan = {k for k in evname if int(scans_by_event.get(k, 0)) < 1000}   # events with no usable scan data (e.g. the 2022 Atlas friendly)
    cap_s = cap_table("season", H[~H.EventKey.isin(no_scan)])
    cap_s["scanned_entries"] = [int(scans_by_event.reindex([k for k, n in evname.items() if n[:4] == str(y) and k not in no_scan]).fillna(0).sum()) for y in cap_s.index]
    cap_s["utilization"] = cap_s.scanned_entries / cap_s.capacity
    cap_s["sell_through"] = cap_s.sold / cap_s.capacity
    cap_s["scan_rate_of_sold"] = cap_s.scanned_of_sold / cap_s.sold
    cap_s = cap_s.reset_index()
    cap_m = cap_table("EventKey", H)
    cap_m["scan_data"] = ~cap_m.index.isin(no_scan)
    cap_m["event"] = cap_m.index.map(evname)
    cap_m["date"] = cap_m.event.str[:10]
    cap_m["opponent"] = cap_m.event.str[11:]
    cap_m["scanned_entries"] = scans_by_event.reindex(cap_m.index).fillna(0).astype(int)
    cap_m["utilization"] = cap_m.scanned_entries / A.Q2_CAPACITY
    cap_m["scan_rate_of_sold"] = cap_m.scanned_of_sold / cap_m.sold
    cap_m = cap_m.sort_values("date").reset_index()
    A.log(f"  {len(H):,} holder seats at {H.EventKey.nunique()} events; scan rate of sold {H.scanned.mean():.1%}")

    # ------------------------------------------------------------------ 5. customer equity
    curves = pd.read_excel(ltv_path, sheet_name="Projected_Curves")
    active = ind[(ind.season == LAST_COMPLETE) & (ind.rev > 0)]
    banked = ind.groupby("acct").rev.sum()
    eq_rows, sens = [], {}

    def ext(seg_name, K):
        c = curves[curves.segment.eq(seg_name)].sort_values("age")
        r = c.share_active.to_numpy(dtype=float); v = c.revenue_per_starting_member.to_numpy(dtype=float)
        f = r[-1] / r[-2] if r[-2] else 1.0
        while len(r) < K:
            r = np.append(r, r[-1] * f); v = np.append(v, v[-1] * f)
        return r, v

    def remaining(seg_name, age, horizon, rate):
        r, v = ext(seg_name, age + horizon + 1)
        js = np.arange(1, horizon + 1)
        return float((v[age + js] / r[age] / (1 + rate) ** js).sum()) if r[age] else 0.0

    for sg in SEGMENTS:
        x = active[active.segment.eq(sg)]
        for age, g in x.groupby("age"):
            n = len(g)
            eq_rows.append({"segment": sg, "age": int(age), "first_season": int(LAST_COMPLETE - age), "active_accounts": n,
                            "banked_to_date": float(banked.reindex(g.acct).sum()),
                            "revenue_2025": float(g.rev.sum()),
                            "remaining_per_member": remaining(sg, int(age), BASE_HORIZON, BASE_RATE),
                            "remaining_total": n * remaining(sg, int(age), BASE_HORIZON, BASE_RATE)})
            for h in HORIZONS:
                for rt in RATES:
                    sens[(h, rt)] = sens.get((h, rt), 0.0) + n * remaining(sg, int(age), h, rt)
    E = pd.DataFrame(eq_rows)
    eq_seg = E.groupby("segment").agg(active_accounts=("active_accounts", "sum"), banked_to_date=("banked_to_date", "sum"),
                                      revenue_2025=("revenue_2025", "sum"), remaining_total=("remaining_total", "sum")).reindex(SEGMENTS).reset_index()
    eq_seg["remaining_per_member"] = eq_seg.remaining_total / eq_seg.active_accounts
    tot = {"segment": "All individual accounts active in 2025", "active_accounts": int(eq_seg.active_accounts.sum()),
           "banked_to_date": float(eq_seg.banked_to_date.sum()), "revenue_2025": float(eq_seg.revenue_2025.sum()),
           "remaining_total": float(eq_seg.remaining_total.sum())}
    tot["remaining_per_member"] = tot["remaining_total"] / tot["active_accounts"]
    eq_seg = pd.concat([eq_seg, pd.DataFrame([tot])], ignore_index=True)
    grp25 = AS[AS.segment.eq(GROUP_SEG) & (AS.season == LAST_COMPLETE) & (AS.rev > 0)]
    ind_cohort = cohort[~cohort.index.isin(group_accts)]
    lapsed = int((~ind_cohort.index.isin(active.acct) & (ind_cohort <= LAST_COMPLETE)).sum())
    eq_sens = pd.DataFrame([{"horizon_seasons": h, "discount_rate": rt, "customer_equity": v} for (h, rt), v in sorted(sens.items())])
    eq_memo = pd.DataFrame([
        {"measure": "Group / corporate accounts paying in 2025 (not projected)", "value": int(grp25.acct.nunique())},
        {"measure": "Their 2025 club revenue", "value": float(grp25.rev.sum())},
        {"measure": "Individual accounts that paid before 2025 but not in 2025 (lapsed, no value assigned)", "value": lapsed},
        {"measure": "Individual accounts whose first paying season is 2026 (not yet in the base)", "value": int((ind_cohort > LAST_COMPLETE).sum())},
    ])

    # ------------------------------------------------------------------ write
    notes = pd.DataFrame({"note": [
        "Club revenue counted once: plan rows + single tickets outside a plan + plan game rows with no plan row. Resale is secondary spend; transfers carry $0.",
        "Segments: first paying season's purchase (plan = Season-ticket member; waitlist fee only = Waitlist member; else Single-ticket buyer); accounts with more than 8 items on one product are Group / corporate.",
        "Seasons: plan products by the year in their name; match tickets by the event year from the audit crosswalk; plan game rows without one by their plan row's season; else transaction year with December rolled forward. Floor 2021.",
        f"Revenue bridge: Ticket rows on Austin FC match products by match season, {BRIDGE_FROM} to {BRIDGE_TO}. Types: plan (subscription_instance_id), group (account with more than 8 items on one product), then price_type_group class; $0 singles are comps. Reservation-confirmation rows (plan holders' reserved cup and playoff seats) are plan game rows and are kept. Volume + mix + price = change.",
        "By transaction year the same match-ticket rows read $42.4M (2022) to $34.5M (2025): plan renewals for a season are paid in the prior calendar year, so the calendar view moves revenue between years. The season view is the one to use.",
        "Capacity: final holder per (event, seat) using the library's holder rule; scanned = that seat was scanned at that event; unused = sold and not scanned. Capacity is 20,738 per match; sold exceeds it because GA slots and hospitality places sit on top of fixed seats. "
        f"Events with fewer than 1,000 scans are left out of the season totals and flagged scan_data = False in the match table: {', '.join(sorted(evname[k] for k in no_scan)) or 'none'}.",
        f"Customer equity: individual accounts paying in {LAST_COMPLETE}, by first-season segment and seasons since first purchase; remaining value per active member from the fan_ltv projected curves conditional on being active at that age, {BASE_HORIZON} seasons at {BASE_RATE:.0%}. Gross ticketing revenue at nominal prices, not profit. Lapsed and group accounts are listed as memo items only.",
        "Data window 2019-08-15 to 2026-09-18; 2026 is partial. All figures are gross revenue to the club's ticketing system; no costs are in the data.",
    ]})
    out = os.path.join(a.data_dir, f"value_statements_{dt.date.today():%Y%m%d}.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as xw_:
        contrib.to_excel(xw_, sheet_name="1_Contribution_Statement", index=False)
        season_rev.to_excel(xw_, sheet_name="1_Revenue_by_Season", index=False)
        tri_all.to_excel(xw_, sheet_name="2_Cohort_Revenue_All", index=False)
        tri_seg.to_excel(xw_, sheet_name="2_Cohort_Revenue_by_Segment", index=False)
        bridge.to_excel(xw_, sheet_name="3_Revenue_Bridge", index=False)
        det.to_excel(xw_, sheet_name="3_Bridge_Detail", index=False)
        cap_s.to_excel(xw_, sheet_name="4_Capacity_by_Season", index=False)
        cap_m.drop(columns=["event"]).to_excel(xw_, sheet_name="4_Capacity_by_Match", index=False)
        eq_seg.to_excel(xw_, sheet_name="5_Customer_Equity", index=False)
        E.to_excel(xw_, sheet_name="5_Equity_by_Segment_Age", index=False)
        eq_sens.to_excel(xw_, sheet_name="5_Equity_Sensitivity", index=False)
        eq_memo.to_excel(xw_, sheet_name="5_Equity_Memo", index=False)
        notes.to_excel(xw_, sheet_name="Notes", index=False)
    A.autofilter(out)
    A.log(f"Wrote {out}  (aggregate only)")

    if a.json_out:
        def rec(df):
            return json.loads(df.to_json(orient="records", date_format="iso"))
        json.dump({"contribution": rec(contrib), "segments": cols, "zero_accounts": zero, "season_revenue": rec(season_rev),
                   "cohort_all": rec(tri_all), "cohort_seg": rec(tri_seg), "bridge": rec(bridge), "bridge_detail": rec(det),
                   "capacity_season": rec(cap_s), "capacity_match": rec(cap_m.drop(columns=["event"])),
                   "equity": rec(eq_seg), "equity_detail": rec(E), "equity_sens": rec(eq_sens), "equity_memo": rec(eq_memo),
                   "notes": notes.note.tolist(), "source": os.path.basename(out), "ltv_source": os.path.basename(ltv_path),
                   "bridge_years": [BRIDGE_FROM, BRIDGE_TO], "base": {"horizon": BASE_HORIZON, "rate": BASE_RATE, "season": LAST_COMPLETE}},
                  open(a.json_out, "w"), indent=1)
        A.log(f"Wrote {a.json_out}")

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(contrib.to_string(index=False)); print(bridge.to_string(index=False)); print(cap_s.to_string(index=False)); print(eq_seg.to_string(index=False))


if __name__ == "__main__":
    main()
