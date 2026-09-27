"""
Austin FC TVOF | Fan value and tenure (v2, 2026-09-26)
-------------------------------------------------------------
Dollar value and duration per customer, through three lenses:

  PAYER      every Sales internal_account_id (complete: needs no linking). Split into
             individual/household buyers (never more than GROUP_MAX_ITEMS items for one
             product) and group/corporate/broker buyers.
  ATTENDEE   every fan who scanned into a match: the original sale value of each ticket
             they scanned (scan_sale_join_LOCAL_ONLY.csv), credited to the person who used
             it, whoever paid. Matches only (no parking / other events), 2022-2026.
  LINKED FAN Fan Info person with Sales accounts attached by internal_account_id (the
             2026-09-24 Sales export shares Fan Info's account IDs), then by the usable seat
             bridge for any account the direct ID does not reach (older exports).

Club revenue (counted once):
  plan rows (item_type Subscription)                    plan payment, on the plan row
  + single tickets (item_type Ticket, no subscription)  includes parking, other stadium events
  + plan game rows whose plan row is missing            (973 plans)
  Plan game rows are NOT added: their total_payment is the plan payment split per game
  (plan row == SUM(game rows) for 99.9% of plans).
  Resale rows are the buyer's spend paid to the seller (club share unknown): reported
  separately as secondary_spend, never in club revenue. Transfers carry $0.

Duration:
  account  first -> last Sales transaction of any kind
  fan      earliest of (seatgeek_since_date, first purchase, first scan)
           -> latest of (last purchase, last scan)
  Data runs 2019-08-15 to 2026-09-18, so every value and duration is TO DATE
  (right-censored), not lifetime.

Usage:
    python tvof_fan_value.py --data-dir "<your folder of client CSVs>"

Outputs (data folder):
    fan_value_summary_<date>.xlsx   aggregate statistics only
    fan_value_LOCAL_ONLY.csv        one row per fan with hashed IDs: keep local
"""
import argparse, datetime as dt, os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tvof_data_audit as A   # noqa: E402
from tvof_account_bridge import hex_lookup   # noqa: E402

BRIDGE_FILE = "account_bridge_LOCAL_ONLY.csv"
JOIN_FILE = "scan_sale_join_LOCAL_ONLY.csv"
GROUP_MAX_ITEMS = 8   # more than this many items bought for one product = group/corporate/broker buyer
PCTS = [0.01, 0.10, 0.25, 0.75, 0.90, 0.99]


def describe(x, name, unit, mode_round=0):
    """Mean / median / mode / min / max and spread for one measure."""
    x = pd.Series(x, dtype="float64").dropna()
    if x.empty:
        return {"measure": name, "unit": unit, "n": 0}
    mv = x.round(mode_round).value_counts()
    q = x.quantile(PCTS)
    lo, hi = x.quantile([0.01, 0.99])
    return {"measure": name, "unit": unit, "n": len(x),
            "mean": x.mean(), "median": x.median(),
            "mode": mv.index[0], "mode_share": mv.iloc[0] / len(x),
            "min": x.min(), "max": x.max(), "std": x.std(),
            **{f"p{int(p * 100)}": q[p] for p in PCTS},
            "mean_winsorized_1_99": x.clip(lo, hi).mean(),
            "total": x.sum(), "top1pct_share_of_total": x[x >= hi].sum() / x.sum() if x.sum() else np.nan}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=A.DEFAULT_DIR)
    ap.add_argument("--chunksize", type=int, default=A.CHUNK)
    a = ap.parse_args()
    files = A.find_files(a.data_dir)
    for k in ("fan", "attendance", "sales"):
        if k not in files:
            sys.exit(f"{k} file not found in {a.data_dir}")
    bridge_path = os.path.join(a.data_dir, BRIDGE_FILE)
    have_bridge = os.path.exists(bridge_path)
    if not have_bridge:
        A.log(f"  {BRIDGE_FILE} not found: linking by account ID only")

    A.log("Loading sales ...")
    s = A.load("sales", files["sales"], a.chunksize)
    it = s.item_type.astype(object).str.lower().to_numpy()
    sub_na = s.subscription_instance_id.isna().to_numpy()
    pay = s.total_payment.fillna(0).to_numpy()
    plan_ids = set(s.subscription_instance_id[it == "subscription"].dropna().to_numpy())
    orphan_game = (it == "ticket") & ~sub_na & ~s.subscription_instance_id.isin(plan_ids).to_numpy()
    cls = A.cat_apply(s.product_description, lambda x: x.map(A.product_class)).astype(object).to_numpy()

    club = np.where((it == "subscription") | ((it == "ticket") & sub_na) | orphan_game, pay, 0.0)
    R = pd.DataFrame({"acct": s.internal_account_id.array, "t": s.transaction_date.to_numpy(),
                      "club": club, "secondary": np.where(it == "resale", pay, 0.0),
                      "plan": np.where(it == "subscription", pay, 0.0) + np.where(orphan_game, pay, 0.0),
                      "match_single": np.where((it == "ticket") & sub_na & (cls == "Austin FC match"), pay, 0.0),
                      "parking_single": np.where((it == "ticket") & sub_na & (cls == "parking"), pay, 0.0),
                      "other_event_single": np.where((it == "ticket") & sub_na & (cls == "other stadium event"), pay, 0.0)})
    R["other_single"] = R.club - R.plan - R.match_single - R.parking_single - R.other_event_single
    R["yr"] = pd.to_datetime(R.t).dt.year
    R["product"] = s.product_id.astype(object).to_numpy()
    A.log(f"  club revenue ${R.club.sum():,.0f} (plans ${R.plan.sum():,.0f}); secondary spend ${R.secondary.sum():,.0f}; "
          f"rows with no account ${R.club[R.acct.isna()].sum():,.0f}")
    del s
    R = R.dropna(subset=["acct"])
    g = R.groupby("acct")
    acc = g.agg(club_revenue=("club", "sum"), secondary_spend=("secondary", "sum"),
                plan_revenue=("plan", "sum"), match_single=("match_single", "sum"),
                parking_single=("parking_single", "sum"), other_event_single=("other_event_single", "sum"),
                other_single=("other_single", "sum"), first_txn=("t", "min"), last_txn=("t", "max"),
                rows=("t", "size"))
    acc["paying_years"] = R[R.club > 0].groupby("acct").yr.nunique().reindex(acc.index).fillna(0).astype(int)
    acc["tenure_days"] = (acc.last_txn - acc.first_txn).dt.total_seconds() / 86400
    acc["tenure_years"] = acc.tenure_days / 365.25
    acc["club_revenue_per_paying_year"] = acc.club_revenue / acc.paying_years.replace(0, np.nan)
    acc["max_items_one_product"] = R[R.club != 0].groupby(["acct", "product"]).size().groupby(level=0).max() \
        .reindex(acc.index).fillna(0).astype(int)
    acc["segment"] = np.where(acc.max_items_one_product > GROUP_MAX_ITEMS, "group/corporate", "individual/household")
    del R, g

    A.log("Loading fan info and attendance ...")
    fan = A.load("fan", files["fan"], a.chunksize)
    F = pd.DataFrame({"fid": fan.internal_fan_id.array, "aid": fan.internal_account_id.array,
                      "since": pd.to_datetime(fan.seatgeek_since_date)})
    del fan
    F["fan_key"] = np.where(F.fid.notna(), "f" + F.fid.astype(str), "a" + F.aid.astype(str))
    F = F[F.fid.notna() | F.aid.notna()]
    a2f = F.dropna(subset=["aid"]).drop_duplicates("aid").set_index("aid").fan_key
    att = A.load("attendance", files["attendance"], a.chunksize)
    scans = pd.DataFrame({"aid": att.internal_account_id.array, "t": att.AttendedDatetime.to_numpy()}).dropna()
    del att
    scans["fan_key"] = scans.aid.map(a2f)
    sc = scans.dropna(subset=["fan_key"]).groupby("fan_key").agg(first_scan=("t", "min"), last_scan=("t", "max"),
                                                                scans=("t", "size"))

    A.log("Attaching Sales accounts to fans: account ID first, seat bridge for the rest ...")
    keys = a2f.to_numpy(dtype=object)
    pos = pd.Index(a2f.index).get_indexer(acc.index)          # exact UInt64 lookup, no float round trip
    acc["fan_key"] = np.where(pos >= 0, keys[np.clip(pos, 0, None)], None)
    acc["link"] = np.where(pos >= 0, "account_id", None)
    n_direct, n_bridge = int((pos >= 0).sum()), 0
    if have_bridge:
        br = pd.read_csv(bridge_path, dtype=str)
        usable = br.usable.str.lower().eq("true") if "usable" in br else br.confidence.isin(["High", "Medium"])
        br = br[usable]
        br["s"] = A._hash(br.sales_account_id.str.strip().to_numpy(dtype=object))
        br["f"] = A._hash(br.fan_account_id.str.strip().to_numpy(dtype=object))
        bpos = pd.Index(a2f.index).get_indexer(pd.Index(br.f))
        br["fan_key"] = np.where(bpos >= 0, keys[np.clip(bpos, 0, None)], None)
        link = br.dropna(subset=["fan_key"]).drop_duplicates("s").set_index("s").fan_key
        spos = pd.Index(link.index).get_indexer(acc.index)
        fill = acc.fan_key.isna().to_numpy() & (spos >= 0)
        acc.loc[fill, "fan_key"] = link.to_numpy(dtype=object)[np.clip(spos, 0, None)][fill]
        acc.loc[fill, "link"] = "seat_bridge"
        n_bridge = int(fill.sum())
    A.log(f"  {n_direct:,} of {len(acc):,} accounts linked by account ID, {n_bridge:,} more by the seat bridge")
    linked = acc.dropna(subset=["fan_key"])
    fv = linked.groupby("fan_key").agg(
        sales_accounts=("club_revenue", "size"), club_revenue=("club_revenue", "sum"),
        secondary_spend=("secondary_spend", "sum"), plan_revenue=("plan_revenue", "sum"),
        match_single=("match_single", "sum"), parking_single=("parking_single", "sum"),
        other_event_single=("other_event_single", "sum"), other_single=("other_single", "sum"),
        first_txn=("first_txn", "min"), last_txn=("last_txn", "max"))
    fv = fv.join(F.groupby("fan_key").since.min().rename("seatgeek_since")).join(sc)
    fv["start"] = fv[["seatgeek_since", "first_txn", "first_scan"]].min(axis=1)
    fv["end"] = fv[["last_txn", "last_scan"]].max(axis=1)
    fv["tenure_days"] = (fv.end - fv.start).dt.total_seconds() / 86400
    fv["tenure_years"] = fv.tenure_days / 365.25
    py = fv.index.map(linked.groupby("fan_key").paying_years.max())
    fv["club_revenue_per_paying_year"] = fv.club_revenue / pd.Series(py, index=fv.index).replace(0, np.nan)

    A.log("Attendee lens: value of the tickets each fan scanned ...")
    jp = os.path.join(a.data_dir, JOIN_FILE)
    if not os.path.exists(jp):
        sys.exit(f"{JOIN_FILE} not found: run tvof_scan_sale_join.py first")
    J = pd.read_csv(jp, usecols=["scan_internal_account_id", "orig_total_payment", "AttendedDatetime",
                                 "EventKey", "SeasonKey"], dtype={"scan_internal_account_id": str})
    J["t"] = A._parse_dt(J.AttendedDatetime.astype(str))
    fr = pd.read_csv(files["fan"], usecols=["internal_fan_id", "internal_account_id"], dtype=str,
                     keep_default_na=False).apply(lambda x: x.str.strip())
    fr = fr.mask(fr.apply(lambda x: x.str.lower().isin(A.NULL_TOKENS))).dropna(subset=["internal_account_id"])
    fr["fan"] = fr.internal_fan_id.fillna(fr.internal_account_id)
    J["fan"] = J.scan_internal_account_id.map(fr.drop_duplicates("internal_account_id")
                                              .set_index("internal_account_id").fan)
    J["fan"] = J.fan.fillna(J.scan_internal_account_id)
    at = J.groupby("fan").agg(ticket_value_used=("orig_total_payment", "sum"), scans=("t", "size"),
                              matches=("EventKey", "nunique"), seasons=("SeasonKey", "nunique"),
                              first_scan=("t", "min"), last_scan=("t", "max"))
    at["value_per_season_attended"] = at.ticket_value_used / at.seasons
    at["value_per_match_attended"] = at.ticket_value_used / at.matches
    at["tenure_days"] = (at.last_scan - at.first_scan).dt.total_seconds() / 86400
    at["tenure_years"] = at.tenure_days / 365.25
    A.log(f"  {len(at):,} attending fans; ticket value used ${at.ticket_value_used.sum():,.0f}")
    del J

    A.log("Statistics ...")
    paying = acc[acc.club_revenue > 0]
    ind = paying[paying.segment == "individual/household"]
    grp = paying[paying.segment == "group/corporate"]
    payer_measures = [("club_revenue", "Club revenue to date", "$", 0),
                      ("club_revenue_per_paying_year", "Club revenue per paying year", "$", 0),
                      ("secondary_spend", "Secondary (resale) spend to date", "$", 0),
                      ("tenure_days", "Tenure (first to last transaction)", "days", 0),
                      ("tenure_years", "Tenure (first to last transaction)", "years", 1)]
    att_measures = [("ticket_value_used", "Value of tickets used to date", "$", 0),
                    ("value_per_season_attended", "Value per season attended", "$", 0),
                    ("value_per_match_attended", "Value per match attended", "$", 0),
                    ("matches", "Matches attended", "matches", 0),
                    ("tenure_days", "Tenure (first to last scan)", "days", 0),
                    ("tenure_years", "Tenure (first to last scan)", "years", 1)]
    rows = []
    for label, df, ms in [("PAYER: individual/household accounts, paying", ind, payer_measures),
                          ("PAYER: all paying accounts", paying, payer_measures),
                          ("PAYER: group/corporate accounts, paying", grp, payer_measures),
                          ("PAYER: all Sales accounts incl. $0 (transfer/resale receivers)", acc, payer_measures),
                          ("ATTENDEE: fans who scanned into a match", at, att_measures),
                          ("LINKED FAN: Fan Info people with a linked Sales account", fv, payer_measures)]:
        for col_, nm, unit, r in ms:
            d = describe(df[col_], nm, unit, r)
            rows.append({"population": label, **d})
    S = pd.DataFrame(rows)
    num = S.select_dtypes("number").columns.difference(["n"])
    S[num] = S[num].astype(float).round(2)

    tot_club = acc.club_revenue.sum()
    cov = pd.DataFrame([
        {"measure": "Sales accounts", "value": len(acc)},
        {"measure": "Sales accounts paying club revenue > $0", "value": len(paying)},
        {"measure": f"  individual/household (<= {GROUP_MAX_ITEMS} items for any one product)", "value": len(ind)},
        {"measure": "  group/corporate/broker", "value": len(grp)},
        {"measure": "  group/corporate share of club revenue", "value": round(grp.club_revenue.sum() / acc.club_revenue.sum(), 4)},
        {"measure": "Attending fans (scanned into a match)", "value": len(at)},
        {"measure": "Ticket value used by attending fans (matches, 2022-2026)", "value": round(at.ticket_value_used.sum(), 2)},
        {"measure": "Sales accounts linked to a fan", "value": int(acc.fan_key.notna().sum())},
        {"measure": "  by internal_account_id (direct)", "value": n_direct},
        {"measure": "  by the seat bridge only", "value": n_bridge},
        {"measure": "Share of club revenue on linked accounts", "value": round(linked.club_revenue.sum() / tot_club, 4)},
        {"measure": "Fans valued (Fan Info people with a linked Sales account)", "value": len(fv)},
        {"measure": "Fans with 2+ linked Sales accounts", "value": int((fv.sales_accounts > 1).sum())},
        {"measure": "Club revenue, total (counted once)", "value": round(tot_club, 2)},
        {"measure": "  of which plans", "value": round(acc.plan_revenue.sum(), 2)},
        {"measure": "  of which single Austin FC match tickets", "value": round(acc.match_single.sum(), 2)},
        {"measure": "  of which single parking", "value": round(acc.parking_single.sum(), 2)},
        {"measure": "  of which single other stadium events", "value": round(acc.other_event_single.sum(), 2)},
        {"measure": "  of which other single tickets (add-ons etc.)", "value": round(acc.other_single.sum(), 2)},
        {"measure": "Secondary (resale) spend, total, not club revenue", "value": round(acc.secondary_spend.sum(), 2)},
    ])
    # value bands: where the money sits
    bands = [-np.inf, 0, 100, 500, 1000, 2500, 5000, 10000, 25000, 100000, np.inf]
    labels = ["<= $0", "$0-100", "$100-500", "$500-1k", "$1k-2.5k", "$2.5k-5k", "$5k-10k", "$10k-25k", "$25k-100k", "> $100k"]
    B = []
    for label, v in [("PAYER: individual/household, paying", ind.club_revenue),
                     ("PAYER: all paying", paying.club_revenue),
                     ("ATTENDEE: value of tickets used", at.ticket_value_used)]:
        b = pd.cut(v, bands, labels=labels).value_counts(sort=False)
        rv = v.groupby(pd.cut(v, bands, labels=labels), observed=False).sum()
        B.append(pd.DataFrame({"population": label, "band": labels, "count": b.to_numpy(),
                               "pct_of_count": (b / b.sum()).round(4).to_numpy(),
                               "revenue": rv.round(0).to_numpy(),
                               "pct_of_revenue": (rv / rv.sum()).round(4).to_numpy()}))
    B = pd.concat(B, ignore_index=True)
    T = pd.concat([pd.cut(df.tenure_years, [-0.01, 0, 1, 2, 3, 4, 5, 8], labels=["0 (one day)", "<1", "1-2", "2-3", "3-4", "4-5", "5-7"])
                   .value_counts(sort=False).rename(lbl) for lbl, df in
                   [("PAYER: individual/household, paying", ind), ("ATTENDEE: fans who scanned", at)]], axis=1)
    T = T.rename_axis("tenure_years").reset_index()
    notes = pd.DataFrame({"note": [
        "Club revenue = plan rows + single tickets not in a plan + plan game rows with no plan row. Plan game rows are the plan payment split per game, so they are not added again.",
        "Resale payments go to the seller; shown as secondary_spend, never in club revenue. Returns are not in the data (per the dictionary).",
        "PAYER lens is complete (every Sales account). Group/corporate = bought more than 8 items for one product (match or plan).",
        "ATTENDEE lens credits each scanned ticket's ORIGINAL sale value (plan tickets: the plan payment split per game) to the fan who scanned it. Covers matches 2022-2026 only; no parking or other events.",
        f"LINKED FAN lens: {acc.fan_key.notna().mean():.1%} of Sales accounts and {linked.club_revenue.sum() / tot_club:.1%} of club revenue reach a Fan Info person "
        f"({n_direct:,} accounts by internal_account_id, {n_bridge:,} by the seat bridge).",
        "Data window 2019-08-15 to 2026-09-18: values and tenures are to date, not lifetime. 2026 is a partial season.",
        "Mode: dollars rounded to $1, days to 1 day, years to 0.1 year. Top-1% share shows how concentrated revenue is.",
        "Single tickets include parking and non-Austin FC stadium events; see Coverage for the split."]})

    out = os.path.join(a.data_dir, f"fan_value_summary_{dt.date.today():%Y%m%d}.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        S.to_excel(xw, sheet_name="Statistics", index=False)
        cov.to_excel(xw, sheet_name="Coverage", index=False)
        B.to_excel(xw, sheet_name="Value_Bands", index=False)
        T.to_excel(xw, sheet_name="Tenure_Bands", index=False)
        notes.to_excel(xw, sheet_name="Notes", index=False)
    A.autofilter(out)
    A.log(f"Wrote {out}  (aggregate only)")

    A.log("Writing row-level file ...")
    fh = hex_lookup(files["fan"], "internal_fan_id", [int(k[1:]) for k in fv.index if k.startswith("f")])
    ah = hex_lookup(files["fan"], "internal_account_id", [int(k[1:]) for k in fv.index if k.startswith("a")])
    fv.insert(0, "fan_id", [fh.get(int(k[1:])) if k.startswith("f") else None for k in fv.index])
    fv.insert(1, "fan_account_id_if_no_fan_id", [ah.get(int(k[1:])) if k.startswith("a") else None for k in fv.index])
    loc = os.path.join(a.data_dir, "fan_value_LOCAL_ONLY.csv")
    fv.reset_index(drop=True).to_csv(loc, index=False, float_format="%.2f")
    A.log(f"Wrote {loc}  ({len(fv):,} fans, ROW-LEVEL: keep local)")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(cov.to_string(index=False))
    print(S[["population", "measure", "unit", "n", "mean", "median", "mode", "mode_share", "min", "max",
             "p10", "p25", "p75", "p90", "p99", "mean_winsorized_1_99", "top1pct_share_of_total"]].to_string(index=False))
    print(B.to_string(index=False))
    print(T.to_string(index=False))


if __name__ == "__main__":
    main()
