"""
Austin FC TVOF | Scan-to-sale join (v1, 2026-09-23)
-------------------------------------------------------------
Links every Attendance scan to the Sales row for the same seat at the same match:

    1. clean seat labels   'Section 101' / 'Row 9' / 'Seat 4'  ->  101 / 9 / 4   (seat_norm)
    2. add the match       Sales.product_id -> Attendance.EventKey via the event crosswalk
    3. final holder only   last valid row per match + seat (drops plan rows, Resold seller
                           rows, Pending/Canceled transfers), same rule as the account bridge

Each scan also carries the ORIGINAL sale in its ticket chain (earliest row with the same
primary_ticket_id), since the final holder's row is often a transfer or resale.

Usage:
    python tvof_scan_sale_join.py --data-dir "<your folder of client CSVs>" --crosswalk-overrides crosswalk_overrides_DRAFT.csv

Outputs (data folder):
    scan_sale_join_LOCAL_ONLY.csv        one row per scan. ROW LEVEL: keep local. It has more rows
                                        than Excel can hold: do NOT open and save it in Excel.
    scan_sale_join_summary_<date>.xlsx   aggregate only
"""
import argparse, datetime as dt, os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tvof_data_audit as A        # noqa: E402
from tvof_account_bridge import hex_lookup   # noqa: E402

ATT_COLS = ["SeasonKey", "EventDateKey", "EventKey", "EventName", "MasterEventName", "AttendedDatetime",
            "SeatingArea", "SectionCategory", "SectionName", "RowName", "SeatName"]
SALE_COLS = ["product_description", "item_type", "product_type", "sale_type", "transfer_status",
             "resale_status", "price_type", "price_type_group", "application_channel",
             "transaction_date", "total_payment", "total_plan_amount"]
ORIG_COLS = ["item_type", "price_type", "price_type_group", "application_channel",
             "transaction_date", "total_payment", "total_plan_amount"]


def final_holder_rows(sal, pmap, key):
    """Positions of the final holder row per (EventKey, seat), same rules as A.final_holder_frame."""
    p, it, ts, rs = A.col(sal, "product_id"), A.col(sal, "item_type"), A.col(sal, "transfer_status"), A.col(sal, "resale_status")
    low = lambda c: sal[c].astype(object).str.lower()
    ok = sal[p].isin(pmap.index)
    if it:
        ok &= ~low(it).eq("subscription").fillna(False)
    if rs:
        ok &= ~low(rs).eq("resold").fillna(False)
    if ts and it:
        ok &= ~(low(it).eq("transfer").fillna(False) & low(ts).isin(["pending", "canceled", "cancelled"]).fillna(False))
    idx = np.flatnonzero(ok.to_numpy())
    F = pd.DataFrame({"ri": idx, "key": key[idx],
                      "EventKey": sal[p].astype(object).to_numpy()[idx],
                      "td": sal[A.col(sal, "transaction_date")].to_numpy()[idx]})
    F["EventKey"] = F.EventKey.map(pmap.astype(str))
    F = F.dropna(subset=["EventKey", "key"]).sort_values("td", kind="stable")
    return F.drop_duplicates(["EventKey", "key"], keep="last").drop(columns="td")


def chain_info(sal):
    """Per row: position of the original sale in its primary_ticket_id chain, and chain length."""
    pt = sal[A.col(sal, "primary_ticket_id")]
    S = pd.DataFrame({"pt": pt.array, "td": sal[A.col(sal, "transaction_date")].to_numpy(),
                      "ri": np.arange(len(sal))}).dropna(subset=["pt"])
    S = S.sort_values("td", kind="stable")
    first = S.drop_duplicates("pt").set_index("pt").ri.rename("orig_ri")
    size = S.groupby("pt").size().rename("chain_rows")
    return pt, first, size


def take(sal, cols, ri, prefix):
    out = {}
    for c in cols:
        sc = A.col(sal, c)
        if sc:
            v = sal[sc].iloc[ri]
            # .array keeps UInt64 hashes exact (to_numpy can turn them into lossy floats)
            out[prefix + c] = v.astype(object).to_numpy() if isinstance(v.dtype, pd.CategoricalDtype) else v.array
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=A.DEFAULT_DIR)
    ap.add_argument("--crosswalk-overrides", default=None)
    ap.add_argument("--chunksize", type=int, default=A.CHUNK)
    a = ap.parse_args()
    if a.crosswalk_overrides:
        A.OVERRIDES = pd.read_csv(a.crosswalk_overrides, dtype=str)[["product_id", "EventKey"]].dropna()
    files = A.find_files(a.data_dir)
    for k in ("sales", "attendance"):
        if k not in files:
            sys.exit(f"{k} file not found in {a.data_dir}")

    A.log("Loading attendance and sales ...")
    att = A.load("attendance", files["attendance"], a.chunksize)
    sal = A.load("sales", files["sales"], a.chunksize)

    A.log("Building event crosswalk ...")
    A.crosswalk_and_seats(sal, att)
    if A.CROSSWALK is None:
        sys.exit("Event crosswalk could not be built; nothing to join.")
    X = A.CROSSWALK
    pmap = X[X.confidence.isin(["High", "Medium", "Manual override"])].set_index("product_id").EventKey.astype(str)
    A.log(f"  {len(pmap)} match products trusted")

    A.log("Final holder per match + seat ...")
    skey = A.seat_key(sal[A.col(sal, "section")], sal[A.col(sal, "row")], sal[A.col(sal, "seat")])
    F = final_holder_rows(sal, pmap, skey)
    pt, first, size = chain_info(sal)
    fpt = pd.Series(pt.array[F.ri.to_numpy()])
    F["orig_ri"] = fpt.map(first).fillna(pd.Series(F.ri.to_numpy())).astype(int).to_numpy()
    F["chain_rows"] = fpt.map(size).fillna(1).astype(int).to_numpy()
    F = pd.concat([F.reset_index(drop=True),
                   take(sal, ["internal_account_id"], F.ri, "sales_"),
                   take(sal, SALE_COLS, F.ri, "final_"),
                   take(sal, ORIG_COLS, F.orig_ri, "orig_")], axis=1).drop(columns=["ri", "orig_ri"])

    A.log("Joining scans to sales ...")
    acol = A.col(att, "internal_account_id")
    S = pd.DataFrame({c: (att[c].astype(object) if isinstance(att[c].dtype, pd.CategoricalDtype) else att[c]).to_numpy()
                      for c in ATT_COLS if c in att.columns})
    S["scan_account"] = att[acol].array
    S["key"] = A.seat_key(att[A.col(att, "SectionName")], att[A.col(att, "RowName")], att[A.col(att, "SeatName")])
    S["EventKey"] = S.EventKey.astype(str)
    J = S.merge(F, on=["EventKey", "key"], how="left").drop(columns="key")
    J.insert(0, "sale_found", J.final_item_type.notna() | J.chain_rows.notna())
    del S, F, att

    A.log("Restoring account IDs ...")
    for c, path in [("scan_account", files["attendance"]), ("sales_internal_account_id", files["sales"])]:
        h = pd.Series(J[c].array).dropna().astype("uint64")
        m = hex_lookup(path, "internal_account_id", h.unique())
        J[c] = pd.Series(J[c].array).astype("UInt64").astype(object).map(lambda v: m.get(int(v)) if pd.notna(v) else None)
    J = J.rename(columns={"scan_account": "scan_internal_account_id",
                          "sales_internal_account_id": "final_holder_sales_account_id"})

    out = os.path.join(a.data_dir, "scan_sale_join_LOCAL_ONLY.csv")
    J.to_csv(out, index=False)
    A.log(f"Wrote {out}  ({len(J):,} rows, ROW-LEVEL: keep local; too many rows for Excel)")

    A.log("Summary ...")
    J["season"] = J.SeasonKey if "SeasonKey" in J else pd.to_datetime(J.AttendedDatetime).dt.year
    by = J.groupby("season").agg(scans=("sale_found", "size"), sale_found=("sale_found", "sum"),
                                 final_total_payment=("final_total_payment", "sum"),
                                 orig_total_payment=("orig_total_payment", "sum"),
                                 orig_plan_amount_per_game=("orig_total_plan_amount", "sum")).reset_index()
    by["match_rate"] = (by.sale_found / by.scans).round(4)
    fi = J.groupby(["final_item_type", "orig_item_type"], dropna=False).agg(
        scans=("sale_found", "size"), orig_total_payment=("orig_total_payment", "sum")).reset_index()
    ch = J.groupby("chain_rows", dropna=False).size().rename("scans").reset_index()
    notes = pd.DataFrame({"note": [
        "One row per Attendance scan. Seat + match joined to the final holder row in Sales.",
        "final_* = final holder's Sales row (often a transfer or resale); orig_* = earliest row in the same primary_ticket_id chain.",
        "orig_total_plan_amount on plan game rows is a per-game allocation, not the plan total.",
        "Summing orig_total_payment across scans does not double-count: each match + seat is scanned once.",
        "Revenue for no-shows, parking and non-match products is not in this file (no scan to join to)."]})
    summ = os.path.join(a.data_dir, f"scan_sale_join_summary_{dt.date.today():%Y%m%d}.xlsx")
    with pd.ExcelWriter(summ, engine="openpyxl") as xw:
        by.to_excel(xw, sheet_name="By_Season", index=False)
        fi.to_excel(xw, sheet_name="Final_vs_Original_Type", index=False)
        ch.to_excel(xw, sheet_name="Chain_Length", index=False)
        notes.to_excel(xw, sheet_name="Notes", index=False)
    A.autofilter(summ)
    A.log(f"Wrote {summ}  (aggregate only)")
    print(by.to_string(index=False))


if __name__ == "__main__":
    main()
