"""
Austin FC TVOF | Sales account bridge (v1, 2026-09-21)
------------------------------------------------------
Sales.internal_account_id is hashed differently from Fan Info / Attendance (0% match,
confirmed by tvof_id_diagnostic.py). This builds a PROBABILISTIC bridge:

    same match (EventKey) + same seat  ->  Sales final holder  <->  Attendance scanner

Scanner and final holder are usually the same person, so a Sales account that keeps
sharing seats with one Attendance account across several matches is linked to it.
This is record linkage, not a guaranteed identity. Use only High/Medium links, and
prefer an official corrected extract from Austin FC when it arrives.

Needs tvof_data_audit.py (v2.3+) in the same folder.

    python tvof_account_bridge.py --data-dir "<your folder of client CSVs>" --crosswalk-overrides crosswalk_overrides_DRAFT.csv

Outputs (in the data folder):
    account_bridge_LOCAL_ONLY.csv    row-level hashed IDs: keep with the raw files, never share
    account_bridge_summary_<date>.xlsx   aggregate counts only (shareable)
"""
import argparse, os, sys, datetime as dt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tvof_data_audit as A   # noqa: E402

MIN_EVENTS_HIGH, MIN_SHARE_HIGH = 3, 0.80
MIN_EVENTS_MED, MIN_SHARE_MED = 2, 0.60


def hex_lookup(path, colname, wanted, chunksize=200_000):
    """Map uint64 hash -> original hex string for the IDs we need to write out."""
    out = {}
    wanted = set(int(x) for x in wanted)
    for ch in pd.read_csv(path, usecols=[colname], dtype=str, keep_default_na=False, chunksize=chunksize):
        u = pd.Series(ch[colname].str.strip().unique())
        u = u[~u.str.lower().isin(A.NULL_TOKENS)]
        h = pd.util.hash_array(u.to_numpy(dtype=object), categorize=True)
        for hv, sv in zip(h, u):
            if int(hv) in wanted:
                out[int(hv)] = sv
    return out


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
    X = A.CROSSWALK
    trusted = X[X.confidence.isin(["High", "Medium", "Manual override"])]
    pmap = trusted.set_index("product_id").EventKey.astype(str)
    A.log(f"  {len(pmap)} match products trusted")

    A.log("Pairing final holders with scanners seat by seat ...")
    T = A.final_holder_frame(sal, pmap).dropna(subset=["acct"])
    ek = A.col(att, "EventKey")
    Au = pd.DataFrame({"EventKey": att[ek].astype(str),
                       "key": A.seat_key(att[A.col(att, "SectionName")], att[A.col(att, "RowName")],
                                         att[A.col(att, "SeatName")]),
                       "att_acct": att[A.col(att, "internal_account_id")]}).dropna()
    Au = Au.drop_duplicates(["EventKey", "key"])
    P = T.merge(Au, on=["EventKey", "key"])[["acct", "att_acct", "EventKey"]]
    P.columns = ["s", "f", "EventKey"]
    del T, Au

    # events each pair shares, and each side's total scanned-seat events
    pe = P.drop_duplicates(["s", "f", "EventKey"]).groupby(["s", "f"]).size().rename("events_together").reset_index()
    s_tot = P.drop_duplicates(["s", "EventKey"]).groupby("s").size().rename("s_events")
    f_tot = P.drop_duplicates(["f", "EventKey"]).groupby("f").size().rename("f_events")
    pe = pe.join(s_tot, on="s").join(f_tot, on="f")
    pe["share_of_sales_acct_events"] = pe.events_together / pe.s_events
    pe["share_of_fan_acct_events"] = pe.events_together / pe.f_events
    # best partner each way; keep mutual bests
    bs = pe.sort_values(["events_together", "share_of_sales_acct_events"], ascending=False).drop_duplicates("s")
    bf = pe.sort_values(["events_together", "share_of_fan_acct_events"], ascending=False).drop_duplicates("f")
    link = bs.merge(bf[["s", "f"]], on=["s", "f"])
    minshare = link[["share_of_sales_acct_events", "share_of_fan_acct_events"]].min(axis=1)
    link["confidence"] = np.select(
        [(link.events_together >= MIN_EVENTS_HIGH) & (minshare >= MIN_SHARE_HIGH),
         (link.events_together >= MIN_EVENTS_MED) & (minshare >= MIN_SHARE_MED)],
        ["High", "Medium"], "Low")

    n_s = int(sal[A.col(sal, "internal_account_id")].dropna().nunique())
    n_f = int(att[A.col(att, "internal_account_id")].dropna().nunique())
    summ = [("Sales accounts (distinct)", n_s, None),
            ("Sales accounts with >=1 scanned final-holder seat", int(P.s.nunique()), n_s),
            ("Attendance accounts (distinct)", n_f, None),
            ("Attendance accounts paired to any sales holder", int(P.f.nunique()), n_f)]
    for c in ["High", "Medium", "Low"]:
        k = int((link.confidence == c).sum())
        summ.append((f"Mutual-best links: {c}", k, n_s))
    hm = link[link.confidence.isin(["High", "Medium"])]
    summ.append(("Sales accounts linked High/Medium (usable)", len(hm), n_s))
    summ.append(("Attendance accounts linked High/Medium", int(hm.f.nunique()), n_f))
    summ.append(("Seat-events explained by High/Medium links (holder == scanner)",
                 int(P.merge(hm[["s", "f"]], on=["s", "f"]).shape[0]), len(P)))
    S = pd.DataFrame(summ, columns=["measure", "count", "denominator"])
    S["pct"] = (S["count"] / S["denominator"]).round(4)
    dist = link.groupby(["confidence", pd.cut(link.events_together, [0, 1, 2, 3, 5, 10, 20, 50, 1000])],
                        observed=True).size().rename("links").reset_index()
    dist["events_together"] = dist.events_together.astype(str)

    A.log("Writing outputs ...")
    sh = hex_lookup(files["sales"], A.col(sal, "internal_account_id"), link.s.astype("uint64"))
    fh = hex_lookup(files["attendance"], A.col(att, "internal_account_id"), link.f.astype("uint64"))
    out = pd.DataFrame({"sales_account_id": link.s.astype("uint64").map(lambda v: sh.get(int(v))),
                        "fan_account_id": link.f.astype("uint64").map(lambda v: fh.get(int(v))),
                        "events_together": link.events_together,
                        "share_of_sales_acct_events": link.share_of_sales_acct_events.round(4),
                        "share_of_fan_acct_events": link.share_of_fan_acct_events.round(4),
                        "confidence": link.confidence})
    p1 = os.path.join(a.data_dir, "account_bridge_LOCAL_ONLY.csv")
    out.to_csv(p1, index=False)
    p2 = os.path.join(a.data_dir, f"account_bridge_summary_{dt.date.today():%Y%m%d}.xlsx")
    with pd.ExcelWriter(p2, engine="openpyxl") as xw:
        S.to_excel(xw, sheet_name="Bridge_Summary", index=False)
        dist.to_excel(xw, sheet_name="Links_by_Events", index=False)
        pd.DataFrame(A.LOG).to_excel(xw, sheet_name="Run_Log", index=False)
    A.log(f"Wrote {p1}  (ROW-LEVEL: keep local)")
    A.log(f"Wrote {p2}  (aggregate only)")
    print(S.to_string(index=False))


if __name__ == "__main__":
    main()
