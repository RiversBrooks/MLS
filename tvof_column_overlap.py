"""
Austin FC TVOF | Cross-table column overlap scan (v1, 2026-09-23)
-------------------------------------------------------------
Tests EVERY column of every client table against every column of the other tables
for shared values, to find joins the data dictionary does not list.

Each column is compared in three forms:
    raw     trimmed, lower-case
    label   'Section 101' / 'Row 09' / 'Seat 4' -> '101' / '9' / '4'
    date    parsed to YYYYMMDD (only for columns that are mostly dates)

Values are hashed (64-bit) so the scan fits in memory; no values are written out.

Usage:
    python tvof_column_overlap.py --data-dir "<your folder of client CSVs>"

Output: column_overlap_<date>.xlsx in the data folder (aggregate only).
"""
import argparse, datetime as dt, os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tvof_data_audit as A   # noqa: E402

MIN_DISTINCT = 1   # keep every column; low-cardinality pairs are flagged, not dropped


def forms(s):
    """Series of stripped strings (NaN = null) -> {form: normalized Series}."""
    raw = s.str.lower()
    lab = raw.str.replace(r"^(section|sec|row|seat)\s*", "", regex=True).str.strip()
    num = lab.str.fullmatch(r"\d+", na=False)
    lab = lab.where(~num, lab.str.lstrip("0").replace("", "0"))
    out = {"raw": raw, "label": lab}
    v = s.dropna()
    if len(v) and v.str.contains(r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}|^\d{8}$", regex=True).mean() > 0.8:
        d = pd.to_datetime(s.str.replace(r"(T?\d{2}:\d{2}:\d{2}):(\d{3})$", r"\1.\2", regex=True),
                           errors="coerce", format="mixed")
        d8 = s.where(s.str.fullmatch(r"\d{8}", na=False))
        out["date"] = d.dt.strftime("%Y%m%d").where(d.notna(), d8)
    return out


def column_sets(name, path, chunksize):
    """{(column, form): sorted unique uint64 hashes}, plus per-column stats."""
    acc, stats = {}, {}
    for ch in pd.read_csv(path, dtype=str, keep_default_na=False, chunksize=chunksize,
                          encoding_errors="replace"):
        for c in ch.columns:
            s = ch[c].str.strip()
            s = s.mask(s.str.lower().isin(A.NULL_TOKENS))
            st = stats.setdefault(c, {"rows": 0, "non_null": 0, "max_len": 0})
            st["rows"] += len(s)
            st["non_null"] += int(s.notna().sum())
            if s.notna().any():
                st["max_len"] = max(st["max_len"], int(s.str.len().max()))
            for f, v in forms(s).items():
                v = v.dropna()
                if len(v):
                    h = np.unique(pd.util.hash_array(v.to_numpy(dtype=object), categorize=True))
                    lst = acc.setdefault((c, f), [])
                    lst.append(h)
                    if len(lst) >= 8:
                        acc[(c, f)] = [np.unique(np.concatenate(lst))]
        A.log(f"  {name}: {stats[ch.columns[0]]['rows']:,} rows read")
    sets = {k: np.unique(np.concatenate(v)) for k, v in acc.items()}
    for c, st in stats.items():
        st["distinct"] = len(sets.get((c, "raw"), []))
    return sets, stats


def verdict(r):
    small = min(r.left_distinct, r.right_distinct)
    if r.known:
        return r.known
    if small < 50:
        return "Low-cardinality overlap (codes, flags, small numbers) - not a key"
    if r.shared < 0.01 * small:
        return "Coincidental overlap (<1% of the smaller column)"
    return "CANDIDATE JOIN - review"


KNOWN = {
    frozenset({"fan.internal_account_id", "attendance.internal_account_id"}): "Known join: account (100% of attendance)",
    frozenset({"fan.internal_fan_id", "activation.internal_fan_id"}): "Known join: fan (32% of activation fans)",
    frozenset({"attendance.SectionName", "sales.section"}): "Known join: seat part (use with match + row + seat)",
    frozenset({"attendance.RowName", "sales.row"}): "Known join: seat part (use with match + section + seat)",
    frozenset({"attendance.SeatName", "sales.seat"}): "Known join: seat part (use with match + section + row)",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=A.DEFAULT_DIR)
    ap.add_argument("--chunksize", type=int, default=A.CHUNK)
    a = ap.parse_args()
    files = A.find_files(a.data_dir)
    sets, stats = {}, {}
    for name in ["fan", "activation", "attendance", "sales"]:
        if name not in files:
            sys.exit(f"{name} file not found in {a.data_dir}")
        A.log(f"Reading {name} ...")
        s, st = column_sets(name, files[name], a.chunksize)
        sets[name], stats[name] = s, st

    A.log("Comparing every cross-table column pair ...")
    names = list(sets)
    rows = []
    for i, ln in enumerate(names):
        for rn in names[i + 1:]:
            for (lc, lf), lh in sets[ln].items():
                for (rc, rf), rh in sets[rn].items():
                    if lf != rf:
                        continue
                    shared = len(np.intersect1d(lh, rh, assume_unique=True))
                    if shared == 0:
                        continue
                    rows.append({"left": f"{ln}.{lc}", "right": f"{rn}.{rc}", "form": lf,
                                 "left_distinct": len(lh), "right_distinct": len(rh), "shared": shared,
                                 "pct_of_left": round(shared / len(lh), 4),
                                 "pct_of_right": round(shared / len(rh), 4)})
    R = pd.DataFrame(rows)
    # keep the best form per column pair
    R = R.sort_values("shared", ascending=False).drop_duplicates(["left", "right"])
    R["known"] = [KNOWN.get(frozenset({l, r}), "") for l, r in zip(R.left, R.right)]
    R["verdict"] = R.apply(verdict, axis=1)
    R = R.drop(columns="known").sort_values(["verdict", "shared"], ascending=[True, False])

    prof = pd.DataFrame([{"table": t, "column": c, **st} for t, d in stats.items() for c, st in d.items()])
    seen = set(R.left) | set(R.right)
    prof["overlaps_another_table"] = [f"{t}.{c}" in seen for t, c in zip(prof.table, prof.column)]

    out = os.path.join(a.data_dir, f"column_overlap_{dt.date.today():%Y%m%d}.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        R.to_excel(xw, sheet_name="Overlapping_Pairs", index=False)
        prof.to_excel(xw, sheet_name="Columns", index=False)
    A.autofilter(out)
    A.log(f"Wrote {out}  (aggregate only)")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(R.to_string(index=False))


if __name__ == "__main__":
    main()
