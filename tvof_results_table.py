"""
Austin FC TVOF | Everything in one table (v1, 2026-09-27)
-------------------------------------------------------------
Stacks every sheet of the latest dated results workbook of each kind, plus the small aggregate CSVs the pipeline
writes, into ONE long table that filters on every column:

    workbook      the kind of workbook (audit, fan_ltv, value_statements, ...)
    file          the exact file the row came from
    run_date      the date in the file name
    sheet         the sheet inside the workbook (or "csv")
    row           1-based row inside that sheet, so a whole row can be reassembled with a filter
    row_label     the first column's value on that row (segment, measure, cohort ...), for quick filtering
    column        the column heading
    value         the cell as written (numbers stay numbers in the workbook)
    value_num     the value as a number when it is one, blank otherwise
    value_type    number / date / bool / text

Aggregate only: the raw client CSVs and every *_LOCAL_ONLY file are never opened, so the output is shareable.

Usage:
    python tvof_results_table.py [--data-dir D] [--out-dir D2] [--json-out p] [--json-max-rows N]

Output: results_master_<date>.xlsx (sheets Everything, Index, Notes) and results_master_<date>.csv in --out-dir
(default: the data dir).
"""
import argparse, datetime as dt, glob, json, os, re, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tvof_data_audit as A   # noqa: E402

# kinds of results workbook -> file-name prefix (latest dated file of each is used; _HHMM copies sort last, so they win)
WORKBOOKS = ["TVOF_audit_results", "account_bridge_summary", "scan_sale_join_summary", "column_overlap", "fan_value_summary",
             "fan_ltv", "peer_benchmarks", "value_statements", "ltv_montecarlo", "ltv_validation", "insights",
             "activation_cohort", "Peer_Data_Audit_enriched", "fan_linkage"]
CSVS = ["home_games_by_match", "purchasers_by_zip"]
EXCEL_MAX_ROWS = 1_048_575


def latest(data_dir, prefix, ext):
    pat = re.compile(rf"^{re.escape(prefix)}_(\d{{8}})(?:_(\d{{4}}))?\.{ext}$", re.I)
    hits = []
    for f in glob.glob(os.path.join(data_dir, f"{prefix}_*.{ext}")):
        b = os.path.basename(f)
        if "LOCAL_ONLY" in b.upper():
            continue
        m = pat.match(b)
        if m:
            hits.append(((m.group(1), m.group(2) or ""), f))
    return max(hits)[1] if hits else None


def vtype(v):
    if isinstance(v, (bool, np.bool_)):
        return "bool"
    if isinstance(v, (int, float, np.integer, np.floating)) and not (isinstance(v, float) and np.isnan(v)):
        return "number"
    if isinstance(v, (pd.Timestamp, dt.date, dt.datetime, np.datetime64)):
        return "date"
    return "text"


def melt_sheet(df, kind, file, run_date, sheet):
    """One sheet -> long rows. Empty cells are dropped; everything else is kept as written."""
    if df is None or df.empty or df.shape[1] == 0:
        return None
    d = df.copy()
    d.columns = [str(c) for c in d.columns]          # sheets may have columns named row, column or value: use placeholders
    first = d.columns[0]
    d.insert(0, "__row", np.arange(1, len(d) + 1))
    d.insert(1, "__label", d[first].map(lambda v: "" if pd.isna(v) else str(v)))
    long = d.melt(id_vars=["__row", "__label"], var_name="__column", value_name="__value")
    long = long[long["__value"].notna()]
    if long.empty:
        return None
    long = long[~long["__value"].map(lambda v: isinstance(v, str) and v.strip() == "")]
    long = long.rename(columns={"__row": "row", "__label": "row_label", "__column": "column", "__value": "value"})
    long.insert(0, "workbook", kind)
    long.insert(1, "file", file)
    long.insert(2, "run_date", run_date)
    long.insert(3, "sheet", sheet)
    long["value_type"] = long["value"].map(vtype)
    num = pd.to_numeric(long["value"].where(long["value_type"].eq("number")), errors="coerce")
    long["value_num"] = num.astype(float)
    long["value"] = long["value"].map(lambda v: v.date() if isinstance(v, pd.Timestamp) and v.normalize() == v else v)
    return long[["workbook", "file", "run_date", "sheet", "row", "row_label", "column", "value", "value_num", "value_type"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=A.DEFAULT_DIR)
    ap.add_argument("--out-dir", default=None, help="where to write (default: the data dir)")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--json-max-rows", type=int, default=30000, help="rows of the long table to include in the JSON (the index is always complete)")
    a = ap.parse_args()
    out_dir = a.out_dir or a.data_dir
    os.makedirs(out_dir, exist_ok=True)

    parts, index = [], []
    for kind in WORKBOOKS:
        f = latest(a.data_dir, kind, "xlsx")
        if not f:
            A.log(f"  {kind}: no dated workbook found, skipped")
            continue
        run_date = re.search(r"_(\d{8})", os.path.basename(f)).group(1)
        try:
            sheets = pd.read_excel(f, sheet_name=None)
        except Exception as e:
            A.log(f"  {kind}: could not read {os.path.basename(f)}: {e}")
            continue
        for sheet, df in sheets.items():
            long = melt_sheet(df, kind, os.path.basename(f), run_date, sheet)
            n = 0 if long is None else len(long)
            index.append({"workbook": kind, "file": os.path.basename(f), "run_date": run_date, "sheet": sheet,
                          "rows": int(len(df)), "columns": int(df.shape[1]), "cells_kept": n})
            if long is not None:
                parts.append(long)
        A.log(f"  {kind}: {os.path.basename(f)}, {len(sheets)} sheets")
    for kind in CSVS:
        f = latest(a.data_dir, kind, "csv")
        if not f:
            A.log(f"  {kind}: no dated csv found, skipped")
            continue
        run_date = re.search(r"_(\d{8})", os.path.basename(f)).group(1)
        df = pd.read_csv(f)
        long = melt_sheet(df, kind, os.path.basename(f), run_date, "csv")
        index.append({"workbook": kind, "file": os.path.basename(f), "run_date": run_date, "sheet": "csv",
                      "rows": int(len(df)), "columns": int(df.shape[1]), "cells_kept": 0 if long is None else len(long)})
        if long is not None:
            parts.append(long)
        A.log(f"  {kind}: {os.path.basename(f)}")
    if not parts:
        sys.exit("nothing found to stack")
    E = pd.concat(parts, ignore_index=True)
    E.insert(0, "n", np.arange(1, len(E) + 1))
    I = pd.DataFrame(index)
    notes = pd.DataFrame({"note": [
        "One row per non-empty cell of every sheet of the latest dated results workbook of each kind, plus the pipeline's small aggregate CSVs.",
        "Filter on workbook + sheet + row to reassemble one row of the original sheet; row_label carries that row's first-column value.",
        "value keeps the cell as written; value_num is the numeric copy for >= / <= filters; value_type says what the cell is.",
        "Aggregate only: the raw client exports and every *_LOCAL_ONLY file are never opened. Safe to share with the team.",
        f"Built {dt.date.today():%Y-%m-%d} by tvof_results_table.py from {os.path.abspath(a.data_dir)}.",
    ]})
    stamp = f"{dt.date.today():%Y%m%d}"
    csv_out = os.path.join(out_dir, f"results_master_{stamp}.csv")
    E.to_csv(csv_out, index=False)
    A.log(f"Wrote {csv_out}  ({len(E):,} rows)")
    xlsx_out = os.path.join(out_dir, f"results_master_{stamp}.xlsx")
    if len(E) > EXCEL_MAX_ROWS:
        A.log(f"  {len(E):,} rows exceed Excel's limit; the workbook holds the first {EXCEL_MAX_ROWS:,} rows, the CSV is complete")
    with pd.ExcelWriter(xlsx_out, engine="openpyxl") as xw:
        E.head(EXCEL_MAX_ROWS).to_excel(xw, sheet_name="Everything", index=False)
        I.to_excel(xw, sheet_name="Index", index=False)
        notes.to_excel(xw, sheet_name="Notes", index=False)
    A.autofilter(xlsx_out)
    A.log(f"Wrote {xlsx_out}  (aggregate only)")
    if a.json_out:
        J = E.head(a.json_max_rows).copy()
        J["value"] = J.value.map(lambda v: v.isoformat() if isinstance(v, (dt.date, dt.datetime, pd.Timestamp)) else (None if isinstance(v, float) and np.isnan(v) else v))
        J["value_num"] = J.value_num.where(J.value_num.notna(), None)
        json.dump({"index": json.loads(I.to_json(orient="records")), "rows_total": int(len(E)), "rows_in_json": int(len(J)),
                   "rows": json.loads(J.to_json(orient="records")), "notes": notes.note.tolist(),
                   "csv": os.path.basename(csv_out), "xlsx": os.path.basename(xlsx_out)}, open(a.json_out, "w"), indent=0)
        A.log(f"Wrote {a.json_out}")
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
    print(I.to_string(index=False))
    print(f"\nTOTAL: {len(E):,} rows from {I.workbook.nunique()} workbooks and {len(I)} sheets; "
          f"{(E.value_type == 'number').mean():.0%} numeric cells; csv {os.path.getsize(csv_out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
