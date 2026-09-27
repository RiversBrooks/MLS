"""
profile_dataset.py — Draft Profile Sheet generator for the Austin FC TVOF project.

Runs the automatable parts of the ten-step framework on one CSV / Excel file:
  Step 1 Dataset · Step 2 Grain (candidate keys) · Step 3 Time Period ·
  Step 4 Fan IDs · Step 6 Missingness · Step 7 Duplicates · Step 8 (numeric field summary)
Steps 5, 9 and 10 require judgment and are filled in by hand.

Usage:
  python profile_dataset.py path/to/file.csv
  python profile_dataset.py path/to/file.xlsx --sheet "Sales"
  python profile_dataset.py file.csv --id account_id,email --date sale_date,event_date --money price,total
  python profile_dataset.py file.csv --out profiles/tix_2024.md

If --id / --date / --money are omitted the script guesses from column names.
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ID_PAT = re.compile(r"(account|customer|fan|contact|member|patron|buyer|user|person|household|email|_id$|id$|number$)", re.I)
DATE_PAT = re.compile(r"(date|time|_dt$|_ts$|created|updated|renew|expire|scan)", re.I)
MONEY_PAT = re.compile(r"(price|amount|revenue|total|fee|paid|cost|value|charge|net|gross|\$)", re.I)
NULL_TOKENS = {"", "na", "n/a", "null", "none", "unknown", "-", "--", "0000-00-00"}


def load(path: Path, sheet=None) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        xl = pd.ExcelFile(path)
        if sheet is None:
            if len(xl.sheet_names) > 1:
                print(f"NOTE: workbook has sheets {xl.sheet_names}; profiling '{xl.sheet_names[0]}'. Use --sheet to pick another.")
            sheet = xl.sheet_names[0]
        return pd.read_excel(path, sheet_name=sheet)
    return pd.read_csv(path, low_memory=False)


def guess(cols, pat, override):
    if override:
        return [c.strip() for c in override.split(",") if c.strip() in cols]
    return [c for c in cols if pat.search(str(c))]


def section(title):
    return f"\n## {title}\n"


def profile(df: pd.DataFrame, name: str, id_cols, date_cols, money_cols) -> str:
    out = [f"# Draft Profile Sheet — {name}", "_Auto-generated. Verify every line; complete Steps 5, 9, 10 by hand._"]

    # Step 1
    out.append(section("1. Dataset"))
    out.append(f"- File: {name}\n- Rows × Columns: {df.shape[0]:,} × {df.shape[1]}")
    out.append("- Columns (dtype, non-null %):")
    for c in df.columns:
        out.append(f"  - `{c}` — {df[c].dtype}, {df[c].notna().mean()*100:.1f}% populated")

    # Placeholder-null cleanup for analysis (does not modify source)
    work = df.copy()
    for c in [c for c in work.columns if not pd.api.types.is_numeric_dtype(work[c]) and not pd.api.types.is_datetime64_any_dtype(work[c])]:
        s = work[c].astype(str).str.strip().str.lower()
        work.loc[s.isin(NULL_TOKENS), c] = pd.NA

    # Step 2 — candidate keys
    out.append(section("2. Grain / Level"))
    out.append("- One row = ______ (fill in)")
    exact_dups = int(df.duplicated().sum())
    uniq_single = [c for c in df.columns if df[c].notna().all() and df[c].is_unique]
    out.append(f"- Single columns that are unique & fully populated: {uniq_single or 'none'}")
    if not uniq_single:
        pairs = []
        cand = [c for c in df.columns if df[c].nunique() > df.shape[0] * 0.01][:12]
        for i, a in enumerate(cand):
            for b in cand[i + 1:]:
                if not df.duplicated(subset=[a, b]).any():
                    pairs.append((a, b))
        out.append(f"- Column pairs that are unique: {pairs[:8] or 'none found among top candidates'}")
        if not pairs:
            out.append("- No unique key found → file is aggregated or contains duplicate rows (see Step 7).")

    # Step 3 — dates
    out.append(section("3. Time Period"))
    if not date_cols:
        out.append("- No date-like columns detected. **Flag for Step 10: cannot place rows in a calendar year.**")
    for c in date_cols:
        d = pd.to_datetime(work[c], errors="coerce")
        if d.notna().sum() == 0:
            out.append(f"- `{c}`: could not parse as date")
            continue
        yrs = d.dt.year.value_counts().sort_index()
        bad = d[(d.dt.year < 2000) | (d.dt.year > 2035)].count()
        out.append(f"- `{c}`: {d.min().date()} → {d.max().date()}, unparseable {d.isna().mean()*100:.1f}%"
                   + (f", **{bad} rows with implausible years (1900/1970 placeholders?)**" if bad else ""))
        out.append("  - Rows per calendar year: " + ", ".join(f"{int(y)}: {n:,}" for y, n in yrs.items() if 2000 <= y <= 2035))
    if len(date_cols) >= 2:
        out.append("- Attribution question: which of these dates defines the calendar year for value? (Step 10)")

    # Step 4 — IDs
    out.append(section("4. Fan / Customer IDs"))
    if not id_cols:
        out.append("- No ID-like columns detected. **Flag for Step 10: no way to attribute rows to a fan.**")
    for c in id_cols:
        s = work[c]
        top = s.value_counts().head(3)
        share_top = top.iloc[0] / s.notna().sum() * 100 if s.notna().sum() else 0
        out.append(f"- `{c}`: {s.notna().mean()*100:.1f}% populated, {s.nunique():,} distinct, "
                   f"top value `{top.index[0]}` = {top.iloc[0]:,} rows ({share_top:.1f}%)"
                   + ("  ← **possible placeholder / broker / house account**" if share_top > 5 else ""))
        lens = s.dropna().astype(str).str.len()
        if lens.nunique() > 3:
            out.append(f"  - Mixed ID lengths ({lens.min()}–{lens.max()}) — check format consistency / leading zeros")
    out.append("- PII present? ______  Handling: ______")

    # Step 5
    out.append(section("5. How Files Connect"))
    out.append("- Joins to → ______ on `______` (type ____, match rate ____%)\n- Role in entity map: master / feed / lookup")

    # Step 6 — missingness
    out.append(section("6. Missingness"))
    miss = (work.isna().mean() * 100).sort_values(ascending=False)
    out.append("- Null % by column (after treating placeholders as null):")
    for c, v in miss.items():
        if v > 0:
            out.append(f"  - `{c}`: {v:.1f}%" + ("  ← **fully empty**" if v >= 99.9 else ""))
    if (miss == 0).all():
        out.append("  - none")
    if date_cols:
        d = pd.to_datetime(work[date_cols[0]], errors="coerce").dt.year
        worst = miss[(miss > 0) & (miss < 99.9)].head(5).index
        if len(worst):
            out.append(f"- Null % by calendar year of `{date_cols[0]}` for top-missing columns:")
            tab = work[worst].isna().groupby(d).mean().mul(100).round(1)
            tab = tab[(tab.index >= 2000) & (tab.index <= 2035)]
            out.append("```\n" + tab.to_string() + "\n```")

    # Step 7 — duplicates
    out.append(section("7. Duplicates"))
    out.append(f"- Exact duplicate rows: {exact_dups:,} ({exact_dups/len(df)*100:.2f}%)")
    for c in uniq_single[:1] + [c for c in id_cols if c not in uniq_single][:2]:
        k = int(df.duplicated(subset=[c]).sum())
        out.append(f"- Duplicate values in `{c}`: {k:,}" + ("  (expected if grain is finer than this ID)" if c in id_cols else ""))
    out.append("- Cause: ______   Rule applied: ______   Rows before/after: ______")

    # Step 8 — money / numeric
    out.append(section("8. Revenue / Behavior Fields"))
    num = [c for c in money_cols if pd.api.types.is_numeric_dtype(df[c])]
    for c in money_cols:
        if c not in num:
            out.append(f"- `{c}`: not numeric ({df[c].dtype}) — strip $ / commas before use")
    if num:
        desc = df[num].describe().T[["count", "mean", "min", "50%", "max"]].round(2)
        out.append("```\n" + desc.to_string() + "\n```")
        for c in num:
            neg = (df[c] < 0).mean() * 100
            zero = (df[c] == 0).mean() * 100
            out.append(f"- `{c}`: {zero:.1f}% zero, {neg:.1f}% negative — confirm comps / refunds treatment")
        if date_cols:
            d = pd.to_datetime(work[date_cols[0]], errors="coerce").dt.year
            tot = df[num].groupby(d).sum().round(0)
            tot = tot[(tot.index >= 2000) & (tot.index <= 2035)]
            out.append(f"- Totals by calendar year of `{date_cols[0]}` (sanity-check against known figures):")
            out.append("```\n" + tot.to_string() + "\n```")
    else:
        out.append("- No money-like numeric columns detected.")
    out.append("- Behavior columns: ______   Segmentation dimensions: ______")

    # Steps 9–10
    out.append(section("9. SOW Questions Supported (F / P / —)"))
    out.append("- Q1 Retention/LTV: __  - Q2 Acquisition/Conversion: __  - Q3 Segmentation: __\n- Q4 Ticketing/Demand: __  - Q5 Peer/Market: __  - Q6 Engagement→Value: __")
    out.append(section("10. Issues / Questions"))
    out.append("1. [Blocker/Clarification/Note] — observation / why it matters / need from Austin FC / owner / working assumption")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--sheet")
    ap.add_argument("--id", help="comma-separated fan/customer ID columns")
    ap.add_argument("--date", help="comma-separated date columns")
    ap.add_argument("--money", help="comma-separated revenue columns")
    ap.add_argument("--out", help="write markdown here instead of stdout")
    a = ap.parse_args()

    path = Path(a.file)
    df = load(path, a.sheet)
    cols = list(df.columns)
    id_cols = guess(cols, ID_PAT, a.id)
    date_cols = guess(cols, DATE_PAT, a.date)
    money_cols = guess(cols, MONEY_PAT, a.money)
    # date columns should not be treated as IDs
    id_cols = [c for c in id_cols if c not in date_cols]

    md = profile(df, path.name, id_cols, date_cols, money_cols)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(md, encoding="utf-8")   # Windows defaults to cp1252, which can't encode → or —
        print(f"wrote {a.out}")
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        print(md)


if __name__ == "__main__":
    sys.exit(main())
