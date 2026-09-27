"""
Austin FC TVOF | Group 2 Data Audit Runner  (v2.7, 2026-09-25, memory-safe)
-------------------------------------------------------------
Runs the full audit on the four client CSVs LOCALLY and writes an
AGGREGATE-ONLY Excel workbook (no row-level IDs, no hashed keys).

Usage (Windows, from any folder):
    python tvof_data_audit.py --data-dir "<your folder of client CSVs>"

Requirements: pandas, openpyxl  (pip install pandas openpyxl)

What it produces (in the data folder unless --out is given):
    TVOF_audit_results_<date>.xlsx   tabs: Run_Log, Column_Profile, Key_Checks,
                                    Match_Rates, Quality_Checks, Dataset_Checks,
                                    Event_Crosswalk, Reconciliation
Nothing is modified in the source files.

v2 (memory-safe): files are read in chunks; ID columns are stored as 64-bit
hashes (same string -> same hash in every file, so joins still work), dates and
amounts are parsed per chunk, text columns become categoricals. Columns not used
by any check are profiled and then dropped. Typical footprint for a 5M-row sales
file: ~1 to 2 GB instead of 10+ GB.
"""
import argparse, gc, glob, json, os, re, sys, datetime as dt
from collections import Counter
import numpy as np
import pandas as pd
from pandas.api.types import union_categoricals, CategoricalDtype
from openpyxl.utils import get_column_letter

# Categorical group keys must never expand to unobserved combinations
# (pandas 2.x default observed=False can explode memory). Force observed=True.
for _cls in (pd.DataFrame, pd.Series):
    _orig = _cls.groupby
    def _gb(self, *a, _orig=_orig, **k):
        k.setdefault("observed", True)
        return _orig(self, *a, **k)
    _cls.groupby = _gb

# Point at your own copy of the client files: set TVOF_DATA_DIR, or pass --data-dir.
DEFAULT_DIR = os.environ.get("TVOF_DATA_DIR", "./data")
NULL_TOKENS = {"", "null", "nan", "none", "n/a", "na", "#n/a", "undefined"}
Q2_CAPACITY = 20738            # from Group 2 MLS stadium audit
CDT_OFFSET_H = 5               # UTC -> CDT (summer 2026)
MAX_GAP_DAYS = 400             # crosswalk: an event more than this after the typical purchase is another season's match
MIN_INFER_SEATS = 500          # crosswalk: undated products need at least this many sold seats to infer an event
CHUNK = 100_000

# Stored as uint64 hashes (joins/uniqueness work, strings not kept in memory)
HASH_COLS = {"internal_fan_id", "internal_account_id", "mls_id", "ticketid", "sales_item_id",
             "primary_ticket_id", "subscription_instance_id", "product_item_id", "transaction_id"}
DATE_COLS = {"signup_datetime", "ticket_scan_datetime", "attendeddatetime", "transaction_date",
             "seatgeek_since_date", "seatgeek_stm_since_date"}
NUM_COLS = {"total_payment", "total_plan_amount"}
# Profiled, then dropped (no check needs them row-level)
DROP_AFTER_PROFILE = {"sales": {"product_item_id", "transaction_id", "sales_details", "price_level",
                                "sales_rep", "sales_rep.1"}}
STATS = {}   # (dataset, column) -> streaming profile stats

# filename keyword -> logical dataset
FILE_PATTERNS = {
    "fan":        ["fan_info", "fan info", "fan_dataset", "_fan_", "fan"],
    "activation": ["activation", "world_cup", "worldcup", "fevo"],
    "attendance": ["attendance", "scan"],
    "sales":      ["sales", "history", "ticket_sales"],
}

LOG, PROFILE, KEYS, MATCH, QUALITY, DSCHECK, RECON = [], [], [], [], [], [], []


def log(msg):
    print(msg)
    LOG.append({"time": dt.datetime.now().strftime("%H:%M:%S"), "message": msg})


# ------------------------------------------------------------------ loading
def autofilter(path):
    """Project convention: every table we produce is filterable on every column. Turns on Excel AutoFilter over the
    used range of every sheet in the workbook and freezes the header row. Call it after each ExcelWriter block."""
    from openpyxl import load_workbook
    try:
        wb = load_workbook(path)
    except Exception as e:  # never let a cosmetic step break a run
        log(f"  autofilter skipped for {os.path.basename(path)}: {e}")
        return
    for ws in wb.worksheets:
        if ws.max_row > 1 and ws.max_column >= 1:
            ws.auto_filter.ref = ws.dimensions
            ws.freeze_panes = "A2"
    wb.save(path)


def find_files(data_dir):
    csvs = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    # skip superseded copies, e.g. "..._2026_09_18 (OLD).csv"
    csvs = [f for f in csvs if not re.search(r"\b(old|backup|bak|truncated)\b", os.path.basename(f).lower())]
    # skip files these scripts write into the data folder (e.g. scan_sale_join_LOCAL_ONLY.csv matches "scan")
    csvs = [f for f in csvs if not re.search(r"local_only|_summary|account_bridge|scan_sale_join|fan_value|column_overlap|home_games|purchasers_by_zip|peer_benchmarks|insights",
                                             os.path.basename(f).lower())]
    found = {}
    for key in ["activation", "attendance", "sales", "fan"]:  # fan last: broadest word
        hits = [f for f in csvs if f not in found.values()
                and any(p in os.path.basename(f).lower() for p in FILE_PATTERNS[key])]
        if hits:
            found[key] = hits[0]
            if len(hits) > 1:
                log(f"  WARNING: {len(hits)} files match '{key}'; using {os.path.basename(hits[0])}. "
                    f"Pass --{key} or rename the others to be sure.")
    return found


def _is_id(c):
    return any(h in c.lower() for h in ID_HINTS)


def _hash(vals):
    """object array (may contain None) -> nullable UInt64 hash."""
    vals = np.asarray(vals, dtype=object)
    mask = pd.isna(vals)
    h = pd.util.hash_array(np.where(mask, "", vals).astype(object), categorize=True)
    return pd.array(h, dtype="UInt64").copy() if not mask.any() else \
        pd.arrays.IntegerArray(h, mask)


def _parse_dt(s):
    s = s.str.replace(r"(T?\d{2}:\d{2}:\d{2}):(\d{3})$", r"\1.\2", regex=True)  # '17:08:32:00.000' style
    d = pd.to_datetime(s, errors="coerce", format="ISO8601")
    bad = s.notna() & d.isna()
    if bad.any():
        d[bad] = pd.to_datetime(s[bad], errors="coerce", format="mixed")
    return d


def load(name, path, chunksize=CHUNK):
    header = pd.read_csv(path, nrows=0, encoding_errors="replace").columns.tolist()
    dups = [c for c in header if re.match(r".+\.\d+$", c)]
    if dups:
        q(name, "duplicate column names in source header (pandas renamed)", len(dups), None, "High",
          ", ".join(dups) + "  (dictionary lists sales_rep twice: ID and name)")
    drop = {d.lower() for d in DROP_AFTER_PROFILE.get(name, set())}
    parts = {c: [] for c in header if c.lower() not in drop}
    n = 0
    for ch in pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False,
                          chunksize=chunksize, encoding_errors="replace"):
        n += len(ch)
        for c in ch.columns:
            st = STATS.setdefault((name, c), {"nn": 0, "lit": 0, "lens": Counter(), "hashes": [],
                                             "vc": Counter(), "vc_ok": not _is_id(c), "parse_fail": 0})
            s = ch[c].str.strip()
            is_null = s.str.lower().isin(NULL_TOKENS)
            st["lit"] += int((is_null & (s != "")).sum())
            s = s.mask(is_null)
            v = s.dropna()
            st["nn"] += len(v)
            if len(v):
                st["lens"].update(v.str.len().value_counts().to_dict())
                st["hashes"].append(np.unique(pd.util.hash_array(v.to_numpy(dtype=object), categorize=True)))
                if len(st["hashes"]) >= 8:
                    st["hashes"] = [np.unique(np.concatenate(st["hashes"]))]
                if st["vc_ok"]:
                    st["vc"].update(v.value_counts().to_dict())
                    if len(st["vc"]) > 50_000:
                        st["vc_ok"], st["vc"] = False, Counter()
            lc = c.lower()
            if lc in drop:
                continue
            if lc in HASH_COLS:
                parts[c].append(_hash(s.to_numpy(dtype=object)))
            elif lc in DATE_COLS:
                d = _parse_dt(s)
                st["parse_fail"] += int((s.notna() & d.isna()).sum())
                parts[c].append(d.reset_index(drop=True))
            elif lc in NUM_COLS:
                x = pd.to_numeric(s.str.replace(r"[$,]", "", regex=True), errors="coerce")
                st["parse_fail"] += int((s.notna() & x.isna()).sum())
                parts[c].append(x.reset_index(drop=True))
            else:
                parts[c].append(s.astype("category").array)
        del ch
        gc.collect()
    out = {}
    for c, lst in parts.items():
        if not lst:
            continue
        first = lst[0]
        if isinstance(first, pd.Categorical):
            out[c] = pd.Series(union_categoricals(lst, ignore_order=True))
        elif isinstance(first, pd.Series):
            out[c] = pd.concat(lst, ignore_index=True)
        else:
            out[c] = pd.Series(pd.concat([pd.Series(a) for a in lst], ignore_index=True))
        parts[c] = []  # release chunk references
        gc.collect()
    df = pd.DataFrame(out, copy=False)
    del out
    gc.collect()
    # profile rows from streaming stats
    for c in header:
        st = STATS[(name, c)]
        distinct = len(np.unique(np.concatenate(st["hashes"]))) if st["hashes"] else 0
        st["distinct"] = distinct
        st["hashes"] = None
        lens = st["lens"]
        PROFILE.append({
            "dataset": name, "column": c, "rows": n, "non_null": st["nn"],
            "null_pct": round(1 - st["nn"] / n, 4) if n else None,
            "literal_null_tokens": st["lit"], "distinct": distinct,
            "distinct_pct_of_non_null": round(distinct / st["nn"], 4) if st["nn"] else None,
            "min_len": min(lens) if lens else None, "max_len": max(lens) if lens else None,
            "parse_failures": st["parse_fail"],
            "kept_in_memory_as": ("dropped after profile" if c.lower() in drop else
                                  "uint64 hash" if c.lower() in HASH_COLS else
                                  "datetime" if c.lower() in DATE_COLS else
                                  "float" if c.lower() in NUM_COLS else "category"),
            "top_values (masked for IDs)": "masked" if _is_id(c) else
                ("; ".join(f"{k} ({v})" for k, v in st["vc"].most_common(3))[:250] if st["vc_ok"]
                 else "high cardinality"),
        })
        if st["parse_fail"]:
            q(name, f"{c} unparseable (non-null)", st["parse_fail"], st["nn"], "Moderate")
    return df


def col(df, *cands):
    """Return first matching column (case-insensitive)."""
    low = {c.lower(): c for c in df.columns}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    return None


# ------------------------------------------------------------------ profiling
ID_HINTS = ("id", "key", "zip", "rep")


def key_check(name, df, cols, label=None):
    cols = [c for c in cols if c]
    if not cols:
        return
    sub = df[cols]
    all_present = sub.notna().all(axis=1)
    dup = sub[all_present].duplicated(keep=False)
    KEYS.append({
        "dataset": name, "candidate_key": label or " + ".join(cols),
        "rows": len(df), "rows_with_all_key_parts": int(all_present.sum()),
        "distinct_key_values": int(sub[all_present].drop_duplicates().shape[0]),
        "rows_in_duplicate_groups": int(dup.sum()),
        "is_unique": bool(dup.sum() == 0 and all_present.all()),
    })


def match_rate(label, left_name, lk, right_name, rk, note=""):
    ls, rs = lk.dropna(), rk.dropna()
    lu, ru = set(ls.unique()), set(rs.unique())
    inter = lu & ru
    MATCH.append({
        "join": label, "left": left_name, "right": right_name,
        "left_rows_non_null_key": len(ls),
        "left_rows_matched": int(ls.isin(ru).sum()),
        "left_row_match_rate": round(ls.isin(ru).mean(), 4) if len(ls) else None,
        "left_distinct_keys": len(lu), "right_distinct_keys": len(ru),
        "distinct_keys_in_both": len(inter),
        "left_distinct_match_rate": round(len(inter) / len(lu), 4) if lu else None,
        "right_distinct_match_rate": round(len(inter) / len(ru), 4) if ru else None,
        "note": note,
    })


def q(dataset, check, value, denom=None, severity="", note=""):
    QUALITY.append({"dataset": dataset, "check": check,
                    "count": value, "denominator": denom,
                    "pct": round(value / denom, 4) if denom else None,
                    "severity": severity, "note": note})


# ------------------------------------------------------------------ helpers
def cat_apply(s, fn):
    """Apply fn to categories only (cheap), then expand by codes."""
    if isinstance(s.dtype, CategoricalDtype):
        cats = pd.Series(s.cat.categories.astype(object))
        res = fn(cats).reset_index(drop=True)
        return res.reindex(s.cat.codes.to_numpy()).set_axis(s.index)
    return fn(s)


def fill_null(s, token="<null>"):
    if isinstance(s.dtype, CategoricalDtype):
        if token not in s.cat.categories:
            s = s.cat.add_categories([token])
        return s.fillna(token)
    return s.astype(object).fillna(token)


def _seat_norm(s):
    s = s.astype(object).where(s.notna()).astype("string").str.upper().str.strip()
    s = s.str.replace(r"^(SECTION|SEC|ROW|SEAT)\s*", "", regex=True).str.strip()
    num = s.str.fullmatch(r"\d+", na=False)
    s = s.where(~num, s.str.lstrip("0").replace("", "0"))
    return s.astype(object)


def seat_norm(s):
    """'Section 114' -> '114', 'Row 17' -> '17', ' 004' -> '4'."""
    return cat_apply(s, _seat_norm)


# A 3-4 digit numeric zip is a 5-digit US zip that lost its leading zero(s) in export when the restored
# prefix is one the US uses: 005 (Holtsville), 006-009 (PR/VI), 010-069 (New England), 070-089 (NJ), 090-098 (AE).
US_ZIP3_RESTORED = ("005", "098")
# Well-formed postal codes of other countries: valid addresses, just not US ZIPs.
FOREIGN_POSTAL = (r"(?i)^(?:[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}"   # UK
                  r"|[A-Z]\d[A-Z]\s?\d[A-Z]\d"                   # Canada
                  r"|\d{4}\s?[A-Z]{2}"                           # Netherlands
                  r"|\d{3}\s\d{2}"                               # Sweden
                  r"|\d{5}-?\d{3}"                               # Brazil
                  r"|\d{4}-\d{3}"                                # Portugal
                  r"|\d{3}-?\d{4}"                               # Japan
                  r"|\d{6}"                                      # India, China
                  r"|[A-Z]\d{2}\s?[A-Z\d]{4})$")                 # Ireland


def _restorable(z):
    """3-4 digit numeric values whose zero-padded ZIP3 prefix is one the US uses."""
    short = z.str.fullmatch(r"\d{3,4}", na=False)
    p3 = z.str.zfill(5).str[:3]
    return (short & p3.between(*US_ZIP3_RESTORED)).fillna(False).astype(bool)


def _zip_class(z):
    z = z.astype("string").str.strip()
    out = pd.Series("missing", index=z.index, dtype=object)
    out[z.str.fullmatch(r"\d{5}", na=False)] = "5-digit"
    out[z.str.fullmatch(r"\d{5}-?\d{4}", na=False)] = "ZIP+4"
    short = z.str.fullmatch(r"\d{3,4}", na=False)
    rest = _restorable(z)
    out[rest] = "5-digit (leading zero restored)"
    out[short & ~rest] = "3-4 digit, not a US prefix"
    out[(out == "missing") & z.str.fullmatch(FOREIGN_POSTAL, na=False).fillna(False).astype(bool)] = "non-US postal code (recognised format)"
    out[z.notna() & (out == "missing")] = "malformed"
    return out


def zip_class(z):
    return cat_apply(z, _zip_class).fillna("missing")


def _zip5(z):
    """5-digit zip: ZIP+4 truncated, lost leading zeros restored; <NA> otherwise."""
    z = z.astype("string").str.strip()
    ok = z.str.fullmatch(r"\d{5}(-?\d{4})?", na=False)
    return z.str[:5].where(ok, z.str.zfill(5).where(_restorable(z))).astype(object)


def zip5(z):
    return cat_apply(z, _zip5)


def to_dt(s, **kw):
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    return cat_apply(s, lambda x: pd.to_datetime(x, errors="coerce", **kw))


def to_num(s):
    if pd.api.types.is_numeric_dtype(s):
        return s
    return cat_apply(s, lambda x: pd.to_numeric(x.astype("string").str.replace(r"[$,]", "", regex=True),
                                                errors="coerce"))


# ------------------------------------------------------------------ dataset checks
def check_fan(df):
    n = len(df)
    fid, aid, mid = col(df, "internal_fan_id"), col(df, "internal_account_id"), col(df, "mls_id")
    key_check("fan", df, [fid]); key_check("fan", df, [aid]); key_check("fan", df, [mid])
    key_check("fan", df, [fid, aid]); key_check("fan", df, [fid, aid, mid])
    pres = pd.DataFrame({k: (df[c].notna() if c else pd.Series(False, index=df.index)) for k, c in
                         [("fan_id", fid), ("account_id", aid), ("mls_id", mid)]})
    combo = pres.apply(lambda r: "+".join(k for k, v in r.items() if v) or "NONE", axis=1)
    for k, v in combo.value_counts().items():
        DSCHECK.append({"dataset": "fan", "check": "ID presence pattern", "value": k,
                        "rows": int(v), "pct": round(v / n, 4)})
    q("fan", "rows with no ID at all", int((combo == "NONE").sum()), n, "High")
    if fid and aid:
        m = df[[fid, aid]].dropna()
        per_fan = m.groupby(fid)[aid].nunique()
        per_acct = m.groupby(aid)[fid].nunique()
        q("fan", "fan_ids mapped to >1 account_id", int((per_fan > 1).sum()), len(per_fan), "High",
          "One person, many SeatGeek accounts (or shared hash). Decide identity rule.")
        q("fan", "account_ids mapped to >1 fan_id", int((per_acct > 1).sum()), len(per_acct), "High",
          "Shared/household/corporate accounts or email join fan-out.")
    if mid:
        lens = STATS[("fan", mid)]["lens"]
        DSCHECK.append({"dataset": "fan", "check": "mls_id length distribution",
                        "value": json.dumps({int(k): int(v) for k, v in lens.most_common(5)}),
                        "rows": int(df[mid].notna().sum()), "pct": None})
    sgz, mlz = col(df, "seatgeek_zip_code"), col(df, "mls_zip_code")
    for c in [sgz, mlz]:
        if c:
            for k, v in zip_class(df[c]).value_counts().items():
                DSCHECK.append({"dataset": "fan", "check": f"{c} format", "value": k,
                                "rows": int(v), "pct": round(v / n, 4)})
    if sgz and mlz:
        a, b = zip5(df[sgz]), zip5(df[mlz])
        both = a.notna() & b.notna()
        q("fan", "seatgeek_zip vs mls_zip disagree (both present)", int((a[both] != b[both]).sum()),
          int(both.sum()), "Moderate", "Apply Source-of-Truth rule: SeatGeek wins for transactors.")
    for c in [col(df, "seatgeek_since_date"), col(df, "seatgeek_stm_since_date")]:
        if c:
            d = to_dt(df[c])
            if d.notna().any():
                DSCHECK.append({"dataset": "fan", "check": f"{c} range",
                                "value": f"{d.min().date()} to {d.max().date()}",
                                "rows": int(d.notna().sum()), "pct": None})
                q("fan", f"{c} in the future (> extract date)", int((d > pd.Timestamp("2026-09-18")).sum()),
                  int(d.notna().sum()), "Moderate")
    ss, stm = col(df, "seatgeek_since_date"), col(df, "seatgeek_stm_since_date")
    if ss and stm:
        a, b = to_dt(df[ss]), to_dt(df[stm])
        q("fan", "STM since date earlier than SeatGeek since date", int((b < a).sum()),
          int((a.notna() & b.notna()).sum()), "Moderate", "Tenure logic conflict.")
    opt = col(df, "mls_club_fan_marketing_optin_flag")
    if opt:
        for k, v in fill_null(df[opt]).astype(str).str.lower().value_counts().items():
            DSCHECK.append({"dataset": "fan", "check": "marketing opt-in values", "value": k,
                            "rows": int(v), "pct": round(v / n, 4)})


def check_activation(df):
    n = len(df)
    tid, fid, eid = col(df, "ticketId"), col(df, "internal_fan_id"), col(df, "eventID")
    en, ed = col(df, "event_name"), col(df, "event_date")
    su, sc = col(df, "signup_datetime"), col(df, "ticket_scan_datetime")
    z = col(df, "fevo_zip_code", "zip_code")
    key_check("activation", df, [tid]); key_check("activation", df, [fid, eid])
    key_check("activation", df, [fid, eid, ed], "internal_fan_id + eventID + event_date")
    sud, scd = to_dt(df[su]), to_dt(df[sc])
    g = pd.DataFrame({"event": df[en], "scan": scd.notna(), "edate_null": df[ed].isna(),
                      "fan": df[fid]})
    agg = g.groupby("event").agg(rows=("scan", "size"), fans=("fan", "nunique"),
                                 scanned=("scan", "sum"), event_date_null=("edate_null", "sum"))
    agg["scan_rate"] = (agg.scanned / agg.rows).round(4)
    for ev, r in agg.iterrows():
        DSCHECK.append({"dataset": "activation", "check": "scan rate by event", "value": ev,
                        "rows": int(r.rows), "pct": r.scan_rate})
    both = sud.notna() & scd.notna()
    raw_neg = int((sud[both] > scd[both]).sum())
    shifted = sud - pd.Timedelta(hours=CDT_OFFSET_H)
    adj_neg = int((shifted[both] > scd[both] + pd.Timedelta(minutes=2)).sum())
    q("activation", "signup after scan (raw timestamps)", raw_neg, int(both.sum()), "High",
      "If this drops to ~0 after a -5h shift, signup_datetime is UTC, not Central.")
    q("activation", "signup after scan (signup shifted -5h)", adj_neg, int(both.sum()), "Info")
    zc = zip_class(df[z]) if z else pd.Series(dtype=str)
    for k, v in zc.value_counts().items():
        DSCHECK.append({"dataset": "activation", "check": "zip format", "value": k,
                        "rows": int(v), "pct": round(v / n, 4)})


def check_attendance(df):
    n = len(df)
    ek, dk, sk = col(df, "EventKey"), col(df, "EventDateKey"), col(df, "SeasonKey")
    aid, at = col(df, "internal_account_id", "AttendedAccountKey"), col(df, "AttendedDatetime")
    sec, row, seat = col(df, "SectionName"), col(df, "RowName"), col(df, "SeatName")
    key_check("attendance", df, [ek, sec, row, seat], "EventKey + Section + Row + Seat (hypothesis)")
    key_check("attendance", df, [ek, sec, row, seat, at], "... + AttendedDatetime")
    key_check("attendance", df, [ek, aid, sec, row, seat, at], "... + account + time (full row)")
    if ek and sec:
        k = df[[ek, sec, row, seat]].dropna()
        dup = k.duplicated(keep="first").sum()
        q("attendance", "extra scans of an already-scanned seat (re-entry or duplicate)",
          int(dup), len(k), "High", "COUNT(*) overstates attendance by this amount.")
    if ek:
        ev = df.groupby(ek).size().rename("scans").to_frame()
        uniq = df.dropna(subset=[ek, sec, row, seat]).drop_duplicates([ek, sec, row, seat]).groupby(ek).size()
        ev["unique_seats_scanned"] = uniq
        if dk: ev["event_date"] = df.groupby(ek)[dk].first()
        if sk: ev["season"] = df.groupby(ek)[sk].first()
        nm = col(df, "MasterEventName", "EventName")
        if nm: ev["event_name"] = df.groupby(ek)[nm].first()
        ev["pct_of_capacity_unique_seats"] = (ev.unique_seats_scanned / Q2_CAPACITY).round(4)
        ev = ev.reset_index().sort_values("event_date" if dk else ek)
        RECON.append(("Attendance_by_Event", ev))
        if sk:
            s = ev.groupby("season").agg(events=(ek, "count"), scans=("scans", "sum"),
                                         unique_seats=("unique_seats_scanned", "sum")).reset_index()
            RECON.append(("Attendance_by_Season", s))
        q("attendance", "events with unique scanned seats > Q2 capacity 20,738",
          int((ev.unique_seats_scanned > Q2_CAPACITY).sum()), len(ev), "High")
        if dk:
            q("attendance", "EventKeys with >1 EventDateKey",
              int((df.groupby(ek)[dk].nunique() > 1).sum()), int(df[ek].nunique()), "Moderate")
    if dk and at:
        d_evt = to_dt(df[dk], format="%Y%m%d")
        d_scan = to_dt(df[at])
        diff = (d_scan.dt.normalize() - d_evt).dt.days
        q("attendance", "scan date not on event date (|days|>=1)", int((diff.abs() >= 1).sum()),
          int(diff.notna().sum()), "Moderate", "Late-night scans/TZ or mis-keyed events.")
    for c in [col(df, "SectionCategory"), col(df, "SeatingArea")]:
        if c:
            for k, v in fill_null(df[c]).value_counts().head(30).items():
                DSCHECK.append({"dataset": "attendance", "check": f"{c} values", "value": k,
                                "rows": int(v), "pct": round(v / n, 4)})
    if aid:
        q("attendance", "scans with null internal_account_id", int(df[aid].isna().sum()), n, "Moderate")


def check_sales(df):
    n = len(df)
    sid, ptid, sub = col(df, "sales_item_id"), col(df, "primary_ticket_id"), col(df, "subscription_instance_id")
    pid, aid = col(df, "product_id"), col(df, "internal_account_id")
    it, pt, st = col(df, "item_type"), col(df, "product_type"), col(df, "sale_type")
    ts, rs = col(df, "transfer_status"), col(df, "resale_status")
    pay, plan = col(df, "total_payment"), col(df, "total_plan_amount")
    ptype, ptg = col(df, "price_type"), col(df, "price_type_group")
    tdate = col(df, "transaction_date")
    key_check("sales", df, [sid]); key_check("sales", df, [ptid])
    key_check("sales", df, [pid, col(df, "section"), col(df, "row"), col(df, "seat")], "product_id + section + row + seat")
    for a, b in [(it, pt), (it, st)]:
        if a and b:
            ct = pd.crosstab(fill_null(df[a]).astype(str), fill_null(df[b]).astype(str))
            RECON.append((f"Sales_{a}_x_{b}"[:31], ct.reset_index()))
    for c in [ts, rs, col(df, "application_channel"), col(df, "is_hospitality"), ptg]:
        if c:
            for k, v in fill_null(df[c]).value_counts().head(40).items():
                DSCHECK.append({"dataset": "sales", "check": f"{c} values", "value": k,
                                "rows": int(v), "pct": round(v / n, 4)})
    if pay:
        p = to_num(df[pay])
        q("sales", "total_payment negative", int((p < 0).sum()), n, "Moderate")
        q("sales", "total_payment = 0", int((p == 0).sum()), n, "Info", "Comps, transfers, or plan-generated game rows.")
        # Revenue reconciliation: three candidate definitions
        rec = [{"definition": "A. Naive SUM(total_payment) all rows  (DO NOT USE)", "rows": int(p.notna().sum()),
                "amount": round(p.sum(), 2)}]
        orig = True
        if it:
            orig = df[it].str.lower().isin(["ticket", "subscription"])
            rec.append({"definition": "B. Original sales only (item_type in Ticket, Subscription)",
                        "rows": int(orig.sum()), "amount": round(p[orig].sum(), 2)})
            rsl = df[it].str.lower().eq("resale")
            rec.append({"definition": "   Resale rows (secondary market, club share unknown)",
                        "rows": int(rsl.sum()), "amount": round(p[rsl].sum(), 2)})
        if plan and sub:
            pa = to_num(df[plan])
            per = pd.DataFrame({"sub": df[sub], "pa": pa}).dropna()
            nvals = per.groupby("sub").pa.nunique()
            q("sales", "subscription_instance_id with >1 distinct total_plan_amount", int((nvals > 1).sum()),
              len(nvals), "High", "Plan value should repeat identically per plan.")
            rec.append({"definition": "C. Naive SUM(total_plan_amount) all rows  (DO NOT USE)",
                        "rows": int(pa.notna().sum()), "amount": round(pa.sum(), 2)})
            rec.append({"definition": "D. Plan revenue: MAX(total_plan_amount) per subscription_instance_id",
                        "rows": int(per["sub"].nunique()), "amount": round(per.groupby("sub").pa.max().sum(), 2)})
        if tdate:
            yr = to_dt(df[tdate]).dt.year
            by = pd.DataFrame({"year": yr, "p": p, "orig": orig})
            y = by[by.orig].groupby("year").p.agg(["count", "sum"]).reset_index()
            y.columns = ["transaction_year", "original_sale_rows", "total_payment_sum"]
            RECON.append(("Sales_Revenue_by_Year", y))
        RECON.append(("Revenue_Definitions", pd.DataFrame(rec)))
        pdsc = col(df, "product_description")
        if pdsc and it and tdate:
            cls = cat_apply(df[pdsc], lambda x: x.map(product_class))
            rv = pd.DataFrame({"year": to_dt(df[tdate]).dt.year, "item_type": df[it].astype(object),
                               "product_class": cls, "total_payment": p})
            t = rv.groupby(["product_class", "item_type", "year"]).total_payment.agg(["count", "sum"]).reset_index()
            t.columns = ["product_class", "item_type", "transaction_year", "rows", "total_payment_sum"]
            RECON.append(("Revenue_by_Class_Item_Year", t))
            del rv
        if plan and sub and it:
            ps = pd.DataFrame({"sub": df[sub], "is_plan_row": df[it].astype(object).str.lower().eq("subscription"),
                               "pa": to_num(df[plan]), "pay": p}).dropna(subset=["sub"])
            g = pd.DataFrame(ps.groupby(["sub", "is_plan_row"]).agg(n=("pa", "size"), pa_sum=("pa", "sum"),
                                                                   pa_max=("pa", "max"), pay_sum=("pay", "sum")).unstack())
            g.columns = [f"{a}_{'plan' if b else 'game'}" for a, b in g.columns]
            summ = [{"metric": c_, "plans_with_value": int(g[c_].notna().sum()),
                     "total": round(float(g[c_].sum()), 2), "median_per_plan": round(float(g[c_].median()), 2)}
                    for c_ in g.columns]
            if {"pa_max_plan", "pa_sum_game"} <= set(g.columns):
                both = g.dropna(subset=["pa_max_plan", "pa_sum_game"])
                summ.append({"metric": "plans where plan-row amount = SUM(game-row amounts) +/- $1",
                             "plans_with_value": len(both),
                             "total": int(((both.pa_max_plan - both.pa_sum_game).abs() <= 1).sum()),
                             "median_per_plan": None})
            if {"pay_sum_plan", "pay_sum_game"} <= set(g.columns):
                both = g.dropna(subset=["pay_sum_plan", "pay_sum_game"])
                summ.append({"metric": "plans with total_payment on BOTH plan row and game rows (double-count risk)",
                             "plans_with_value": len(both),
                             "total": int(((both.pay_sum_plan > 0) & (both.pay_sum_game > 0)).sum()),
                             "median_per_plan": None})
            RECON.append(("Plan_Amount_Structure", pd.DataFrame(summ)))
            del ps, g
    if ptype:
        comp = df[ptype].str.contains("comp", case=False, na=False)
        q("sales", "rows with 'comp' in price_type", int(comp.sum()), n, "Info",
          "Exclude from revenue/ASP; keep for distribution/attendance.")
    if ptid:
        chain = df.groupby(ptid).size()
        for k, v in chain.value_counts().sort_index().head(10).items():
            DSCHECK.append({"dataset": "sales", "check": "rows per primary_ticket_id (chain length)",
                            "value": str(k), "rows": int(v), "pct": round(v / len(chain), 4)})
    if it and ptid:
        q("sales", "non-subscription rows with null primary_ticket_id",
          int((df[it].str.lower().ne("subscription") & df[ptid].isna()).sum()), n, "Moderate")
    if aid:
        q("sales", "rows with null internal_account_id", int(df[aid].isna().sum()), n, "Moderate",
          "Cannot attribute to a fan.")
    if tdate:
        d = to_dt(df[tdate])
        if d.notna().any():
            DSCHECK.append({"dataset": "sales", "check": "transaction_date range",
                            "value": f"{d.min()} to {d.max()}", "rows": int(d.notna().sum()), "pct": None})
    if aid and pay:
        # broker screen: accounts with many resale-out or very high ticket counts
        acct = df.groupby(aid).size()
        q("sales", "accounts with >= 500 rows (broker/group screen)", int((acct >= 500).sum()), len(acct), "Moderate",
          "Behavioral screen only; confirm with club.")


# ------------------------------------------------------------------ cross-file
def cross_file(D):
    fan, act, att, sal = D.get("fan"), D.get("activation"), D.get("attendance"), D.get("sales")
    fk = lambda df, *c: df[col(df, *c)] if df is not None and col(df, *c) else pd.Series(dtype=str)
    if act is not None and fan is not None:
        match_rate("J1 Activation -> Fan Info", "activation", fk(act, "internal_fan_id"),
                   "fan", fk(fan, "internal_fan_id"), "Bridge for all activation->ticketing analysis")
    if fan is not None and sal is not None:
        match_rate("J2 Fan Info -> Sales", "fan", fk(fan, "internal_account_id"),
                   "sales", fk(sal, "internal_account_id"))
        match_rate("J2r Sales -> Fan Info", "sales", fk(sal, "internal_account_id"),
                   "fan", fk(fan, "internal_account_id"), "Orphan sales accounts = not in Fan Info")
    if fan is not None and att is not None:
        match_rate("J3 Fan Info -> Attendance", "fan", fk(fan, "internal_account_id"),
                   "attendance", fk(att, "internal_account_id"))
        match_rate("J3r Attendance -> Fan Info", "attendance", fk(att, "internal_account_id"),
                   "fan", fk(fan, "internal_account_id"))
    if sal is not None and att is not None:
        match_rate("J4 Attendance -> Sales (account level)", "attendance", fk(att, "internal_account_id"),
                   "sales", fk(sal, "internal_account_id"), "Scanning account should exist in sales")
    # J5: activation -> fan -> sales conversion bridge
    if act is not None and fan is not None and sal is not None:
        a_f, f_f, f_a = col(act, "internal_fan_id"), col(fan, "internal_fan_id"), col(fan, "internal_account_id")
        s_a, s_d = col(sal, "internal_account_id"), col(sal, "transaction_date")
        br = fan[[f_f, f_a]].dropna().drop_duplicates()
        su = act[[a_f]].copy()
        su["su_ct"] = to_dt(act[col(act, "signup_datetime")]) - pd.Timedelta(hours=CDT_OFFSET_H)
        first_su = su.groupby(a_f).su_ct.min().rename("first_signup").reset_index()
        m = first_su.merge(br, left_on=a_f, right_on=f_f, how="left")
        s = sal[[s_a, s_d]].dropna(subset=[s_a]).copy()
        s["td"] = to_dt(s[s_d])
        acct = s.groupby(s_a).td.agg(["min", "max"]).reset_index()
        m = m.merge(acct, left_on=f_a, right_on=s_a, how="left")
        fans = m.groupby(a_f).agg(has_acct=(f_a, lambda x: x.notna().any()),
                                  any_sale=("min", lambda x: x.notna().any()),
                                  first_sale=("min", "min"), last_sale=("max", "max"),
                                  su=("first_signup", "first"))
        N = len(fans)
        fans["pre_existing_buyer"] = fans.first_sale < fans.su
        fans["new_buyer_after_signup"] = (fans.first_sale >= fans.su)
        fans["any_purchase_after_signup"] = fans.last_sale >= fans.su
        ss = col(fan, "seatgeek_since_date")
        if ss:
            sg = pd.DataFrame({"fid": fan[f_f], "since": to_dt(fan[ss])}).dropna().groupby("fid").since.min()
            fans["sg_since"] = sg.reindex(fans.index).to_numpy()
            fans["sg_before"] = fans.sg_since < fans.su.dt.normalize()
            fans["sg_after"] = fans.sg_since >= fans.su.dt.normalize()
        rows = [("activation fans (distinct)", N),
                ("... with a SeatGeek account in Fan Info", int(fans.has_acct.sum())),
                ("... SeatGeek account created BEFORE first signup (existing customers)",
                 int(fans["sg_before"].sum()) if "sg_before" in fans else None),
                ("... SeatGeek account created ON/AFTER first signup (new-account proxy for conversion)",
                 int(fans["sg_after"].sum()) if "sg_after" in fans else None),
                ("... with any Sales row  (INVALID until Sales account IDs are fixed)", int(fans.any_sale.sum())),
                ("... first purchase BEFORE signup (existing buyers)", int(fans.pre_existing_buyer.sum())),
                ("... first-ever purchase AFTER signup (candidate conversions)", int(fans.new_buyer_after_signup.sum())),
                ("... any purchase after signup (existing or new)", int(fans.any_purchase_after_signup.sum()))]
        RECON.append(("Activation_Funnel", pd.DataFrame(
            [{"stage": a, "fans": b, "pct_of_activation_fans": round(b / N, 4) if b is not None else None}
             for a, b in rows])))
    # J6: sales <-> attendance seat-level via event crosswalk
    if sal is not None and att is not None:
        crosswalk_and_seats(sal, att)


MONTHS_MD = re.compile(r"(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?")
STOP = {"at", "vs", "v", "austin", "fc", "the", "sc", "cf", "home", "game", "match"}


def _tokens(x):
    return {t for t in re.findall(r"[a-z]+", str(x).lower()) if t not in STOP and len(t) > 2}


OVERRIDES = None
CROSSWALK = None        # set by crosswalk_and_seats (product_id -> EventKey, confidence)
ACCOUNT_BRIDGE = None   # sales-hash -> fan-space-hash (UInt64), set by --account-bridge


def _holder_mask(sal):
    """Rows that can hold a seat: not plan rows, not seller rows marked Resold, not Pending/Canceled transfers."""
    it, ts, rs = col(sal, "item_type"), col(sal, "transfer_status"), col(sal, "resale_status")
    ok = pd.Series(True, index=sal.index)
    if it:
        ok &= ~sal[it].astype(object).str.lower().eq("subscription").fillna(False)
    if rs:
        ok &= ~sal[rs].astype(object).str.lower().eq("resold").fillna(False)
    if ts and it:
        ok &= ~(sal[it].astype(object).str.lower().eq("transfer").fillna(False)
                & sal[ts].astype(object).str.lower().isin(["pending", "canceled", "cancelled"]).fillna(False))
    return ok


def final_holder_frame(sal, pmap):
    """One row per (EventKey, seat): the latest valid holder row for match tickets.
    Drops: plan rows, seller rows marked Resold, transfers still Pending or Canceled."""
    p, td, acct = col(sal, "product_id"), col(sal, "transaction_date"), col(sal, "internal_account_id")
    ok = _holder_mask(sal) & sal[p].isin(pmap.index)
    idx = ok.to_numpy()
    T = pd.DataFrame({"product_id": sal[p].astype(object).to_numpy()[idx],
                      "key": seat_key(sal[col(sal, "section")], sal[col(sal, "row")], sal[col(sal, "seat")])[idx],
                      "acct": sal[acct].array[idx] if acct else pd.NA,
                      "td": sal[td].to_numpy()[idx] if td else pd.NaT})
    T["EventKey"] = T.product_id.map(pmap.astype(str))
    T = T.dropna(subset=["EventKey", "key"]).sort_values("td")
    return T.drop_duplicates(["EventKey", "key"], keep="last").drop(columns="td")



def product_class(d):
    d = str(d)
    if re.search(r"PRK|parking", d, re.I): return "parking"
    if re.search(r"\d+\s*(EXT|FB)\b|^(EXT|FB)\b", d, re.I): return "match add-on (EXT/FB)"
    if re.search(r"membership|full season|extra game|deposit", d, re.I): return "plan/membership"
    if re.search(r"FC2\b|Austin FC II\b|at Austin FC I$", d): return "Austin FC II match (not at Q2)"
    if re.search(r"party", d, re.I): return "other stadium event"
    if re.search(r"at Austin FC|playoff|round \d", d, re.I): return "Austin FC match"
    return "other stadium event"


def seat_key(sec, row, seat):
    """uint64 key of normalized section|row|seat; <NA> if any part missing."""
    parts = pd.DataFrame({"a": seat_norm(sec), "b": seat_norm(row), "c": seat_norm(seat)})
    valid = parts.notna().all(axis=1).to_numpy()
    h = pd.util.hash_pandas_object(parts.fillna(""), index=False).to_numpy()
    del parts
    return pd.arrays.IntegerArray(h, ~valid)


def crosswalk_and_seats(sal, att):
    p, ek = col(sal, "product_id"), col(att, "EventKey")
    ssec, srow, sseat = col(sal, "section"), col(sal, "row"), col(sal, "seat")
    asec, arow, aseat = col(att, "SectionName"), col(att, "RowName"), col(att, "SeatName")
    sacct, aacct = col(sal, "internal_account_id"), col(att, "internal_account_id", "AttendedAccountKey")
    desc, tdate = col(sal, "product_description"), col(sal, "transaction_date")
    dk, nm = col(att, "EventDateKey"), col(att, "MasterEventName", "EventName")
    if not all([p, ek, ssec, srow, sseat, asec, arow, aseat, desc, dk]):
        log("  J6 skipped: required columns missing")
        return
    # raw vs normalized label agreement (category level, cheap)
    raw_s = set(sal[ssec].dropna().astype(object).unique()); raw_a = set(att[asec].dropna().astype(object).unique())
    q("cross", "section labels matching exactly (raw, before normalization)", len(raw_s & raw_a), len(raw_a),
      "High", "Low value = format mismatch (e.g. '114' vs 'Section 114').")
    ns = set(_seat_norm(pd.Series(sorted(raw_s))).dropna()); na_ = set(_seat_norm(pd.Series(sorted(raw_a))).dropna())
    q("cross", "section labels matching after normalization", len(ns & na_), len(na_), "Info")

    # ---- 1. event crosswalk: M/D parsed from product_description, year by purchase timing
    P = pd.DataFrame({"product_id": sal[p], "desc": sal[desc],
                      "td": to_dt(sal[tdate]) if tdate else pd.NaT}).dropna(subset=["product_id"])
    P = P.groupby("product_id").agg(desc=("desc", "first"), rows=("desc", "size"), tx_med=("td", "median")).reset_index()
    P["product_id"] = P.product_id.astype(object)
    P["desc"] = P.desc.astype(object)
    md = P.desc.astype(str).apply(lambda d: MONTHS_MD.findall(d)[-1] if MONTHS_MD.findall(d) else None)
    P["m"] = md.apply(lambda t: int(t[0]) if t else None)
    P["d"] = md.apply(lambda t: int(t[1]) if t else None)
    P["y"] = md.apply(lambda t: (int(t[2]) + (2000 if len(t[2]) == 2 else 0)) if t and t[2] else None)
    E = pd.DataFrame({"EventKey": att[ek], "date": to_dt(att[dk], format="%Y%m%d"),
                      "name": att[nm] if nm else ""}).groupby("EventKey").agg(
        date=("date", "first"), name=("name", "first"), scans=("date", "size")).reset_index()
    E["EventKey"] = E.EventKey.astype(object)
    E["name"] = E.name.astype(object)
    E["m"], E["d"], E["y"] = E.date.dt.month, E.date.dt.day, E.date.dt.year
    C = P.dropna(subset=["m"]).merge(E, on=["m", "d"], suffixes=("", "_evt"))
    C = C[C.y.isna() | (C.y == C.y_evt)]
    gap = (C.date - C.tx_med).dt.days
    C = C[gap.isna() | (gap >= -30)]          # an event can't precede most of its purchases
    gap = (C.date - C.tx_med).dt.days
    C["gap"] = gap.fillna(0)                  # earliest event on/after typical purchase wins
    C = C.sort_values(["product_id", "gap"]).drop_duplicates("product_id")
    C["opp_overlap"] = [len(_tokens(a) & _tokens(b)) for a, b in zip(C.desc, C.name)]
    X = P.merge(C[["product_id", "EventKey", "date", "name", "gap", "opp_overlap"]], on="product_id", how="left")
    X["EventKey"] = X.EventKey.map(lambda v: None if pd.isna(v) else str(v)).astype(object)
    emap = E.assign(k=E.EventKey.astype(str)).set_index("k")
    X["override"] = False
    if OVERRIDES is not None:
        ov = OVERRIDES.set_index("product_id").EventKey.astype(str)
        hit = X.product_id.isin(ov.index)
        X.loc[hit, "EventKey"] = X.loc[hit, "product_id"].map(ov)
        X.loc[hit, "date"] = X.loc[hit, "EventKey"].map(emap.date)
        X.loc[hit, "name"] = X.loc[hit, "EventKey"].map(emap.name)
        X.loc[hit, "override"] = True
        log(f"  applied {int(hit.sum())} crosswalk overrides")
    X["product_class"] = X.desc.map(product_class)
    # purchase-timing gate: a same-M/D event more than MAX_GAP_DAYS after the typical purchase is another season's match
    X["gap"] = (X.date - X.tx_med).dt.days
    too_far = (X.EventKey.notna() & ~X.override & X.gap.gt(MAX_GAP_DAYS)).to_numpy(dtype=bool)
    X.loc[too_far, "EventKey"] = None
    X.loc[too_far, "date"] = pd.NaT
    X.loc[too_far, "name"] = None
    X["reason"] = np.where(X.m.isna(), "no M/D in description (plan, parking, non-game?)",
                  np.where(too_far, f"M/D matched only an event more than {MAX_GAP_DAYS} days after purchase (other season)",
                  np.where(X.EventKey.isna(), "M/D found but no attendance event on that date", "mapped")))

    # attendance: unique scanned seats per event
    Au = pd.DataFrame({"EventKey": att[ek].astype(str), "key": seat_key(att[asec], att[arow], att[aseat]),
                       "acct": att[aacct] if aacct else pd.NA}).dropna(subset=["EventKey", "key"])
    Au = Au.drop_duplicates(["EventKey", "key"])

    # ---- 1b. undated match products: infer the event from the seats that were scanned (conservative, flagged for review)
    undated = X[X.m.isna() & X.EventKey.isna() & X.product_class.eq("Austin FC match") & X.tx_med.notna()]
    inferred = {}
    if len(undated):
        ev_seats = {k: set(g.key.to_numpy()) for k, g in Au.groupby("EventKey")}
        hold = _holder_mask(sal).to_numpy()
        is_cat = hasattr(sal[p], "cat")
        codes = sal[p].cat.codes.to_numpy() if is_cat else sal[p].astype(object).to_numpy()
        for r in undated.itertuples():
            if is_cat:
                if r.product_id not in sal[p].cat.categories:
                    continue
                rows = hold & (codes == sal[p].cat.categories.get_loc(r.product_id))
            else:
                rows = hold & (codes == r.product_id)
            if rows.sum() < MIN_INFER_SEATS:
                continue
            keys = set(pd.Series(seat_key(sal[ssec][rows], sal[srow][rows], sal[sseat][rows])).dropna().unique())
            if len(keys) < MIN_INFER_SEATS:
                continue
            lo, hi = r.tx_med - pd.Timedelta(days=30), r.tx_med + pd.Timedelta(days=MAX_GAP_DAYS)
            scores = []
            for e in E[(E.date >= lo) & (E.date <= hi)].itertuples():
                es = ev_seats.get(str(e.EventKey))
                if es:
                    o = len(keys & es)
                    scores.append((o / len(es), o / len(keys), str(e.EventKey)))
            scores.sort(reverse=True)
            if len(scores) >= 2 and scores[0][0] >= 0.8 and scores[0][1] >= 0.4 and scores[1][0] <= 0.5:
                inferred[r.product_id] = scores[0][2]
        if inferred:
            m_ = X.product_id.isin(inferred)
            X.loc[m_, "EventKey"] = X.loc[m_, "product_id"].map(inferred)
            X.loc[m_, "date"] = X.loc[m_, "EventKey"].map(emap.date)
            X.loc[m_, "name"] = X.loc[m_, "EventKey"].map(emap.name)
            X.loc[m_, "reason"] = "no M/D; event inferred from scanned seats (review)"
            log(f"  seat-inferred {len(inferred)} undated match products")
    X["inferred"] = X.product_id.isin(inferred)
    X["opp_overlap"] = [len(_tokens(a) & _tokens(b)) if isinstance(b, str) else np.nan for a, b in zip(X.desc, X.name)]
    # why is there no event for the unmapped match products? (drives the "of available events" match rate)
    att_min, att_max = E.date.min(), E.date.max()

    def _avail(r):
        if r.product_class != "Austin FC match":
            return "not a Q2 match product"
        if pd.notna(r.EventKey):
            return "event available"
        if pd.isna(r.m) or pd.isna(r.tx_med):
            return "no date in description and no seat-inferable event"
        # first calendar date with this month/day on or after the typical purchase (minus a month)
        y0 = (r.tx_med - pd.Timedelta(days=7)).year
        est = None
        for y in (y0, y0 + 1, y0 + 2):
            try:
                cand = pd.Timestamp(year=y, month=int(r.m), day=int(r.d))
            except ValueError:
                continue
            if cand >= r.tx_med - pd.Timedelta(days=7):   # purchases sit before the match; Nov-Dec renewals are for next season
                est = cand
                break
        if est is None:
            return "date could not be placed"
        if est < att_min:
            return f"event before attendance coverage (starts {att_min:%Y-%m-%d})"
        if est > att_max:
            return f"event after the data window (ends {att_max:%Y-%m-%d})"
        return "event missing from the attendance export"
    X["availability"] = X.apply(_avail, axis=1)

    # ---- 2. validate every mapping with the seats actually scanned
    pmap = X[X.product_class.eq("Austin FC match")].dropna(subset=["EventKey"]).set_index("product_id").EventKey
    if pmap.empty:
        RECON.append(("Event_Crosswalk", X))
        q("cross", "sales product_ids mapped to an attendance EventKey", 0, len(X), "Critical",
          "No product_description contained a parsable M/D date. Seat-level checks skipped; see Event_Crosswalk.")
        return
    T = final_holder_frame(sal, pmap)
    mt = T.merge(Au[["EventKey", "key"]].assign(scanned=True), on=["EventKey", "key"], how="left")
    v = mt.groupby("product_id").agg(sold_seats=("key", "size"), scanned_seats=("scanned", "count")).reset_index()
    X = X.merge(v, on="product_id", how="left")
    X["seat_scan_share"] = (X.scanned_seats / X.sold_seats).round(4)
    X["event_scanned_seats"] = X.EventKey.map(Au.groupby("EventKey").size())
    X["event_side_share"] = (X.scanned_seats / X.event_scanned_seats).round(4)
    share, eside, nsc = X.seat_scan_share.fillna(0), X.event_side_share.fillna(0), X.scanned_seats.fillna(0)
    named = X.opp_overlap.fillna(0) > 0
    date_match = X.m.notna() & X.date.notna() & (X.m == X.date.dt.month) & (X.d == X.date.dt.day)
    seat_ok = (share >= 0.4) | ((eside >= 0.8) & (nsc >= 100))
    conds = [X.EventKey.isna().to_numpy(dtype=bool), (seat_ok & named).to_numpy(dtype=bool), seat_ok.to_numpy(dtype=bool),
             (date_match & named).to_numpy(dtype=bool), X.override.to_numpy(dtype=bool)]
    X["confidence"] = np.select(conds, ["Unmapped - review", "High", "Medium", "Medium", "Manual override"], "Low - review")
    X["basis"] = np.select(conds, ["", "seats scanned + opponent name", "seats scanned (product or event side)",
                                   "date + opponent name; scans too sparse to validate", "manual override; seat evidence weak"],
                           "weak evidence")
    ovm = (X.override & X.EventKey.notna() & X.confidence.ne("Manual override")).to_numpy(dtype=bool)
    X.loc[ovm, "basis"] = "manual override, validated by " + X.loc[ovm, "basis"]
    inf = X.inferred.to_numpy(dtype=bool)
    X.loc[inf, "basis"] = "seat-inferred, no date in description; " + X.loc[inf, "basis"]
    X.loc[X.product_class.ne("Austin FC match") & X.EventKey.notna(), "confidence"] = "Not a match ticket (excluded)"
    X = X[["desc", "product_class", "product_id", "rows", "tx_med", "EventKey", "date", "name", "opp_overlap",
           "sold_seats", "scanned_seats", "seat_scan_share", "event_scanned_seats", "event_side_share",
           "confidence", "basis", "override", "inferred", "reason", "availability"]]
    X.columns = ["sales_product_description", "product_class", "product_id", "sales_rows", "median_transaction_date", "EventKey",
                 "event_date", "attendance_MasterEventName", "opponent_word_overlap", "sold_seats_final_holder",
                 "of_which_scanned", "seat_scan_share", "event_scanned_seats", "event_side_share",
                 "confidence", "basis", "manual_override", "seat_inferred", "reason", "event_availability"]
    global CROSSWALK
    CROSSWALK = X
    RECON.append(("Event_Crosswalk", X.sort_values(["event_date", "sales_product_description"])))
    TRUST = ["High", "Medium", "Manual override"]
    q("cross", "Austin FC match products mapped to an attendance EventKey (High/Medium/Override)",
      int(X.confidence.isin(TRUST).sum()), int(X.product_class.eq("Austin FC match").sum()), "High",
      "Unmapped = plans/non-game products or events outside attendance coverage; review Event_Crosswalk.")
    isq = X.product_class.eq("Austin FC match")
    avail = isq & ~X.event_availability.str.startswith(("event before", "event after"))
    q("cross", "Austin FC match products mapped, of those whose event falls inside attendance coverage",
      int((isq & X.confidence.isin(TRUST)).sum()), int(avail.sum()), "High",
      "Excludes products for 2021 matches and for fixtures after the data window; the remainder unmapped are events missing from the attendance export.")
    q("cross", "Austin FC match product rows mapped, of rows whose event falls inside attendance coverage",
      int(X.sales_rows[isq & X.confidence.isin(TRUST)].sum()), int(X.sales_rows[avail].sum()), "Info")
    q("cross", "attendance EventKeys with no mapped sales product",
      int((~E.EventKey.astype(str).isin(set(X.loc[X.confidence.isin(TRUST), "EventKey"].dropna().astype(str)))).sum()),
      len(E), "High", "Fill these via --crosswalk-overrides (product_id,EventKey CSV).")
    un = E[~E.EventKey.astype(str).isin(set(X.loc[X.confidence.isin(TRUST), "EventKey"].dropna().astype(str)))]
    RECON.append(("Unmapped_Events", un))

    # ---- 3. seat-level match on trusted mappings
    trusted = set(X.loc[X.confidence.isin(TRUST), "product_id"])
    good = set(X.loc[X.confidence.isin(TRUST), "EventKey"].astype(str))
    T = T[T.product_id.isin(trusted)]
    mt = mt[mt.product_id.isin(trusted)]
    yr = X.set_index("product_id").event_date.dt.year
    ys = mt.assign(year=mt.product_id.map(yr)).groupby("year").agg(
        sold_seats=("key", "size"), scanned=("scanned", "count")).reset_index()
    ys["scan_share"] = (ys.scanned / ys.sold_seats).round(4)
    RECON.append(("Seat_Scan_Share_by_Year", ys))
    q("cross", "sold seats (final holder, mapped events) with a scan", int(mt.scanned.notna().sum()), len(mt), "Info",
      "1 - this = seat-level no-show rate (upper bound; missed scans count as no-shows).")
    Ag = Au[Au.EventKey.isin(good)]
    back = Ag.merge(T[["EventKey", "key"]].assign(sold=True), on=["EventKey", "key"], how="left")
    q("cross", "scanned seats (mapped events) with no matching sale row", int(back.sold.isna().sum()), len(back),
      "High", "Should be near 0; high = comps/hospitality missing from sales or format mismatch.")
    if sacct and aacct:
        h = T[T.EventKey.isin(good)].merge(Ag[["EventKey", "key", "acct"]], on=["EventKey", "key"],
                                          suffixes=("_holder", "_scanner"))
        same = (h.acct_holder == h.acct_scanner).fillna(False)
        q("cross", "scanned seats where scanning account == final sales holder", int(same.sum()), len(h), "Info",
          "Tests whether attendance measures holder or buyer (Glossary: UNRESOLVED).")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DIR)
    ap.add_argument("--out", default=None)
    ap.add_argument("--chunksize", type=int, default=CHUNK, help="rows per read chunk (lower = less RAM)")
    ap.add_argument("--account-bridge", default=None,
                    help="LOCAL-ONLY CSV from tvof_account_bridge.py (sales_account_id, fan_account_id)")
    ap.add_argument("--crosswalk-overrides", default=None,
                    help="CSV with columns product_id,EventKey to map products the parser cannot")
    for k in FILE_PATTERNS:
        ap.add_argument(f"--{k}", default=None, help=f"explicit path to the {k} CSV")
    a = ap.parse_args()
    try:
        import openpyxl  # noqa: F401  (needed to write the results workbook)
    except ImportError:
        sys.exit("openpyxl is not installed in this environment. Run:  pip install openpyxl   then rerun.")
    global OVERRIDES, ACCOUNT_BRIDGE
    if a.account_bridge:
        b = pd.read_csv(a.account_bridge, dtype=str)
        if "confidence" in b:
            b = b[b.confidence.isin(["High", "Medium"])]
        ACCOUNT_BRIDGE = pd.DataFrame({"sales": _hash(b.sales_account_id.to_numpy(dtype=object)),
                                       "fan": _hash(b.fan_account_id.to_numpy(dtype=object))})
    if a.crosswalk_overrides:
        OVERRIDES = pd.read_csv(a.crosswalk_overrides, dtype=str)[["product_id", "EventKey"]].dropna()
    files = find_files(a.data_dir)
    for k in FILE_PATTERNS:
        if getattr(a, k):
            files[k] = getattr(a, k)
    log(f"Data dir: {a.data_dir}")
    for k in ["fan", "activation", "attendance", "sales"]:
        log(f"  {k:<11} -> {files.get(k, 'NOT FOUND')}")
    D = {}
    for k, f in files.items():
        log(f"Loading {k} ...")
        df = load(k, f, a.chunksize)
        D[k] = df
        lit = sum(v["lit"] for (ds, _), v in STATS.items() if ds == k)
        mb = df.memory_usage(deep=True).sum() / 1e6
        log(f"  {k}: {len(df):,} rows x {df.shape[1]} cols kept; literal null tokens: {lit:,}; in memory: {mb:,.0f} MB")
        rh = pd.util.hash_pandas_object(df, index=False)
        q(k, "fully duplicated rows (kept columns)", int(rh.duplicated().sum()), len(df), "Moderate")
        del rh
        gc.collect()
    if ACCOUNT_BRIDGE is not None and "sales" in D:
        sa = col(D["sales"], "internal_account_id")
        before = D["sales"][sa].notna().sum()
        # exact uint64 lookup (Series.map would round through float64 and corrupt the hashes)
        keys = D["sales"][sa]
        pos = pd.Index(ACCOUNT_BRIDGE.sales.to_numpy(dtype="uint64")).get_indexer(
            keys.fillna(0).to_numpy(dtype="uint64"))
        pos[keys.isna().to_numpy()] = -1
        D["sales"][sa] = pd.Series(ACCOUNT_BRIDGE.fan.array.take(pos, allow_fill=True), index=keys.index)
        after = D["sales"][sa].notna().sum()
        log(f"  account bridge applied: {after:,} of {before:,} sales rows now carry a Fan-Info-space account ID")
        q("sales", "rows translated by account bridge", int(after), int(before), "Info",
          "Rows without a bridged account are treated as unlinked (NULL) in all fan-level joins.")
    checks = {"fan": check_fan, "activation": check_activation, "attendance": check_attendance, "sales": check_sales}
    for k, fn in checks.items():
        if k in D:
            log(f"Checking {k} ...")
            try:
                fn(D[k])
            except Exception as e:
                import traceback; traceback.print_exc()
                log(f"  !! {k} check failed: {e!r}")
            gc.collect()
    log("Cross-file joins ...")
    try:
        cross_file(D)
    except Exception as e:
        import traceback; traceback.print_exc()
        log(f"  !! cross-file failed: {e!r}")
    out = a.out or os.path.join(a.data_dir, f"TVOF_audit_results_{dt.date.today():%Y%m%d}.xlsx")
    if os.path.exists(out):                      # a file open in Excel cannot be overwritten
        try:
            with open(out, "a"):
                pass
        except OSError:
            out = out[:-5] + dt.datetime.now().strftime("_%H%M") + ".xlsx"
            log(f"  previous results file is locked (open in Excel?); writing {os.path.basename(out)} instead")
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        pd.DataFrame(LOG).to_excel(xw, sheet_name="Run_Log", index=False)
        pd.DataFrame(PROFILE).to_excel(xw, sheet_name="Column_Profile", index=False)
        pd.DataFrame(KEYS).to_excel(xw, sheet_name="Key_Checks", index=False)
        pd.DataFrame(MATCH).to_excel(xw, sheet_name="Match_Rates", index=False)
        pd.DataFrame(QUALITY).to_excel(xw, sheet_name="Quality_Checks", index=False)
        pd.DataFrame(DSCHECK).to_excel(xw, sheet_name="Dataset_Checks", index=False)
        for name, t in RECON:
            t.to_excel(xw, sheet_name=name[:31], index=False)
        for ws in xw.book.worksheets:
            for i, c in enumerate(ws.columns, start=1):
                ws.column_dimensions[get_column_letter(i)].width = min(
                    60, max(10, max(len(str(x.value or "")) for x in c[:200]) + 2))
            ws.freeze_panes = "A2"
    autofilter(out)
    log(f"Wrote {out}  (aggregate only: no row-level IDs)")


if __name__ == "__main__":
    main()
