"""
Austin FC TVOF | Account-ID diagnostic (2026-09-21)
Why does Sales.internal_account_id match 0% of Fan Info and Attendance?
Prints COUNTS ONLY (no ID values). Runtime: a few minutes (reads ID columns only).

    python tvof_id_diagnostic.py --data-dir "<your folder of client CSVs>"
"""
import argparse, glob, hashlib, os, re
from collections import Counter
import pandas as pd

NULLS = {"", "null", "nan", "none", "n/a", "na"}
HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")


def find(d: str, word: str) -> str:
    for f in glob.glob(os.path.join(d, "*.csv")):
        if word in os.path.basename(f).lower():
            return f
    raise SystemExit(f"No CSV with '{word}' in its name found in {d}")


def read_ids(path, cols, chunksize=200_000):
    hdr = pd.read_csv(path, nrows=0).columns
    use = [c for c in hdr if c.split(".")[0].lower() in {x.lower() for x in cols}]
    out = {c: set() for c in use}
    for ch in pd.read_csv(path, usecols=use, dtype=str, keep_default_na=False, chunksize=chunksize):
        for c in use:
            s = ch[c].str.strip()
            out[c].update(s[~s.str.lower().isin(NULLS)].unique())
    return out


def shape(ids):
    lens = Counter(len(x) for x in ids)
    kinds = Counter("hex-lower" if HEX64.match(x) and x == x.lower() else
                    "hex-UPPER" if HEX64.match(x) and x == x.upper() else
                    "hex-mixed" if HEX64.match(x) else
                    "guid" if re.match(r"^[0-9a-fA-F-]{36}$", x) else
                    "numeric" if x.isdigit() else "other" for x in ids)
    return f"n={len(ids):,} lengths={dict(lens.most_common(4))} kinds={dict(kinds)}"


def sha(x):
    return hashlib.sha256(x.encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    a = ap.parse_args()
    fan_f, att_f, sal_f = find(a.data_dir, "fan"), find(a.data_dir, "attendance"), find(a.data_dir, "sales")
    print("Reading ID columns only ...")
    fan = read_ids(fan_f, ["internal_account_id", "internal_fan_id", "mls_id"])
    att = read_ids(att_f, ["internal_account_id"])
    F = fan.get("internal_account_id", set())
    A = next(iter(att.values()))

    print("\n1) ID SHAPES")
    for k, v in fan.items(): print(f"  fan.{k:<22} {shape(v)}")
    print(f"  attendance.internal_account_id {shape(A)}")

    # every Sales column that looks like an ID, first 300k rows
    samp = pd.read_csv(sal_f, dtype=str, keep_default_na=False, nrows=300_000)
    print(f"\n  sales header ({len(samp.columns)} cols): {list(samp.columns)}")
    idcols = [c for c in samp.columns
            if samp[c].str.strip().str.len().between(32, 70).mean() > 0.5]
    print("\n2) EVERY ID-LIKE SALES COLUMN vs Fan/Attendance account IDs (first 300k sales rows)")
    FA = F | A
    for c in idcols:
        s = set(samp[c].str.strip()) - {""} - {x for x in samp[c] if x.lower() in NULLS}
        print(f"  {c:<26} {shape(s)}")
        print(f"      exact match to fan/att accounts: {len(s & FA):,}   "
            f"lower(): {len({x.lower() for x in s} & FA):,}   upper(): {len({x.upper() for x in s} & FA):,}")
        print(f"      match to fan.internal_fan_id: {len(s & fan.get('internal_fan_id', set())):,}   "
            f"to fan.mls_id: {len(s & fan.get('mls_id', set())):,}")

    print("\n3) FULL Sales.internal_account_id tests")
    S = next(iter(read_ids(sal_f, ["internal_account_id"]).values()))
    print(f"  sales.internal_account_id {shape(S)}")
    tests = {
        "exact": len(S & F),
        "case-insensitive": len({x.lower() for x in S} & {x.lower() for x in F}),
        "sales == sha256(fan id)  [fan hashed once more]": None,
        "fan == sha256(sales id)  [sales hashed once more]": None,
        "sales vs attendance exact": len(S & A),
    }
    sample_F = list(F)[:50_000]
    sample_S = list(S)[:50_000]
    tests["sales == sha256(fan id)  [fan hashed once more]"] = \
        f"{sum(sha(x) in S or sha(x.upper()) in S for x in sample_F):,} of {len(sample_F):,} sampled"
    tests["fan == sha256(sales id)  [sales hashed once more]"] = \
        f"{sum(sha(x) in F or sha(x.upper()) in F for x in sample_S):,} of {len(sample_S):,} sampled"
    for k, v in tests.items():
        print(f"  {k:<50} {v}")
    print("\nPaste this whole output back. No ID values are printed.")


if __name__ == "__main__":
    main()
