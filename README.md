# TVOF Data Audit Scripts (Austin FC capstone, Group 2)

Reproducible audit of the four client CSVs. **Code only. No client data lives in this repo.**

## What each script does

| Script | Purpose | Output |
|---|---|---|
| `tvof_data_audit.py` | Full audit: column profile, key checks, cross-file match rates, quality checks, event crosswalk, revenue definitions, activation funnel | `TVOF_audit_results_<date>.xlsx` (aggregate only) |
| `tvof_id_diagnostic.py` | Why Sales `internal_account_id` matches nothing in Fan Info or Attendance | console output, counts only |
| `tvof_account_bridge.py` | Interim seat-based link between the Sales and Fan Info ID spaces | `account_bridge_LOCAL_ONLY.csv` (row level, keep local) + aggregate summary |

`crosswalk_overrides_DRAFT.csv` hand-maps 6 matches the date parser cannot resolve. Each row states its basis. **Verify before relying on it.**

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Put your own copy of the client CSVs in a folder outside this repo (or in `./data/`, which git ignores). The scripts find files by keyword in the filename: `fan`, `activation`, `attendance`, `sales`.

## Run

```bash
python tvof_data_audit.py --data-dir "<your folder of client CSVs>" --crosswalk-overrides crosswalk_overrides_DRAFT.csv
python tvof_id_diagnostic.py --data-dir "<your folder of client CSVs>"
python tvof_account_bridge.py --data-dir "<your folder of client CSVs>" --crosswalk-overrides crosswalk_overrides_DRAFT.csv
```

Or set `TVOF_DATA_DIR` once and drop `--data-dir`.

Useful flags: `--chunksize 50000` (less RAM), `--account-bridge <csv>` (translate Sales accounts into Fan Info ID space), `--out <path>`.

## Requirements and runtime

Python 3.9+, about 1.3 GB RAM peak, roughly 5 minutes for the 5.3M-row sales file. Files are read in chunks; ID columns are stored as 64-bit hashes, so a full load is never attempted.

## Rules for this repo

1. **Private repo only.** The code names the client and describes their data model.
2. **Never commit data.** `.gitignore` blocks csv/xlsx/parquet, the bridge file, and all results workbooks. Check `git status` before every commit.
3. **Results workbooks are aggregate only** and safe to share with the team; the bridge file is row level and is not.
4. Anything AI-assisted gets verified by a person before it reaches the client.

## Known issues the scripts surface

- Sales `internal_account_id` matches 0% of Fan Info and Attendance (different hash). Client fix pending.
- `total_plan_amount` on game rows is a per-game allocation, not the plan total.
- Sales covers all Q2 Stadium products, not only Austin FC matches.
- Attendance covers 2022 to 2026 only; no 2021 season.
