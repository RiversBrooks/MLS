# TVOF Data Audit Scripts (Austin FC capstone, Group 2)

Reproducible audit and fan-value pipeline over the four client CSVs (Fan Info, World Cup Activation, Attendance, Sales History). **Code only. No client data lives in this repo.**

## What each script does

`tvof_data_audit.py` is the shared library: every other `tvof_*.py` imports it for the chunked loader, the joins and the event crosswalk. Outputs go to the data folder; `<date>` is `YYYYMMDD`.

| Script | Purpose | Output |
|---|---|---|
| `tvof_data_audit.py` | Full audit: column profile, key checks, cross-file match rates, quality checks, event crosswalk (Sales product to Attendance event), revenue definitions, activation funnel | `TVOF_audit_results_<date>.xlsx` (aggregate only) |
| `tvof_id_diagnostic.py` | Standalone check of how Sales `internal_account_id` compares with Fan Info and Attendance | console output, counts only |
| `tvof_account_bridge.py` | Seat-based link between Sales accounts and Attendance/Fan Info accounts, with a date-confirmation check; validation only since the 2026-09-24 Sales export shares the account key | `account_bridge_LOCAL_ONLY.csv` (row level, keep local) + `account_bridge_summary_<date>.xlsx` |
| `tvof_scan_sale_join.py` | One row per scan with the final holder and the original sale of that seat | `scan_sale_join_LOCAL_ONLY.csv` (row level) + `scan_sale_join_summary_<date>.xlsx` |
| `tvof_fan_value.py` | Value per person through three lenses (payer, attendee, linked fan): revenue counted once, tenure, distributions, coverage | `fan_value_summary_<date>.xlsx` + `fan_value_LOCAL_ONLY.csv` |
| `tvof_fan_ltv.py` | Cohort chain ladder and lifetime value by first-season segment (season-ticket member, waitlist member, single-ticket buyer) | `fan_ltv_<date>.xlsx` |
| `tvof_peer_benchmarks.py` | Joins Attendance events to the MLS match benchmark (date + opponent) and builds the peer tables from `data/peer/` | `peer_benchmarks_<date>.xlsx` |
| `tvof_fan_statements.py` | Five value statements: fan contribution, cohort revenue triangle, revenue bridge 2022 to 2025, capacity utilization, customer equity | `value_statements_<date>.xlsx` |
| `tvof_ltv_montecarlo.py` | 20,000-run Monte Carlo around the LTV and customer-equity point estimates | `ltv_montecarlo_<date>.xlsx` |
| `tvof_ltv_validation.py` | Rolling-origin backtest of the chain ladder and the Monte Carlo, tornado sensitivity, Sobol variance decomposition | `ltv_validation_<date>.xlsx` |
| `tvof_analyst_models.py` | Five models past the chain ladder: state-transition (Markov) model, usage and renewal, transfer network, resale premium, activation uplift with a placebo year. Reads the four files once into a local cache, then runs in seconds | `analyst_models_<date>.xlsx` (aggregate only) + `analyst_models_cache_LOCAL_ONLY.pkl` (row level by account code, keep local) |
| `tvof_results_table.py` | Stacks every sheet of the latest results workbook of each kind (plus the small aggregate CSVs) into one long table that filters on every column | `results_master_<date>.xlsx` + `.csv` (aggregate only; a copy is kept in `results/`) |
| `tvof_fan_linkage.py` | Fan-level linkage workbook (one sheet per question); bridge columns fill in when the bridge file exists | `fan_linkage_<date>.xlsx` |
| `tvof_column_overlap.py` | Every column against every column across the four tables, with a verdict per pair | `column_overlap_<date>.xlsx` |
| `profile_dataset.py` | Draft profile sheet for one CSV or workbook sheet (the ten-step profiling framework) | console or `--out` markdown |

`crosswalk_overrides_DRAFT.csv` hand-maps the match products the date parser cannot resolve. Each row states its basis and is validated against scanned seats when the audit runs. **Verify before relying on it.**

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Put your own copy of the client CSVs in a folder outside this repo (or in `./data/`, which git ignores). The scripts find files by keyword in the filename: `fan`, `activation`, `attendance`, `sales`; names containing `old`, `backup`, `bak` or `truncated` are skipped. Peer reference files (MLS match data, stadium info, ACS ZCTA workbooks) go in `data/peer/`.

## Run

`./data` is the default data folder (or set `TVOF_DATA_DIR`), so `--data-dir` can be omitted. In dependency order:

```bash
python tvof_data_audit.py      --crosswalk-overrides crosswalk_overrides_DRAFT.csv
python tvof_account_bridge.py  --crosswalk-overrides crosswalk_overrides_DRAFT.csv   # optional since the 09-24 export
python tvof_scan_sale_join.py  --crosswalk-overrides crosswalk_overrides_DRAFT.csv
python tvof_fan_value.py
python tvof_fan_ltv.py
python tvof_peer_benchmarks.py
python tvof_fan_statements.py
python tvof_ltv_montecarlo.py
python tvof_ltv_validation.py
python tvof_analyst_models.py [--refresh]   # five analyst models; --refresh rebuilds the cache from the client files (about ten minutes)
python tvof_results_table.py            # everything in one table -> results_master_<date>.xlsx + .csv

# independent tools
python tvof_fan_linkage.py
python tvof_column_overlap.py
python tvof_id_diagnostic.py --data-dir data
python profile_dataset.py <one csv/xlsx> [--sheet S] [--id a,b] [--date c] [--money d] [--out x.md]
```

Useful flags: `--chunksize 50000` (less RAM), `--out <path>`, `--fan/--activation/--attendance/--sales <file>` to name a file explicitly, and `--account-bridge <csv>` on the audit to translate Sales accounts through the seat bridge. The Monte Carlo and validation scripts take `--sims N` and `--json-out <path>`.

## Requirements and runtime

Python 3.9+ (developed on 3.12 with pandas 3.x), about 1.3 GB RAM peak, roughly 5 minutes for the 5.3M-row sales file. Files are read in chunks; ID columns are stored as 64-bit hashes, so a full load is never attempted. The LTV, statements, Monte Carlo and validation scripts run in seconds from the workbooks the earlier steps write.

## Rules for this repo

1. **Private repo only.** The code names the client and describes their data model.
2. **Never commit data.** `.gitignore` blocks csv/xlsx/parquet, every `*_LOCAL_ONLY` file and all results workbooks. Check `git status` before every commit.
3. **Results workbooks are aggregate only** and safe to share with the team; `results/` holds the stacked one-table copy of them (the only data files tracked here); anything named `*_LOCAL_ONLY.*` is row level and re-identifying, so it stays in the data folder.
4. Anything AI-assisted gets verified by a person before it reaches the client.

## Data facts the scripts rely on

- The 2026-09-24 Sales export shares `internal_account_id` with Fan Info and Attendance (99.85% of Sales accounts match); exports up to 2026-09-18 used a different hash and matched 0%, which is why the seat-based account bridge exists.
- Club revenue is counted once: plan rows plus single tickets outside a plan plus plan-game rows whose plan row is missing. Plan-game rows carry the plan payment split per game, and `total_plan_amount` on them is a per-game allocation. Resale payments go to the seller and are reported separately.
- Sales covers all Q2 Stadium products, not only Austin FC matches; Attendance covers 2022 to 2026 only (no 2021 season); the data window ends 2026-09-18, so every value is to date, not lifetime.
- Match tickets are assigned to the season of the match, not the transaction year: plan renewals for a season are paid in the prior calendar year.
