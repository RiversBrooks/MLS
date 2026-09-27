"""
Austin FC TVOF | Peer benchmarks (v1, 2026-09-25)
--------------------------------------------------
J10: joins the Attendance events (match date + opponent) to the MLS match benchmark, and builds peer comparisons
from the reference files in data/peer/:

    MLS_Match_Data_cleaned_flagged.xlsx     match-level attendance, 2018-2026 regular season (Home_Std / Away_Std canonical)
    MLS_Stadium_Info_cleaned.xlsx           capacity and metro population per club
    MLS_Club_General_Stats_20212026.xlsx    club performance per season
    MLS_Official_Attendance_Reference.csv   league fact-book all-time home / road attendance per club
    EPL_Match_Data_cleaned.xlsx             cross-league context
    Austin_FC_TVOF_Peer_Data_Audit.xlsx     the team's five-club comparison (Austin, St. Louis, Houston, Dallas, Cincinnati)

Also reads the attendance CSV (scans and unique scanned seats per event) and the latest TVOF_audit_results_<date>.xlsx
(sold seats per event from Event_Crosswalk).

Usage:
    python tvof_peer_benchmarks.py [--data-dir data] [--peer-dir data/peer] [--json-out path]

Output: peer_benchmarks_<date>.xlsx in the data dir (aggregate only) and, with --json-out, a JSON summary for the page.
"""
import argparse, datetime as dt, glob, json, os, re, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tvof_data_audit as A   # noqa: E402

DATA_END = pd.Timestamp("2026-09-18")
Q2_CAPACITY = A.Q2_CAPACITY
GA_SECTIONS = {"101", "102", "103", "104", "105"}          # supporters sections: rows GA1..GA38, slots not seats
HOSP_RE = r"sro|standing|suite|boardroom|conference|corner|loft|oak"   # suites, lofts, hospitality rooms, standing room


def _norm_cat(series):
    """A._seat_norm applied per category, mapped back to the rows."""
    c = series.astype("category")
    cats = pd.Series(c.cat.categories.astype(str))
    return c.astype(str).map(pd.Series(A._seat_norm(cats).to_numpy(), index=c.cat.categories.astype(str)))


def seat_inventory(sales_path, p2e, evname):
    """Final-holder seats per event split into fixed seats, GA slots and hospitality/SRO places (one pass over Sales)."""
    parts = []
    for ch in pd.read_csv(sales_path, keep_default_na=False, chunksize=1_000_000,
                          usecols=["product_id", "section", "row", "seat", "item_type", "transfer_status", "resale_status", "transaction_date"],
                          dtype={"product_id": "category", "section": "category", "row": "category", "seat": "category", "item_type": "category",
                                 "transfer_status": "category", "resale_status": "category", "transaction_date": str}):
        ek = ch.product_id.astype(str).map(p2e)
        m = ek.notna()
        if not m.any():
            continue
        c = ch[m]
        it, ts, rs = c.item_type.astype(str).str.lower(), c.transfer_status.astype(str).str.lower(), c.resale_status.astype(str).str.lower()
        hold = ~it.eq("subscription") & ~rs.eq("resold") & ~(it.eq("transfer") & ts.isin(["pending", "canceled", "cancelled"]))
        c = c[hold]
        parts.append(pd.DataFrame({"EventKey": ek[m][hold].to_numpy(), "sec": _norm_cat(c.section).to_numpy(), "row": _norm_cat(c.row).to_numpy(),
                                   "seat": _norm_cat(c.seat).to_numpy(), "td": c.transaction_date.to_numpy()}))
    H = pd.concat(parts, ignore_index=True)
    H["td"] = pd.to_datetime(H.td, errors="coerce")
    H = H.dropna(subset=["sec", "row", "seat"]).sort_values("td").drop_duplicates(["EventKey", "sec", "row", "seat"], keep="last")
    H["kind"] = np.where(H.sec.isin(GA_SECTIONS), "GA slot (sections 101-105)",
                np.where(H.sec.str.contains(HOSP_RE, case=False, regex=True), "suite / loft / hospitality / SRO", "fixed seat"))
    inv = H.groupby(["EventKey", "kind"]).size().unstack(fill_value=0).reset_index()
    inv["sold_total"] = inv.drop(columns="EventKey").sum(axis=1)
    inv["excess_over_capacity"] = inv.sold_total - Q2_CAPACITY
    inv["event"] = inv.EventKey.map(evname)
    inv["season"] = inv.event.str[:4]
    return inv.sort_values("event")

PEERS5 = ["Austin FC", "St. Louis City SC", "Houston Dynamo FC", "FC Dallas", "FC Cincinnati"]
LIGA_MX = {"mazatlan", "juarez", "pumas", "unam", "monterrey", "tijuana", "puebla", "pachuca", "america", "atlas", "leon", "tigres", "cruz", "azul", "chivas", "toluca", "necaxa", "queretaro", "santos"}
ALIASES = {"lafc": {"los", "angeles", "football", "club"}, "nycfc": {"new", "york", "city"}, "rsl": {"real", "salt", "lake"},
           "skc": {"sporting", "kansas", "city"}, "red": {"bull", "bulls"}, "bulls": {"bull"}, "bull": {"bulls"}}


def tokens(x):
    import unicodedata
    x = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode()
    t = A._tokens(x)
    for k in list(t):
        t |= ALIASES.get(k, set())
    return t


def event_kind(name, opp_tokens, matched):
    n = str(name).lower()
    if matched:
        return "MLS regular season"
    if "playoff" in n:
        return "MLS Cup Playoffs"
    if "open cup" in n:
        return "US Open Cup"
    if "friendly" in n or "preseason" in n:
        return "friendly"
    if "leagues cup" in n or opp_tokens & LIGA_MX:
        return "Leagues Cup or friendly vs Liga MX"
    if opp_tokens & {"violette", "locomotive", "paso"}:
        return "friendly / cup vs non-MLS club"
    return "other competition (cup tie vs MLS club)"


def opponent_crosswalk(att_opponents, xwalk):
    """Map the opponent parsed from the Attendance event name to the canonical club name.
    Order: exact canonical name; exact Match_Data_Name in the team's crosswalk; token match (with aliases); else unmapped."""
    canon = sorted(set(xwalk.Canonical_Name.astype(str)))
    md = {str(k).strip().lower(): str(v) for k, v in zip(xwalk.Match_Data_Name, xwalk.Canonical_Name)}
    canon_l = {c.lower(): c for c in canon}
    canon_tok = {c: tokens(c) for c in canon}
    rows = []
    for o in att_opponents:
        ol = str(o).strip().lower()
        if ol in canon_l:
            rows.append((o, canon_l[ol], "exact canonical name"))
            continue
        if ol in md:
            rows.append((o, md[ol], "team crosswalk (Match_Data_Name)"))
            continue
        ot = tokens(o)
        best, score = None, 0.0
        for c, ct in canon_tok.items():
            if not ct or not ot:
                continue
            j = len(ot & ct) / len(ot | ct)
            if j > score:
                best, score = c, j
        if best and score >= 0.4:
            rows.append((o, best, f"token match ({score:.2f})"))
        else:
            rows.append((o, None, "unmapped (not an MLS club or not a league fixture)"))
    return pd.DataFrame(rows, columns=["attendance_opponent", "canonical_club", "how"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=A.DEFAULT_DIR)
    ap.add_argument("--peer-dir", default=None)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()
    peer = a.peer_dir or os.path.join(a.data_dir, "peer")
    files = A.find_files(a.data_dir)
    if "attendance" not in files:
        sys.exit(f"attendance file not found in {a.data_dir}")

    # ---- attendance events: scans and unique scanned seats
    att = pd.read_csv(files["attendance"], usecols=["EventKey", "EventDateKey", "EventName", "SectionName", "RowName", "SeatName"], dtype="category")
    ev = att.groupby("EventKey", observed=True).agg(date_key=("EventDateKey", "first"), name=("EventName", "first"), scans=("EventName", "size")).reset_index()
    seats = att.drop_duplicates(["EventKey", "SectionName", "RowName", "SeatName"]).groupby("EventKey", observed=True).size()
    ev["scanned_seats"] = ev.EventKey.map(seats).astype(int)
    ev["EventKey"] = ev.EventKey.astype(str)
    ev["date"] = pd.to_datetime(ev.date_key.astype(str), format="%Y%m%d", errors="coerce")
    ev["name"] = ev.name.astype(str)
    ev["opponent"] = ev.name.str.replace(r"^\d{4}-\d{2}-\d{2}\s*", "", regex=True)
    ev = ev.sort_values("date").reset_index(drop=True)
    A.log(f"attendance: {len(ev)} events, {int(ev.scans.sum()):,} scans")

    # ---- sold seats per event from the latest audit workbook
    sold = pd.Series(dtype=float)
    audits = sorted(glob.glob(os.path.join(a.data_dir, "TVOF_audit_results_*.xlsx")))
    if audits:
        xw = pd.read_excel(audits[-1], sheet_name="Event_Crosswalk")
        tr = xw[xw.confidence.isin(["High", "Medium", "Manual override"]) & xw.product_class.eq("Austin FC match")]
        ekey = pd.to_numeric(tr.EventKey, errors="coerce").map(lambda v: str(int(v)) if pd.notna(v) else None)
        sold = tr.groupby(ekey).sold_seats_final_holder.sum()
        A.log(f"sold seats per event from {os.path.basename(audits[-1])}")
    ev["sold_seats"] = ev.EventKey.map(sold)
    inv = seat_inventory(files["sales"], {str(p_): str(int(e_)) for p_, e_ in zip(tr.product_id, tr.EventKey)},
                         {str(int(e_)): n_ for e_, n_ in zip(tr.EventKey, tr.attendance_MasterEventName)}) if audits and "sales" in files else None
    if inv is not None:
        inv_season = inv.groupby("season").agg(events=("EventKey", "size"), sold_total=("sold_total", "mean"), fixed_seats=("fixed seat", "mean"),
                                               ga_slots=("GA slot (sections 101-105)", "mean"), hospitality_sro=("suite / loft / hospitality / SRO", "mean"),
                                               excess_over_capacity=("excess_over_capacity", "mean"),
                                               events_over_capacity=("excess_over_capacity", lambda x: int((x > 0).sum()))).round(0).reset_index()
        A.log("seat inventory per season: " + "; ".join(f"{r.season}: fixed {int(r.fixed_seats):,}, GA {int(r.ga_slots):,}, hospitality {int(r.hospitality_sro):,}" for r in inv_season.itertuples()))

    # ---- MLS match data
    m = pd.read_excel(os.path.join(peer, "MLS_Match_Data_cleaned_flagged.xlsx"), sheet_name="Combined (all seasons)")
    m["Date"] = pd.to_datetime(m.Date, errors="coerce")
    m["Attendance"] = pd.to_numeric(m.Attendance, errors="coerce")
    m["played"] = m.Score.notna() & ~m.Future_Fixture_Flag.astype(bool)
    aus = m[m.Home_Std.eq("Austin FC")].copy().sort_values("Date")
    A.log(f"MLS match data: {len(m):,} matches {int(m.Season.min())}-{int(m.Season.max())}; Austin home {len(aus)}")

    # ---- J10 per spec: parse opponent from the event name, map it through the team-name crosswalk, join on date + opponent (1:1)
    xwalk = pd.read_excel(os.path.join(peer, "MLS_Match_Data_cleaned_flagged.xlsx"), sheet_name="Team Name Crosswalk")
    oc = opponent_crosswalk(sorted(ev.opponent.unique()), xwalk)
    ev = ev.merge(oc, left_on="opponent", right_on="attendance_opponent", how="left").drop(columns="attendance_opponent")
    A.log(f"opponent crosswalk: {int(oc.canonical_club.notna().sum())} of {len(oc)} distinct Attendance opponents map to a canonical club "
          f"({', '.join(f'{k}: {v}' for k, v in oc.how.str.split(' (', regex=False).str[0].value_counts().items())})")
    key_cols = ["Season", "Date", "Away", "Away_Std", "Score", "Attendance", "played", "Future_Fixture_Flag", "Day", "Time"]
    j = ev.merge(aus[key_cols], left_on=["date", "canonical_club"], right_on=["Date", "Away_Std"], how="left")
    j["matched"] = j.Date.notna()
    j["benchmark"] = np.where(j.matched, "MLS regular season (match data)", None)
    j["competition"] = np.where(j.matched, "MLS regular season", None)
    # supplementary benchmark: cup, playoff and friendly home fixtures (Wikipedia wikitext, verify before use)
    supp_path = os.path.join(peer, "Austin_FC_nonleague_home_matches_wikipedia.csv")
    n_supp = 0
    if os.path.exists(supp_path):
        sp = pd.read_csv(supp_path)
        sp["date"] = pd.to_datetime(sp.date)
        sp["announced_attendance"] = pd.to_numeric(sp.announced_attendance, errors="coerce")
        for r in sp.itertuples():
            hit = (~j.matched) & (j.date == r.date)
            if not hit.any():
                continue
            ok = hit & j.opponent.map(lambda o: bool(tokens(o) & tokens(r.opponent)) or str(o).lower().startswith(("us open cup", "mls playoffs", "leagues cup")))
            if not ok.any():
                A.log(f"  supplementary row {r.date.date()} {r.opponent}: same date as an Attendance event but opponent does not agree; not joined")
                continue
            j.loc[ok, ["matched", "Date", "Attendance", "Score", "competition"]] = [True, r.date, r.announced_attendance, r.score, r.competition]
            j.loc[ok, "benchmark"] = "cup / playoff / friendly (Wikipedia, verify)"
            j.loc[ok, "canonical_club"] = j.loc[ok, "canonical_club"].where(j.loc[ok, "canonical_club"].notna(), r.opponent)
            n_supp += int(ok.sum())
        A.log(f"supplementary benchmark: {len(sp)} non-league home fixtures on file; {n_supp} Attendance events joined through it")
    # cardinality: each event at most one match, each match at most one event
    dup_ev = j[j.matched].groupby("EventKey").size()
    dup_m = j[j.matched].groupby(["Date", "Away_Std"]).size()
    one_to_one = bool((dup_ev <= 1).all() and (dup_m <= 1).all())
    A.log(f"J10 cardinality: {'1:1 confirmed' if one_to_one else 'NOT 1:1'} ({int((dup_ev > 1).sum())} events with >1 match, {int((dup_m > 1).sum())} matches with >1 event)")
    date_only = ev.merge(aus[["Date"]], left_on="date", right_on="Date", how="inner")
    A.log(f"     date-only join would give {date_only.EventKey.nunique()} events; date + opponent gives {int(j.matched.sum())}")
    j["opp_ok"] = j.matched
    j["kind"] = [event_kind(n, tokens(o), mt and c == "MLS regular season") if not (mt and c and c != "MLS regular season") else c
                 for n, o, mt, c in zip(j.name, j.opponent, j.matched, j.competition)]
    j["official_attendance"] = j.Attendance
    j["scanned_share_of_official"] = (j.scanned_seats / j.official_attendance).round(4)
    j["sold_share_of_official"] = (j.sold_seats / j.official_attendance).round(4)
    j["season"] = j.date.dt.year
    n_ev, n_match = len(j), int(j.matched.sum())
    A.log(f"J10: {n_match} of {n_ev} Attendance events match a benchmark fixture ({n_match - n_supp} MLS regular season + {n_supp} cup/playoff/friendly); opponent agrees on {int((j.opp_ok == True).sum())}")
    A.log("     unmatched by kind: " + ", ".join(f"{k}: {v}" for k, v in j[~j.matched].kind.value_counts().items()))
    # reverse: MLS home matches inside the attendance window that have an event
    win = aus[(aus.Date >= ev.date.min()) & (aus.Date <= DATA_END)].copy()
    win["has_event"] = win.Date.isin(set(ev.date))
    A.log(f"     MLS home matches {ev.date.min():%Y-%m-%d} to {DATA_END:%Y-%m-%d}: {int(win.has_event.sum())} of {len(win)} have an Attendance event")
    missing_events = win[~win.has_event][["Season", "Date", "Away_Std", "Score", "Attendance", "played"]]

    events_out = j[["season", "EventKey", "date", "name", "opponent", "canonical_club", "how", "kind", "matched", "benchmark", "Away_Std", "Score", "official_attendance",
                    "scans", "scanned_seats", "sold_seats", "scanned_share_of_official", "sold_share_of_official", "opp_ok"]].rename(
        columns={"Away_Std": "mls_opponent", "Score": "score", "opp_ok": "opponent_name_agrees", "how": "crosswalk_method"})
    league = j[j.matched & j.competition.eq("MLS regular season") & j.official_attendance.notna()]
    season_sum = league.groupby("season").agg(league_matches=("EventKey", "size"), official_avg=("official_attendance", "mean"),
                                              scanned_seats_avg=("scanned_seats", "mean"), sold_seats_avg=("sold_seats", "mean"),
                                              scanned_share=("scanned_share_of_official", "mean")).round(3).reset_index()
    all_kinds = j.groupby(["season", "kind"]).size().unstack(fill_value=0).reset_index()

    # ---- peers: per club per season from match data + stadium info
    st = pd.read_excel(os.path.join(peer, "MLS_Stadium_Info_cleaned.xlsx"), sheet_name="Stadium Info")
    st = st.rename(columns={"Team": "club", "Stadium_Capacity_Primary": "capacity", "Metro Area Population": "metro_pop", "Multi_Stadium_Flag": "multi_stadium"})
    played = m[m.played & m.Attendance.notna()]
    peers = played.groupby(["Season", "Home_Std"]).agg(home_matches=("Attendance", "size"), avg_attendance=("Attendance", "mean"), total_attendance=("Attendance", "sum")).reset_index()
    peers = peers.rename(columns={"Season": "season", "Home_Std": "club"}).merge(st[["club", "capacity", "metro_pop", "multi_stadium", "Stadium"]], on="club", how="left")
    # Austin FC: use the club's scanned entries (distinct seats scanned per league match) instead of the announced 20,738
    scanned_by_season = {int(r.season): float(r.scanned_seats_avg) for r in season_sum.itertuples()}
    sold_by_season = {int(r.season): float(r.sold_seats_avg) for r in season_sum.itertuples()}
    peers["attendance_basis"] = "announced (tickets distributed)"
    peers_announced = peers.copy()
    peers_announced["rank_avg_attendance"] = peers_announced.groupby("season").avg_attendance.rank(ascending=False, method="min").astype(int)
    ia = peers.club.eq("Austin FC")
    peers.loc[ia, "avg_attendance"] = peers.loc[ia, "season"].map(scanned_by_season)
    peers.loc[ia, "total_attendance"] = (peers.loc[ia, "avg_attendance"] * peers.loc[ia, "home_matches"]).round(0)
    peers.loc[ia, "attendance_basis"] = "scanned entries (club data)"
    no_scan = ia & peers.avg_attendance.isna()
    if no_scan.any():
        A.log(f"Austin FC seasons without scan data dropped from the peer tables: {sorted(peers.loc[no_scan, 'season'].astype(int))}")
        peers = peers[~no_scan]
    peers["utilization"] = (peers.avg_attendance / peers.capacity).round(4)
    peers["attendance_per_1000_metro"] = (peers.avg_attendance / peers.metro_pop * 1000).round(2)
    peers["rank_avg_attendance"] = peers.groupby("season").avg_attendance.rank(ascending=False, method="min").astype(int)
    peers["rank_utilization"] = peers.groupby("season").utilization.rank(ascending=False, method="min")
    peers["clubs_in_season"] = peers.groupby("season").club.transform("size")
    peers["season_status"] = np.where(peers.season == 2026, "in progress (to 2026-09-07 export)", "complete")
    peers = peers.sort_values(["season", "rank_avg_attendance"])
    austin_rank = peers[peers.club.eq("Austin FC")][["season", "home_matches", "avg_attendance", "capacity", "utilization", "rank_avg_attendance", "rank_utilization", "clubs_in_season", "metro_pop", "attendance_per_1000_metro"]]
    league_avg = peers.groupby("season").agg(clubs=("club", "size"), league_avg_attendance=("avg_attendance", "mean"), league_median_attendance=("avg_attendance", "median"),
                                             league_avg_utilization=("utilization", "mean")).round(3).reset_index()
    five = peers[peers.club.isin(PEERS5) & peers.season.between(2021, 2026)].pivot_table(index="club", columns="season", values="avg_attendance").round(0).reindex(PEERS5)
    five_util = peers[peers.club.isin(PEERS5)].pivot_table(index="club", columns="season", values="utilization").round(3).reindex(PEERS5)

    # ---- the team's five-club audit sheet, for cross-check against the match data (2025 averages)
    audit_sheet = pd.read_excel(os.path.join(peer, "Austin_FC_TVOF_Peer_Data_Audit.xlsx"), sheet_name="Data Audit")
    row = audit_sheet[audit_sheet.Metric.astype(str).str.startswith("Avg. home attendance, 2025")]
    team_2025 = {c: pd.to_numeric(str(row.iloc[0][c]).replace(",", ""), errors="coerce") for c in ["Austin FC", "St. Louis City SC", "Houston Dynamo FC", "FC Dallas", "FC Cincinnati"]} if len(row) else {}
    ours_2025 = peers[peers.season.eq(2025)].set_index("club").avg_attendance
    xcheck = pd.DataFrame({"club": PEERS5, "team_audit_2025_avg": [team_2025.get(c) for c in PEERS5], "match_data_2025_avg": [round(float(ours_2025.get(c, np.nan)), 0) for c in PEERS5]})
    xcheck["difference"] = xcheck.match_data_2025_avg - xcheck.team_audit_2025_avg

    # ---- official fact-book reference vs match data (Austin FC through 2025)
    ref = pd.read_csv(os.path.join(peer, "MLS_Official_Attendance_Reference.csv"))
    ref_aus = ref[ref.Team_As_Printed.eq("Austin FC")].iloc[0]
    aus25 = aus[aus.Season.between(2021, 2025)]
    recon = pd.DataFrame([
        ("fact book: home dates through 2025", int(ref_aus.Home_Dates)), ("fact book: total home attendance", int(ref_aus.Home_Total_Attendance)), ("fact book: home average", int(ref_aus.Home_Average)),
        ("match data: Austin home rows 2021-2025", len(aus25)), ("match data: rows with attendance", int(aus25.Attendance.notna().sum())),
        ("match data: rows with blank attendance (all 2021)", int(aus25.Attendance.isna().sum())), ("match data: total attendance recorded", int(aus25.Attendance.sum())),
        ("gap to fact book total", int(ref_aus.Home_Total_Attendance - aus25.Attendance.sum())), ("gap divided by 20,738", round(float((ref_aus.Home_Total_Attendance - aus25.Attendance.sum()) / 20738), 3)),
    ], columns=["item", "value"])

    # ---- club performance context (Austin)
    cs = pd.read_excel(os.path.join(peer, "MLS_Club_General_Stats_20212026.xlsx"), sheet_name="All Seasons", header=3)
    perf = cs[cs.Club.astype(str).str.strip().eq("Austin")][["Season", "GP", "PTS", "W", "L", "T", "G", "GA", "GD"]].rename(columns={"Season": "season"})
    perf = perf.merge(season_sum, on="season", how="left").merge(austin_rank[["season", "avg_attendance", "utilization", "rank_avg_attendance", "clubs_in_season"]], on="season", how="left")

    # ---- EPL context
    e = pd.read_excel(os.path.join(peer, "EPL_Match_Data_cleaned.xlsx"), sheet_name="EPL Match Data (cleaned)")
    e["Date"] = pd.to_datetime(e.Date, errors="coerce")
    e["Attendance"] = pd.to_numeric(e.Attendance, errors="coerce")
    e["season"] = np.where(e.Date.dt.month >= 7, e.Date.dt.year, e.Date.dt.year - 1)
    epl = e[e.Attendance.notna()].groupby("season").agg(matches=("Attendance", "size"), avg_attendance=("Attendance", "mean"), median_attendance=("Attendance", "median")).round(0).reset_index()
    epl["season"] = epl.season.astype(str) + "-" + (epl.season + 1).astype(str).str[2:]

    # ---- demand vs performance: Austin results, form and kickoff vs turnout, per matched league match
    def goals(score, home_is_austin):
        if pd.isna(score):
            return (np.nan, np.nan)
        g = re.findall(r"\d+", str(score))
        if len(g) < 2:
            return (np.nan, np.nan)
        h, aw = int(g[0]), int(g[1])
        return (h, aw) if home_is_austin else (aw, h)
    allaus = m[(m.Home_Std.eq("Austin FC") | m.Away_Std.eq("Austin FC")) & m.played & m.Score.notna()].copy().sort_values("Date")
    ga = [goals(sc, h == "Austin FC") for sc, h in zip(allaus.Score, allaus.Home_Std)]
    allaus["gf"], allaus["ga"] = [x[0] for x in ga], [x[1] for x in ga]
    allaus["result"] = np.select([allaus.gf > allaus.ga, allaus.gf == allaus.ga], ["W", "D"], "L")
    allaus["pts"] = allaus.result.map({"W": 3, "D": 1, "L": 0})
    allaus["form_pts_last5"] = allaus.groupby("Season").pts.transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
    allaus["season_pts_before"] = allaus.groupby("Season").pts.transform(lambda x: x.shift(1).fillna(0).cumsum())
    allaus["season_gp_before"] = allaus.groupby("Season").cumcount()
    dvp = league.merge(allaus[["Date", "Home_Std", "gf", "ga", "result", "pts", "form_pts_last5", "season_pts_before", "season_gp_before"]], on="Date", how="left")
    dvp["ppg_before"] = (dvp.season_pts_before / dvp.season_gp_before.replace(0, np.nan)).round(2)
    dvp["kickoff_hour"] = pd.to_numeric(dvp.Time.astype(str).str.extract(r"^(\d{1,2}):")[0], errors="coerce")
    dvp["weekday"] = dvp.Day
    dvp_out = dvp[["season", "date", "opponent", "canonical_club", "weekday", "kickoff_hour", "result", "gf", "ga", "form_pts_last5", "ppg_before",
                   "official_attendance", "sold_seats", "scanned_seats", "scanned_share_of_official"]].sort_values("date")
    by_result = dvp.groupby("result").agg(matches=("date", "size"), scanned_share=("scanned_share_of_official", "mean"), scanned_avg=("scanned_seats", "mean")).round(3).reset_index()
    by_day = dvp.groupby("weekday").agg(matches=("date", "size"), scanned_share=("scanned_share_of_official", "mean")).round(3).reset_index().sort_values("matches", ascending=False)
    dvp["form_band"] = pd.cut(dvp.form_pts_last5, [-1, 4, 7, 10, 15], labels=["0-4 pts (poor)", "5-7", "8-10", "11-15 (strong)"])
    by_form = dvp.groupby("form_band", observed=True).agg(matches=("date", "size"), scanned_share=("scanned_share_of_official", "mean")).round(3).reset_index()
    corr_form = float(dvp[["form_pts_last5", "scanned_share_of_official"]].dropna().corr().iloc[0, 1]) if dvp.form_pts_last5.notna().sum() > 3 else float("nan")
    corr_ppg = float(dvp[["ppg_before", "scanned_share_of_official"]].dropna().corr().iloc[0, 1]) if dvp.ppg_before.notna().sum() > 3 else float("nan")
    season_perf = perf.merge(dvp.groupby("season").agg(home_W=("result", lambda x: int((x == "W").sum())), home_D=("result", lambda x: int((x == "D").sum())),
                                                     home_L=("result", lambda x: int((x == "L").sum()))).reset_index(), on="season", how="left")
    A.log(f"demand vs performance: corr(turnout, last-5 form) = {corr_form:.2f}; corr(turnout, season ppg before match) = {corr_ppg:.2f}")

    # ---- sellout check: announced vs ticketed vs turnstile, per match and season
    sc = league.copy()
    sc["announced_sellout"] = sc.official_attendance >= Q2_CAPACITY
    sc["ticketed_sellout"] = sc.sold_seats >= Q2_CAPACITY
    if inv is not None:
        fx = inv.set_index("EventKey")
        sc["fixed_seats_sold"] = sc.EventKey.map(fx["fixed seat"])
        sc["ga_slots_sold"] = sc.EventKey.map(fx["GA slot (sections 101-105)"])
        fixed_cap = float(sc.fixed_seats_sold.quantile(0.95))
        sc["fixed_seat_fill"] = (sc.fixed_seats_sold / fixed_cap).round(3)
    sc["turnstile_90"] = sc.scanned_share_of_official >= 0.9
    sellout_season = sc.groupby("season").agg(league_matches=("EventKey", "size"), announced_sellouts=("announced_sellout", "sum"), ticketed_sellouts=("ticketed_sellout", "sum"),
                                              matches_scanned_90pct=("turnstile_90", "sum"), sold_avg=("sold_seats", "mean"), scanned_share=("scanned_share_of_official", "mean"),
                                              **({"fixed_seat_fill_avg": ("fixed_seat_fill", "mean")} if inv is not None else {})).round(3).reset_index()
    sellout_out = sc[["season", "date", "opponent", "official_attendance", "sold_seats", "scanned_seats", "scanned_share_of_official", "announced_sellout", "ticketed_sellout"]
                     + (["fixed_seats_sold", "ga_slots_sold", "fixed_seat_fill"] if inv is not None else [])].sort_values("date")
    A.log("sellout check: " + "; ".join(f"{int(r.season)}: announced {int(r.announced_sellouts)}/{int(r.league_matches)}, ticketed {int(r.ticketed_sellouts)}, scanned 90%+ {int(r.matches_scanned_90pct)}" for r in sellout_season.itertuples()))

    # ---- club x season pivot and Austin by opponent
    piv = peers.pivot_table(index="club", columns="season", values="avg_attendance", aggfunc="mean").round(0)
    piv["capacity"] = peers.drop_duplicates("club").set_index("club").capacity
    piv["utilization_2025"] = peers[peers.season.eq(2025)].set_index("club").utilization
    piv["rank_2025"] = peers[peers.season.eq(2025)].set_index("club").rank_avg_attendance
    piv = piv.sort_values(2025, ascending=False).reset_index()
    piv.columns = [str(c) for c in piv.columns]
    ann = peers_announced[peers_announced.club.eq("Austin FC")].set_index("season")
    scn = peers[peers.club.eq("Austin FC")].set_index("season")
    austin_actual = pd.DataFrame({"season": list(scanned_by_season), "announced_avg": [20738.0] * len(scanned_by_season),
                                  "sold_seats_avg": [sold_by_season[y] for y in scanned_by_season], "scanned_entries_avg": list(scanned_by_season.values()),
                                  "scanned_share_of_announced": [round(scanned_by_season[y] / Q2_CAPACITY, 4) for y in scanned_by_season],
                                  "rank_announced": [int(ann.rank_avg_attendance.get(y, 0)) for y in scanned_by_season],
                                  "rank_on_scanned_entries": [int(scn.rank_avg_attendance.get(y, 0)) for y in scanned_by_season],
                                  "clubs": [int(scn.clubs_in_season.get(y, 0)) for y in scanned_by_season]})
    A.log("Austin actual vs announced: " + "; ".join(f"{int(r.season)}: scanned {r.scanned_entries_avg:,.0f} ({r.scanned_share_of_announced:.0%}), rank {r.rank_announced} announced vs {r.rank_on_scanned_entries} on scans" for r in austin_actual.itertuples()))
    opp = league.groupby("Away_Std").agg(matches=("EventKey", "size"), official_avg=("official_attendance", "mean"), sold_avg=("sold_seats", "mean"),
                                        scanned_avg=("scanned_seats", "mean"), scanned_share=("scanned_share_of_official", "mean"),
                                        seasons=("season", lambda x: ", ".join(str(v) for v in sorted(set(x))))).round(3).sort_values("scanned_share", ascending=False).reset_index().rename(columns={"Away_Std": "opponent"})

    # ---- enriched copy of the team's Data Audit workbook (original untouched)
    import openpyxl
    src = os.path.join(peer, "Austin_FC_TVOF_Peer_Data_Audit.xlsx")
    wb = openpyxl.load_workbook(src)
    ws = wb["Data Audit"]
    hdr = [c.value for c in ws[1]]
    col_of = {str(h).strip(): i + 1 for i, h in enumerate(hdr) if h}
    r0 = ws.max_row + 2
    ws.cell(r0, 1, "Added 2026-09-25 from data/peer via tvof_peer_benchmarks.py (played MLS regular-season matches; Austin's official figure is a capacity assumption in the source)")
    r = r0 + 1
    by_cs = peers.set_index(["club", "season"])
    def put(metric, values, source, conf, note):
        nonlocal r
        ws.cell(r, col_of["Metric"], metric)
        for club, v in values.items():
            if club in col_of:
                ws.cell(r, col_of[club], v)
        ws.cell(r, col_of.get("Source(s)", 7), source)
        ws.cell(r, col_of.get("Confidence", 8), conf)
        ws.cell(r, col_of.get("Notes / Caveats", 9), note)
        r += 1
    for season in [2021, 2022, 2023, 2024, 2025, 2026]:
        vals = {c: (int(round(by_cs.loc[(c, season), "avg_attendance"])) if (c, season) in by_cs.index else "n/a") for c in PEERS5}
        put(f"Avg. home attendance, {season} season{' to date' if season == 2026 else ''} (peers: match data; Austin: scanned entries)", vals,
            "MLS_Match_Data_cleaned_flagged.xlsx; Austin FC attendance scans", "Medium (peers announced; Austin actual)",
            "Peers: mean announced attendance of played home matches. Austin: distinct seats scanned per league match (announced figure is 20,738 every match)." + (" 2026 export dated 2026-09-07." if season == 2026 else ""))
    put("Announced home attendance, every season (published), Austin only", {"Austin FC": 20738, "St. Louis City SC": "n/a", "Houston Dynamo FC": "n/a", "FC Dallas": "n/a", "FC Cincinnati": "n/a"},
        "MLS_Match_Data_cleaned_flagged.xlsx; club sellout announcements", "High", "Reference only: the club reports a 20,738 sellout for every home match; the rows above use scanned entries for Austin instead.")
    put("Capacity utilization, 2025 (match data)", {c: (round(float(by_cs.loc[(c, 2025), "utilization"]), 3) if (c, 2025) in by_cs.index else "n/a") for c in PEERS5},
        "match data / MLS_Stadium_Info_cleaned.xlsx", "Medium", "Average attendance over Stadium_Capacity_Primary. FC Dallas distorted by the temporary capacity cut.")
    put("Rank by avg. home attendance, 2025 (of 30)", {c: (int(by_cs.loc[(c, 2025), "rank_avg_attendance"]) if (c, 2025) in by_cs.index else "n/a") for c in PEERS5},
        "match data", "Medium", "")
    put("Metro area population (Stadium Info file)", {c: (int(st.set_index("club").metro_pop.get(c, 0)) or "n/a") for c in PEERS5}, "MLS_Stadium_Info_cleaned.xlsx", "Medium", "Differs from the Census figures in the row above; keep one source per comparison.")
    turn = {int(x["season"]): x["scanned_share"] for x in season_sum.to_dict(orient="records")}
    put("Turnstile rate, 2025 (scanned seats / announced), Austin only", {"Austin FC": round(turn.get(2025, float("nan")), 3), "St. Louis City SC": "n/a", "Houston Dynamo FC": "n/a", "FC Dallas": "n/a", "FC Cincinnati": "n/a"},
        "Austin FC attendance scans vs match data", "High for Austin", "Distinct scanned seats per league match over the announced 20,738; other clubs have no scan data here.")
    put("League mean avg. home attendance, 2025", {"Austin FC": int(round(league_avg.set_index("season").loc[2025, "league_avg_attendance"]))}, "match data", "Medium", "Same value applies to all clubs; placed in the first column.")
    def add_sheet(name, df):
        w = wb.create_sheet(name)
        w.append(list(df.columns))
        for row in df.itertuples(index=False):
            w.append([None if (isinstance(v, float) and v != v) else (v.item() if hasattr(v, "item") else v) for v in row])
    add_sheet("Match Data Benchmarks", piv)
    add_sheet("Austin by Opponent", opp)
    add_sheet("Austin Turnstile by Season", season_sum)
    enriched = os.path.join(a.data_dir, f"Peer_Data_Audit_enriched_{dt.date.today():%Y%m%d}.xlsx")
    wb.save(enriched)
    A.autofilter(enriched)
    A.log(f"Wrote {enriched}  (copy of the team's Data Audit workbook with match-data rows and three sheets)")

    # ---- write
    out = os.path.join(a.data_dir, f"peer_benchmarks_{dt.date.today():%Y%m%d}.xlsx")
    notes = pd.DataFrame([
        "J10 (per spec): the opponent is parsed from the Attendance event name (the text after the date), mapped to the canonical club through the team's Team Name Crosswalk, extended for Attendance's long-form names (Opponent_Crosswalk sheet shows each mapping and how it was made), and the join to Austin FC home rows is on match date + canonical opponent. Cardinality is checked to be 1:1.",
        "The benchmark is the MLS regular-season file plus Austin_FC_nonleague_home_matches_wikipedia.csv: cup, playoff and friendly home fixtures with the attendance printed in the Wikipedia match templates (season pages, competition pages, the 2025 Open Cup final page). Those rows were AI-extracted and carry a verify flag; announced attendance for those fixtures is used in J10 and the per-game table only, not in the league turnout analyses.",
        "Demand vs performance uses all played Austin matches (home and away) to compute result, points, last-5 form and season points-per-game entering each home match; turnout is scanned seats over the announced figure. Sellout check compares announced (official >= 20,738), ticketed (final-holder seats >= 20,738) and turnstile (scanned share) per match.",
        "The MLS file is regular season only, so playoff, US Open Cup, Leagues Cup and friendly events in Attendance are expected to stay unmatched; their kind is inferred from the event name and opponent.",
        "Official attendance for Austin FC is the file's figure, which the source flags as an assumption: every Austin home match is recorded at Q2 capacity (20,738).",
        "scanned_seats = distinct section/row/seat scanned at the event; sold_seats = final-holder seats from the audit's Event_Crosswalk (trusted products only).",
        "Peer averages use played matches with a recorded attendance; 2026 is in progress. Capacity is Stadium_Capacity_Primary; NYCFC plays in two venues.",
        "Austin FC's attendance in every peer table is the club's scanned entries (distinct seats scanned per league match, 2022 onward; 2021 has no scan data and is left blank), not the announced 20,738. Every other club's figure is announced attendance (tickets distributed), so Austin's rank, utilization and per-capita figures are understated relative to peers by the size of their own unobserved no-show gap. Austin_Actual_vs_Announced keeps both bases side by side.",
        "Sold seats exceed the 20,738 capacity on most matches because ticketed inventory is not all fixed seating: about 3,700 general-admission slots per match in supporters sections 101-105 (rows GA1-GA38) and about 850 suite, loft, conference-room and standing-room places sit on top of roughly 16,200 fixed seats. Section labels are consistent between Sales and Attendance (177 shared); there are no duplicate seat labels (Seat_Inventory sheets).",
        "The fact-book reconciliation shows the 85 vs 81 home-date gap is the four 2021 rows with blank attendance: the total gap is exactly four sellouts of 20,738.",
        "Team-audit cross-check compares the Data Audit sheet's 2025 averages with the match data; FC Dallas 2025 is distorted by a temporary capacity cut (see the team's flags).",
    ], columns=["note"])
    try:
        xw_ = pd.ExcelWriter(out, engine="openpyxl")
    except PermissionError:
        out = out.replace(".xlsx", f"_{dt.datetime.now():%H%M}.xlsx")
        xw_ = pd.ExcelWriter(out, engine="openpyxl")
    with xw_:
        events_out.to_excel(xw_, sheet_name="J10_Event_Match", index=False)
        oc.to_excel(xw_, sheet_name="Opponent_Crosswalk", index=False)
        dvp_out.to_excel(xw_, sheet_name="Demand_vs_Performance", index=False)
        pd.concat([by_result.assign(cut="result"), by_form.rename(columns={"form_band": "result"}).assign(cut="last-5 form"), by_day.rename(columns={"weekday": "result"}).assign(cut="weekday")]).to_excel(xw_, sheet_name="Turnout_by_Cut", index=False)
        season_perf.to_excel(xw_, sheet_name="Season_Performance_Turnout", index=False)
        sellout_out.to_excel(xw_, sheet_name="Sellout_Check", index=False)
        sellout_season.to_excel(xw_, sheet_name="Sellout_by_Season", index=False)
        season_sum.to_excel(xw_, sheet_name="Austin_League_by_Season", index=False)
        all_kinds.to_excel(xw_, sheet_name="Events_by_Kind", index=False)
        missing_events.to_excel(xw_, sheet_name="MLS_Matches_No_Event", index=False)
        peers.to_excel(xw_, sheet_name="Peers_by_Season", index=False)
        austin_rank.to_excel(xw_, sheet_name="Austin_Rank", index=False)
        league_avg.to_excel(xw_, sheet_name="League_Averages", index=False)
        five.to_excel(xw_, sheet_name="Five_Peers_Avg_Attendance")
        five_util.to_excel(xw_, sheet_name="Five_Peers_Utilization")
        piv.to_excel(xw_, sheet_name="Avg_Attendance_Club_x_Season", index=False)
        austin_actual.to_excel(xw_, sheet_name="Austin_Actual_vs_Announced", index=False)
        opp.to_excel(xw_, sheet_name="Austin_by_Opponent", index=False)
        xcheck.to_excel(xw_, sheet_name="Team_Audit_Crosscheck", index=False)
        recon.to_excel(xw_, sheet_name="Fact_Book_Reconciliation", index=False)
        perf.to_excel(xw_, sheet_name="Austin_Performance_Context", index=False)
        epl.to_excel(xw_, sheet_name="EPL_Context", index=False)
        if inv is not None:
            inv_season.to_excel(xw_, sheet_name="Seat_Inventory_by_Season", index=False)
            inv.to_excel(xw_, sheet_name="Seat_Inventory_by_Event", index=False)
        notes.to_excel(xw_, sheet_name="Notes", index=False)
    A.autofilter(out)
    A.log(f"Wrote {out}  (aggregate only)")

    if a.json_out:
        lg = league.copy()
        summary = dict(
            events=n_ev, matched=n_match, opponent_agrees=int((j.opp_ok == True).sum()),
            unmatched_by_kind=j[~j.matched].kind.value_counts().to_dict(), matched_supp=n_supp, matched_league=n_match - n_supp,
            matched_by_competition=j[j.matched].kind.value_counts().to_dict(),
            mls_window_matches=len(win), mls_window_with_event=int(win.has_event.sum()),
            missing_events=[[str(r.Date.date()), r.Away_Std, (None if pd.isna(r.Attendance) else int(r.Attendance))] for r in missing_events.itertuples()],
            official_avg=float(lg.official_attendance.mean()), scanned_avg=float(lg.scanned_seats.mean()), scanned_share=float(lg.scanned_share_of_official.mean()),
            sold_avg=float(lg.sold_seats.mean()) if lg.sold_seats.notna().any() else None,
            season_summary=season_sum.to_dict(orient="records"),
            austin_rank=austin_rank.to_dict(orient="records"), league_avg=league_avg.to_dict(orient="records"),
            five=five.reset_index().to_dict(orient="records"), five_util=five_util.reset_index().to_dict(orient="records"),
            xcheck=xcheck.to_dict(orient="records"), recon=recon.to_dict(orient="records"), perf=perf.to_dict(orient="records"),
            epl=epl.to_dict(orient="records"),
            matches=[dict(season=int(r.season), date=str(r.date.date()), opponent=r.opponent, kind=r.kind, benchmark=r.benchmark, official=(None if pd.isna(r.official_attendance) else int(r.official_attendance)),
                          scanned=int(r.scanned_seats), scans=int(r.scans), sold=(None if pd.isna(r.sold_seats) else int(r.sold_seats)), score=(None if pd.isna(r.score) else str(r.score)))
                     for r in events_out.itertuples()],
            peers_2025=peers[peers.season.eq(2025)][["club", "home_matches", "avg_attendance", "capacity", "utilization", "metro_pop", "attendance_per_1000_metro", "rank_avg_attendance"]].round(3).to_dict(orient="records"),
            source_file="MLS_Match_Data_cleaned_flagged.xlsx (regular season 2018-2026, exported 2026-09-07)",
            inventory=(inv_season.to_dict(orient="records") if inv is not None else None),
            club_season=piv.to_dict(orient="records"), by_opponent=opp.to_dict(orient="records"), austin_actual=austin_actual.to_dict(orient="records"),
            one_to_one=one_to_one, date_only_events=int(date_only.EventKey.nunique()),
            crosswalk_how=oc.how.str.split(" (", regex=False).str[0].value_counts().to_dict(), crosswalk=oc.to_dict(orient="records"),
            by_result=by_result.to_dict(orient="records"), by_form=by_form.astype({"form_band": str}).to_dict(orient="records"), by_day=by_day.to_dict(orient="records"),
            corr_form=corr_form, corr_ppg=corr_ppg, season_perf=season_perf.to_dict(orient="records"), sellout_season=sellout_season.to_dict(orient="records"),
        )
        json.dump(summary, open(a.json_out, "w"), indent=1, default=str)
        A.log(f"Wrote {a.json_out}")
    pd.set_option("display.width", 200)
    print(season_sum.to_string(index=False))
    print(austin_rank.to_string(index=False))
    print(xcheck.to_string(index=False))
    print(recon.to_string(index=False))


if __name__ == "__main__":
    main()
