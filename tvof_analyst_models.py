"""
Austin FC TVOF | Analyst models (v1, 2026-09-29)
-------------------------------------------------------------
Five models that go past the cohort chain ladder, each answering one question:

  1  State-transition (Markov) model   plan, waitlist, single-ticket, lapsed 1 season, lapsed 2+: one matrix holds
                                       waitlist-to-plan conversion, upgrades, lapses and returns. LTV by entry state,
                                       customer equity including the lapsed pool, a backtest and an interval.
  2  Usage and renewal                 discrete-time survival: does a plan holder who leaves seats unused renew less?
                                       Renewal by share of seats wasted, logistic hazard model, survival by usage tercile.
  3  Transfer network                  who passes seats to whom: reach of a plan account, receive-only accounts and how
                                       many later buy, repeat partners, resellers, household-sized clusters.
  4  Resale premium                    the buyer's resale payment against the original sale of the same seat, by season,
                                       opponent, weekday, lead time, zone and face-price quartile: the willingness-to-pay signal.
  5  Activation uplift                 matched comparison of World Cup activation accounts with like accounts on
                                       pre-period attendance, with a placebo year to net out selection.

Stage A (slow, about ten minutes) reads the four client files once and caches compact frames keyed by an integer
account code (no IDs, no hashes) in <data>/analyst_models_cache_LOCAL_ONLY.pkl. Stage B (seconds) runs the models.

Usage:
    python tvof_analyst_models.py [--data-dir D] [--refresh] [--prep-only] [--sims 5000] [--json-out p]

Output: analyst_models_<date>.xlsx in the data folder (aggregate only).
"""
import argparse, datetime as dt, json, os, pickle, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tvof_data_audit as A   # noqa: E402
from tvof_fan_ltv import season_revenue, GROUP_MAX_ITEMS, SEGMENTS, FIRST_SEASON, LAST_COMPLETE, BASE_RATE, BASE_HORIZON   # noqa: E402
from tvof_fan_statements import by_cat, newest   # noqa: E402

CACHE = "analyst_models_cache_LOCAL_ONLY.pkl"
MIN_SCANS_EVENT = 1000          # events with fewer scanned seats have no usable scan data
TIE_MATCHES, HUB_PARTNERS = 5, 25   # network: a recurring tie is 5+ matches; an account with over 25 partners is a hub
IT = {"ticket": 0, "transfer": 1, "resale": 2, "subscription": 3}
TRUST = ["High", "Medium", "Manual override"]
WINDOW_END = "09-19"            # revenue windows stop at the data end (2026-09-18) in both years


# =============================================================================================== stage A: prepare
def codes_of(idx, arr):
    """Exact integer codes of a nullable UInt64 hash array in idx; -1 for missing. No float round trip."""
    a = pd.array(arr, dtype="UInt64")
    na = np.asarray(a.isna())
    v = a.to_numpy(dtype="uint64", na_value=0)
    c = idx.get_indexer(v).astype(np.int64)
    c[na] = -1
    return c


def cat_lower(series):
    """categorical -> lower-case object array (None where missing)."""
    return by_cat(series, lambda c: str(c).strip().lower(), default=None)


def prep(a):
    files = A.find_files(a.data_dir)
    for k in ("sales", "attendance", "fan", "activation"):
        if k not in files:
            sys.exit(f"{k} file not found in {a.data_dir}")
    audit_path = newest(a.data_dir, "TVOF_audit_results")
    xw = pd.read_excel(audit_path, sheet_name="Event_Crosswalk")
    tr = xw[xw.confidence.isin(TRUST) & xw.product_class.eq("Austin FC match")]
    p2e = dict(zip(tr.product_id.astype(str), tr.EventKey.astype(int).astype(str)))
    A.log(f"Crosswalk: {len(p2e)} match products ({os.path.basename(audit_path)})")

    # ------------------------------------------------------------------ attendance
    A.log("Loading attendance ...")
    att = A.load("attendance", files["attendance"], a.chunksize)
    ek = att[A.col(att, "EventKey")].astype(object).astype(str).to_numpy()
    dk = att[A.col(att, "EventDateKey")].astype(object).astype(str).to_numpy()
    nm = att[A.col(att, "MasterEventName", "EventName")].astype(object).astype(str).to_numpy()
    ev = pd.DataFrame({"EventKey": ek, "d": dk, "name": nm}).groupby("EventKey").agg(
        date=("d", "first"), name=("name", "first"), scan_rows=("d", "size")).reset_index()
    ev["date"] = pd.to_datetime(ev.date, format="%Y%m%d", errors="coerce")
    ev = ev.sort_values("date").reset_index(drop=True)
    ev["ev"] = np.arange(len(ev))
    ev_code = dict(zip(ev.EventKey, ev.ev))
    aev = pd.Series(ek).map(ev_code).to_numpy()
    akey = A.seat_key(att[A.col(att, "SectionName")], att[A.col(att, "RowName")], att[A.col(att, "SeatName")])
    att_acct = att[A.col(att, "internal_account_id")].array
    sec_a = pd.Series(A._seat_norm(att[A.col(att, "SectionName")].astype(object)).to_numpy())
    zc = pd.DataFrame({"sec": sec_a, "z": att[A.col(att, "SectionCategory")].astype(object).to_numpy()}).dropna()
    zone_map = zc.groupby("sec").z.agg(lambda x: x.value_counts().index[0]).to_dict()
    del att, zc, sec_a

    # ------------------------------------------------------------------ fan info, activation
    A.log("Loading fan info and activation ...")
    fan = A.load("fan", files["fan"], a.chunksize)
    fan_acct, fan_fid = fan[A.col(fan, "internal_account_id")].array, fan[A.col(fan, "internal_fan_id")].array
    fan_since = pd.to_datetime(fan[A.col(fan, "seatgeek_since_date")]).to_numpy()
    del fan
    act = A.load("activation", files["activation"], a.chunksize)
    act_fid = act[A.col(act, "internal_fan_id")].array
    act_sign = pd.to_datetime(act[A.col(act, "signup_datetime")]).to_numpy()
    act_scan = pd.to_datetime(act[A.col(act, "ticket_scan_datetime")]).notna().to_numpy()
    del act

    # ------------------------------------------------------------------ sales
    A.log("Loading sales ...")
    s = A.load("sales", files["sales"], a.chunksize)
    n = len(s)
    sal_acct = s.internal_account_id.array

    def u64(x):
        return pd.Series(pd.array(x, dtype="UInt64")).dropna().astype("uint64").to_numpy()
    idx = pd.Index(pd.unique(np.concatenate([u64(sal_acct), u64(att_acct), u64(fan_acct)])))
    A.log(f"  {len(idx):,} distinct accounts across sales, attendance and fan info")
    sa, aa, fa = codes_of(idx, sal_acct), codes_of(idx, att_acct), codes_of(idx, fan_acct)

    it_s = cat_lower(s.item_type)
    itc = np.array([IT.get(v, 4) for v in it_s], dtype=np.int8)
    sub_na = s.subscription_instance_id.isna().to_numpy()
    pay = s.total_payment.fillna(0).to_numpy(dtype=float)
    td = s.transaction_date.to_numpy()
    plan_ids = set(s.subscription_instance_id[itc == 3].dropna().to_numpy())
    orphan = (itc == 0) & ~sub_na & ~s.subscription_instance_id.isin(plan_ids).to_numpy()
    game = (itc == 0) & ~sub_na                                   # plan game rows (orphans included: still plan seats)
    club = np.where((itc == 3) | ((itc == 0) & sub_na) | orphan, pay, 0.0)
    evs = by_cat(s.product_id, lambda p: ev_code.get(p2e.get(str(p))), default=None)
    sev = np.array([-1 if v is None else int(v) for v in evs], dtype=np.int32)
    skey = A.seat_key(s[A.col(s, "section")], s[A.col(s, "row")], s[A.col(s, "seat")])
    skey_ok = ~np.asarray(skey.isna())
    skey_v = skey.to_numpy(dtype="uint64", na_value=0)
    cats = pd.Series(s.section.cat.categories.astype(str))
    zone_by_cat = np.array([zone_map.get(v) for v in A._seat_norm(cats).to_numpy()], dtype=object)
    zc_codes = s.section.cat.codes.to_numpy()
    zone = np.full(n, None, dtype=object); okz = zc_codes >= 0; zone[okz] = zone_by_cat[zc_codes[okz]]
    ts_s = cat_lower(s.transfer_status)
    dead_transfer = (itc == 1) & np.isin(ts_s.astype(str), ["pending", "canceled", "cancelled"])
    hold = A._holder_mask(s).to_numpy()
    pt = pd.array(s.primary_ticket_id.array, dtype="UInt64")
    pt_ok = ~np.asarray(pt.isna()); pt_v = pt.to_numpy(dtype="uint64", na_value=0)

    # account-season revenue, the fan_ltv season rule
    A.log("Account-season revenue ...")
    R = season_revenue(s)
    R["a"] = codes_of(idx, R.a.array)
    mx = R.groupby(["a", "prod"]).size().groupby(level=0).max()
    group_accts = mx[mx > GROUP_MAX_ITEMS].index.to_numpy()
    AS = R.groupby(["a", "season"]).agg(rev=("club", "sum"), plan=("plan", "sum"), wait=("wait", "sum")).reset_index()
    del R, s

    # ------------------------------------------------------------------ seats: plan originals, final holders, scans
    A.log("Seats: plan originals, final holders, scans ...")
    AU = pd.DataFrame({"ev": aev, "key": akey}).dropna()
    AU["key"] = AU.key.astype("uint64"); AU["ev"] = AU.ev.astype(np.int32)
    AU = AU.drop_duplicates()
    ev["scanned_seats"] = ev.ev.map(AU.groupby("ev").size()).fillna(0).astype(int)
    no_scan = set(ev.ev[ev.scanned_seats < MIN_SCANS_EVENT])
    m = game & (sev >= 0) & skey_ok & (sa >= 0)
    O = pd.DataFrame({"ev": sev[m], "key": skey_v[m], "a": sa[m], "pay": pay[m], "td": td[m]}).sort_values("td", kind="stable")
    O = O.drop_duplicates(["ev", "key"], keep="first").drop(columns="td")
    m = hold & (sev >= 0) & skey_ok
    H = pd.DataFrame({"ev": sev[m], "key": skey_v[m], "a_final": sa[m], "path": itc[m], "td": td[m]}).sort_values("td", kind="stable")
    H = H.drop_duplicates(["ev", "key"], keep="last").drop(columns="td")
    ev["sold_seats"] = ev.ev.map(H.groupby("ev").size()).fillna(0).astype(int)
    U = O.merge(H, on=["ev", "key"], how="left").merge(AU.assign(scanned=True), on=["ev", "key"], how="left")
    U["scanned"] = U.scanned.fillna(False).astype(bool)
    U = U[~U.ev.isin(no_scan)]
    U["season"] = U.ev.map(ev.set_index("ev").date.dt.year).astype(int)
    kept = U.path.eq(0) & U.a_final.eq(U.a)
    U["kept_scanned"] = kept & U.scanned
    U["kept_unscanned"] = kept & ~U.scanned
    U["transferred"] = U.path.eq(1)
    U["transferred_scanned"] = U.path.eq(1) & U.scanned
    U["resold"] = U.path.eq(2)
    U["resold_scanned"] = U.path.eq(2) & U.scanned
    U["other"] = ~(kept | U.transferred | U.resold)
    usage = U.groupby(["a", "season"]).agg(seats=("key", "size"), events=("ev", "nunique"), face=("pay", "sum"),
                                           kept_scanned=("kept_scanned", "sum"), kept_unscanned=("kept_unscanned", "sum"),
                                           transferred=("transferred", "sum"), transferred_scanned=("transferred_scanned", "sum"),
                                           resold=("resold", "sum"), resold_scanned=("resold_scanned", "sum"),
                                           other=("other", "sum")).reset_index()
    A.log(f"  {len(U):,} plan seat-events, {usage.a.nunique():,} plan accounts with seats")
    del O, H, U

    # ------------------------------------------------------------------ ticket chains: sender -> receiver
    A.log("Ticket chains ...")
    m = pt_ok & (itc != 3) & ~dead_transfer & (sa >= 0)
    C = pd.DataFrame({"pt": pt_v[m], "td": td[m], "a": sa[m], "it": itc[m], "pay": pay[m], "ev": sev[m],
                      "zone": zone[m], "plan_row": game[m]})
    C = C.sort_values(["pt", "td"], kind="stable").reset_index(drop=True)
    ptv, av = C.pt.to_numpy(), C.a.to_numpy()
    same = np.r_[False, ptv[1:] == ptv[:-1]]
    prev = np.where(same, np.r_[-1, av[:-1]], -1)
    grp = np.cumsum(~same) - 1
    first = np.flatnonzero(~same)
    C["sender"] = prev
    C["face"] = C.pay.to_numpy()[first][grp]
    C["orig_plan"] = C.plan_row.to_numpy()[first][grp]
    C["orig_a"] = av[first][grp]
    C["hop"] = np.arange(len(C)) - first[grp]
    edges = C[C.it.isin([1, 2]) & (C.sender >= 0) & (C.sender != C.a)].rename(columns={"a": "receiver"})
    edges = edges[["sender", "receiver", "it", "ev", "td", "pay", "face", "orig_plan", "orig_a", "zone", "hop"]].reset_index(drop=True)
    A.log(f"  {len(C):,} chain rows, {len(edges):,} sender-to-receiver rows ({int((edges.it == 1).sum()):,} transfers, {int((edges.it == 2).sum()):,} resales)")
    del C

    # ------------------------------------------------------------------ account attributes
    A.log("Account attributes ...")
    na_acc = len(idx)
    td_i = td.astype("datetime64[ns]").astype("int64")
    big = np.iinfo(np.int64).max
    nat = np.datetime64("NaT").astype("datetime64[ns]").astype("int64")
    ok = (sa >= 0) & ~np.isnat(td)
    first_td = np.full(na_acc, big); np.minimum.at(first_td, sa[ok], td_i[ok])
    okc = ok & (club > 0)
    first_club = np.full(na_acc, big); np.minimum.at(first_club, sa[okc], td_i[okc])
    okr = ok & np.isin(itc, [1, 2]) & ~dead_transfer
    first_recv = np.full(na_acc, big); np.minimum.at(first_recv, sa[okr], td_i[okr])
    since_i = fan_since.astype("datetime64[ns]").astype("int64")
    oks = (fa >= 0) & ~np.isnat(fan_since)
    since = np.full(na_acc, big); np.minimum.at(since, fa[oks], since_i[oks])
    acc = pd.DataFrame({"first_td": pd.to_datetime(np.where(first_td == big, nat, first_td)),
                        "first_club_td": pd.to_datetime(np.where(first_club == big, nat, first_club)),
                        "first_recv_td": pd.to_datetime(np.where(first_recv == big, nat, first_recv)),
                        "since": pd.to_datetime(np.where(since == big, nat, since)),
                        "club_total": np.bincount(sa[sa >= 0], weights=club[sa >= 0], minlength=na_acc)})
    for y in (2025, 2026):
        mm = ok & (td >= np.datetime64(f"{y}-07-08")) & (td < np.datetime64(f"{y}-{WINDOW_END}"))
        acc[f"club_after_{y}"] = np.bincount(sa[mm], weights=club[mm], minlength=na_acc)
        single = np.where((itc == 0) & sub_na, pay, 0.0)             # tickets outside a plan: what a campaign can move in ten weeks
        acc[f"single_after_{y}"] = np.bincount(sa[mm], weights=single[mm], minlength=na_acc)
    acc["group"] = False
    acc.loc[group_accts[group_accts >= 0], "group"] = True
    acc["in_sales"] = False; acc.loc[np.unique(sa[sa >= 0]), "in_sales"] = True
    acc["in_fan_info"] = False; acc.loc[np.unique(fa[fa >= 0]), "in_fan_info"] = True

    att_ev = pd.DataFrame({"a": aa, "ev": aev}).dropna()
    att_ev = att_ev[att_ev.a >= 0].astype({"ev": np.int32})
    att_ev = att_ev.groupby(["a", "ev"]).size().rename("seats").reset_index()

    F = pd.DataFrame({"fid": pd.array(fan_fid, dtype="UInt64"), "a": fa})
    F = F[(F.a >= 0) & F.fid.notna()]
    T = pd.DataFrame({"fid": pd.array(act_fid, dtype="UInt64"), "sign": act_sign, "scan": act_scan}).dropna(subset=["fid"])
    T = T.groupby("fid").agg(first_signup=("sign", "min"), wp_scanned=("scan", "max"), signups=("sign", "size")).reset_index()
    treated = F.merge(T, on="fid", how="inner").groupby("a").agg(first_signup=("first_signup", "min"), wp_scanned=("wp_scanned", "max")).reset_index()
    A.log(f"  activation: {len(T):,} fans, {len(treated):,} linked accounts")

    out = {"AS": AS, "usage": usage, "edges": edges, "events": ev, "att_ev": att_ev, "acc": acc, "treated": treated,
           "n_accounts": na_acc, "activation_fans": int(len(T)), "built": dt.datetime.now().isoformat(timespec="seconds"),
           "sources": {k: os.path.basename(v) for k, v in files.items()}, "audit": os.path.basename(audit_path)}
    path = os.path.join(a.data_dir, CACHE)
    with open(path, "wb") as fh:
        pickle.dump(out, fh, protocol=pickle.HIGHEST_PROTOCOL)
    A.log(f"Wrote {path}  ({os.path.getsize(path) / 1e6:,.0f} MB, ROW LEVEL by account code: keep local)")
    return out


def get_cache(a):
    path = os.path.join(a.data_dir, CACHE)
    if a.refresh or not os.path.exists(path):
        return prep(a)
    with open(path, "rb") as fh:
        D = pickle.load(fh)
    A.log(f"Cache {CACHE} built {D['built']} from {D['sources'].get('sales')}")
    return D


# =============================================================================================== stage B: models
STATES = ["Plan member", "Waitlist only", "Single-ticket buyer", "Lapsed 1 season", "Lapsed 2+ seasons"]
P_, W_, S_, L1_, L2_ = range(5)
ENTRY = [(SEGMENTS[0], P_), (SEGMENTS[1], W_), (SEGMENTS[2], S_)]
PANEL_LAST = LAST_COMPLETE + 1      # 2026 enters only as the renewal outcome of 2025 plan holders
NK = PANEL_LAST - FIRST_SEASON + 1
SEASONS = FIRST_SEASON + np.arange(NK)


def state_panel(D):
    """Individual and household accounts x seasons: state, club revenue, cohort. Same population and season rule as fan_ltv."""
    AS, acc = D["AS"], D["acc"]
    x = AS[(AS.rev > 0) & AS.season.between(FIRST_SEASON, PANEL_LAST) & (AS.a >= 0)]
    x = x[~acc.group.to_numpy()[x.a.to_numpy()]]
    accts = np.unique(x.a.to_numpy())
    pos = pd.Index(accts).get_indexer(x.a.to_numpy())
    rev, plan, wait = np.zeros((len(accts), NK)), np.zeros((len(accts), NK)), np.zeros((len(accts), NK))
    j = x.season.to_numpy() - FIRST_SEASON
    rev[pos, j], plan[pos, j], wait[pos, j] = x.rev.to_numpy(), x.plan.to_numpy(), x.wait.to_numpy()
    active = rev > 0
    st = np.full(rev.shape, -1, dtype=np.int8)
    st[active & (plan > 0)] = P_
    st[active & (plan <= 0) & (rev - wait > 1e-9)] = S_
    st[active & (plan <= 0) & ~(rev - wait > 1e-9)] = W_
    seen = np.maximum.accumulate(active, axis=1)
    for k in range(1, NK):
        lap = seen[:, k - 1] & ~active[:, k]
        st[lap & active[:, k - 1], k] = L1_
        st[lap & ~active[:, k - 1], k] = L2_
    cohort = FIRST_SEASON + active.argmax(axis=1)
    return accts, st, rev, cohort


def trans_counts(st, years):
    """Counts of state(t) -> state(t+1) for each season t in years."""
    M = np.zeros((5, 5))
    for y in years:
        a, b = st[:, y - FIRST_SEASON], st[:, y + 1 - FIRST_SEASON]
        ok = (a >= 0) & (b >= 0)
        np.add.at(M, (a[ok], b[ok]), 1)
    return M


def row_norm(M):
    s = M.sum(axis=1, keepdims=True)
    return np.divide(M, s, out=np.zeros_like(M), where=s > 0)


def revenue_vectors(st, rev, cohort, fit_last):
    """Mean club revenue by state: in the acquisition season (first) and in later seasons, seasons <= fit_last."""
    is_first = cohort[:, None] == SEASONS[None, :]
    inwin = (SEASONS <= fit_last)[None, :]
    out = {}
    for nm, mask in (("first", is_first), ("later", ~is_first)):
        mean, se, n = np.zeros(5), np.zeros(5), np.zeros(5)
        for s_ in (P_, W_, S_):
            x = rev[(st == s_) & mask & inwin]
            if len(x):
                mean[s_], n[s_] = x.mean(), len(x)
                se[s_] = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
        out[nm] = (mean, se, n)
    return out


def ltv_entry(Pm, r_first, r_later, H, rate):
    out = np.zeros(5)
    for s_ in (P_, W_, S_):
        pi = np.zeros(5); pi[s_] = 1.0
        v = r_first[s_]
        for k in range(1, H):
            pi = pi @ Pm
            v += (pi @ r_later) / (1 + rate) ** k
        out[s_] = v
    return out


def remaining_value(Pm, r_later, H, rate):
    """Expected club revenue over the next H seasons by current state (the current season is excluded)."""
    V, Pk = np.zeros(5), np.eye(5)
    for k in range(1, H + 1):
        Pk = Pk @ Pm
        V += (Pk @ r_later) / (1 + rate) ** k
    return V


def model_markov(D, a, rng):
    accts, st, rev, cohort = state_panel(D)
    fit_years = list(range(FIRST_SEASON, LAST_COMPLETE))            # t = 2021..2024, transitions into 2022..2025
    M = trans_counts(st, fit_years)
    Pm = row_norm(M)
    RV = revenue_vectors(st, rev, cohort, LAST_COMPLETE)
    r_first, r_later = RV["first"][0], RV["later"][0]
    H, rate = BASE_HORIZON, BASE_RATE

    T1 = pd.DataFrame([{"from_state": STATES[i], "accounts_at_risk": int(M[i].sum()),
                        **{f"to {STATES[j]}": Pm[i, j] for j in range(5)}, **{f"n to {STATES[j]}": int(M[i, j]) for j in range(5)}}
                       for i in range(5)])
    yr = []
    for y in fit_years:
        My = trans_counts(st, [y]); Py = row_norm(My)
        ret = (My[L1_, :3].sum() + My[L2_, :3].sum()) / max(My[L1_].sum() + My[L2_].sum(), 1)
        yr.append({"from_season": y, "to_season": y + 1, "plan_stays_plan": Py[P_, P_], "plan_at_risk": int(My[P_].sum()),
                   "waitlist_to_plan": Py[W_, P_], "waitlist_at_risk": int(My[W_].sum()),
                   "single_to_plan": Py[S_, P_], "single_stays_active": Py[S_, :3].sum(), "single_at_risk": int(My[S_].sum()),
                   "lapsed1_returns": Py[L1_, :3].sum(), "lapsed2_returns": Py[L2_, :3].sum(), "lapsed_any_returns": ret,
                   "lapsed_at_risk": int(My[L1_].sum() + My[L2_].sum())})
    T2 = pd.DataFrame(yr)
    T3 = pd.DataFrame([{"state": STATES[s_], "revenue_first_season": RV["first"][0][s_], "accounts_first_season": int(RV["first"][2][s_]),
                        "revenue_later_seasons": RV["later"][0][s_], "account_seasons_later": int(RV["later"][2][s_])} for s_ in (P_, W_, S_)])

    # interval: resample the transition years (drift between years) and draw each row from its Dirichlet (sampling error)
    n = a.sims
    ltv_s, rem_s = np.zeros((n, 5)), np.zeros((n, 5))
    Mys = [trans_counts(st, [y]) for y in fit_years]
    for i in range(n):
        pick = rng.integers(0, len(Mys), len(Mys))
        Mb = sum(Mys[k] for k in pick)
        Pb = np.vstack([rng.dirichlet((Mb[r] if Mb[r].sum() >= 30 else M[r]) + 0.5) for r in range(5)])   # a resample with no lapsed accounts falls back to the pooled row
        rf = rng.normal(RV["first"][0], RV["first"][1]); rl = rng.normal(RV["later"][0], RV["later"][1])
        ltv_s[i] = ltv_entry(Pb, rf, rl, H, rate); rem_s[i] = remaining_value(Pb, rl, H, rate)
    ltv_pt, rem_pt = ltv_entry(Pm, r_first, r_later, H, rate), remaining_value(Pm, r_later, H, rate)
    P_last = row_norm(trans_counts(st, [LAST_COMPLETE - 1]))            # the latest transition alone: retention has drifted down
    ltv_last, rem_last = ltv_entry(P_last, r_first, r_later, H, rate), remaining_value(P_last, r_later, H, rate)

    ltv_path = newest(a.data_dir, "fan_ltv")
    cl = pd.read_excel(ltv_path, sheet_name="LTV_Base_Case").set_index("segment")
    T4 = pd.DataFrame([{"entry_state": STATES[s_], "chain_ladder_segment": seg, "markov_ltv": ltv_pt[s_],
                        "markov_p5": np.quantile(ltv_s[:, s_], 0.05), "markov_p50": np.quantile(ltv_s[:, s_], 0.5),
                        "markov_p95": np.quantile(ltv_s[:, s_], 0.95), "markov_ltv_latest_year": ltv_last[s_],
                        "chain_ladder_ltv": float(cl.loc[seg, "ltv"]), "markov_over_chain_ladder": ltv_pt[s_] / float(cl.loc[seg, "ltv"]) - 1,
                        "first_season_value": r_first[s_]} for seg, s_ in ENTRY])

    s25 = st[:, LAST_COMPLETE - FIRST_SEASON]
    cnt = np.array([(s25 == k).sum() for k in range(5)])
    eq_s = rem_s * cnt[None, :]
    st_path = newest(a.data_dir, "value_statements")
    eqs = pd.read_excel(st_path, sheet_name="5_Customer_Equity")
    cl_eq = float(eqs[eqs.segment.str.startswith("All")].remaining_total.iloc[0])
    rows = [{"state_in_2025": STATES[k], "accounts": int(cnt[k]), "remaining_per_account": rem_pt[k], "remaining_total": rem_pt[k] * cnt[k],
             "p5": np.quantile(eq_s[:, k], 0.05), "p95": np.quantile(eq_s[:, k], 0.95), "remaining_total_latest_year": rem_last[k] * cnt[k]} for k in range(5)]
    act_tot, lap_tot = eq_s[:, :3].sum(axis=1), eq_s[:, 3:].sum(axis=1)
    rows.append({"state_in_2025": "Active in 2025 (plan, waitlist, single)", "accounts": int(cnt[:3].sum()), "remaining_per_account": (rem_pt[:3] * cnt[:3]).sum() / max(cnt[:3].sum(), 1),
                 "remaining_total": (rem_pt[:3] * cnt[:3]).sum(), "p5": np.quantile(act_tot, 0.05), "p95": np.quantile(act_tot, 0.95),
                 "remaining_total_latest_year": (rem_last[:3] * cnt[:3]).sum()})
    rows.append({"state_in_2025": "Lapsed pool (paid before, not in 2025)", "accounts": int(cnt[3:].sum()), "remaining_per_account": (rem_pt[3:] * cnt[3:]).sum() / max(cnt[3:].sum(), 1),
                 "remaining_total": (rem_pt[3:] * cnt[3:]).sum(), "p5": np.quantile(lap_tot, 0.05), "p95": np.quantile(lap_tot, 0.95),
                 "remaining_total_latest_year": (rem_last[3:] * cnt[3:]).sum()})
    rows.append({"state_in_2025": "All accounts that ever paid", "accounts": int(cnt.sum()), "remaining_per_account": (rem_pt * cnt).sum() / max(cnt.sum(), 1),
                 "remaining_total": (rem_pt * cnt).sum(), "p5": np.quantile(act_tot + lap_tot, 0.05), "p95": np.quantile(act_tot + lap_tot, 0.95),
                 "remaining_total_latest_year": (rem_last * cnt).sum()})
    rows.append({"state_in_2025": "Chain ladder, active base only (value_statements)", "accounts": int(eqs[eqs.segment.str.startswith("All")].active_accounts.iloc[0]),
                 "remaining_per_account": float(eqs[eqs.segment.str.startswith("All")].remaining_per_member.iloc[0]), "remaining_total": cl_eq})
    T5 = pd.DataFrame(rows)

    # backtest: fit on seasons up to T, predict the seasons after for the accounts that existed at T
    entry = st[np.arange(len(st)), cohort - FIRST_SEASON]
    val = None
    try:
        val = pd.read_excel(newest(a.data_dir, "ltv_validation"), sheet_name="Backtest_by_Season")
    except SystemExit:
        pass
    bt = []
    for T in range(FIRST_SEASON + 1, LAST_COMPLETE):
        Mt = trans_counts(st, range(FIRST_SEASON, T)); Pt = row_norm(Mt)
        Pl = row_norm(trans_counts(st, [T - 1]))
        rl = revenue_vectors(st, rev, cohort, T)["later"][0]
        sT = st[:, T - FIRST_SEASON]
        exist = (cohort <= T) & (sT >= 0)
        for y in range(T + 1, LAST_COMPLETE + 1):
            v = np.linalg.matrix_power(Pt, y - T) @ rl
            pred_i = v[np.clip(sT, 0, 4)]
            pred_l = (np.linalg.matrix_power(Pl, y - T) @ rl)[np.clip(sT, 0, 4)]
            for seg, s_ in ENTRY + [("All individual buyers", -1)]:
                g = exist & ((entry == s_) if s_ >= 0 else True)
                actual, pred = rev[g, y - FIRST_SEASON].sum(), pred_i[g].sum()
                act_n, pred_n = int((st[g, y - FIRST_SEASON] <= S_).sum()), float((np.linalg.matrix_power(Pt, y - T)[np.clip(sT[g], 0, 4)][:, :3]).sum())
                cle = np.nan
                if val is not None:
                    h = val[(val.cutoff == T) & (val.season == y) & (val.segment == seg)]
                    cle = float(h.error_pct.iloc[0]) if len(h) else np.nan
                bt.append({"data_through": T, "season_predicted": y, "seasons_ahead": y - T, "first_season_segment": seg, "accounts": int(g.sum()),
                           "actual_revenue": actual, "markov_predicted": pred, "markov_error": pred / actual - 1 if actual else np.nan,
                           "markov_latest_year_error": pred_l[g].sum() / actual - 1 if actual else np.nan,
                           "chain_ladder_error": cle, "actual_active": act_n, "markov_active": pred_n,
                           "markov_active_error": pred_n / act_n - 1 if act_n else np.nan})
    T6 = pd.DataFrame(bt)

    # duration dependence: does plan retention rise with seasons held? (the Markov chain assumes it does not)
    run = np.zeros(st.shape, dtype=int)
    for k in range(NK):
        run[:, k] = np.where(st[:, k] == P_, (run[:, k - 1] if k else 0) + 1, 0)
    dur = []
    for t_ in (1, 2, 3, 4):
        at, stay = 0, 0
        for y in fit_years:
            k = y - FIRST_SEASON
            g = (st[:, k] == P_) & (run[:, k] == t_)
            at += int(g.sum()); stay += int((st[g, k + 1] == P_).sum())
        if at:
            dur.append({"consecutive_seasons_as_plan_member": t_, "accounts_at_risk": at, "renewed": stay, "renewal_rate": stay / at})
    T7 = pd.DataFrame(dur)
    ent = pd.Series(entry[(cohort >= FIRST_SEASON) & (cohort <= LAST_COMPLETE)]).map(dict(enumerate(STATES))).value_counts()
    A.log("  Markov: entry states of the 2021-2025 cohorts: " + ", ".join(f"{k} {v:,}" for k, v in ent.items()))
    return {"1_Markov_Transitions": T1, "1_Markov_By_Year": T2, "1_State_Revenue": T3, "1_Markov_LTV": T4, "1_Markov_Equity": T5,
            "1_Markov_Backtest": T6, "1_Plan_Retention_by_Tenure": T7}, {"panel": (accts, st, rev, cohort), "rem": rem_pt, "cnt": cnt}


# ------------------------------------------------------------------------------------ 2 usage and renewal
def logit_fit(X, y, iters=60):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ b)))
        W = p * (1 - p)
        Hm = (X * W[:, None]).T @ X + 1e-9 * np.eye(X.shape[1])
        step = np.linalg.solve(Hm, X.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-9:
            break
    p = np.clip(1 / (1 + np.exp(-(X @ b))), 1e-12, 1 - 1e-12)
    cov = np.linalg.inv((X * (p * (1 - p))[:, None]).T @ X + 1e-9 * np.eye(X.shape[1]))
    ll = float((y * np.log(p) + (1 - y) * np.log(1 - p)).sum())
    return b, np.sqrt(np.diag(cov)), ll, p


def auc(y, p):
    r = pd.Series(p).rank().to_numpy()
    n1, n0 = y.sum(), (1 - y).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan


BANDS = [(-0.001, 0.05, "0 to 5%"), (0.05, 0.10, "5 to 10%"), (0.10, 0.20, "10 to 20%"), (0.20, 0.30, "20 to 30%"),
         (0.30, 0.50, "30 to 50%"), (0.50, 1.01, "over 50%")]


def model_usage(D, a, ctx):
    accts, st, rev, cohort = ctx["panel"]
    U = D["usage"].copy()
    pos = pd.Index(accts).get_indexer(U.a.to_numpy())
    U = U[(pos >= 0) & U.season.between(FIRST_SEASON + 1, LAST_COMPLETE)].copy()
    pos = pd.Index(accts).get_indexer(U.a.to_numpy())
    k = U.season.to_numpy() - FIRST_SEASON
    U["state"] = st[pos, k]
    U["renewed"] = (st[pos, k + 1] == P_).astype(float)
    U["next_state"] = st[pos, k + 1]
    U["tenure"] = U.season.to_numpy() - cohort[pos]
    U = U[(U.state == P_) & (U.seats >= 10)].copy()
    for c in ("kept_scanned", "kept_unscanned", "transferred", "resold", "other"):
        U["sh_" + c] = U[c] / U.seats
    U["seats_per_event"] = U.seats / U.events
    U["face_per_seat"] = U.face / U.seats
    U["band"] = pd.cut(U.sh_kept_unscanned, [b[0] for b in BANDS] + [BANDS[-1][1]], labels=[b[2] for b in BANDS])

    by = U.groupby(["season", "band"]).agg(plan_accounts=("a", "size"), renewed=("renewed", "sum"), seats=("seats", "sum"),
                                           wasted_share=("sh_kept_unscanned", "mean"), passed_on_share=("sh_transferred", "mean"),
                                           resold_share=("sh_resold", "mean")).reset_index()
    by["renewal_rate"] = by.renewed / by.plan_accounts
    allb = U.groupby("band").agg(plan_accounts=("a", "size"), renewed=("renewed", "sum"), seats=("seats", "sum"),
                                 wasted_share=("sh_kept_unscanned", "mean"), passed_on_share=("sh_transferred", "mean"),
                                 resold_share=("sh_resold", "mean")).reset_index()
    allb["renewal_rate"] = allb.renewed / allb.plan_accounts
    allb.insert(0, "season", "all")
    by["season"] = by.season.astype(str)
    T1 = pd.concat([allb, by], ignore_index=True)
    T1["share_of_plan_accounts"] = T1.plan_accounts / T1.groupby("season").plan_accounts.transform("sum")

    summ = U.groupby("season").agg(plan_accounts=("a", "size"), seats=("seats", "sum"), used_by_holder=("kept_scanned", "sum"),
                                   wasted=("kept_unscanned", "sum"), passed_on=("transferred", "sum"), passed_on_scanned=("transferred_scanned", "sum"),
                                   resold=("resold", "sum"), resold_scanned=("resold_scanned", "sum"), renewed=("renewed", "sum")).reset_index()
    for c in ("used_by_holder", "wasted", "passed_on", "resold"):
        summ[c + "_share"] = summ[c] / summ.seats
    summ["renewal_rate"] = summ.renewed / summ.plan_accounts
    summ["renewal_outcome"] = summ.season.map(lambda y: f"plan in {y + 1}" + (" (2026 plans as sold by 2026-09-18)" if y + 1 > LAST_COMPLETE else ""))

    seasons = sorted(U.season.unique())
    X = np.column_stack([np.ones(len(U)), U.sh_kept_unscanned * 10, U.sh_transferred * 10, U.sh_resold * 10, U.tenure,
                         np.log(U.seats_per_event), np.log(U.face_per_seat.clip(lower=1))]
                        + [(U.season == y).astype(float) for y in seasons[1:]])
    names = ["intercept", "seats wasted, per 10 points of the plan", "seats passed on by transfer, per 10 points", "seats resold, per 10 points",
             "seasons since first purchase", "log seats per match", "log payment per seat"] + [f"season {y}" for y in seasons[1:]]
    y = U.renewed.to_numpy()
    b, se, ll, p = logit_fit(X, y)
    b0, _, ll0, _ = logit_fit(np.ones((len(U), 1)), y)
    ame = (p * (1 - p)).mean() * b
    T2 = pd.DataFrame({"term": names, "coefficient": b, "std_error": se, "z": b / se, "odds_ratio": np.exp(b),
                       "avg_marginal_effect_on_renewal": ame})
    T2.loc[0, ["odds_ratio", "avg_marginal_effect_on_renewal"]] = np.nan
    fit = {"person_seasons": int(len(U)), "accounts": int(U.a.nunique()), "renewal_rate": float(y.mean()), "log_likelihood": ll,
           "mcfadden_r2": 1 - ll / ll0, "auc": auc(y, p)}
    grid = []
    xm = X.mean(axis=0)
    for w in (0.0, 0.05, 0.10, 0.20, 0.30, 0.50):
        x = xm.copy(); x[1] = w * 10
        grid.append({"seats_wasted": w, "predicted_renewal": float(1 / (1 + np.exp(-(x @ b))))})
    T3 = pd.DataFrame(grid)

    # survival of the 2022 plan holders by how much of the plan they wasted in 2022
    c22 = U[U.season == FIRST_SEASON + 1].copy()
    c22["tercile"] = pd.qcut(c22.sh_kept_unscanned.rank(method="first"), 3, labels=["lowest third wasted", "middle third", "highest third wasted"])
    pos22 = pd.Index(accts).get_indexer(c22.a.to_numpy())
    sv = []
    for t_, g in c22.groupby("tercile"):
        gp = pos22[c22.tercile.to_numpy() == t_]
        row = {"wasted_in_2022": t_, "plan_accounts_2022": int(len(g)), "mean_wasted_share_2022": float(g.sh_kept_unscanned.mean())}
        alive = np.ones(len(gp), dtype=bool)
        for yk in range(FIRST_SEASON + 2, PANEL_LAST + 1):
            alive &= st[gp, yk - FIRST_SEASON] == P_
            row[f"still_plan_{yk}"] = float(alive.mean())
        sv.append(row)
    T4 = pd.DataFrame(sv)

    n_plan = int(ctx["cnt"][P_])
    lever = float(-ame[1])
    T5 = pd.DataFrame([{"measure": "Plan accounts, individual and household, 2025", "value": n_plan},
                       {"measure": "Remaining value per plan account, Markov, 10 seasons at 8%", "value": float(ctx["rem"][P_])},
                       {"measure": "Renewal change per 10 points less wasted, average marginal effect", "value": lever},
                       {"measure": "Extra renewals a season if wasted share fell 10 points", "value": lever * n_plan},
                       {"measure": "Equity attached to those renewals", "value": lever * n_plan * float(ctx["rem"][P_])},
                       {"measure": "Person-seasons in the model", "value": fit["person_seasons"]},
                       {"measure": "Renewal rate in the model", "value": fit["renewal_rate"]},
                       {"measure": "McFadden R2", "value": fit["mcfadden_r2"]}, {"measure": "AUC", "value": fit["auc"]}])
    return {"2_Usage_by_Season": summ, "2_Renewal_by_Wasted_Share": T1, "2_Renewal_Model": T2, "2_Renewal_vs_Wasted": T3,
            "2_Survival_by_Usage": T4, "2_Lever": T5}, {"fit": fit}


# ------------------------------------------------------------------------------------ 3 transfer network
def components(u, v):
    nodes, inv = np.unique(np.r_[u, v], return_inverse=True)
    parent = np.arange(len(nodes))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    h = len(u)
    for i in range(h):
        ra, rb = find(inv[i]), find(inv[h + i])
        if ra != rb:
            parent[ra] = rb
    return nodes, np.array([find(i) for i in range(len(nodes))])


def model_network(D, a, ctx):
    E, acc, ev, att = D["edges"].copy(), D["acc"], D["events"], D["att_ev"]
    accts = ctx["panel"][0]
    E["kind"] = np.where(E.it == 1, "transfer", "resale")
    evy = ev.set_index("ev").date.dt.year
    E["season"] = np.where(E.ev >= 0, E.ev.map(evy), pd.to_datetime(E.td).dt.year).astype(float)
    rows = []
    for kind, g in list(E.groupby("kind")) + [("all", E)]:
        pr = g.groupby(["sender", "receiver"]).size()
        rows.append({"kind": kind, "tickets_moved": int(len(g)), "senders": int(g.sender.nunique()), "receivers": int(g.receiver.nunique()),
                     "sender_receiver_pairs": int(len(pr)), "tickets_per_pair_mean": float(pr.mean()), "tickets_per_pair_median": float(pr.median()),
                     "share_of_tickets_in_pairs_with_3plus_tickets": float(pr[pr >= 3].sum() / pr.sum()),
                     "match_tickets": int((g.ev >= 0).sum()), "payments_to_sellers": float(g.pay[g.it == 2].sum())})
    T1 = pd.DataFrame(rows)

    tr = E[E.it == 1]
    pr = tr.groupby(["sender", "receiver"]).agg(tickets=("it", "size"), matches=("ev", "nunique")).reset_index()
    pr["band"] = pd.cut(pr.tickets, [0, 1, 2, 5, 20, 100, 10 ** 9], labels=["1 ticket", "2", "3 to 5", "6 to 20", "21 to 100", "over 100"])
    key = set(zip(pr.sender.to_numpy(), pr.receiver.to_numpy()))
    pr["reciprocal"] = [(r, s_) in key for s_, r in zip(pr.sender.to_numpy(), pr.receiver.to_numpy())]
    T2 = pr.groupby("band").agg(pairs=("tickets", "size"), tickets=("tickets", "sum"), reciprocal_pairs=("reciprocal", "sum")).reset_index()
    T2["share_of_pairs"] = T2.pairs / T2.pairs.sum(); T2["share_of_tickets"] = T2.tickets / T2.tickets.sum()
    T2["reciprocal_share"] = T2.reciprocal_pairs / T2.pairs

    # reach of a plan account: distinct accounts it hands its own plan seats to, per season
    ind = np.zeros(len(acc), dtype=bool); ind[accts] = True
    first_hop = E[(E.it == 1) & E.orig_plan & (E.sender == E.orig_a) & (E.ev >= 0)]
    reach = first_hop.groupby(["sender", "season"]).agg(recipients=("receiver", "nunique"), tickets=("it", "size")).reset_index()
    US = D["usage"][["a", "season", "seats"]].rename(columns={"a": "sender"})
    US = US[ind[US.sender.to_numpy()] & (US.seats >= 10)]
    RR = US.merge(reach, on=["sender", "season"], how="left").fillna({"recipients": 0, "tickets": 0})
    T3 = RR.groupby("season").agg(plan_accounts=("sender", "size"), share_passing_on_any=("recipients", lambda x: float((x > 0).mean())),
                                  recipients_mean=("recipients", "mean"), recipients_median=("recipients", "median"),
                                  recipients_p90=("recipients", lambda x: float(x.quantile(0.9))), distinct_recipient_links=("recipients", "sum"),
                                  tickets_passed_on=("tickets", "sum"), plan_seats=("seats", "sum")).reset_index()
    T3["share_of_plan_seats_passed_on"] = T3.tickets_passed_on / T3.plan_seats
    T3["season"] = T3.season.astype(int)

    # receive-only accounts: in Sales, never paid the club, received at least one ticket
    recv = E.groupby("receiver").agg(tickets=("it", "size"), by_transfer=("it", lambda x: int((x == 1).sum())), by_resale=("it", lambda x: int((x == 2).sum())),
                                     paid_sellers=("pay", "sum"), senders=("sender", "nunique")).reset_index().rename(columns={"receiver": "a"})
    recv["paid_club"] = acc.club_total.to_numpy()[recv.a.to_numpy()] > 0
    matches = att.groupby("a").ev.nunique()
    recv["matches"] = recv.a.map(matches).fillna(0)
    ro = recv[~recv.paid_club].copy()
    ro["route"] = np.select([(ro.by_transfer > 0) & (ro.by_resale == 0), (ro.by_transfer == 0) & (ro.by_resale > 0)], ["transfers only", "resale only"], "both")
    T4 = ro.groupby("route").agg(accounts=("a", "size"), tickets_received=("tickets", "sum"), attended_any=("matches", lambda x: int((x > 0).sum())),
                                 matches_mean=("matches", "mean"), attended_3plus=("matches", lambda x: int((x >= 3).sum())),
                                 paid_to_sellers=("paid_sellers", "sum")).reset_index()
    tot = pd.DataFrame([{"route": "all receive-only accounts", "accounts": len(ro), "tickets_received": int(ro.tickets.sum()), "attended_any": int((ro.matches > 0).sum()),
                         "matches_mean": float(ro.matches.mean()), "attended_3plus": int((ro.matches >= 3).sum()), "paid_to_sellers": float(ro.paid_sellers.sum())}])
    T4 = pd.concat([T4, tot], ignore_index=True)
    T4["attended_share"] = T4.attended_any / T4.accounts

    # conversion: the account's first Sales row was a ticket received from someone else; did it later buy from the club?
    fr = acc[acc.in_sales & acc.first_recv_td.notna() & (acc.first_recv_td <= acc.first_td)].copy()
    fe = E.sort_values("td", kind="stable").drop_duplicates("receiver").set_index("receiver").kind
    fr["route"] = fe.reindex(fr.index).to_numpy()
    fr["year"] = fr.first_recv_td.dt.year
    fr["converted"] = fr.first_club_td.notna() & (fr.first_club_td > fr.first_recv_td)
    fr["days"] = (fr.first_club_td - fr.first_recv_td).dt.days.where(fr.converted)
    fr["rev"] = fr.club_total.where(fr.converted, 0.0)
    g = fr.dropna(subset=["route"]).groupby(["route", "year"])
    T5 = g.agg(accounts=("converted", "size"), later_bought_from_club=("converted", "sum"), median_days_to_first_purchase=("days", "median"),
               club_revenue_from_converts=("rev", "sum")).reset_index()
    ga = fr.dropna(subset=["route"]).groupby("route").agg(accounts=("converted", "size"), later_bought_from_club=("converted", "sum"),
                                                        median_days_to_first_purchase=("days", "median"), club_revenue_from_converts=("rev", "sum")).reset_index()
    ga.insert(1, "year", "all")
    T5["year"] = T5.year.astype(int).astype(str)
    T5 = pd.concat([ga, T5], ignore_index=True)
    T5["conversion_rate"] = T5.later_bought_from_club / T5.accounts
    T5["revenue_per_convert"] = T5.club_revenue_from_converts / T5.later_bought_from_club.replace(0, np.nan)

    # resellers
    rs = E[E.it == 2].groupby("sender").agg(tickets=("it", "size"), buyers=("receiver", "nunique"), proceeds=("pay", "sum"), face=("face", "sum")).reset_index()
    rs["group"] = acc.group.to_numpy()[rs.sender.to_numpy()]
    rs["band"] = pd.cut(rs.tickets, [0, 5, 20, 100, 500, 10 ** 9], labels=["1 to 5 tickets", "6 to 20", "21 to 100", "101 to 500", "over 500"])
    T6 = rs.groupby("band").agg(sellers=("sender", "size"), tickets_resold=("tickets", "sum"), buyers_reached=("buyers", "sum"),
                                proceeds=("proceeds", "sum"), face_value=("face", "sum"), group_accounts=("group", "sum")).reset_index()
    T6["share_of_sellers"] = T6.sellers / T6.sellers.sum(); T6["share_of_tickets"] = T6.tickets_resold / T6.tickets_resold.sum()
    T6["proceeds_over_face"] = T6.proceeds / T6.face_value.replace(0, np.nan)

    # clusters: accounts that pass seats to each other at 5 or more matches; hubs with over 25 partners are set aside
    mp = tr[tr.ev >= 0].groupby(["sender", "receiver"]).agg(tickets=("it", "size"), matches=("ev", "nunique")).reset_index()
    deg = pd.concat([mp.sender, mp.receiver]).value_counts()
    hubs = set(deg[deg > HUB_PARTNERS].index)
    hub_tickets = int(mp.tickets[mp.sender.isin(hubs) | mp.receiver.isin(hubs)].sum())
    strong = mp[(mp.matches >= TIE_MATCHES) & ~mp.sender.isin(hubs) & ~mp.receiver.isin(hubs)]
    nodes, root = components(strong.sender.to_numpy(), strong.receiver.to_numpy())
    size = pd.Series(root).value_counts()
    cs = pd.cut(size, [1, 2, 4, 9, 10 ** 9], labels=["2 accounts", "3 to 4", "5 to 9", "10 or more"])
    T7 = pd.DataFrame({"cluster_size": cs.to_numpy(), "accounts": size.to_numpy()}).groupby("cluster_size").agg(clusters=("accounts", "size"), accounts=("accounts", "sum")).reset_index()
    T7["share_of_accounts"] = T7.accounts / T7.accounts.sum()
    meta = {"receive_only": int(len(ro)), "receive_only_attended": int((ro.matches > 0).sum()), "largest_cluster": int(size.max()),
            "clustered_accounts": int(len(nodes)), "clusters": int(len(size)), "recurring_pairs": int(len(strong)), "match_pairs": int(len(mp)),
            "hubs": int(len(hubs)), "hub_share_of_transferred_match_tickets": hub_tickets / max(int(mp.tickets.sum()), 1),
            "tie_matches": TIE_MATCHES, "hub_partners": HUB_PARTNERS}
    return {"3_Network_Summary": T1, "3_Pair_Strength": T2, "3_Plan_Reach": T3, "3_Receive_Only": T4, "3_Conversion": T5,
            "3_Resellers": T6, "3_Clusters": T7}, meta


# ------------------------------------------------------------------------------------ 4 resale premium
def model_resale(D, a, ctx):
    E, ev = D["edges"], D["events"].set_index("ev")
    if "sold_seats" not in ev:                      # cache built before sold seats were stored per event
        ev = ev.assign(sold_seats=np.nan)
    R = E[(E.it == 2) & (E.ev >= 0) & (E.face > 0) & (E.pay > 0)].copy()
    n_all = int(((E.it == 2) & (E.ev >= 0)).sum())
    R["ratio"] = R.pay / R.face
    odd = int((R.ratio > 20).sum())
    R = R[R.ratio <= 20]
    R["date"] = R.ev.map(ev.date); R["season"] = R.date.dt.year
    opp = R.ev.map(ev.name).str[11:]
    canon = opp.groupby(opp.str.lower()).agg(lambda x: x.value_counts().index[0])
    R["opponent"] = opp.str.lower().map(canon)
    R["weekday"] = R.date.dt.day_name()
    R["lead"] = (R.date.dt.normalize() - pd.to_datetime(R.td).dt.normalize()).dt.days
    R["lead_band"] = pd.cut(R.lead, [-10 ** 6, 1, 7, 30, 90, 10 ** 6], labels=["match day or day before", "2 to 7 days", "8 to 30 days", "31 to 90 days", "over 90 days"])
    R["origin"] = np.where(R.orig_plan, "plan seat", "single ticket")
    R["face_quartile"] = R.groupby("season").face.transform(lambda x: pd.qcut(x.rank(method="first"), 4, labels=["lowest face quartile", "second", "third", "highest face quartile"]))

    def tab(by, minn=0):
        g = R.groupby(by)
        t = g.agg(resales=("ratio", "size"), paid_by_buyers=("pay", "sum"), original_sale_value=("face", "sum"), median_ratio=("ratio", "median"),
                  p25_ratio=("ratio", lambda x: float(x.quantile(0.25))), p75_ratio=("ratio", lambda x: float(x.quantile(0.75))),
                  below_original=("ratio", lambda x: float((x < 1).mean())), median_paid=("pay", "median"), median_original=("face", "median")).reset_index()
        t["markup"] = t.paid_by_buyers - t.original_sale_value
        t["paid_over_original"] = t.paid_by_buyers / t.original_sale_value
        return t[t.resales >= minn]
    T1 = tab("season"); T1["season"] = T1.season.astype(int)
    T2 = tab("opponent", 300).sort_values("median_ratio", ascending=False)
    T3 = tab("weekday"); T4 = tab("lead_band"); T5 = tab("zone"); T6 = tab("face_quartile"); T7 = tab("origin")
    M = tab("ev")
    M["date"] = M.ev.map(ev.date).dt.strftime("%Y-%m-%d"); M["opponent"] = M.ev.map(ev.name).str[11:]
    M["scanned_seats"] = M.ev.map(ev.scanned_seats); M["sold_seats"] = M.ev.map(ev.sold_seats)
    M["scan_rate_of_sold"] = M.scanned_seats / M.sold_seats.replace(0, np.nan)
    M["resold_share_of_sold"] = M.resales / M.sold_seats.replace(0, np.nan)
    M = M[M.scanned_seats >= MIN_SCANS_EVENT].sort_values("date").drop(columns="ev")
    okm = M.scan_rate_of_sold.notna() & (M.resales >= 300)
    c1 = float(np.corrcoef(M.median_ratio[okm], M.scan_rate_of_sold[okm])[0, 1]) if okm.sum() > 2 else np.nan
    c2 = float(np.corrcoef(M.resold_share_of_sold[okm], M.scan_rate_of_sold[okm])[0, 1]) if okm.sum() > 2 else np.nan
    meta = {"resale_rows_at_matches": n_all, "with_both_prices": int(len(R)), "ratio_over_20_dropped": odd,
            "corr_ratio_vs_scan_rate": c1, "corr_resold_share_vs_scan_rate": c2,
            "median_ratio": float(R.ratio.median()), "below_original": float((R.ratio < 1).mean()),
            "paid": float(R.pay.sum()), "original": float(R.face.sum())}
    return {"4_Resale_by_Season": T1, "4_Resale_by_Opponent": T2, "4_Resale_by_Weekday": T3, "4_Resale_by_Lead_Time": T4,
            "4_Resale_by_Zone": T5, "4_Resale_by_Face_Quartile": T6, "4_Resale_by_Origin": T7, "4_Resale_by_Match": M}, meta


# ------------------------------------------------------------------------------------ 5 activation uplift
def matched(df, strata, outcomes):
    """Exact matching on strata. ATT = treated-weighted mean of within-stratum differences; strata need both groups."""
    out = []
    g = df.groupby(strata + ["treated"])
    size = g.size().unstack("treated")
    ok = size.notna().all(axis=1)
    nT, nC = size.loc[ok, True], size.loc[ok, False]
    w = nT / nT.sum()
    for y in outcomes:
        m = g[y].mean().unstack("treated").loc[ok]; v = g[y].var().unstack("treated").loc[ok].fillna(0)
        att = float((w * (m[True] - m[False])).sum())
        se = float(np.sqrt((w ** 2 * (v[True] / nT + v[False] / nC)).sum()))
        raw_t, raw_c = df.loc[df.treated, y].mean(), df.loc[~df.treated, y].mean()
        out.append({"outcome": y, "treated_accounts": int(df.treated.sum()), "treated_matched": int(nT.sum()), "controls_used": int(nC.sum()),
                    "strata": int(ok.sum()), "treated_mean": float((w * m[True]).sum()), "matched_control_mean": float((w * m[False]).sum()),
                    "difference_matched": att, "std_error": se, "z": att / se if se else np.nan,
                    "raw_treated_mean": float(raw_t), "raw_other_mean": float(raw_c), "difference_raw": float(raw_t - raw_c)})
    return pd.DataFrame(out)


def frame(D, Y, post_from, n_matches=None):
    acc, ev, att, AS = D["acc"], D["events"], D["att_ev"], D["AS"]
    cut = pd.Timestamp(f"{Y}-05-12")
    since = acc.since if "since" in acc else acc.first_td
    since = since.fillna(acc.first_td)
    U = pd.DataFrame({"a": np.flatnonzero((acc.in_fan_info & since.notna() & (since < cut)).to_numpy())})
    d = ev.set_index("ev").date
    e = att.assign(date=att.ev.map(d))

    def cnt(lo, hi):
        x = e[(e.date >= lo) & (e.date <= hi)].groupby("a").ev.nunique()
        return U.a.map(x).fillna(0).to_numpy()
    after = d[d >= pd.Timestamp(post_from)].sort_values()
    after = after[after.dt.year == Y]
    if n_matches:
        after = after.iloc[:n_matches]
    post_lo, post_to = after.min(), after.max()
    U["pre"] = cnt(pd.Timestamp(f"{Y}-01-01"), pd.Timestamp(f"{Y}-05-31"))
    U["post_matches"] = cnt(post_lo, post_to)
    U["post_any"] = (U.post_matches > 0).astype(float)
    U["prev"] = cnt(pd.Timestamp(f"{Y - 1}-01-01"), pd.Timestamp(f"{Y - 1}-12-31"))
    col = f"club_after_{Y}"
    U["post_club_revenue"] = acc[col].to_numpy()[U.a.to_numpy()] if col in acc else np.nan
    U["post_bought"] = (U.post_club_revenue > 0).astype(float)
    col = f"single_after_{Y}"
    U["post_single_revenue"] = acc[col].to_numpy()[U.a.to_numpy()] if col in acc else np.nan
    U["post_plan_revenue"] = U.post_club_revenue - U.post_single_revenue
    U["post_bought_single"] = (U.post_single_revenue > 0).astype(float).where(U.post_single_revenue.notna())
    plan = AS[(AS.season == Y) & (AS.plan > 0)].a.unique()
    U["plan"] = U.a.isin(plan)
    fc = acc.first_club_td.to_numpy()[U.a.to_numpy()]
    U["paid_before"] = pd.Series(fc).notna().to_numpy() & (fc < np.datetime64(cut))
    yrs = (cut - pd.Series(pd.to_datetime(since.to_numpy()[U.a.to_numpy()]))).dt.days.to_numpy() / 365.25
    U["age"] = pd.cut(yrs, [-1, 1, 2, 4, 100], labels=["under 1 year", "1 to 2", "2 to 4", "over 4"]).astype(str)
    U["pre_b"] = np.minimum(U.pre, 5).astype(int)
    U["prev_b"] = pd.cut(U.prev, [-1, 0, 2, 5, 10, 100], labels=["0", "1-2", "3-5", "6-10", "11+"]).astype(str)
    U["group"] = acc.group.to_numpy()[U.a.to_numpy()]
    n_pre = int(((d >= pd.Timestamp(f"{Y}-01-01")) & (d <= pd.Timestamp(f"{Y}-05-31"))).sum())
    n_post = int(len(after))
    return U, n_pre, n_post, f"{post_lo:%Y-%m-%d} to {post_to:%Y-%m-%d}"


def model_activation(D, a, ctx):
    tr = D["treated"].set_index("a")
    strata = ["pre_b", "prev_b", "plan", "paid_before", "age"]
    outs = ["post_matches", "post_any", "post_single_revenue", "post_bought_single", "post_plan_revenue", "post_club_revenue", "post_bought"]
    res, meta = [], {}
    n_real = None
    for label, Y, lo in (("2026, the activation year", 2026, "2026-07-22"), ("2025, placebo year", 2025, "2025-07-08")):
        U, n_pre, n_post, win = frame(D, Y, lo, n_real)
        n_real = n_real or n_post
        U = U[~U.group]
        U["treated"] = U.a.isin(tr.index)
        U["wp"] = U.a.map(tr.wp_scanned).fillna(False).astype(bool)
        oo = [o for o in outs if U[o].notna().any()]
        for who, sub in (("all linked activation accounts", U), ("attended a watch party", U[~U.treated | U.wp]), ("signed up only", U[~U.treated | ~U.wp])):
            t = matched(sub, strata, oo)
            t.insert(0, "treated_group", who); t.insert(0, "window", f"{win}, {n_post} home matches"); t.insert(0, "year", label)
            res.append(t)
        meta[Y] = {"universe": int(len(U)), "treated": int(U.treated.sum()), "pre_matches": n_pre, "post_matches": n_post, "window": win}
    T1 = pd.concat(res, ignore_index=True)
    real = T1[T1.year.str.startswith("2026")].set_index(["treated_group", "outcome"])
    plc = T1[T1.year.str.startswith("2025")].set_index(["treated_group", "outcome"])
    adj = []
    for k in real.index:
        if k in plc.index:
            r, p_ = real.loc[k], plc.loc[k]
            scale = meta[2026]["post_matches"] / meta[2025]["post_matches"] if k[1] == "post_matches" else 1.0
            pl = float(p_.difference_matched) * scale
            adj.append({"treated_group": k[0], "outcome": k[1], "matched_difference_2026": float(r.difference_matched), "std_error_2026": float(r.std_error),
                        "placebo_difference_2025": float(p_.difference_matched), "placebo_scaled_to_2026_window": pl,
                        "net_of_placebo": float(r.difference_matched) - pl, "std_error_net": float(np.sqrt(r.std_error ** 2 + (p_.std_error * scale) ** 2)),
                        "treated_matched": int(r.treated_matched), "matched_control_mean_2026": float(r.matched_control_mean)})
    T2 = pd.DataFrame(adj)
    T2["z_net"] = T2.net_of_placebo / T2.std_error_net
    T2["net_total_over_treated"] = T2.net_of_placebo * T2.treated_matched

    # accounts the activation created: linked accounts opened on or after 2026-05-12, no pre-period to match on
    acc, att, ev = D["acc"], D["att_ev"], D["events"].set_index("ev")
    since = (acc.since if "since" in acc else acc.first_td).fillna(acc.first_td)
    new = tr.index[(since.reindex(tr.index) >= pd.Timestamp("2026-05-12")).to_numpy()]
    e = att[att.a.isin(new)].assign(date=lambda x: x.ev.map(ev.date))
    post = e[e.date >= pd.Timestamp("2026-07-22")].groupby("a").ev.nunique()
    T3 = pd.DataFrame([{"measure": "Linked activation accounts", "value": int(len(tr))},
                       {"measure": "Opened before 2026-05-12 (matched comparison)", "value": int(len(tr) - len(new))},
                       {"measure": "Opened on or after 2026-05-12 (new accounts)", "value": int(len(new))},
                       {"measure": "New accounts that scanned at a match from 2026-07-22", "value": int(len(post))},
                       {"measure": "New accounts that paid the club from 2026-07-08", "value": int((acc.club_after_2026.reindex(new) > 0).sum()) if "club_after_2026" in acc else np.nan},
                       {"measure": "Club revenue from new accounts from 2026-07-08", "value": float(acc.club_after_2026.reindex(new).sum()) if "club_after_2026" in acc else np.nan},
                       {"measure": "Club revenue from new accounts, to date", "value": float(acc.club_total.reindex(new).sum())}])
    return {"5_Activation_Matched": T1, "5_Activation_Net": T2, "5_Activation_New_Accounts": T3}, meta


NOTES = [
    "Population for models 1 and 2: individual and household accounts (never more than 8 items of one product), the fan_ltv population. Seasons follow the fan_ltv rule: plans take the year in their name, other purchases the transaction year with December rolled forward.",
    "1 States per account and season: plan member (paid plan revenue), single-ticket buyer (other club revenue), waitlist only (paid nothing but the waitlist fee), lapsed 1 season, lapsed 2+ seasons. An account that pays the waitlist fee and also buys single tickets is a single-ticket buyer, as in fan_ltv.",
    "1 The matrix pools the four transitions 2021-22 to 2024-25. LTV by entry state = first-season revenue plus ten seasons of expected revenue by state, discounted at 8%. Remaining value excludes the current season. The interval resamples the four transition years and draws each row from a Dirichlet.",
    "1 A first-order chain has no memory: plan retention is assumed the same in every season held. The tenure table shows how far that is from the data.",
    "2 Plan seats are plan game rows mapped to a match. For each seat and match: used by the holder (kept and scanned), wasted (kept and not scanned), passed on (final holder row is a transfer), resold (final holder row is a resale). Matches with fewer than 1,000 scanned seats are left out. Plan accounts with fewer than 10 mapped seats in a season are left out.",
    "2 Renewal = the account pays plan revenue for the next season. The 2025 rows use 2026 plans as sold by 2026-09-18. The model is a logistic hazard on account-seasons; shares are entered per 10 points of the plan with 'used by the holder' as the reference. It shows association, not cause: a fan who is losing interest both skips matches and does not renew.",
    "3 A sender-to-receiver row is a transfer or resale row in a ticket chain (primary_ticket_id), preceded by a different account. Pending and cancelled transfers are left out. Receive-only accounts are in Sales, have never paid the club, and have received at least one ticket.",
    "3 Conversion: accounts whose first Sales row is a received ticket, and whether they later paid the club. Recent years have had less time to convert.",
    "4 Ratio = the resale buyer's total_payment over the total_payment of the first row in the same ticket chain (the plan payment split per game for plan seats). Whether the buyer's payment includes fees is not stated in the data dictionary. Rows with a ratio over 20 are dropped as errors.",
    "5 Exact matching on five characteristics measured before the activation: home matches attended January to May of the year (0 to 5+), matches attended the year before (5 bands), plan member that season, paid the club before 12 May, account age (4 bands). The difference is weighted by the treated accounts in each cell. The placebo repeats the design one year earlier for the same accounts, when no activation took place; what it finds is selection, and it is subtracted, scaled to the number of matches in the window.",
    "5 Revenue windows run 8 July to 18 September in both years. Accounts opened on or after 12 May 2026 have no pre-period and are reported separately.",
    "All five are gross ticketing revenue at nominal prices. Row-level frames stay in the local cache; this workbook is aggregate only.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=A.DEFAULT_DIR)
    ap.add_argument("--chunksize", type=int, default=A.CHUNK)
    ap.add_argument("--refresh", action="store_true", help="rebuild the cache from the client files")
    ap.add_argument("--prep-only", action="store_true")
    ap.add_argument("--sims", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()
    D = get_cache(a)
    if a.prep_only:
        for k, v in D.items():
            if isinstance(v, pd.DataFrame):
                print(f"{k:10s} {v.shape}  {list(v.columns)}")
        return
    rng = np.random.default_rng(a.seed)
    sheets, meta = {}, {}
    A.log("1 State-transition model ...")
    t, ctx = model_markov(D, a, rng); sheets.update(t)
    A.log("2 Usage and renewal ...")
    t, meta["usage"] = model_usage(D, a, ctx); sheets.update(t)
    A.log("3 Transfer network ...")
    t, meta["network"] = model_network(D, a, ctx); sheets.update(t)
    A.log("4 Resale premium ...")
    t, meta["resale"] = model_resale(D, a, ctx); sheets.update(t)
    A.log("5 Activation uplift ...")
    t, meta["activation"] = model_activation(D, a, ctx); sheets.update(t)
    notes = pd.DataFrame({"note": NOTES + [f"Built {dt.date.today():%Y-%m-%d} from {D['sources']} and {D['audit']}; cache built {D['built']}; seed {a.seed}, {a.sims:,} draws."]})
    out = os.path.join(a.data_dir, f"analyst_models_{dt.date.today():%Y%m%d}.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        for k, v in sheets.items():
            v.to_excel(xw, sheet_name=k[:31], index=False)
        notes.to_excel(xw, sheet_name="Notes", index=False)
    A.autofilter(out)
    A.log(f"Wrote {out}  (aggregate only)")
    if a.json_out:
        json.dump({"sheets": {k: json.loads(v.to_json(orient="records", date_format="iso")) for k, v in sheets.items()},
                   "meta": json.loads(json.dumps(meta, default=float)), "notes": notes.note.tolist(), "source": os.path.basename(out),
                   "horizon": BASE_HORIZON, "rate": BASE_RATE, "sims": a.sims, "seed": a.seed}, open(a.json_out, "w"), indent=0)
        A.log(f"Wrote {a.json_out}")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 300)
    for k, v in sheets.items():
        print(f"\n=== {k}  {v.shape}")
        print(v.head(40).to_string(index=False))
    print("\nMETA", json.dumps(meta, default=float, indent=1))


if __name__ == "__main__":
    main()
