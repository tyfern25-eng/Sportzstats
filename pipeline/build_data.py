#!/usr/bin/env python3
"""
Sportzstats data pipeline.

Downloads free NFL data (nflverse), computes usage and advanced metrics,
builds PPR projections with matchup adjustments, and writes JSON files
that the website reads from /data.

Run:  python pipeline/build_data.py            (writes ./data/*.json)
      python pipeline/build_data.py --backtest  (also scores the model on past weeks)
"""
import json
import math
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- settings
SEASON = int(os.environ.get("SEASON", 2026))
BASE = "https://github.com/nflverse/nflverse-data/releases/download"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.environ.get("CACHE_DIR", os.path.join(ROOT, ".cache"))
OUT = os.path.join(ROOT, "data")

HALF_LIFE = 4.0          # weeks: how fast older games fade for usage
PRIOR_SEASON_W = 0.35    # weight of last season in defense / QB profiles
LAST_FANTASY_WEEK = 17   # most ESPN leagues end their playoffs in week 17
SKILL = ["QB", "RB", "WR", "TE"]

# ESPN standard PPR
SC = dict(pass_yd=0.04, pass_td=4, int=-2, rush_yd=0.1, rec_yd=0.1, td=6, rec=1, fum=-2)

TEAM_NAMES = {
    "ARI": "Cardinals", "ATL": "Falcons", "BAL": "Ravens", "BUF": "Bills", "CAR": "Panthers",
    "CHI": "Bears", "CIN": "Bengals", "CLE": "Browns", "DAL": "Cowboys", "DEN": "Broncos",
    "DET": "Lions", "GB": "Packers", "HOU": "Texans", "IND": "Colts", "JAX": "Jaguars",
    "KC": "Chiefs", "LA": "Rams", "LAC": "Chargers", "LV": "Raiders", "MIA": "Dolphins",
    "MIN": "Vikings", "NE": "Patriots", "NO": "Saints", "NYG": "Giants", "NYJ": "Jets",
    "PHI": "Eagles", "PIT": "Steelers", "SEA": "Seahawks", "SF": "49ers", "TB": "Buccaneers",
    "TEN": "Titans", "WAS": "Commanders",
}
GROUP_WORD = {"RB": "running backs", "WR": "wide receivers", "TE": "tight ends"}


# ---------------------------------------------------------------- download
def fetch(path, refresh=False):
    os.makedirs(CACHE, exist_ok=True)
    local = os.path.join(CACHE, path.replace("/", "__"))
    fresh = os.path.exists(local) and time.time() - os.path.getmtime(local) < 3 * 3600
    if refresh or not fresh:
        url = f"{BASE}/{path}"
        print("downloading", url)
        for attempt in range(3):
            try:
                urllib.request.urlretrieve(url, local)
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    if os.path.exists(local):
                        print("  using cached copy:", e)
                        break
                    raise
                time.sleep(3)
    return pd.read_parquet(local)


def load_raw():
    pbp, ftn = [], []
    for s in (SEASON - 1, SEASON):
        pbp.append(fetch(f"pbp/play_by_play_{s}.parquet"))
        try:
            ftn.append(fetch(f"ftn_charting/ftn_charting_{s}.parquet"))
        except Exception as e:  # noqa: BLE001
            print("no charting for", s, e)
    raw = dict(
        pbp=pd.concat(pbp, ignore_index=True),
        ftn=pd.concat(ftn, ignore_index=True) if ftn else None,
        stats=fetch(f"stats_player/stats_player_week_{SEASON}.parquet"),
        snaps=fetch(f"snap_counts/snap_counts_{SEASON}.parquet"),
        roster=fetch(f"weekly_rosters/roster_weekly_{SEASON}.parquet"),
        games=fetch("schedules/games.parquet"),
        players=fetch("players/players.parquet"),
    )
    try:
        raw["inj"] = fetch(f"injuries/injuries_{SEASON}.parquet")
    except Exception:  # noqa: BLE001
        raw["inj"] = None
    return raw


# ---------------------------------------------------------------- helpers
def rnd(x, n=1):
    if x is None:
        return None
    try:
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
    except Exception:  # noqa: BLE001
        pass
    return round(float(x), n)


def shrink(num, den, prior, k):
    """Bayesian-style shrinkage of a rate num/den toward prior with k pseudo-observations."""
    return (num + k * prior) / (den + k) if den + k > 0 else prior


def ordinal(n):
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def grade(mult):
    for cut, g in [(1.15, "A+"), (1.08, "A"), (1.03, "B+"), (0.98, "B"), (0.94, "C+"),
                   (0.90, "C"), (0.85, "D")]:
        if mult >= cut:
            return g
    return "F"


def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# ---------------------------------------------------------------- play prep
def prep_plays(raw):
    p = raw["pbp"]
    p = p[(p.season_type == "REG") & p.play_type.isin(["pass", "run"])].copy()
    p = p[p.two_point_attempt.fillna(0) == 0]
    ftn = raw["ftn"]
    if ftn is not None:
        f = ftn[["nflverse_game_id", "nflverse_play_id", "n_blitzers", "n_pass_rushers"]].rename(
            columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
        f = f.drop_duplicates(["game_id", "play_id"])
        p = p.merge(f, on=["game_id", "play_id"], how="left")
        nb = p.n_blitzers
        p["blitz"] = np.where(nb.isna() | (nb > 11), np.nan, (nb > 0).astype(float))
    else:
        p["blitz"] = np.nan

    pos = raw["players"].set_index("gsis_id").position.to_dict()
    p["rec_pos"] = p.receiver_player_id.map(pos).replace({"FB": "RB"})
    p["is_target"] = ((p.pass_attempt == 1) & (p.sack.fillna(0) == 0) & p.receiver_player_id.notna()).astype(int)
    p["is_att"] = ((p.pass_attempt == 1) & (p.sack.fillna(0) == 0)).astype(int)
    p["is_carry"] = ((p.rush_attempt == 1) & p.rusher_player_id.notna()).astype(int)
    p["designed_run"] = ((p.rush_attempt == 1) & (p.qb_scramble.fillna(0) == 0)).astype(int)
    p["rec_yds"] = p.receiving_yards.fillna(0)
    p["rush_yds"] = p.rushing_yards.fillna(0)
    p["pass_yds"] = p.passing_yards.fillna(0)
    p["db_yds"] = np.where(p.qb_dropback == 1, p.yards_gained.fillna(0), 0)
    p["pressure"] = ((p.sack.fillna(0) == 1) | (p.qb_hit.fillna(0) == 1)).astype(int)
    p["short"] = ((p.is_att == 1) & (p.air_yards.fillna(99) <= 5)).astype(int)
    for c in ["qb_dropback", "sack", "interception", "pass_touchdown", "rush_touchdown",
              "complete_pass", "fumble_lost", "epa", "air_yards", "yards_after_catch"]:
        p[c] = p[c].fillna(0)
    return p


def add_weights(p, through):
    """Recency weight per play. Current season fades with HALF_LIFE; last season is a weak prior."""
    cur = (p.season == SEASON) & (p.week <= through)
    w = np.where(cur, 0.5 ** ((through - p.week) / HALF_LIFE), 0.0)
    w = np.where(p.season == SEASON - 1, PRIOR_SEASON_W * 0.5 ** ((18 - p.week) / 8), w)
    p = p.assign(w=w)
    return p[p.w > 0]


# ---------------------------------------------------------------- core model
class Model:
    def __init__(self, raw, through):
        self.raw = raw
        self.through = through
        allp = prep_plays(raw)
        self.allp = add_weights(allp, through)
        self.cur = self.allp[self.allp.season == SEASON]
        games = raw["games"]
        self.sched = games[(games.season == SEASON) & (games.game_type == "REG")].copy()
        self._league()
        self._teams()
        self._defenses()
        self._qb_profiles()
        self._players()

    # ---- league baselines
    def _league(self):
        c = self.cur
        t = c[c.is_target == 1]
        self.lg = dict(
            blitz=c[(c.qb_dropback == 1) & c.blitz.notna()].blitz.mean(),
            pressure=c[c.qb_dropback == 1].pressure.mean(),
            ypa=c[c.is_att == 1].pass_yds.sum() / max(1, c.is_att.sum()),
            int_rate=c.interception.sum() / max(1, c.is_att.sum()),
            ypc=c[c.designed_run == 1].rush_yds.mean(),
        )
        self.lg["catch"], self.lg["ypt"] = {}, {}
        for g in ["RB", "WR", "TE"]:
            tg = t[t.rec_pos == g]
            self.lg["catch"][g] = tg.complete_pass.mean()
            self.lg["ypt"][g] = tg.rec_yds.mean()
        ypc = {}
        for pos in ["RB", "QB", "WR"]:
            ids = [k for k, v in self.raw["players"].set_index("gsis_id").position.items() if v == pos]
            r = c[(c.is_carry == 1) & c.rusher_player_id.isin(set(ids))]
            ypc[pos] = r.rush_yds.mean() if len(r) else self.lg["ypc"]
        ypc["TE"] = ypc["WR"]
        self.lg["ypc_pos"] = ypc

        # league target mix when blitzed vs not, by receiver group
        a = self.allp[(self.allp.is_target == 1) & self.allp.blitz.notna() & self.allp.rec_pos.isin(["RB", "WR", "TE"])]
        mix = {}
        for b in (0, 1):
            s = a[a.blitz == b]
            tot = s.w.sum()
            mix[b] = {g: s[s.rec_pos == g].w.sum() / tot for g in ["RB", "WR", "TE"]}
        tot = a.w.sum()
        mix["all"] = {g: a[a.rec_pos == g].w.sum() / tot for g in ["RB", "WR", "TE"]}
        self.lg["mix"] = mix

    # ---- team offense volume
    def _teams(self):
        c = self.cur
        gg = c.groupby(["game_id", "week", "posteam"]).agg(
            plays=("play_id", "count"), targets=("is_target", "sum"), att=("is_att", "sum"),
            carries=("is_carry", "sum"), dropbacks=("qb_dropback", "sum"),
            pass_td=("pass_touchdown", "sum"), rush_td=("rush_touchdown", "sum"),
            rz_targets=("is_target", lambda s: s[c.loc[s.index, "yardline_100"] <= 20].sum()),
            gl_carries=("is_carry", lambda s: s[c.loc[s.index, "yardline_100"] <= 10].sum()),
            air=("air_yards", lambda s: s[c.loc[s.index, "is_target"] == 1].sum()),
            w=("w", "first"),
        ).reset_index()
        # points per game from schedule
        sc = self.sched[self.sched.home_score.notna()]
        pts = pd.concat([
            sc[["game_id", "home_team", "home_score", "away_score"]].rename(columns={"home_team": "team", "home_score": "pf", "away_score": "pa"}),
            sc[["game_id", "away_team", "away_score", "home_score"]].rename(columns={"away_team": "team", "away_score": "pf", "home_score": "pa"}),
        ])
        gg = gg.merge(pts, left_on=["game_id", "posteam"], right_on=["game_id", "team"], how="left")
        self.team_games = gg
        lg_pts = gg.pf.mean()
        lg_plays = gg.plays.mean()
        self.lg["pts"] = lg_pts
        self.lg["plays"] = lg_plays
        self.lg["td_per_pt"] = (gg.pass_td + gg.rush_td).sum() / gg.pf.sum()
        self.lg["pass_td_frac"] = gg.pass_td.sum() / (gg.pass_td + gg.rush_td).sum()

        teams = {}
        for tm, d in gg.groupby("posteam"):
            W = d.w.sum()
            avg = lambda col: (d[col] * d.w).sum() / W  # noqa: E731
            n = len(d)
            pf = shrink((d.pf * d.w).sum(), W, lg_pts, 1.5)
            td = (d.pass_td + d.rush_td).sum()
            teams[tm] = dict(
                games=n, plays=avg("plays"), targets=avg("targets"), att=avg("att"),
                carries=avg("carries"), dropbacks=avg("dropbacks"), pts=pf,
                raw_ppg=d.pf.mean(), pa_g=d.pa.mean(),
                pass_rate=(d.dropbacks * d.w).sum() / (d.plays * d.w).sum(),
                td_per_pt=shrink(td, d.pf.sum(), self.lg["td_per_pt"], 60),
                pass_td_frac=shrink(d.pass_td.sum(), td, self.lg["pass_td_frac"], 8),
            )
        # plays allowed by defense
        da = gg.merge(self.sched[["game_id", "home_team", "away_team"]], on="game_id")
        da["defteam"] = np.where(da.posteam == da.home_team, da.away_team, da.home_team)
        for dt_, d in da.groupby("defteam"):
            if dt_ in teams:
                teams[dt_]["plays_allowed"] = (d.plays * d.w).sum() / d.w.sum()
                teams[dt_]["pa_shrunk"] = shrink((d.pf * d.w).sum(), d.w.sum(), lg_pts, 1.5)
        self.teams = teams

    # ---- defense tendencies, opponent-adjusted
    def _defenses(self):
        a = self.allp
        # offense baselines per team-season (to adjust for who each defense faced)
        t = a[(a.is_target == 1) & a.rec_pos.isin(["RB", "WR", "TE"])]
        off_ypt = t.groupby(["season", "posteam", "rec_pos"]).rec_yds.mean()
        off_catch = t.groupby(["season", "posteam", "rec_pos"]).complete_pass.mean()
        t = t.join(off_ypt.rename("off_ypt"), on=["season", "posteam", "rec_pos"])
        t = t.join(off_catch.rename("off_catch"), on=["season", "posteam", "rec_pos"])
        t["r_ypt"] = t.rec_yds - t.off_ypt
        t["r_catch"] = t.complete_pass - t.off_catch

        r = a[a.designed_run == 1]
        r = r.join(r.groupby(["season", "posteam"]).rush_yds.mean().rename("off_ypc"), on=["season", "posteam"])
        r["r_ypc"] = r.rush_yds - r.off_ypc

        d = a[a.qb_dropback == 1]
        d = d.join(d.groupby(["season", "posteam"]).db_yds.mean().rename("off_ydb"), on=["season", "posteam"])
        d["r_ydb"] = d.db_yds - d.off_ydb
        lg_ydb = d.db_yds.mean()

        # fantasy points allowed by position (current season, raw) for display
        st = self.raw["stats"]
        st = st[(st.season_type == "REG") & (st.week <= self.through)]
        st = st.assign(pos=st.position.replace({"FB": "RB"}))
        fpa = (st[st.pos.isin(SKILL)].groupby(["opponent_team", "pos", "week"]).fantasy_points_ppr.sum()
               .groupby(["opponent_team", "pos"]).mean())

        defs = {}
        for tm in TEAM_NAMES:
            dd = d[d.defteam == tm]
            known = dd[dd.blitz.notna()]
            rec = dict(team=tm, name=TEAM_NAMES[tm])
            rec["blitz_rate"] = shrink((known.blitz * known.w).sum(), known.w.sum(), self.lg["blitz"], 40)
            rec["pressure_rate"] = shrink((dd.pressure * dd.w).sum(), dd.w.sum(), self.lg["pressure"], 60)
            rec["sack_rate"] = shrink((dd.sack * dd.w).sum(), dd.w.sum(), 0.065, 80)
            rec["int_rate"] = shrink((dd.interception * dd.w).sum(), dd.w.sum(), self.lg["int_rate"], 150)
            rec["pass_factor"] = 1 + shrink((dd.r_ydb * dd.w).sum(), dd.w.sum(), 0, 120) / lg_ydb
            rr = r[r.defteam == tm]
            rec["rush_factor"] = 1 + shrink((rr.r_ypc * rr.w).sum(), rr.w.sum(), 0, 100) / self.lg["ypc"]
            tt = t[t.defteam == tm]
            for g in ["RB", "WR", "TE"]:
                tg = tt[tt.rec_pos == g]
                rec[f"ypt_{g}"] = 1 + shrink((tg.r_ypt * tg.w).sum(), tg.w.sum(), 0, 45) / self.lg["ypt"][g]
                rec[f"catch_{g}"] = 1 + shrink((tg.r_catch * tg.w).sum(), tg.w.sum(), 0, 45) / self.lg["catch"][g]
            for pos in SKILL:
                rec[f"fpa_{pos}"] = fpa.get((tm, pos), np.nan)
            cw = d[(d.defteam == tm) & (d.season == SEASON)]
            rec["dropbacks_faced"] = int(len(cw))
            rec["blitz_raw"] = cw[cw.blitz.notna()].blitz.mean() if len(cw) else None
            defs[tm] = rec
        # ranks (1 = highest)
        df = pd.DataFrame(defs).T
        for col in ["blitz_rate", "pressure_rate"]:
            df[col + "_rank"] = df[col].astype(float).rank(ascending=False, method="min").astype(int)
        for col in ["pass_factor", "rush_factor", "ypt_RB", "ypt_WR", "ypt_TE"] + [f"fpa_{p}" for p in SKILL]:
            # rank 1 = easiest matchup (allows the most)
            df[col + "_rank"] = df[col].astype(float).rank(ascending=False, method="min")
        self.defs = {k: v for k, v in df.to_dict("index").items()}

    # ---- QB behaviour when blitzed
    def _qb_profiles(self):
        a = self.allp[(self.allp.qb_dropback == 1) & self.allp.passer_player_id.notna()]
        prof = {}
        mix = self.lg["mix"]
        for qb, d in a.groupby("passer_player_id"):
            if d[d.season == SEASON].is_att.sum() < 15:
                continue
            k = d[d.blitz.notna()]
            out = {"n_db": int(len(d[d.season == SEASON]))}
            ypa_all = (d.db_yds * d.w).sum() / d.w.sum()
            for b, lab in ((1, "b"), (0, "nb")):
                s = k[k.blitz == b]
                W = s.w.sum()
                lg_ratio = 1.0
                if b == 1:
                    lg_ratio = self._lg_ydb_ratio(1)
                else:
                    lg_ratio = self._lg_ydb_ratio(0)
                out[f"ydb_{lab}"] = shrink((s.db_yds * s.w).sum(), W, ypa_all * lg_ratio, 40)
                out[f"sack_{lab}"] = shrink((s.sack * s.w).sum(), W, 0.065 * (1.35 if b else 0.9), 40)
                out[f"short_{lab}"] = shrink((s.short * s.w).sum(), (s.is_att * s.w).sum(), 0.42, 30)
                out[f"epa_{lab}"] = shrink((s.epa * s.w).sum(), W, 0.0, 40)
                tg = s[(s.is_target == 1) & s.rec_pos.isin(["RB", "WR", "TE"])]
                TW = tg.w.sum()
                for g in ["RB", "WR", "TE"]:
                    out[f"mix_{lab}_{g}"] = shrink(tg[tg.rec_pos == g].w.sum(), TW, mix[b][g], 30)
                out[f"n_{lab}"] = int(len(s[s.season == SEASON]))
            tg = d[(d.is_target == 1) & d.rec_pos.isin(["RB", "WR", "TE"])]
            for g in ["RB", "WR", "TE"]:
                out[f"mix_all_{g}"] = shrink(tg[tg.rec_pos == g].w.sum(), tg.w.sum(), mix["all"][g], 30)
            out["blitz_faced"] = shrink((k.blitz * k.w).sum(), k.w.sum(), self.lg["blitz"], 30)
            prof[qb] = out
        self.qb_prof = prof
        # each team's current starting QB = most weighted attempts this season
        c = self.cur[self.cur.is_att == 1]
        starters = {}
        for tm, d in c.groupby("posteam"):
            last = d[d.week >= d.week.max() - 1]
            s = (last if len(last) else d).groupby("passer_player_id").w.sum()
            starters[tm] = s.idxmax()
        self.starter = starters

    def _lg_ydb_ratio(self, b):
        a = self.allp[(self.allp.qb_dropback == 1) & self.allp.blitz.notna()]
        return a[a.blitz == b].db_yds.mean() / a.db_yds.mean()

    # ---- player usage and efficiency
    def _players(self):
        raw, c = self.raw, self.cur
        roster = raw["roster"]
        roster = roster[roster.week <= max(self.through + 1, roster.week.min())]
        latest = roster.sort_values("week").groupby("gsis_id").tail(1).set_index("gsis_id")
        players_tbl = raw["players"].set_index("gsis_id")

        st = raw["stats"]
        st = st[(st.season_type == "REG") & (st.week <= self.through)].copy()
        st["pos"] = st.position.replace({"FB": "RB"})
        self.stats = st

        # player-game level from play-by-play
        tg = self.team_games.set_index(["game_id", "posteam"])
        t = c[c.is_target == 1]
        rec = t.groupby(["game_id", "posteam", "receiver_player_id"]).agg(
            tgt=("is_target", "sum"), rec=("complete_pass", "sum"), ryds=("rec_yds", "sum"),
            air=("air_yards", "sum"), yac=("yards_after_catch", "sum"),
            rz=("yardline_100", lambda s: (s <= 20).sum()),
            tgt_b=("blitz", lambda s: (s == 1).sum()), tgt_nb=("blitz", lambda s: (s == 0).sum()),
            epa=("epa", "sum"), w=("w", "first"),
        )
        rec.index.names = ["game_id", "posteam", "pid"]
        ru = c[c.is_carry == 1].groupby(["game_id", "posteam", "rusher_player_id"]).agg(
            car=("is_carry", "sum"), ruyds=("rush_yds", "sum"),
            gl=("yardline_100", lambda s: (s <= 10).sum()), w=("w", "first"),
        )
        ru.index.names = ["game_id", "posteam", "pid"]
        pa = c[c.qb_dropback == 1].groupby(["game_id", "posteam", "passer_player_id"]).agg(
            att=("is_att", "sum"), pyds=("pass_yds", "sum"), ptd=("pass_touchdown", "sum"),
            ints=("interception", "sum"), w=("w", "first"),
        )
        pa.index.names = ["game_id", "posteam", "pid"]

        # team blitz-split target totals per game
        tb = t.groupby(["game_id", "posteam"]).agg(
            t_b=("blitz", lambda s: (s == 1).sum()), t_nb=("blitz", lambda s: (s == 0).sum()))

        # which games each player was active: any stat row, or any pbp involvement
        active = st[["player_id", "week", "team"]].rename(columns={"player_id": "pid", "team": "posteam"})
        gid = self.sched.set_index(["week", "home_team"]).game_id.to_dict()
        gid.update(self.sched.set_index(["week", "away_team"]).game_id.to_dict())
        active["game_id"] = [gid.get((w, tm)) for w, tm in zip(active.week, active.posteam)]
        involved = pd.concat([rec.reset_index()[["game_id", "posteam", "pid"]],
                              ru.reset_index()[["game_id", "posteam", "pid"]],
                              pa.reset_index()[["game_id", "posteam", "pid"]]])
        involved = involved.merge(self.team_games[["game_id", "posteam", "week"]], on=["game_id", "posteam"])
        active = pd.concat([active[["pid", "game_id", "posteam", "week"]], involved]).dropna().drop_duplicates(["pid", "game_id"])

        g = active.merge(self.team_games[["game_id", "posteam", "targets", "att", "carries", "rz_targets",
                                          "gl_carries", "air", "w"]], on=["game_id", "posteam"], how="left")
        g = g.merge(tb, left_on=["game_id", "posteam"], right_index=True, how="left")
        g = g.merge(rec.drop(columns="w").reset_index(), on=["game_id", "posteam", "pid"], how="left")
        g = g.merge(ru.drop(columns="w").reset_index(), on=["game_id", "posteam", "pid"], how="left")
        g = g.merge(pa.drop(columns="w").reset_index(), on=["game_id", "posteam", "pid"], how="left")
        g = g.fillna({k: 0 for k in ["tgt", "rec", "ryds", "air", "yac", "rz", "tgt_b", "tgt_nb", "epa", "car",
                                      "ruyds", "gl", "att_y", "pyds", "ptd", "ints", "t_b", "t_nb"]})
        g = g.rename(columns={"att_x": "team_att", "att_y": "patt", "air_x": "team_air", "air_y": "air"})
        g = g.fillna(0)

        # snaps (matched by name + team + week)
        sn = raw["snaps"]
        sn = sn[sn.game_type == "REG"]
        snap_key = {(n.lower().replace(".", "").replace("'", ""), tm, wk): pct
                    for n, tm, wk, pct in zip(sn.player, sn.team, sn.week, sn.offense_pct)}

        inj = raw.get("inj")
        inj_latest = {}
        if inj is not None and len(inj):
            ij = inj[inj.season_type == "REG"].sort_values("week")
            for r_ in ij.itertuples():
                inj_latest[r_.gsis_id] = dict(week=int(r_.week), status=r_.report_status if isinstance(r_.report_status, str) else None,
                                              injury=r_.report_primary_injury if isinstance(r_.report_primary_injury, str) else None,
                                              practice=r_.practice_status if isinstance(r_.practice_status, str) else None)

        team_last_week = self.team_games.groupby("posteam").week.max().to_dict()
        team_weeks = self.team_games.groupby("posteam").week.apply(sorted).to_dict()

        P = {}
        for pid, d in g.groupby("pid"):
            if pid in latest.index:
                pos = latest.loc[pid, "position"]
                team = latest.loc[pid, "team"]
                name = latest.loc[pid, "full_name"]
                head = latest.loc[pid, "headshot_url"]
                status = latest.loc[pid, "status"]
            elif pid in players_tbl.index:
                pos = players_tbl.loc[pid, "position"]
                team = d.sort_values("week").posteam.iloc[-1]
                name = players_tbl.loc[pid, "display_name"]
                head = players_tbl.loc[pid, "headshot"]
                status = None
            else:
                continue
            pos = "RB" if pos == "FB" else pos
            if pos not in SKILL:
                continue
            d = d.sort_values("week")
            W = d.w
            sw = lambda col: float((d[col] * W).sum())  # noqa: E731
            tgt, ttgt = sw("tgt"), sw("targets")
            if tgt + sw("car") + sw("patt") < 1:
                continue
            grp = pos if pos in ("RB", "WR", "TE") else "RB"
            pr = dict(id=pid, name=name, pos=pos, team=team, headshot=head if isinstance(head, str) else None,
                      roster_status=status if isinstance(status, str) else None, games=int(len(d)),
                      weeks=[int(x) for x in d.week])
            pr["tgt_share"] = tgt / ttgt if ttgt else 0
            pr["air_share"] = sw("air") / sw("team_air") if sw("team_air") else 0
            pr["rz_share"] = sw("rz") / sw("rz_targets") if sw("rz_targets") else pr["tgt_share"]
            pr["carry_share"] = sw("car") / sw("carries") if sw("carries") else 0
            pr["gl_share"] = shrink(sw("gl"), sw("gl_carries"), pr["carry_share"], 3)
            pr["att_share"] = sw("patt") / sw("team_att") if sw("team_att") else 0
            pr["catch"] = shrink(sw("rec"), tgt, self.lg["catch"].get(grp, 0.65), 25)
            pr["ypt"] = shrink(sw("ryds"), tgt, self.lg["ypt"].get(grp, 7.5), 35)
            pr["ypc"] = shrink(sw("ruyds"), sw("car"), self.lg["ypc_pos"].get(pos, self.lg["ypc"]), 45)
            pr["ypa"] = shrink(sw("pyds"), sw("patt"), self.lg["ypa"] * 0.97, 90)
            pr["int_rate"] = shrink(sw("ints"), sw("patt"), self.lg["int_rate"], 250)
            pr["adot"] = sw("air") / tgt if tgt else None
            pr["yac_per_rec"] = sw("yac") / sw("rec") if sw("rec") else None
            pr["wopr"] = 1.5 * pr["tgt_share"] + 0.7 * pr["air_share"]
            pr["epa_per_tgt"] = sw("epa") / tgt if tgt else None
            # blitz split target share (raw, unweighted counts too)
            pr["_tgt_b"], pr["_tgt_nb"] = sw("tgt_b"), sw("tgt_nb")
            pr["_team_b"], pr["_team_nb"] = sw("t_b"), sw("t_nb")
            pr["raw_tgt_b"] = int(d.tgt_b.sum())
            pr["raw_tgt_nb"] = int(d.tgt_nb.sum())
            pr["raw_share_b"] = d.tgt_b.sum() / d.t_b.sum() if d.t_b.sum() else None
            pr["raw_share_nb"] = d.tgt_nb.sum() / d.t_nb.sum() if d.t_nb.sum() else None
            # trend: last 3 games vs season (unweighted)
            l3 = d.tail(3)
            pr["tgt_share_l3"] = l3.tgt.sum() / l3.targets.sum() if l3.targets.sum() else 0
            pr["carry_share_l3"] = l3.car.sum() / l3.carries.sum() if l3.carries.sum() else 0
            pr["tgt_share_season"] = d.tgt.sum() / d.targets.sum() if d.targets.sum() else 0
            pr["carry_share_season"] = d.car.sum() / d.carries.sum() if d.carries.sum() else 0
            # snaps
            key_name = str(name).lower().replace(".", "").replace("'", "")
            snaps = [snap_key.get((key_name, tm, int(wk))) for tm, wk in zip(d.posteam, d.week)]
            snaps = [s for s in snaps if s is not None]
            pr["snap_pct"] = float(np.mean(snaps)) if snaps else None
            pr["snap_pct_l3"] = float(np.mean(snaps[-3:])) if snaps else None
            # availability: missed the team's most recent games?
            tw = team_weeks.get(team, [])
            played = set(pr["weeks"])
            recent = [w_ for w_ in tw if w_ <= self.through][-2:]
            pr["missed_recent"] = [w_ for w_ in recent if w_ not in played]
            pr["last_played"] = max(played) if played else None
            pr["team_changed"] = bool(len(d) and d.posteam.iloc[-1] != team)
            pr["injury"] = inj_latest.get(pid)
            P[pid] = pr
        self.players = P

    # ---- projections
    def matchup_context(self, team, week):
        row = self.sched[(self.sched.week == week) & ((self.sched.home_team == team) | (self.sched.away_team == team))]
        if not len(row):
            return None
        r = row.iloc[0]
        home = r.home_team == team
        opp = r.away_team if home else r.home_team
        T = self.teams.get(team, {})
        O = self.teams.get(opp, {})
        team_pts = T.get("pts", self.lg["pts"])
        if pd.notna(r.spread_line) and pd.notna(r.total_line):
            spread_fav = r.spread_line if home else -r.spread_line
            implied = (r.total_line + spread_fav) / 2
            vegas = True
        else:
            opp_pa = O.get("pa_shrunk", self.lg["pts"])
            implied = team_pts + 0.5 * (opp_pa - self.lg["pts"]) + (0.9 if home else -0.9)
            opp_pts = O.get("pts", self.lg["pts"]) + 0.5 * (T.get("pa_shrunk", self.lg["pts"]) - self.lg["pts"]) + (-0.9 if home else 0.9)
            spread_fav = implied - opp_pts
            vegas = False
        played = pd.notna(r.home_score)
        return dict(opp=opp, home=bool(home), implied=implied, spread_fav=spread_fav, vegas=vegas,
                    game_id=r.game_id, played=bool(played), gameday=r.gameday, gametime=r.gametime)

    def team_volume(self, team, ctx, neutral=False):
        T = self.teams[team]
        if neutral:
            vol, pm, rm, implied = 1.0, 1.0, 1.0, T["pts"]
        else:
            O = self.teams.get(ctx["opp"], {})
            pa = O.get("plays_allowed", self.lg["plays"])
            vol = 1 + 0.4 * (pa / self.lg["plays"] - 1)
            s = ctx["spread_fav"]
            pm = min(1.08, max(0.92, 1 - 0.006 * s))
            rm = min(1.12, max(0.90, 1 + 0.009 * s))
            implied = ctx["implied"]
        tds = implied * T["td_per_pt"]
        return dict(targets=T["targets"] * vol * pm, att=T["att"] * vol * pm, carries=T["carries"] * vol * rm,
                    pass_td=tds * T["pass_td_frac"], rush_td=tds * (1 - T["pass_td_frac"]), implied=implied)

    def receiver_share(self, pr, blitz_rate):
        """Target share given the opponent's blitz rate, using the QB's blitz tendencies."""
        team = pr["team"]
        qb = self.qb_prof.get(self.starter.get(team))
        grp = pr["pos"] if pr["pos"] in ("RB", "WR", "TE") else None
        s = pr["tgt_share"]
        if grp is None or s <= 0:
            return s, 0.0
        if qb:
            m_b = qb[f"mix_b_{grp}"] / qb[f"mix_all_{grp}"]
            m_nb = qb[f"mix_nb_{grp}"] / qb[f"mix_all_{grp}"]
        else:
            mix = self.lg["mix"]
            m_b, m_nb = mix[1][grp] / mix["all"][grp], mix[0][grp] / mix["all"][grp]
        sb = shrink(pr["_tgt_b"], pr["_team_b"], s * m_b, 20)
        snb = shrink(pr["_tgt_nb"], pr["_team_nb"], s * m_nb, 20)
        share = blitz_rate * sb + (1 - blitz_rate) * snb
        base = self.lg["blitz"] * sb + (1 - self.lg["blitz"]) * snb
        # keep overall level anchored to observed share
        adj = share / base if base else 1
        return s * adj, adj - 1

    def project(self, pr, week, neutral=False):
        team = pr["team"]
        ctx = self.matchup_context(team, week)
        if ctx is None:
            return dict(week=week, bye=True, proj=0.0)
        if team not in self.teams:
            return None
        D = self.defs.get(ctx["opp"]) if not neutral else None
        vol = self.team_volume(team, ctx, neutral)
        pos = pr["pos"]
        grp = pos if pos in ("RB", "WR", "TE") else None
        br = D["blitz_rate"] if D else self.lg["blitz"]
        notes = []
        comp = {}

        # receiving
        share, blitz_eff = self.receiver_share(pr, br) if grp else (0, 0)
        if pos == "QB":
            share = pr["tgt_share"]
        tg = vol["targets"] * share
        catch = pr["catch"] * (D[f"catch_{grp}"] if D and grp else 1)
        ypt = pr["ypt"] * (D[f"ypt_{grp}"] if D and grp else 1)
        rec = tg * catch
        ryds = tg * ypt
        rtd_share = 0.55 * pr["rz_share"] + 0.45 * pr["tgt_share"]
        rtd = vol["pass_td"] * rtd_share * (1 + blitz_eff)
        # rushing
        car = vol["carries"] * pr["carry_share"]
        ruyds = car * pr["ypc"] * (D["rush_factor"] if D else 1)
        rutd = vol["rush_td"] * (0.6 * pr["gl_share"] + 0.4 * pr["carry_share"])
        # passing
        pyds = ptd = ints = att = 0.0
        qb_blitz_mult = 1.0
        if pos == "QB" and pr["att_share"] > 0:
            att = vol["att"] * pr["att_share"]
            q = self.qb_prof.get(pr["id"])
            if q and not neutral:
                at = br * q["ydb_b"] + (1 - br) * q["ydb_nb"]
                base = self.lg["blitz"] * q["ydb_b"] + (1 - self.lg["blitz"]) * q["ydb_nb"]
                qb_blitz_mult = at / base if base else 1
            pyds = att * pr["ypa"] * (D["pass_factor"] if D else 1) * qb_blitz_mult
            ptd = vol["pass_td"] * pr["att_share"] * (D["pass_factor"] if D else 1) ** 0.5
            ints = att * pr["int_rate"] * ((D["int_rate"] / self.lg["int_rate"]) ** 0.5 if D else 1)
        fum = (rec + car) * 0.0035

        pts = (pyds * SC["pass_yd"] + ptd * SC["pass_td"] + ints * SC["int"] + ruyds * SC["rush_yd"]
               + ryds * SC["rec_yd"] + (rtd + rutd) * SC["td"] + rec * SC["rec"] + fum * SC["fum"])
        comp = dict(targets=tg, rec=rec, rec_yds=ryds, rec_td=rtd, carries=car, rush_yds=ruyds, rush_td=rutd,
                    pass_att=att, pass_yds=pyds, pass_td=ptd, ints=ints)

        if neutral:
            return dict(week=week, proj=pts)

        # ---- explanations
        opp = ctx["opp"]
        oname = TEAM_NAMES.get(opp, opp)
        if grp and abs(blitz_eff) >= 0.04:
            qbid = self.starter.get(team)
            qbname = self.players.get(qbid, {}).get("name", "the QB") if qbid else "the QB"
            q = self.qb_prof.get(qbid, {})
            lift = (q.get(f"mix_b_{grp}", 0) / q.get(f"mix_nb_{grp}", 1) - 1) if q else 0
            word = "up" if blitz_eff > 0 else "down"
            heavy = br >= self.lg["blitz"]
            lead = (f"{oname} blitz on {br*100:.0f}% of dropbacks ({ordinal(int(D['blitz_rate_rank']))} in NFL)."
                    if heavy else f"{oname} rarely blitz ({br*100:.0f}% of dropbacks, {ordinal(int(D['blitz_rate_rank']))} in NFL).")
            # group tendency agrees with this player's net effect -> explain via QB; otherwise via the player's own split
            agrees = (lift > 0) == (blitz_eff > 0) if heavy else (lift < 0) == (blitz_eff > 0)
            if agrees and abs(lift) >= 0.05:
                why = (f"When blitzed, {qbname} throws to {GROUP_WORD[grp]} {abs(lift)*100:.0f}% "
                       f"{'more' if lift > 0 else 'less'} often")
            elif pr["raw_share_b"] is not None and pr["raw_share_nb"] is not None:
                why = (f"{pr['name']} gets {pr['raw_share_b']*100:.0f}% of targets when his QB is blitzed vs "
                       f"{pr['raw_share_nb']*100:.0f}% when not")
            else:
                why = f"His usage shifts when his QB is blitzed"
            notes.append(dict(kind="blitz", good=blitz_eff > 0,
                              text=f"{lead} {why}, so his projected target share goes {word} {abs(blitz_eff)*100:.0f}%."))
        if grp and D:
            f = D[f"ypt_{grp}"]
            if abs(f - 1) >= 0.06:
                notes.append(dict(kind="coverage", good=f > 1,
                                  text=f"{oname} allow {abs(f-1)*100:.0f}% {'more' if f > 1 else 'fewer'} yards per target to "
                                       f"{GROUP_WORD[grp]} than those offenses normally get."))
        if pr["carry_share"] >= 0.15 and D and abs(D["rush_factor"] - 1) >= 0.05:
            f = D["rush_factor"]
            notes.append(dict(kind="run", good=f > 1,
                              text=f"{oname} run defense allows {abs(f-1)*100:.0f}% {'more' if f > 1 else 'fewer'} yards per carry than average (adjusted for opponents)."))
        if pos == "QB" and D:
            if abs(qb_blitz_mult - 1) >= 0.03:
                notes.append(dict(kind="blitz", good=qb_blitz_mult > 1,
                                  text=f"{oname} blitz {br*100:.0f}% of the time; {pr['name']} averages "
                                       f"{self.qb_prof[pr['id']]['ydb_b']:.1f} yds/dropback vs blitz vs "
                                       f"{self.qb_prof[pr['id']]['ydb_nb']:.1f} without."))
            if abs(D["pass_factor"] - 1) >= 0.05:
                notes.append(dict(kind="coverage", good=D["pass_factor"] > 1,
                                  text=f"{oname} pass defense allows {abs(D['pass_factor']-1)*100:.0f}% {'more' if D['pass_factor'] > 1 else 'fewer'} yards per dropback than average."))
            if D["pressure_rate_rank"] <= 6:
                notes.append(dict(kind="pressure", good=False,
                                  text=f"{oname} pressure the QB on {D['pressure_rate']*100:.0f}% of dropbacks ({ordinal(int(D['pressure_rate_rank']))})."))
        diff = vol["implied"] - self.teams[team]["pts"]
        if abs(diff) >= 2.5:
            notes.append(dict(kind="vegas", good=diff > 0,
                              text=f"{'Vegas' if ctx['vegas'] else 'Model'} expects {vol['implied']:.1f} points for {team} "
                                   f"(they average {self.teams[team]['pts']:.1f})."))
        if pr["team_changed"]:
            notes.append(dict(kind="info", good=None, text="Recently changed teams; usage numbers are from his old team."))

        return dict(week=week, opp=opp, home=ctx["home"], gameday=ctx["gameday"], played=ctx["played"],
                    proj=pts, comp={k: rnd(v, 2) for k, v in comp.items()}, implied=rnd(vol["implied"]),
                    notes=notes, vegas=ctx["vegas"], blitz_eff=blitz_eff)


# ---------------------------------------------------------------- K / DST
def kicker_dst(model, weeks):
    st = model.stats
    out = []
    # kickers
    k = st[st.pos == "K"].copy()
    if len(k):
        k["kpts"] = (3 * (k.fg_made_0_19.fillna(0) + k.fg_made_20_29.fillna(0) + k.fg_made_30_39.fillna(0))
                     + 4 * k.fg_made_40_49.fillna(0) + 5 * (k.fg_made_50_59.fillna(0) + k.fg_made_60_.fillna(0))
                     + k.pat_made.fillna(0) - k.fg_missed.fillna(0) - k.pat_missed.fillna(0))
        lg_k = k.groupby(["team", "week"]).kpts.sum().mean()
        for (pid, name, team), d in k.groupby(["player_id", "player_display_name", "team"]):
            if d.week.max() < model.through - 2:
                continue
            ppg = shrink(d.kpts.sum(), len(d), lg_k, 3)
            T = model.teams.get(team)
            if not T:
                continue
            projs = []
            for w in weeks:
                ctx = model.matchup_context(team, w)
                if ctx is None:
                    projs.append(dict(week=w, bye=True, proj=0.0))
                    continue
                proj = ppg * (ctx["implied"] / T["pts"]) ** 0.8
                projs.append(dict(week=w, opp=ctx["opp"], home=ctx["home"], gameday=ctx["gameday"], played=ctx["played"],
                                  proj=proj, implied=rnd(ctx["implied"]), notes=[], grade=grade((ctx["implied"] / T["pts"]) ** 0.8)))
            out.append(dict(id=pid, name=name, pos="K", team=team, games=int(len(d)), ppg=rnd(d.kpts.mean()),
                            log=[dict(week=int(r.week), opp=r.opponent_team, ppr=rnd(r.kpts)) for r in d.itertuples()],
                            proj=projs))
    # defenses
    c = model.cur
    pts = model.team_games[["game_id", "posteam", "week", "pa"]]
    lg_pa = model.lg["pts"]
    for tm in TEAM_NAMES:
        dd = c[c.defteam == tm]
        D = model.defs[tm]
        g = dd.groupby("week").agg(sacks=("sack", "sum"), ints=("interception", "sum"), fum=("fumble_lost", "sum"))
        pa = pts[pts.posteam == tm].set_index("week").pa

        def pa_pts(x):
            return 5 if x == 0 else 4 if x <= 6 else 3 if x <= 13 else 1 if x <= 17 else 0 if x <= 27 else -1 if x <= 34 else -3 if x <= 45 else -5

        log = []
        for w, r in g.iterrows():
            p_ = pa.get(w, np.nan)
            sc_ = r.sacks + 2 * r.ints + 2 * r.fum + (pa_pts(p_) if pd.notna(p_) else 0)
            log.append(dict(week=int(w), ppr=rnd(sc_), sacks=int(r.sacks), ints=int(r.ints), fum=int(r.fum),
                            pa=None if pd.isna(p_) else int(p_)))
        projs = []
        for w in weeks:
            ctx = model.matchup_context(tm, w)
            if ctx is None:
                projs.append(dict(week=w, bye=True, proj=0.0))
                continue
            opp = ctx["opp"]
            O = model.teams.get(opp, {})
            odb = O.get("dropbacks", 36)
            opp_implied = ctx["implied"] - ctx["spread_fav"]
            sacks = odb * D["sack_rate"]
            to = odb * D["int_rate"] + O.get("plays", 62) * 0.006
            dtd = 0.13 * to / 1.3
            # expected points-allowed tier score, PA ~ Normal(opp_implied, 9.5)
            ev = 0.0
            edges = [(-99, 0.5, 5), (0.5, 6.5, 4), (6.5, 13.5, 3), (13.5, 17.5, 1), (17.5, 27.5, 0),
                     (27.5, 34.5, -1), (34.5, 45.5, -3), (45.5, 99, -5)]
            for lo, hi, v in edges:
                ev += v * (norm_cdf((hi - opp_implied) / 9.5) - norm_cdf((lo - opp_implied) / 9.5))
            proj = sacks + 2 * to + 6 * dtd + ev
            projs.append(dict(week=w, opp=opp, home=ctx["home"], gameday=ctx["gameday"], played=ctx["played"],
                              proj=proj, implied=rnd(opp_implied), notes=[
                                  dict(kind="vegas", good=opp_implied < lg_pa,
                                       text=f"{TEAM_NAMES[opp]} projected for {opp_implied:.1f} points.")]))
        avg = np.mean([x["ppr"] for x in log]) if log else 0
        for p_ in projs:
            if not p_.get("bye"):
                p_["grade"] = grade(p_["proj"] / max(avg, 3)) if avg else "C"
        out.append(dict(id=f"DST-{tm}", name=f"{TEAM_NAMES[tm]} D/ST", pos="DST", team=tm, games=len(log),
                        ppg=rnd(avg), log=log, proj=projs))
    return out


# ---------------------------------------------------------------- weeks
def current_week(sched):
    reg = sched[sched.game_type == "REG"]
    for w, d in reg.groupby("week"):
        if d.home_score.notna().mean() < 0.5:
            return int(w)
    return int(reg.week.max())


def last_complete_week(pbp_season):
    return int(pbp_season.week.max())


# ---------------------------------------------------------------- assemble
def build_outputs(model, target_week):
    raw = model.raw
    weeks = list(range(target_week, LAST_FANTASY_WEEK + 1))
    st = model.stats
    logs = {pid: d.sort_values("week") for pid, d in st.groupby("player_id")}

    players_out = []
    for pid, pr in model.players.items():
        if pr["team"] not in model.teams:
            continue
        d = logs.get(pid)
        log = []
        if d is not None:
            for r in d.itertuples():
                log.append(dict(week=int(r.week), opp=r.opponent_team, ppr=rnd(r.fantasy_points_ppr),
                                tgt=int(r.targets or 0), rec=int(r.receptions or 0), ryd=int(r.receiving_yards or 0),
                                car=int(r.carries or 0), ruyd=int(r.rushing_yards or 0),
                                td=int((r.receiving_tds or 0) + (r.rushing_tds or 0)),
                                patt=int(r.attempts or 0), pyd=int(r.passing_yards or 0), ptd=int(r.passing_tds or 0),
                                ints=int(r.passing_interceptions or 0)))
        ppr_vals = [x["ppr"] for x in log if x["ppr"] is not None]
        ppg = float(np.mean(ppr_vals)) if ppr_vals else 0
        sd = float(np.std(ppr_vals)) if len(ppr_vals) > 1 else 0
        cv_prior = {"QB": 0.38, "RB": 0.52, "WR": 0.58, "TE": 0.62}[pr["pos"]]

        projs = []
        for w in weeks:
            p = model.project(pr, w)
            if p is None:
                continue
            if p.get("bye"):
                projs.append(dict(week=w, bye=True, proj=0.0))
                continue
            n = model.project(pr, w, neutral=True)["proj"]
            mult = p["proj"] / n if n > 0.5 else 1
            proj = p["proj"]
            flags = []
            inj = pr.get("injury")
            out_now = False
            if pr["missed_recent"] and len(pr["missed_recent"]) >= 2:
                flags.append(f"Has not played since week {pr['last_played']}")
                out_now = True
            elif inj and inj.get("status") in ("Out", "Doubtful") and inj["week"] == w:
                flags.append(f"{inj['status']} ({inj.get('injury') or 'injury'})")
                out_now = inj["status"] == "Out"
            if pr.get("roster_status") in ("RES", "INA", "CUT", "RET", "SUS"):
                flags.append(f"Roster status: {pr['roster_status']}")
                out_now = True
            if out_now:
                proj = 0.0
            var_sd = math.sqrt((len(ppr_vals) * sd ** 2 + 4 * (cv_prior * max(proj, 1)) ** 2) / (len(ppr_vals) + 4)) if proj else 0
            var_sd = var_sd * (proj / ppg) ** 0.5 if ppg > 1 and proj > 0 else var_sd
            projs.append(dict(week=w, opp=p["opp"], home=p["home"], gameday=p["gameday"], played=p["played"],
                              proj=rnd(proj), floor=rnd(max(0, proj - 0.84 * var_sd)), ceil=rnd(proj + 0.84 * var_sd),
                              grade=grade(mult), mult=rnd(mult, 3), implied=p["implied"], vegas=p["vegas"],
                              comp=p["comp"], notes=p["notes"], flags=flags))

        # blitz split display
        q = model.qb_prof.get(pid) if pr["pos"] == "QB" else None
        out = dict(
            id=pid, name=pr["name"], pos=pr["pos"], team=pr["team"], headshot=pr["headshot"], games=pr["games"],
            ppg=rnd(ppg), ppg_l3=rnd(np.mean(ppr_vals[-3:])) if ppr_vals else None, total=rnd(sum(ppr_vals)),
            injury=pr["injury"], roster_status=pr["roster_status"], missed_recent=pr["missed_recent"],
            adv=dict(
                snap=rnd(pr["snap_pct"] * 100 if pr["snap_pct"] is not None and pr["snap_pct"] <= 1 else pr["snap_pct"]),
                snap_l3=rnd(pr["snap_pct_l3"] * 100 if pr["snap_pct_l3"] is not None and pr["snap_pct_l3"] <= 1 else pr["snap_pct_l3"]),
                tgt_share=rnd(pr["tgt_share_season"] * 100), tgt_share_l3=rnd(pr["tgt_share_l3"] * 100),
                air_share=rnd(pr["air_share"] * 100), wopr=rnd(pr["wopr"], 2), adot=rnd(pr["adot"]),
                rz_share=rnd(pr["rz_share"] * 100), carry_share=rnd(pr["carry_share_season"] * 100),
                carry_share_l3=rnd(pr["carry_share_l3"] * 100), gl_share=rnd(pr["gl_share"] * 100),
                catch=rnd(pr["catch"] * 100), ypt=rnd(pr["ypt"]), ypc=rnd(pr["ypc"]),
                yac_rec=rnd(pr["yac_per_rec"]), epa_tgt=rnd(pr["epa_per_tgt"], 2),
                ypa=rnd(pr["ypa"]) if pr["pos"] == "QB" else None,
                tgt_share_blitz=rnd(pr["raw_share_b"] * 100) if pr["raw_share_b"] is not None else None,
                tgt_share_noblitz=rnd(pr["raw_share_nb"] * 100) if pr["raw_share_nb"] is not None else None,
                tgt_blitz_n=pr["raw_tgt_b"], tgt_noblitz_n=pr["raw_tgt_nb"],
            ),
            log=log, proj=projs,
        )
        if q:
            out["qb_blitz"] = {k: rnd(v * 100 if k.startswith(("mix", "sack", "short", "blitz_faced")) else v, 2)
                               for k, v in q.items()}
        out["ros"] = ros_summary(out)
        players_out.append(out)

    for kd in kicker_dst(model, weeks):
        for p_ in kd["proj"]:
            p_["proj"] = rnd(p_["proj"])
            if not p_.get("bye"):
                p_["floor"] = rnd(max(0, p_["proj"] * 0.45))
                p_["ceil"] = rnd(p_["proj"] * 1.6)
                p_["flags"] = []
        kd["ros"] = ros_summary(kd)
        players_out.append(kd)

    players_out.sort(key=lambda x: -(x["proj"][0]["proj"] if x["proj"] else 0))
    return players_out


def ros_summary(p):
    pj = [x for x in p["proj"] if not x.get("played")]
    games = [x for x in pj if not x.get("bye")]
    total = sum(x["proj"] or 0 for x in games)
    po = [x for x in games if x["week"] >= 15]
    mults = [x.get("mult") for x in games if x.get("mult")]
    po_m = [x.get("mult") for x in po if x.get("mult")]
    adv = p.get("adv", {})
    trend = None
    if adv and p["pos"] != "QB":
        if p["pos"] in ("WR", "TE") or (p["pos"] == "RB" and (adv.get("tgt_share") or 0) > (adv.get("carry_share") or 0)):
            a, b = adv.get("tgt_share_l3"), adv.get("tgt_share")
        else:
            a, b = adv.get("carry_share_l3"), adv.get("carry_share")
        if a is not None and b:
            trend = rnd((a - b), 1)
    ppg = total / len(games) if games else 0
    sched = float(np.mean(mults)) if mults else None
    po_sched = float(np.mean(po_m)) if po_m else None

    # plain-English outlook
    bits = []
    if ppg:
        bits.append(f"Projects for {ppg:.1f} PPR points per game over his remaining {len(games)} games.")
    if sched:
        if sched >= 1.04:
            bits.append("Remaining schedule is favorable.")
        elif sched <= 0.96:
            bits.append("Remaining schedule is tough.")
        else:
            bits.append("Remaining schedule is about average.")
    if po_sched:
        if po_sched >= 1.04:
            bits.append("Fantasy playoff matchups (weeks 15-17) look good.")
        elif po_sched <= 0.96:
            bits.append("Fantasy playoff matchups (weeks 15-17) are difficult.")
    if trend is not None and abs(trend) >= 4:
        bits.append(f"Usage is trending {'up' if trend > 0 else 'down'} ({'+' if trend > 0 else ''}{trend} share points over the last 3 games).")
    if p.get("missed_recent"):
        wk = ", ".join(str(w) for w in p["missed_recent"])
        bits.append(f"Did not play in week {wk}, so check his status.")
    return dict(total=rnd(total), ppg=rnd(ppg), games=len(games), sched=rnd(sched, 3), playoff_sched=rnd(po_sched, 3),
                trend=trend, outlook=" ".join(bits))


def defenses_out(model):
    out = []
    for tm, d in model.defs.items():
        out.append(dict(
            team=tm, name=TEAM_NAMES[tm],
            blitz=rnd(d["blitz_rate"] * 100), blitz_rank=int(d["blitz_rate_rank"]),
            blitz_raw=rnd(d["blitz_raw"] * 100) if d.get("blitz_raw") == d.get("blitz_raw") and d.get("blitz_raw") is not None else None,
            pressure=rnd(d["pressure_rate"] * 100), pressure_rank=int(d["pressure_rate_rank"]),
            sack=rnd(d["sack_rate"] * 100), int_rate=rnd(d["int_rate"] * 100, 2),
            pass_adj=rnd((d["pass_factor"] - 1) * 100), rush_adj=rnd((d["rush_factor"] - 1) * 100),
            ypt_RB=rnd((d["ypt_RB"] - 1) * 100), ypt_WR=rnd((d["ypt_WR"] - 1) * 100), ypt_TE=rnd((d["ypt_TE"] - 1) * 100),
            fpa={p: rnd(d[f"fpa_{p}"]) for p in SKILL},
            fpa_rank={p: int(d[f"fpa_{p}_rank"]) if d[f"fpa_{p}_rank"] == d[f"fpa_{p}_rank"] else None for p in SKILL},
            pa_g=rnd(model.teams.get(tm, {}).get("pa_shrunk")),
        ))
    return sorted(out, key=lambda x: -x["blitz"])


def teams_out(model, target_week):
    out = []
    for tm, T in model.teams.items():
        qb = model.starter.get(tm)
        q = model.qb_prof.get(qb, {})
        sched = []
        for w in range(1, 19):
            ctx = model.matchup_context(tm, w)
            sched.append(None if ctx is None else dict(week=w, opp=ctx["opp"], home=ctx["home"], played=ctx["played"]))
        out.append(dict(team=tm, name=TEAM_NAMES.get(tm, tm), ppg=rnd(T["raw_ppg"]), plays=rnd(T["plays"]),
                        pass_rate=rnd(T["pass_rate"] * 100), targets=rnd(T["targets"]), carries=rnd(T["carries"]),
                        qb=model.players.get(qb, {}).get("name"), qb_id=qb,
                        qb_blitz_faced=rnd(q.get("blitz_faced", 0) * 100) if q else None,
                        sched=sched))
    return out


# ---------------------------------------------------------------- backtest
def backtest(raw, first=5):
    """Project each completed week using only data from before it; compare to what happened."""
    sched = raw["games"]
    sched = sched[(sched.season == SEASON) & (sched.game_type == "REG")]
    done_weeks = sorted(sched[sched.home_score.notna()].week.unique())
    results = []
    st = raw["stats"]
    for wk in done_weeks:
        if wk <= first or sched[(sched.week == wk)].home_score.isna().any():
            continue
        m = Model(raw, wk - 1)
        actual = st[(st.week == wk) & (st.season_type == "REG")].set_index("player_id").fantasy_points_ppr
        rows = []
        for pid, pr in m.players.items():
            if pid not in actual.index:
                continue
            p = m.project(pr, wk)
            if not p or p.get("bye"):
                continue
            prior = st[(st.player_id == pid) & (st.week < wk)].fantasy_points_ppr
            if len(prior) < 3 or prior.mean() < 5:
                continue
            rows.append(dict(pos=pr["pos"], proj=p["proj"], naive=prior.mean(), actual=actual[pid]))
        df = pd.DataFrame(rows)
        if not len(df):
            continue
        res = dict(week=int(wk), n=int(len(df)),
                   mae_model=rnd((df.proj - df.actual).abs().mean(), 2),
                   mae_naive=rnd((df.naive - df.actual).abs().mean(), 2),
                   corr_model=rnd(df[["proj", "actual"]].corr().iloc[0, 1], 3),
                   corr_naive=rnd(df[["naive", "actual"]].corr().iloc[0, 1], 3),
                   bias={pos: rnd((d.proj - d.actual).mean(), 2) for pos, d in df.groupby("pos")})
        # how often the model picks the better of two same-position players
        pairs = right_m = right_n = 0
        for pos, d in df.groupby("pos"):
            v = d.values
            for i in range(len(d)):
                for j in range(i + 1, len(d)):
                    a, b = d.iloc[i], d.iloc[j]
                    if a.actual == b.actual:
                        continue
                    pairs += 1
                    right_m += (a.proj > b.proj) == (a.actual > b.actual)
                    right_n += (a.naive > b.naive) == (a.actual > b.actual)
        res["startsit_model"] = rnd(right_m / pairs * 100, 1) if pairs else None
        res["startsit_naive"] = rnd(right_n / pairs * 100, 1) if pairs else None
        print("backtest", res)
        results.append(res)
    return results


# ---------------------------------------------------------------- main
def main():
    do_bt = "--backtest" in sys.argv
    raw = load_raw()
    cur_pbp = raw["pbp"][raw["pbp"].season == SEASON]
    through = last_complete_week(cur_pbp)
    sched = raw["games"][(raw["games"].season == SEASON)]
    target = current_week(sched)
    target = max(target, 1)
    print(f"data through week {through}; projecting from week {target}")
    model = Model(raw, through)
    players = build_outputs(model, target)
    os.makedirs(OUT, exist_ok=True)

    def dump(name, obj):
        with open(os.path.join(OUT, name), "w") as f:
            json.dump(obj, f, separators=(",", ":"), default=lambda o: None if o != o else (float(o) if isinstance(o, (np.floating,)) else int(o) if isinstance(o, np.integer) else str(o)))

    dump("players.json", players)
    dump("defenses.json", defenses_out(model))
    dump("teams.json", teams_out(model, target))
    meta = dict(season=SEASON, through_week=through, target_week=target, last_week=LAST_FANTASY_WEEK,
                generated=datetime.now(timezone.utc).isoformat(timespec="minutes"),
                league=dict(blitz=rnd(model.lg["blitz"] * 100), pressure=rnd(model.lg["pressure"] * 100),
                            mix={str(k): {g: rnd(v * 100) for g, v in m.items()} for k, m in model.lg["mix"].items()}))
    bt_path = os.path.join(OUT, "backtest.json")
    if do_bt:
        dump("backtest.json", backtest(raw))
    meta["has_backtest"] = os.path.exists(bt_path)
    dump("meta.json", meta)
    print("wrote", len(players), "players")


if __name__ == "__main__":
    main()
