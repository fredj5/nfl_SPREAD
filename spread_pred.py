import os, sys
import numpy as np
import pandas as pd

DATA_DIR = "./data/"
P_SCORES = os.path.join(DATA_DIR, "spreadspoke_scores.csv")
P_TEAMS  = os.path.join(DATA_DIR, "nfl_teams.csv")
P_WEEK1  = os.path.join(DATA_DIR, "week_1_games.csv")
OUT_PRED = os.path.join(DATA_DIR, "week1_spread_predictions.csv")

def load_team_maps(p):
    t = pd.read_csv(p)
    name_col = "team_name" if "team_name" in t.columns else t.columns[0]
    abbr_col = "team_id"   if "team_id"   in t.columns else t.columns[1]
    full2abbr = {str(n).strip(): str(a).strip() for n,a in zip(t[name_col], t[abbr_col])}
    return full2abbr

def to_abbr(x, full2abbr):
    if pd.isna(x): return x
    s = str(x).strip()
    if s in full2abbr.values(): return s
    return full2abbr.get(s, s)

def build_hist_calibration(scores_csv, teams_csv):
    full2abbr = load_team_maps(teams_csv)
    df = pd.read_csv(scores_csv)
    for c in ["score_home","score_away","spread_favorite"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["team_favorite_id","team_home","team_away",
                           "score_home","score_away","spread_favorite","schedule_date"]).copy()
    df["schedule_date"] = pd.to_datetime(df["schedule_date"], errors="coerce")
    df = df.loc[df["schedule_date"].dt.year >= 2000]

    df["home_abbr"] = df["team_home"].apply(lambda s: to_abbr(s, full2abbr))
    df["away_abbr"] = df["team_away"].apply(lambda s: to_abbr(s, full2abbr))
    fav_abbr = df["team_favorite_id"].astype(str).str.strip()

    fav_is_home = (fav_abbr == df["home_abbr"])
    fav_is_away = (fav_abbr == df["away_abbr"])
    df = df.loc[fav_is_home | fav_is_away].copy()

    score_fav = np.where(fav_is_home, df["score_home"], df["score_away"])
    score_opp = np.where(fav_is_home, df["score_away"], df["score_home"])
    margin = score_fav - score_opp
    spread_abs = df["spread_favorite"].abs()
    cover = (margin > spread_abs).astype(int)

    df["cover_favorite"] = cover
    df["is_home_favorite"] = fav_is_home.astype(int)
    df["spread_abs"] = spread_abs

    bins = [0, 1.5, 2.5, 3.5, 5.5, 7.5, 9.5, 12.5, 20]
    df["bucket"] = pd.cut(df["spread_abs"], bins, right=True, include_lowest=True)

    calib = (df.groupby(["bucket","is_home_favorite"])["cover_favorite"]
               .mean().rename("p_hist").reset_index())

    global_rate = float(df["cover_favorite"].mean())
    return calib, bins, global_rate, full2abbr

def apply_calibration(spread_abs, is_home_fav, calib, bins, global_rate, weight=0.35):
    bucket = pd.cut(spread_abs, bins, right=True, include_lowest=True)
    tmp = pd.DataFrame({"bucket": bucket, "is_home_favorite": is_home_fav})
    tmp = tmp.merge(calib, on=["bucket","is_home_favorite"], how="left")
    p_hist = tmp["p_hist"].fillna(global_rate).to_numpy()
    return (1 - weight) * 0.5 + weight * p_hist

def predict_week1(week1_csv, calib, bins, global_rate, full2abbr):
    wk = pd.read_csv(week1_csv)
    wk["schedule_date"] = pd.to_datetime(wk["schedule_date"], errors="coerce")
    wk["spread_favorite"] = pd.to_numeric(wk["spread_favorite"], errors="coerce")
    wk["over_under_line"] = pd.to_numeric(wk["over_under_line"], errors="coerce")

    wk["home_abbr"] = wk["team_home"].apply(lambda s: to_abbr(s, full2abbr))
    wk["away_abbr"] = wk["team_away"].apply(lambda s: to_abbr(s, full2abbr))
    wk["fav_abbr"]  = wk["team_favorite_id"].apply(lambda s: to_abbr(s, full2abbr))

    wk["is_home_favorite"] = (wk["fav_abbr"] == wk["home_abbr"]).astype(int)
    wk["spread_abs"] = wk["spread_favorite"].abs()

    probs = apply_calibration(
        spread_abs = wk["spread_abs"],
        is_home_fav = wk["is_home_favorite"],
        calib = calib, bins = bins, global_rate = global_rate
    )
    picks = np.where(probs >= 0.5, "FAVORITE", "UNDERDOG")

    out_cols = ["schedule_date","schedule_season","schedule_week","team_home","team_away",
                "team_favorite_id","spread_favorite","over_under_line","stadium"]
    out = wk[out_cols].copy()
    out["prob_favorite_covers"] = probs
    out["pick_against_the_spread"] = picks
    return out

def main():
    calib, bins, global_rate, full2abbr = build_hist_calibration(P_SCORES, P_TEAMS)
    preds = predict_week1(P_WEEK1, calib, bins, global_rate, full2abbr)
    preds["schedule_date"] = preds["schedule_date"].dt.strftime("%Y-%m-%d")
    preds.to_csv(OUT_PRED, index=False)
    print("Wrote", OUT_PRED)

if __name__ == "__main__":
    main()
