import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

FEATURES = [
    # Physical
    "reach_diff", "height_diff", "weight_diff", "reach_ht_ratio_diff",
    # Record / Experience
    "win_diff", "total_fights_diff",
    # Striking
    "str_diff", "sig_str_acc_diff", "absorbed_str_diff", "str_def_diff",
    "ko_rate_diff", "finish_rate_diff", "strike_efficiency_diff",
    # Grappling
    "td_diff", "sub_att_diff", "ctrl_diff", "sub_rate_diff", "td_def_diff",
    # Activity
    "avg_fight_time_diff", "days_since_last_fight_diff",
    # Recent Form
    "last3_win_diff", "last5_win_diff", "win_streak_diff",
    # Context
    "stance_matchup", "weight_class_encoded",
]

WEIGHT_CLASS_MAP = {
    "Strawweight": 0, "Atomweight": 1, "Flyweight": 2,
    "Bantamweight": 3, "Featherweight": 4, "Lightweight": 5,
    "Welterweight": 6, "Middleweight": 7, "Light Heavyweight": 8,
    "Heavyweight": 9, "Super Heavyweight": 10,
    "Women's Strawweight": 11, "Women's Atomweight": 12,
    "Women's Flyweight": 13, "Women's Bantamweight": 14,
    "Women's Featherweight": 15,
}


def parse_height(h):
    """Convert height from feet.inches notation (e.g. 5.11 = 5'11") to total inches."""
    try:
        if pd.isna(h):
            return np.nan
    except (TypeError, ValueError):
        pass
    h = float(str(h).strip())
    feet = int(h)
    inches = round((h - feet) * 100)
    return feet * 12 + inches


def _val(v):
    """Convert a value to a JSON-serializable Python type, or None if NaN."""
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(v, "item"):
        return v.item()
    return v


def encode_stance_matchup(stance_r, stance_b):
    """Encode stance matchup: 0=same, 1=Orth vs South, 2=Switch involved, 3=other."""
    s_r = str(stance_r).strip() if pd.notna(stance_r) else ""
    s_b = str(stance_b).strip() if pd.notna(stance_b) else ""
    if not s_r or not s_b:
        return 0
    if s_r == s_b:
        return 0
    if set([s_r, s_b]) == {"Orthodox", "Southpaw"}:
        return 1
    if "Switch" in (s_r, s_b):
        return 2
    return 3


def compute_fighter_derived_stats(fighters_df, fights_raw, events):
    """Compute per-fighter last3/5 win rates, win streak, days since last fight,
    absorbed strikes, striking defense, and takedown defense from fight history."""
    fights = fights_raw.merge(events[["Event_Id", "Date"]], on="Event_Id", how="left")
    fights["Date"] = pd.to_datetime(fights["Date"], errors="coerce")
    fights = fights.sort_values("Date").reset_index(drop=True)

    all_fids = fighters_df["Fighter_Id"].dropna().unique()
    today = pd.Timestamp.today()
    records = {}

    for fid in all_fids:
        # Fights as Fighter_1 (red corner)
        as_f1 = fights[fights["Fighter_Id_1"] == fid][
            ["Date", "Result_1", "STR_2", "Sig. Str. %_2", "TD_2"]
        ].copy()
        as_f1.columns = ["Date", "won", "opp_str", "opp_str_acc", "opp_td"]
        as_f1["won"] = (as_f1["won"] == "W").astype(float)

        # Fights as Fighter_2 (blue corner)
        as_f2 = fights[fights["Fighter_Id_2"] == fid][
            ["Date", "Result_2", "STR_1", "Sig. Str. %_1", "TD_1"]
        ].copy()
        as_f2.columns = ["Date", "won", "opp_str", "opp_str_acc", "opp_td"]
        as_f2["won"] = (as_f2["won"] == "W").astype(float)

        hist = pd.concat([as_f1, as_f2]).sort_values("Date").reset_index(drop=True)

        if hist.empty:
            records[fid] = {
                "last3_win_rate": np.nan, "last5_win_rate": np.nan,
                "win_streak": 0.0,       "days_since_last_fight": np.nan,
                "absorbed_str": np.nan,  "str_def": np.nan, "td_def": np.nan,
            }
            continue

        wins = hist["won"].tolist()
        last3 = float(np.mean(wins[-3:])) if len(wins) >= 1 else np.nan
        last5 = float(np.mean(wins[-5:])) if len(wins) >= 1 else np.nan

        streak = 0
        for w in reversed(wins):
            if w == 1.0:
                streak += 1
            else:
                break

        last_date = hist["Date"].dropna().max()
        days_since = float((today - last_date).days) if pd.notna(last_date) else np.nan

        absorbed   = float(hist["opp_str"].mean())
        opp_acc    = hist["opp_str_acc"].dropna()
        str_def    = float(1 - opp_acc.mean()) if len(opp_acc) > 0 else np.nan
        td_def_val = float(hist["opp_td"].mean())

        records[fid] = {
            "last3_win_rate":       last3,
            "last5_win_rate":       last5,
            "win_streak":           float(streak),
            "days_since_last_fight": days_since,
            "absorbed_str":         absorbed,
            "str_def":              str_def,
            "td_def":               td_def_val,
        }

    derived = pd.DataFrame.from_dict(records, orient="index")
    derived.index.name = "Fighter_Id"
    return derived.reset_index()


def load_and_prepare_data():
    fighters_base = pd.read_csv(os.path.join(DATA_DIR, "Fighters.csv"))
    fighter_stats = pd.read_csv(os.path.join(DATA_DIR, "Fighters Stats.csv"))
    fights_raw    = pd.read_csv(os.path.join(DATA_DIR, "Fights.csv"))
    events        = pd.read_csv(os.path.join(DATA_DIR, "Events.csv"))

    # Fix height: feet.inches → total inches
    fighters_base["Ht."] = fighters_base["Ht."].apply(parse_height)

    # Merge career stats (drop duplicate columns already in Fighters.csv)
    fighter_stats_clean = fighter_stats.drop(
        columns=["Full Name", "Nickname", "Ht.", "Wt.", "Stance", "W", "L", "D", "Belt"]
    )
    fighters = fighters_base.merge(fighter_stats_clean, on="Fighter_Id", how="left")

    # Compute derived stats from fight history and merge in
    derived = compute_fighter_derived_stats(fighters, fights_raw, events)
    fighters = fighters.merge(derived, on="Fighter_Id", how="left")

    # Additional fighter-level computed columns
    fighters["total_fights"]       = fighters["W"] + fighters["L"] + fighters["D"]
    fighters["finish_rate"]        = fighters["KO Rate"].fillna(0) + fighters["SUB Rate"].fillna(0)
    fighters["strike_efficiency"]  = fighters["STR"] * fighters["Sig. Str. %"]
    fighters["reach_ht_ratio"]     = fighters["Reach"] / fighters["Ht."].replace(0, np.nan)

    # Keep raw fights merged with events for head-to-head display
    h2h_fights = fights_raw.merge(
        events[["Event_Id", "Name", "Date"]], on="Event_Id", how="left"
    )

    # Merge fighters into fights for red corner (Fighter_1) then blue corner (Fighter_2)
    fights = fights_raw.merge(
        fighters, left_on="Fighter_1", right_on="Full Name", how="left", suffixes=("", "_R")
    )
    fights = fights.merge(
        fighters, left_on="Fighter_2", right_on="Full Name", how="left", suffixes=("", "_B")
    )
    fights = fights.copy()  # defragment after double merge

    # ── Differential features (red minus blue) ────────────────────────────────
    fights["reach_diff"]                = fights["Reach"]              - fights["Reach_B"]
    fights["height_diff"]               = fights["Ht."]                - fights["Ht._B"]
    fights["weight_diff"]               = fights["Wt."]                - fights["Wt._B"]
    fights["reach_ht_ratio_diff"]       = fights["reach_ht_ratio"]     - fights["reach_ht_ratio_B"]
    fights["win_diff"]                  = fights["W"]                  - fights["W_B"]
    fights["total_fights_diff"]         = fights["total_fights"]       - fights["total_fights_B"]
    fights["str_diff"]                  = fights["STR"]                - fights["STR_B"]
    fights["sig_str_acc_diff"]          = fights["Sig. Str. %"]        - fights["Sig. Str. %_B"]
    fights["absorbed_str_diff"]         = fights["absorbed_str"]       - fights["absorbed_str_B"]
    fights["str_def_diff"]              = fights["str_def"]            - fights["str_def_B"]
    fights["ko_rate_diff"]              = fights["KO Rate"]            - fights["KO Rate_B"]
    fights["finish_rate_diff"]          = fights["finish_rate"]        - fights["finish_rate_B"]
    fights["strike_efficiency_diff"]    = fights["strike_efficiency"]  - fights["strike_efficiency_B"]
    fights["td_diff"]                   = fights["TD"]                 - fights["TD_B"]
    fights["sub_att_diff"]              = fights["Sub. Att"]           - fights["Sub. Att_B"]
    fights["ctrl_diff"]                 = fights["Ctrl"]               - fights["Ctrl_B"]
    fights["sub_rate_diff"]             = fights["SUB Rate"]           - fights["SUB Rate_B"]
    fights["td_def_diff"]               = fights["td_def"]             - fights["td_def_B"]
    fights["avg_fight_time_diff"]       = fights["Avg Fight Time"]     - fights["Avg Fight Time_B"]
    fights["days_since_last_fight_diff"]= fights["days_since_last_fight"] - fights["days_since_last_fight_B"]
    fights["last3_win_diff"]            = fights["last3_win_rate"]     - fights["last3_win_rate_B"]
    fights["last5_win_diff"]            = fights["last5_win_rate"]     - fights["last5_win_rate_B"]
    fights["win_streak_diff"]           = fights["win_streak"]         - fights["win_streak_B"]

    # Stance matchup (per-fight row)
    fights["stance_matchup"] = fights.apply(
        lambda row: encode_stance_matchup(row.get("Stance"), row.get("Stance_B")), axis=1
    )

    # Weight class from the fight itself
    fights["weight_class_encoded"] = fights["Weight_Class"].map(WEIGHT_CLASS_MAP).fillna(-1)

    X = fights[FEATURES]
    y = fights["Result_1"].apply(lambda x: 1 if x == "W" else 0)

    valid = X.dropna().index
    return fighters, X.loc[valid], y.loc[valid], h2h_fights


def get_head_to_head(red_name, blue_name, h2h_fights):
    mask = (
        ((h2h_fights["Fighter_1"] == red_name)  & (h2h_fights["Fighter_2"] == blue_name)) |
        ((h2h_fights["Fighter_1"] == blue_name) & (h2h_fights["Fighter_2"] == red_name))
    )
    matchups = h2h_fights[mask].copy()
    if matchups.empty:
        return []

    results = []
    for _, row in matchups.iterrows():
        if row["Fighter_1"] == red_name:
            winner = red_name if row["Result_1"] == "W" else blue_name
        else:
            winner = blue_name if row["Result_1"] == "W" else red_name

        results.append({
            "event":  _val(row.get("Name")),
            "date":   _val(row.get("Date")),
            "winner": winner,
            "method": _val(row.get("Method")),
            "round":  _val(row.get("Round")),
        })

    results.sort(key=lambda x: x["date"] or "", reverse=True)
    return results


def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Impute NaNs with column means from training set
    col_means = X_train.mean()
    X_train = X_train.fillna(col_means)
    X_test  = X_test.fillna(col_means)

    lr = LogisticRegression(max_iter=5000, solver="saga")
    lr.fit(X_train, y_train)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
    )
    xgb.fit(X_train, y_train)

    lr_acc  = accuracy_score(y_test, lr.predict(X_test))
    rf_acc  = accuracy_score(y_test, rf.predict(X_test))
    xgb_acc = accuracy_score(y_test, xgb.predict(X_test))

    return lr, rf, xgb, col_means, lr_acc, rf_acc, xgb_acc


def build_feature_vector(red, blue):
    """Build a feature dict from two fighter Series objects."""
    def sd(a, b):
        try:
            av = float(a) if pd.notna(a) else np.nan
            bv = float(b) if pd.notna(b) else np.nan
            return av - bv
        except (TypeError, ValueError):
            return np.nan

    wc = red.get("Weight_Class") or blue.get("Weight_Class")
    wc_encoded = WEIGHT_CLASS_MAP.get(str(wc).strip(), -1) if wc else -1

    return {
        "reach_diff":                sd(red["Reach"],              blue["Reach"]),
        "height_diff":               sd(red["Ht."],                blue["Ht."]),
        "weight_diff":               sd(red["Wt."],                blue["Wt."]),
        "reach_ht_ratio_diff":       sd(red.get("reach_ht_ratio"), blue.get("reach_ht_ratio")),
        "win_diff":                  sd(red["W"],                  blue["W"]),
        "total_fights_diff":         sd(red.get("total_fights"),   blue.get("total_fights")),
        "str_diff":                  sd(red["STR"],                blue["STR"]),
        "sig_str_acc_diff":          sd(red.get("Sig. Str. %"),    blue.get("Sig. Str. %")),
        "absorbed_str_diff":         sd(red.get("absorbed_str"),   blue.get("absorbed_str")),
        "str_def_diff":              sd(red.get("str_def"),        blue.get("str_def")),
        "ko_rate_diff":              sd(red["KO Rate"],            blue["KO Rate"]),
        "finish_rate_diff":          sd(red.get("finish_rate"),    blue.get("finish_rate")),
        "strike_efficiency_diff":    sd(red.get("strike_efficiency"), blue.get("strike_efficiency")),
        "td_diff":                   sd(red.get("TD"),             blue.get("TD")),
        "sub_att_diff":              sd(red.get("Sub. Att"),       blue.get("Sub. Att")),
        "ctrl_diff":                 sd(red.get("Ctrl"),           blue.get("Ctrl")),
        "sub_rate_diff":             sd(red.get("SUB Rate"),       blue.get("SUB Rate")),
        "td_def_diff":               sd(red.get("td_def"),         blue.get("td_def")),
        "avg_fight_time_diff":       sd(red.get("Avg Fight Time"), blue.get("Avg Fight Time")),
        "days_since_last_fight_diff": sd(red.get("days_since_last_fight"), blue.get("days_since_last_fight")),
        "last3_win_diff":            sd(red.get("last3_win_rate"), blue.get("last3_win_rate")),
        "last5_win_diff":            sd(red.get("last5_win_rate"), blue.get("last5_win_rate")),
        "win_streak_diff":           sd(red.get("win_streak"),     blue.get("win_streak")),
        "stance_matchup":            encode_stance_matchup(red.get("Stance"), blue.get("Stance")),
        "weight_class_encoded":      wc_encoded,
    }


CATEGORICAL_FEATURES = {"stance_matchup", "weight_class_encoded"}


def monte_carlo_predict(red_name, blue_name, fighters, lr_model, rf_model, xgb_model, col_means, n_sims=300):
    red_rows  = fighters[fighters["Full Name"] == red_name]
    blue_rows = fighters[fighters["Full Name"] == blue_name]

    if red_rows.empty:
        raise ValueError(f"Fighter not found: {red_name}")
    if blue_rows.empty:
        raise ValueError(f"Fighter not found: {blue_name}")

    red  = red_rows.iloc[0]
    blue = blue_rows.iloc[0]

    feat_dict = build_feature_vector(red, blue)
    feat_df   = pd.DataFrame([feat_dict])[FEATURES].fillna(col_means)
    base      = feat_df.values[0]  # shape (n_features,)

    # Tile the base vector into n_sims rows, then perturb continuous features
    batch = np.tile(base, (n_sims, 1)).astype(float)

    noise_scale = np.abs(base) * 0.08          # 8% of each feature's magnitude
    noise_scale = np.where(noise_scale < 0.01, 0.01, noise_scale)  # floor

    for i, feat_name in enumerate(FEATURES):
        if feat_name not in CATEGORICAL_FEATURES:
            batch[:, i] += np.random.normal(0, noise_scale[i], size=n_sims)

    batch_df  = pd.DataFrame(batch, columns=FEATURES)
    lr_probs  = lr_model.predict_proba(batch_df)[:, 1]
    rf_probs  = rf_model.predict_proba(batch_df)[:, 1]
    xgb_probs = xgb_model.predict_proba(batch_df)[:, 1]
    avg_probs = (lr_probs + rf_probs + xgb_probs) / 3

    return {
        "mc_mean":   round(float(np.mean(avg_probs)), 4),
        "mc_std":    round(float(np.std(avg_probs)),  4),
        "mc_ci_low": round(float(np.percentile(avg_probs, 5)),  4),
        "mc_ci_high":round(float(np.percentile(avg_probs, 95)), 4),
        "mc_sims":   n_sims,
    }


def predict_fight(red_name, blue_name, fighters, lr_model, rf_model, xgb_model, col_means):
    red_rows  = fighters[fighters["Full Name"] == red_name]
    blue_rows = fighters[fighters["Full Name"] == blue_name]

    if red_rows.empty:
        raise ValueError(f"Fighter not found: {red_name}")
    if blue_rows.empty:
        raise ValueError(f"Fighter not found: {blue_name}")

    red  = red_rows.iloc[0]
    blue = blue_rows.iloc[0]

    feat_dict = build_feature_vector(red, blue)
    feat_df   = pd.DataFrame([feat_dict])[FEATURES]
    feat_df   = feat_df.fillna(col_means)

    lr_prob  = float(lr_model.predict_proba(feat_df)[0][1])
    rf_prob  = float(rf_model.predict_proba(feat_df)[0][1])
    xgb_prob = float(xgb_model.predict_proba(feat_df)[0][1])
    avg_prob = (lr_prob + rf_prob + xgb_prob) / 3

    def stats(f):
        return {
            "reach":          _val(f["Reach"]),
            "height":         _val(f["Ht."]),
            "weight":         _val(f["Wt."]),
            "wins":           _val(f["W"]),
            "losses":         _val(f["L"]),
            "str":            _val(f["STR"]),
            "ko_rate":        _val(f["KO Rate"]),
            "fighting_style": _val(f.get("Fighting Style")),
            "weight_class":   _val(f.get("Weight_Class")),
            "sig_str_acc":    _val(f.get("Sig. Str. %")),
            "td":             _val(f.get("TD")),
            "sub_rate":       _val(f.get("SUB Rate")),
            "ctrl":           _val(f.get("Ctrl")),
            "finish_rate":    _val(f.get("finish_rate")),
            "last3_win_rate": _val(f.get("last3_win_rate")),
            "last5_win_rate": _val(f.get("last5_win_rate")),
            "win_streak":     _val(f.get("win_streak")),
            "stance":         _val(f.get("Stance")),
            "total_fights":   _val(f.get("total_fights")),
            "days_since_last_fight": _val(f.get("days_since_last_fight")),
        }

    return {
        "red_name":         red_name,
        "blue_name":        blue_name,
        "lr_red_prob":      round(lr_prob,  4),
        "rf_red_prob":      round(rf_prob,  4),
        "xgb_red_prob":     round(xgb_prob, 4),
        "avg_red_prob":     round(avg_prob, 4),
        "predicted_winner": red_name if avg_prob >= 0.5 else blue_name,
        "red_stats":        stats(red),
        "blue_stats":       stats(blue),
    }


if __name__ == "__main__":
    print("Loading data...")
    fighters, X, y, h2h_fights = load_and_prepare_data()

    print("Training models...")
    lr_model, rf_model, xgb_model, col_means, lr_acc, rf_acc, xgb_acc = train_models(X, y)
    print(f"Logistic Regression Accuracy: {lr_acc:.4f}")
    print(f"Random Forest Accuracy:       {rf_acc:.4f}")
    print(f"XGBoost Accuracy:             {xgb_acc:.4f}")
    print(f"Ensemble Accuracy (avg):      {(lr_acc + rf_acc + xgb_acc) / 3:.4f}")
