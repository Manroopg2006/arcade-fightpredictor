import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

import pandas as pd
import anthropic
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from main import load_and_prepare_data, train_models, predict_fight, get_head_to_head, monte_carlo_predict

app = Flask(__name__)
CORS(app)  # allows React dev server (port 5173) to talk to Flask (port 5000)

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# ── Startup: load data & train once ──────────────────────────────────────────
print("Loading data and training models...")
fighters, X, y, h2h_fights = load_and_prepare_data()
lr_model, rf_model, xgb_model, col_means, lr_acc, rf_acc, xgb_acc = train_models(X, y)
fighter_names = sorted(fighters["Full Name"].dropna().unique().tolist())
print(f"Ready  |  LR: {lr_acc:.2%}  |  RF: {rf_acc:.2%}  |  XGB: {xgb_acc:.2%}  |  {len(fighter_names)} fighters loaded")


# ── Static frontend ───────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_file(os.path.join(FRONTEND_DIR, "index.html"))


# ── Fighters list ─────────────────────────────────────────────────────────────
@app.route("/api/fighters")
def get_fighters():
    return jsonify({"fighters": fighter_names})


# ── Single fighter profile ────────────────────────────────────────────────────
@app.route("/api/fighters/<path:name>")
def get_fighter(name):
    rows = fighters[fighters["Full Name"].str.lower() == name.lower()]
    if rows.empty:
        return jsonify({"error": f"Fighter not found: {name}"}), 404

    f = rows.iloc[0]

    def _v(val):
        try:
            if pd.isna(val):
                return None
        except Exception:
            pass
        return val.item() if hasattr(val, "item") else val

    return jsonify({
        "name":           _v(f["Full Name"]),
        "wins":           _v(f.get("W")),
        "losses":         _v(f.get("L")),
        "draws":          _v(f.get("D")),
        "reach":          _v(f.get("Reach")),
        "height":         _v(f.get("Ht.")),
        "weight":         _v(f.get("Wt.")),
        "stance":         _v(f.get("Stance")),
        "fighting_style": _v(f.get("Fighting Style")),
        "str":            _v(f.get("STR")),
        "ko_rate":        _v(f.get("KO Rate")),
        "sub_attempts":   _v(f.get("Sub. Att")),
        "weight_class":   _v(f.get("Weight_Class")),
        "belt":           _v(f.get("Belt")),
    })


# ── Predict ───────────────────────────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    red  = (data.get("red")  or "").strip()
    blue = (data.get("blue") or "").strip()

    if not red or not blue:
        return jsonify({"error": "Both fighters are required."}), 400
    if red == blue:
        return jsonify({"error": "Please select two different fighters."}), 400

    try:
        result = predict_fight(red, blue, fighters, lr_model, rf_model, xgb_model, col_means)
        result["head_to_head"] = get_head_to_head(red, blue, h2h_fights)
        result.update(monte_carlo_predict(red, blue, fighters, lr_model, rf_model, xgb_model, col_means))
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404


# ── Model accuracy ────────────────────────────────────────────────────────────
@app.route("/api/accuracy")
def get_accuracy():
    return jsonify({
        "lr_accuracy":  round(lr_acc,  4),
        "rf_accuracy":  round(rf_acc,  4),
        "xgb_accuracy": round(xgb_acc, 4),
        "ensemble":     round((lr_acc + rf_acc + xgb_acc) / 3, 4),
    })


# ── AI summary (Claude) ───────────────────────────────────────────────────────
@app.route("/api/summary", methods=["POST"])
def summary():
    data = request.get_json()

    r          = data.get("red_stats",  {})
    b          = data.get("blue_stats", {})
    winner     = data.get("predicted_winner", "")
    red_name   = data.get("red_name",  "")
    blue_name  = data.get("blue_name", "")
    confidence = data.get("avg_red_prob", 0.5)
    h2h        = data.get("head_to_head", [])
    loser      = blue_name if winner == red_name else red_name

    if winner == blue_name:
        confidence = 1 - confidence

    def stat_line(name, s):
        record   = f"{s.get('wins','?')}W-{s.get('losses','?')}L" if s.get("wins") is not None else "N/A"
        reach    = f"{s.get('reach', 'N/A')}\""
        str_pm   = round(s.get("str") or 0, 2)
        ko_rate  = f"{round((s.get('ko_rate') or 0) * 100, 1)}%"
        sub_rate = f"{round((s.get('sub_rate') or 0) * 100, 1)}%"
        style    = s.get("fighting_style", "N/A")
        streak   = s.get("win_streak", 0) or 0
        last3    = f"{round((s.get('last3_win_rate') or 0) * 100, 0):.0f}%" if s.get("last3_win_rate") is not None else "N/A"
        return (f"{name}: {record} | Style: {style} | Reach: {reach} | "
                f"Str/min: {str_pm} | KO%: {ko_rate} | Sub%: {sub_rate} | "
                f"Last-3: {last3} | Streak: {streak}")

    if confidence >= 0.80:
        closeness = f"Frankly, this isn't even close — {winner} has a massive edge here."
    elif confidence >= 0.65:
        closeness = f"This fight isn't particularly competitive on paper — {winner} holds most of the cards."
    else:
        closeness = f"This is a tight matchup that could realistically go either way."

    if h2h:
        h2h_lines = []
        for fight in h2h:
            event  = fight.get("event") or "Unknown Event"
            date   = f" ({fight['date']})" if fight.get("date") else ""
            w      = fight.get("winner") or "Unknown"
            method = fight.get("method") or "?"
            rnd    = fight.get("round") or "?"
            h2h_lines.append(f"  - {event}{date}: {w} won by {method} in round {rnd}")
        h2h_text = "Previous fights between them:\n" + "\n".join(h2h_lines)
    else:
        h2h_text = "These two have never fought each other before."

    prompt = (
        f"You are a charismatic, sharp-tongued UFC analyst who gives hot takes. "
        f"Our ML model predicts {winner} beats {loser} with {confidence:.0%} confidence. "
        f"{closeness}\n\n"
        f"{h2h_text}\n\n"
        f"Write 3-4 punchy sentences breaking down WHY {winner} wins. Reference specific stats "
        f"(reach, record, striking output, KO rate, style). If the fight is lopsided, say so bluntly. "
        f"If they've fought before, call back to that history. No hedging, no filler — just sharp analysis.\n\n"
        f"Fighter stats:\n"
        f"  {stat_line(red_name, r)}\n"
        f"  {stat_line(blue_name, b)}"
    )

    try:
        client  = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=350,
            messages=[{"role": "user", "content": prompt}],
        )
        return jsonify({"summary": message.content[0].text.strip()})
    except Exception as e:
        return jsonify({"error": f"Summary generation failed: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)