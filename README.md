# UFC Fighter Predictor

A full-stack web app that predicts UFC fight outcomes using machine learning models trained on real fighter statistics. Includes an AI-powered fight analyst powered by Claude (Anthropic).

## Features

- Predict fight winners using Logistic Regression, Random Forest, and XGBoost models
- Monte Carlo simulation for win probability confidence
- Fighter stats lookup and head-to-head history
- AI fight analysis via Claude

## Tech Stack

- **Frontend:** React + Vite
- **Backend:** Python + Flask
- **ML:** scikit-learn, XGBoost, pandas, numpy
- **AI:** Anthropic Claude API

---

## Prerequisites

- Python 3.10+
- Node.js 18+
- An [Anthropic API key](https://console.anthropic.com/)

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/Manroopg2006/arcade-fightpredictor.git
cd arcade-fightpredictor
```

### 2. Set up the backend

```bash
cd backend
```

Create a `.env` file in the `backend/` folder:

```
ANTHROPIC_API_KEY=your_api_key_here
```

Install Python dependencies (from the project root):

```bash
pip install -r requirements.txt
```

> You may also need to install xgboost separately:
> ```bash
> pip install xgboost
> ```

### 3. Set up the frontend

From the project root:

```bash
npm install
```

---

## Running the App

You need to run **two servers** at the same time — open two terminals.

### Terminal 1 — Start the Flask backend

```bash
cd backend
python app.py
```

The backend runs on `http://localhost:5000`

### Terminal 2 — Start the React frontend

```bash
npm run dev
```

The frontend runs on `http://localhost:5173`

Then open `http://localhost:5173` in your browser.

---

## Project Structure

```
UFC Fighter Model/
├── backend/
│   ├── app.py          # Flask API server
│   ├── main.py         # ML models and prediction logic
│   └── .env            # Your API key (never commit this)
├── data/
│   └── raw/            # CSV datasets (fighters, fights, events)
├── src/
│   ├── pages/          # React pages (Landing, Predict, Result, etc.)
│   ├── components/     # Reusable React components
│   └── api.js          # Frontend API calls to Flask
├── frontend/           # Static HTML fallback
├── requirements.txt    # Python dependencies
└── package.json        # Node dependencies
```

---

## Data

The `data/raw/` folder contains the following CSV files used to train the models:

- `Fighters Stats.csv` — per-fighter career statistics
- `Fighters.csv` — fighter profiles (height, reach, stance, etc.)
- `Fights.csv` — historical fight results
- `Events.csv` — UFC event metadata

---

## Notes

- The backend trains all three ML models on startup — this takes a few seconds.
- The `.env` file is gitignored and should never be committed.
- Model accuracy is printed to the terminal when the backend starts.
