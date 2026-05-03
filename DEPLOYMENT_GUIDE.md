# NEXUS Market Platform — Full Deployment Guide
## 100% Free Stack: yfinance + FastAPI + Render + GitHub Pages

---

## ARCHITECTURE OVERVIEW

```
[Yahoo Finance]  →  yfinance (free, no API key)
      ↓
[Python Backend]  →  FastAPI + statsmodels + arch + keras + xgboost + prophet
      ↓
[Render.com]      →  Free web service (nexus-market-api.onrender.com)
      ↓
[Frontend HTML]   →  GitHub Pages (yourname.github.io/nexus)
```

---

## STEP 1 — Set Up GitHub (5 minutes)

1. Go to https://github.com and create a free account (or log in)
2. Create TWO new repositories:
   - `nexus-backend`  (for Python API)
   - `nexus-frontend` (for HTML)
3. In `nexus-frontend` repo → Settings → Pages → Source: `main` branch → `/root`
   Your site will be at: `https://YOUR_USERNAME.github.io/nexus-frontend`

---

## STEP 2 — Deploy Backend to Render (10 minutes)

### 2a. Push backend files to GitHub

Upload these files to your `nexus-backend` repo:
```
nexus-backend/
├── main.py
├── requirements.txt
├── render.yaml
├── data/
│   ├── __init__.py
│   └── fetcher.py
└── models/
    ├── __init__.py
    ├── arima_model.py
    ├── garch_model.py
    ├── lstm_model.py
    ├── xgb_model.py
    ├── prophet_model.py
    ├── hybrid_model.py
    └── ensemble.py
```

### 2b. Deploy on Render

1. Go to https://render.com → Sign up free with GitHub
2. Click "New +" → "Web Service"
3. Connect your `nexus-backend` GitHub repo
4. Render auto-detects `render.yaml` — click "Create Web Service"
5. Wait ~5 minutes for first build
6. Your API URL will be: `https://nexus-market-api.onrender.com`
   (or similar — copy the actual URL from Render dashboard)

### 2c. Verify backend works

Visit: `https://YOUR-RENDER-URL.onrender.com/health`
You should see: `{"status":"ok","time":"..."}`

Test a quote:
`https://YOUR-RENDER-URL.onrender.com/quote/EUR%2FUSD`

Test analysis:
`https://YOUR-RENDER-URL.onrender.com/analyze/EUR%2FUSD?horizon=1M`

---

## STEP 3 — Update Frontend with Your API URL (2 minutes)

Open `frontend/index.html` and find line ~340:

```javascript
return 'https://nexus-market-api.onrender.com';
```

Replace `nexus-market-api` with your actual Render service name.

---

## STEP 4 — Deploy Frontend to GitHub Pages (3 minutes)

1. Push `frontend/index.html` to your `nexus-frontend` GitHub repo
2. Go to repo → Settings → Pages
3. Source: Deploy from branch → `main` → `/ (root)`
4. Save → wait 2 minutes
5. Visit: `https://YOUR_USERNAME.github.io/nexus-frontend`

---

## FREE TIER LIMITATIONS & WORKAROUNDS

| Limitation | Workaround |
|---|---|
| Render free tier sleeps after 15min inactivity | First request takes ~30s to wake up. Users see "Connecting…" message |
| yfinance 15-20min data delay | Acceptable for analysis — labelled "15min delayed" in UI |
| 512MB RAM on Render free | TensorFlow-CPU is large; LSTM falls back to Ridge regression if OOM |
| 750 free hours/month on Render | ~31 days — plenty for 1 service running 24/7 |
| GitHub Pages: static only | Fine — all computation is on Render backend |

---

## KEEPING IT FREE FOREVER

- **Render**: Stay on "Free" plan — never upgrade
- **GitHub**: Free public repos + Pages
- **yfinance**: No API key, no cost, no rate limit for normal use
- **Domain**: Use the free `.onrender.com` and `.github.io` domains

---

## OPTIONAL: CUSTOM DOMAIN (Still Free)

1. Get a free domain from https://freenom.com or https://dot.tk
2. In GitHub Pages settings → Custom domain → enter your domain
3. Add CNAME record pointing to `YOUR_USERNAME.github.io`

---

## LOCAL DEVELOPMENT

```bash
# Backend
cd nexus-backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend — just open index.html in browser
# It auto-detects localhost and uses http://localhost:8000
```

---

## TROUBLESHOOTING

**"API offline"** → Render free tier is sleeping. Wait 30 seconds, click "Run Analysis" again.

**"Not enough historical data"** → Some exotic pairs may have limited yfinance data. Try a major pair first.

**LSTM fails / falls back** → Normal on Render free tier (512MB RAM). TensorFlow needs ~400MB. The model gracefully falls back to Ridge regression.

**CORS error** → The backend has `allow_origins=["*"]` so this should never happen.

**Prophet install fails** → Prophet requires `pystan`. If Render build fails, remove `prophet` from requirements.txt and the prophet model will be skipped gracefully.

---

## MODELS EXPLAINED (Real Computation)

| Model | Library | What it actually does |
|---|---|---|
| ARIMA | statsmodels | Auto-selects (p,d,q) by AIC grid search, fits on real prices |
| GARCH(1,1) | arch | Estimates ω, α, β on log-returns, forecasts conditional variance |
| LSTM | Keras/TF | 64→32 unit LSTM, trains on sliding windows of 20 prices |
| XGBoost | xgboost | 15 features (MA, RSI, Bollinger, MACD, ATR, lags), recursive forecast |
| Prophet | prophet | Piecewise linear trend + weekly/annual Fourier seasonality |
| ARIMA+LSTM | both | ARIMA fits linear part, LSTM models the residuals |
| Kalman+XGB | custom+xgboost | Kalman filter smooths noise, XGBoost forecasts clean signal |

## R² IS NOW REAL

R² is computed as: `1 - SS_res / SS_tot` on a held-out validation set (last 20% of data).
Values are clamped to [0, 1]. A value of 0.85 means the model explains 85% of price variance.
