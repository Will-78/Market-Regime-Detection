# Market Regime Detection
 
Unsupervised ML to detect market regimes, distinct periods of market behavior such as bull markets, bear markets, and crises.
 
**Authors:** William Barnett & Preston Smith
 
## Overview
 
We apply four unsupervised algorithms to the same feature set and compare how each one carves up the market into regimes:
 
- **K-Means** — simple, interpretable partitioning (k chosen via elbow method)
- **Gaussian Mixture Model (GMM)** — soft probabilistic assignment (components chosen via BIC); most effective overall
- **HDBSCAN** — density-based, handles noise and irregular shapes
- **Isolation Forest** *(bonus)* — anomaly detection used to validate extreme events
## Data
 
Pulled via `yfinance` across eight ETFs covering equities, bonds, commodities, and sectors:
 
| Ticker | Asset |
|---|---|
| SPY | S&P 500 (large caps) |
| IWM | Russell 2000 (small caps) |
| QQQ | Tech (Nasdaq-100) |
| TLT | Treasury bonds |
| GLD | Gold |
| XLE | Energy |
| XLF | Financials |
| XLK | Tech sector |
 
## Features
 
Seven features computed per trading day on a 20-day rolling window, then standardized with `StandardScaler`:
 
- **Volatility** — rolling std of returns
- **Avg Return** — rolling mean return
- **Dispersion** — cross-asset std (how differently markets move)
- **Correlation** — rolling cross-asset correlation
- **Equity-Bond Spread** — equity returns minus TLT (risk-on vs. risk-off)
- **Drawdown** — distance from rolling high
- **Vol Change** — day-over-day volatility change
The same pipeline feeds every model, so comparisons are fair.
 
## Results
 
All three clustering models independently recovered the same three core regimes:
 
1. **Normal** — low volatility, positive returns
2. **Risk-off** — elevated volatility, negative returns
3. **Crisis** — extreme volatility spike (COVID crash dominates this cluster)
Isolation Forest flagged the same crisis days as anomalies, corroborating the clustering results. The fact that regimes show up consistently across very different algorithms suggests they're a real property of the market rather than an artifact of any single method.
 
**Model comparison:**
- GMM was the most effective overall
- HDBSCAN was best at handling noise and irregular cluster shapes
- K-Means was the simplest and most interpretable
- Isolation Forest was best for validating extreme events
## Running It
 
Requires Python 3. Install dependencies:
 
```bash
pip install -r requirements.txt
```
 
Key packages: `hdbscan`, `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `yfinance`.
 
Launch Jupyter from the project directory:
 
```bash
jupyter notebook
```
 
Then open any of the notebooks:
 
- `kmeans_clustering.ipynb` — K-Means with elbow method (prompts for k after showing the inertia plot)
- `gmm.ipynb` — GMM with BIC-based selection
- `hdbscan.ipynb` — HDBSCAN with grid search over hyperparameters
Each notebook is self-contained and downloads its data automatically. Isolation Forest runs at the end of each notebook — no separate file needed.
