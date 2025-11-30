AutoTSA
=======

AutoTSA is an AutoML toolkit for univariate time-series forecasting with optional exogenous regressors. It builds lag/rolling features, runs a compact model search (tree ensembles, linear models, Gaussian Process; optional LightGBM, Prophet, SARIMA, GARCH, GRU), scores via rolling-origin validation, and returns the best model plus metrics. A Streamlit GUI and CLI are included for fast experimentation.

## Highlights
- Rolling-origin validation with configurable holdout (batch or stepwise) to reduce leakage.
- Automatic feature engineering: target lags, rolling statistics, datetime encodings, holiday indicators, and exogenous lags.
- Model zoo: Gradient Boosting, Random Forest, Ridge/ElasticNet, Gaussian Process; optional LightGBM, Prophet, SARIMA, GARCH, GRU (pytorch); TFT currently disabled.
- Deterministic searches with random seeds; Optuna hyperparameter tuning (optional).
- Exogenous forecasting in the GUI: hold-flat or project each exog forward via a small Gradient Boosting forecaster on its own lags/rolls.
- Exportable search history, holdout plots (CLI), and a modern Streamlit interface.

## Installation
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Optional extras:
#   pip install lightgbm
#   pip install prophet
#   pip install statsmodels
#   pip install arch  # for GARCH
#   pip install torch pytorch-lightning pytorch-forecasting  # for GRU/TFT (TFT disabled by default)
#   pip install holidays  # for holiday features
```

## Quickstart (CLI)
```bash
python -m auto_tsa.cli \
  --data your_data.csv \
  --target y \
  --timestamp ds \
  --freq D \
  --models gbr,rf,ridge \
  --exog temp,volume \
  --holiday-country US \
  --trials 12 \
  --splits 4 \
  --metric smape \
  --predict-next \
  --export-history history.json
```
- `--predict-next` prints a one-step forecast using the fitted model.
- `--plot-holdout path.png` saves an actual vs. predicted plot for the holdout period.
- `--use-optuna --optuna-trials N` enables Optuna tuning if installed.

## Python API
```python
import pandas as pd
from auto_tsa import AutoTSA, AutoTSAConfig

df = pd.read_csv("your_data.csv")
cfg = AutoTSAConfig(
    timestamp_col="ds",
    target_col="y",
    freq="D",
    lags=(1, 2, 3, 7, 14),
    windows=(3, 7, 14),
    exogenous_cols=["temp", "promo"],
    exogenous_lags=(0, 1),
    add_holiday_features=True,
    holidays_country="US",
    rolling_splits=4,
    max_trials_per_model=8,
    scoring="smape",
    use_optuna=False,
)
automl = AutoTSA(cfg).fit(df)
forecast_one_step = automl.predict_next(df)
```

## Streamlit GUI
```bash
streamlit run src/auto_tsa/gui.py
```
What it does:
- Upload a CSV, pick timestamp/target/exogenous columns, horizon, and models.
- Optional toggles: project exogenous forward (per-exog GBR forecaster), add holiday features, enable Optuna (when supported).
- Shows data preview, best model/metric, interactive forecast plot, and forecast table.

## Configuration notes
- Frequency: inferred when possible; provide `freq` explicitly for reliable multi-step forecasts.
- Exogenous handling: exogenous columns are lagged by default; the GUI can forecast exogs forward, otherwise they are held flat.
- Holdout strategy: `holdout_fraction` controls the split; `holdout_stepwise` evaluates one-step-ahead rolling predictions to avoid multi-step leakage.
- Scaling: enabled by default; disable via `scale_features=False` (API) or `--no-scale` (CLI).

## Optional dependencies
- LightGBM (`lightgbm`) for faster tree boosting.
- Prophet (`prophet`) for additive/multiplicative seasonality with holidays.
- SARIMA (`statsmodels`) for classic seasonal ARIMA.
- GARCH (`arch`) for volatility-aware autoregressive mean/variance modeling.
- Holidays (`holidays`) for country-specific holiday indicators.
- Torch stack for GRU; TFT is currently disabled in the catalog.

## Development
- Lint/format: not enforced; keep changes minimal and documented.
- Tests: not provided; validate locally with your data and CLI/GUI runs.
