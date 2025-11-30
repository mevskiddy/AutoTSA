AutoTSA
=======

AutoTSA is an AutoML toolkit for univariate time-series forecasting with optional exogenous regressors. It builds lag/rolling features, searches a compact model zoo (tree ensembles, linear models, Gaussian Process; optional LightGBM, Prophet, SARIMA, GARCH, GRU), scores via rolling-origin validation, and returns the best model plus metrics. A Streamlit GUI and CLI are included for fast experimentation.

## Highlights
- Rolling-origin validation with configurable holdout (batch or stepwise) to reduce leakage.
- Automatic feature engineering: target lags, rolling statistics, datetime encodings, holiday indicators, and exogenous lags.
- Model zoo: Gradient Boosting, Random Forest, Ridge/ElasticNet, Gaussian Process; optional LightGBM, Prophet, SARIMA, GARCH, GRU (torch); TFT currently disabled.
- Deterministic searches with random seeds; Optuna hyperparameter tuning (optional).
- Exogenous forecasting in the GUI: hold-flat or project each exog forward via a small Gradient Boosting forecaster on its own lags/rolls.
- Exportable search history, holdout plots (CLI), cross-val series, and a modern Streamlit interface.

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

## Streamlit GUI (AutoTSA Studio)
```bash
streamlit run src/auto_tsa/gui.py
```
What it delivers
- Upload a CSV, AutoTSA suggests timestamp/target columns, and shows a preview of the first 20 rows.
- Choose horizon length, how much history to plot, and how many latest rows to use for training (focus on recent behavior).
- Toggle models (GBR, RF, Ridge/ElasticNet, LightGBM, Prophet, SARIMA, GARCH, GRU when deps available; TFT hidden without CUDA).
- Exogenous controls: pick regressors, optionally project each forward with a small GBR forecaster, or keep them flat.
- Frequency override box to guarantee regular spacing for multi-step forecasts.
- Add holiday indicators, enable or disable Optuna tuning (auto-disabled for TFT), set Optuna trials.
- Interactive charts: forecast line with history window, holdout plot toggle, rolling CV plot toggle.
- Download the trained pipeline as `.joblib` and view the forecast table inline.

Some GUI Screenshots

<img width="1030" height="570" alt="autotsa1" src="https://github.com/user-attachments/assets/a7fc0407-473d-4612-a0d4-b420087a097e" />
<img width="1025" height="559" alt="autotsa2" src="https://github.com/user-attachments/assets/f1c82c46-5d6a-4352-a135-42847f225258" />
<img width="1423" height="767" alt="autotsa3" src="https://github.com/user-attachments/assets/79c81dcc-6312-4a33-bc62-bf3debe7b2c9" />



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
- `--holdout-stepwise` enforces sequential one-step evaluation on the holdout slice to avoid multi-step leakage.

Common CLI recipes
- Minimal: `python -m auto_tsa.cli --data df.csv --target y --timestamp ds`
- Exogenous: add `--exog reg1,reg2 --exog-lags 0,1,2`
- Faster search: lower `--trials` and `--splits`; set `--models ridge,elasticnet`
- Traditional stats: install extras then use `--models sarima,prophet,garch`

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
next_step = automl.predict_next(df)
```
- Multi-step: ensure `freq` is set or inferred, then call `predict_next` repeatedly with the growing history (GUI automates this).
- Cross-val series for plotting: `cv_df, cv_score = automl.crossval_series(df)`

## Data and feature engineering
- Timestamp column must parse to datetime; target must be numeric (coerced to float with interpolation for gaps).
- Lags/rolling windows build autoregressive features; datetime encodings add calendar signals; holiday indicators require `holidays`.
- Exogenous columns are lagged; GUI can forecast each exog forward with Gradient Boosting to avoid flat-hold assumptions.
- Holdout uses rolling-origin; stepwise holdout predicts sequentially using realized history to reduce multi-step leakage.

## Model catalog
- Built-in: Gradient Boosting, Random Forest, Ridge, ElasticNet, Gaussian Process.
- Optional extras (install first): LightGBM, Prophet, SARIMA (statsmodels), GARCH (arch), GRU (torch stack). TFT is currently disabled.
- Deterministic seeds; Optuna tuning available for supported models.

## Configuration notes
- Frequency: inferred when possible; provide `freq` explicitly for reliable multi-step forecasts.
- Exogenous handling: exogenous columns are lagged by default; the GUI can forecast exogs forward, otherwise they are held flat.
- Holdout strategy: `holdout_fraction` controls the split; `holdout_stepwise` evaluates one-step-ahead rolling predictions.
- Scaling: enabled by default; disable via `scale_features=False` (API) or `--no-scale` (CLI).

## Development
- Lint/format: not enforced; keep changes minimal and documented.
- Tests: not provided; validate locally with your data and CLI/GUI runs.
