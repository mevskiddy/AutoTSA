import argparse
import json
import sys
import os
import tempfile
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp())
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import AutoTSAConfig
from .search import AutoTSA


def parse_args(argv):
    parser = argparse.ArgumentParser(description="AutoTSA - AutoML for time series forecasting.")
    parser.add_argument("--data", required=True, help="Path to CSV file.")
    parser.add_argument("--timestamp", default=None, help="Timestamp column name (optional, inferred if omitted).")
    parser.add_argument("--target", required=True, help="Target column name.")
    parser.add_argument("--freq", default=None, help="Pandas frequency string.")
    parser.add_argument("--seasonal-period", type=int, default=None, help="Fixed seasonal period to enforce (e.g., 7, 12).")
    parser.add_argument("--models", default=None, help="Comma-separated model list.")
    parser.add_argument("--lags", default="1,2,3,6,12,24", help="Comma-separated lags.")
    parser.add_argument("--windows", default="3,6,12,24", help="Comma-separated rolling windows.")
    parser.add_argument("--exog", default=None, help="Comma-separated exogenous regressor columns.")
    parser.add_argument("--exog-lags", default="0,1", help="Comma-separated exogenous lags.")
    parser.add_argument("--holiday-country", default=None, help="Country code for holiday features (e.g., US, GB).")
    parser.add_argument("--trials", type=int, default=8, help="Trials per model for random search.")
    parser.add_argument("--splits", type=int, default=3, help="Rolling validation splits.")
    parser.add_argument("--holdout", type=float, default=0.2, help="Holdout fraction.")
    parser.add_argument("--metric", default="smape", help="Metric: smape|mape|rmse|mae")
    parser.add_argument("--jobs", type=int, default=2, help="Parallel jobs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--no-scale", action="store_true", help="Disable feature scaling.")
    parser.add_argument("--no-date-feats", action="store_true", help="Disable datetime features.")
    parser.add_argument("--no-auto-seasonality", action="store_true", help="Disable automatic seasonal lags/windows.")
    parser.add_argument("--use-optuna", action="store_true", help="Use Optuna for hyperparameter search if installed.")
    parser.add_argument("--optuna-trials", type=int, default=20, help="Trials per model for Optuna.")
    parser.add_argument("--holdout-stepwise", action="store_true", help="Evaluate holdout one-step rolling (no multi-step leakage).")
    parser.add_argument("--verbosity", type=int, default=1, help="Verbosity level.")
    parser.add_argument("--export-history", default=None, help="Path to write search history JSON.")
    parser.add_argument("--predict-next", action="store_true", help="Emit forecast for next step.")
    parser.add_argument("--plot-holdout", default=None, help="Path to save holdout actual vs predicted plot (png).")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    lags = tuple(int(x) for x in args.lags.split(",") if x)
    windows = tuple(int(x) for x in args.windows.split(",") if x)
    models = [m.strip() for m in args.models.split(",")] if args.models else None
    exog_cols = [c.strip() for c in args.exog.split(",")] if args.exog else None
    exog_lags = tuple(int(x) for x in args.exog_lags.split(",") if x)

    df = pd.read_csv(args.data)
    cfg = AutoTSAConfig(
        timestamp_col=args.timestamp,
        target_col=args.target,
        freq=args.freq,
        seasonal_period=args.seasonal_period,
        lags=lags,
        windows=windows,
        exogenous_cols=exog_cols,
        exogenous_lags=exog_lags,
        add_datetime_features=not args.no_date_feats,
        add_holiday_features=bool(args.holiday_country),
        holidays_country=args.holiday_country,
        holdout_fraction=args.holdout,
        auto_seasonality=not args.no_auto_seasonality,
        rolling_splits=args.splits,
        models=models,
        max_trials_per_model=args.trials,
        use_optuna=args.use_optuna,
        optuna_trials=args.optuna_trials,
        holdout_stepwise=args.holdout_stepwise,
        scoring=args.metric,
        n_jobs=args.jobs,
        random_seed=args.seed,
        scale_features=not args.no_scale,
        verbosity=args.verbosity,
    )
    automl = AutoTSA(cfg).fit(df)
    print(f"Best model: {automl.best_model_name} score={automl.best_score:.4f}")
    if args.predict_next:
        forecast = automl.predict_next(df)
        print(f"Next forecast: {forecast:.4f}")
    if args.export_history:
        with open(args.export_history, "w") as fp:
            json.dump(automl.history, fp, indent=2)
        print(f"Wrote history to {args.export_history}")
    if args.plot_holdout and automl.holdout_prediction_frame is not None:
        plot_path = os.path.expanduser(args.plot_holdout)
        fig, ax = plt.subplots(figsize=(10, 4))
        hp = automl.holdout_prediction_frame.sort_index()
        ax.plot(hp.index, hp["actual"], label="actual", linewidth=2)
        ax.plot(hp.index, hp["pred"], label="pred", linewidth=2, linestyle="--")
        ax.set_title(f"Holdout fit - {automl.best_model_name} ({automl.holdout_score:.4f} {cfg.scoring})")
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(plot_path)
        print(f"Holdout plot saved to {plot_path}")


if __name__ == "__main__":
    main()
