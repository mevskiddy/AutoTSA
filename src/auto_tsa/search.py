from __future__ import annotations
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import AutoTSAConfig
from .data import TimeSeriesFrame
from .evaluation import get_metric
from .features import FeatureBuilder
from .models import ModelSpec, get_model_specs

try:
    import optuna
except Exception:
    optuna = None
try:
    import statsmodels.api as sm
except Exception:
    sm = None
try:
    from arch import arch_model
except Exception:
    arch_model = None
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

try:
    import pytorch_lightning as pl
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.metrics import QuantileLoss
    from pytorch_forecasting.models import TemporalFusionTransformer as PFTFT
except Exception:
    pl = None
    TimeSeriesDataSet = None
    QuantileLoss = None
    PFTFT = None
# Disable MPS by default to avoid deadlocks on macOS; prefer CUDA or CPU
os.environ.setdefault("PL_DISABLE_MPS", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def _auto_seasonal_hints(freq: str | None) -> Tuple[List[int], List[int]]:
    if not freq:
        return [], []
    freq = freq.upper()
    if freq.startswith("H"):
        return [1, 6, 12, 24, 48, 168], [3, 6, 12, 24, 48, 168]
    if freq.startswith("D"):
        return [1, 7, 14, 28], [3, 7, 14, 28]
    if freq.startswith("W"):
        return [1, 4, 8, 26, 52], [2, 4, 8, 26, 52]
    if freq.startswith("M"):
        return [1, 6, 12, 24], [3, 6, 12, 24]
    if freq.startswith("Q"):
        return [1, 2, 4, 8], [2, 4, 8]
    return [], []


def _seasonal_period_from_freq(freq: str | None) -> int:
    if not freq:
        return 4  # sensible fallback for quarterly-like data
    freq = freq.upper()
    if freq.startswith("H"):
        return 24
    if freq.startswith("D"):
        return 7
    if freq.startswith("W"):
        return 52
    if freq.startswith("M"):
        return 12
    if freq.startswith("Q"):
        return 4
    return 1


def _seasonal_period_candidates(freq: str | None, override: Optional[int] = None) -> List[int]:
    if override is not None and override > 0:
        return [int(override)]
    base = _seasonal_period_from_freq(freq)
    candidates = {1, base}
    if not freq:
        return sorted([c for c in candidates if c > 0])
    freq = freq.upper()
    if freq.startswith("H"):
        candidates.update([6, 12, 24])
    elif freq.startswith("D"):
        candidates.update([7, 14, 30])
    elif freq.startswith("W"):
        candidates.update([13, 26, 52])
    elif freq.startswith("M"):
        candidates.update([6, 12, 24])
    elif freq.startswith("Q"):
        candidates.update([2, 4])
    return sorted([c for c in candidates if c > 0])


def _manual_seasonal_hints(period: Optional[int], max_len: int) -> Tuple[List[int], List[int]]:
    """
    Derive lag/window hints from a user-specified seasonal period.
    """
    if period is None:
        return [], []
    try:
        p = int(period)
    except (TypeError, ValueError):
        return [], []
    if p <= 0:
        return [], []
    multiples = [p, p * 2, p * 3]
    vals = [v for v in multiples if v < max_len]
    return vals, vals


def _garch_forecast(
    fit_res,
    horizon: int,
    power: float,
    reindex: bool = False,
    align: str = "origin",
):
    """
    arch's analytic forecast only supports power=2; fall back to simulation otherwise.
    """
    method = "analytic" if abs(power - 2.0) < 1e-8 else "simulation"
    kwargs = {"horizon": horizon, "reindex": reindex, "align": align, "method": method}
    if method == "simulation":
        kwargs["simulations"] = 300
    return fit_res.forecast(**kwargs)


class AutoTSA:
    """
    End-to-end AutoML for time series forecasting.
    """

    def __init__(self, config: AutoTSAConfig):
        self.config = config
        self.frame = TimeSeriesFrame(
            timestamp_col=config.timestamp_col,
            target_col=config.target_col,
            freq=config.freq,
        )
        self.feature_builder: FeatureBuilder | None = None
        self.best_model: Any = None
        self.best_score: float | None = None
        self.holdout_score: float | None = None
        self.best_model_name: str | None = None
        self.holdout_prediction_frame: pd.DataFrame | None = None
        self.history: List[Dict[str, Any]] = []
        self._best_params: Dict[str, Any] = {}
        self.cv_prediction_frame: pd.DataFrame | None = None
        self.cv_score: float | None = None

    def _log(self, message: str):
        if self.config.verbosity > 0:
            print(message)

    def _feature_builder(self) -> FeatureBuilder:
        exogenous_cols = self.config.exogenous_cols or []
        return FeatureBuilder(
            target_col=self.config.target_col,
            lags=self.config.lags,
            windows=self.config.windows,
            exogenous_cols=exogenous_cols,
            exogenous_lags=self.config.exogenous_lags,
            add_datetime_features=self.config.add_datetime_features,
            add_holiday_features=self.config.add_holiday_features,
            holidays_country=self.config.holidays_country,
            scale=self.config.scale_features,
        )

    def _max_history(self) -> int:
        return max(
            [1]
            + list(self.config.lags)
            + list(self.config.windows)
            + list(self.config.exogenous_lags)
        )

    def _transform_with_context(self, fb: FeatureBuilder, context_df, df):
        """
        Builds features for df using trailing context from context_df to preserve lagged features.
        """
        max_hist = self._max_history()
        context = context_df.tail(max_hist)
        val_with_ctx = pd.concat([context, df])
        feature_frame, y_series = fb._build_matrix(val_with_ctx)
        feature_frame = feature_frame[feature_frame.index.isin(df.index)]
        idx = feature_frame.index
        y_val = y_series.loc[idx].values
        if fb.scale and fb.scaler:
            X_val = fb.scaler.transform(feature_frame.values)
        else:
            X_val = feature_frame.values
        return X_val, y_val, idx

    def _stepwise_holdout_predictions(self, fb: FeatureBuilder, model, train_df, holdout_df):
        """
        Predict holdout sequentially using realized history (no multi-step leakage).
        """
        context = train_df.copy()
        preds = []
        idxs = []
        for ts, row in holdout_df.iterrows():
            current = pd.DataFrame([row], index=[ts])
            # Remove current target to avoid leakage; use historical target only
            temp = pd.concat([context, current.drop(columns=[self.config.target_col])])
            temp[self.config.target_col] = pd.concat(
                [context[self.config.target_col], pd.Series([np.nan], index=[ts])]
            )
            feature_frame, y_series = fb._build_matrix(temp)
            if ts not in feature_frame.index:
                continue
            features_row = feature_frame.loc[[ts]].values
            if fb.scale and fb.scaler:
                features_row = fb.scaler.transform(features_row)
            pred = model.predict(features_row)[0]
            preds.append(pred)
            idxs.append(ts)
            # update context with actual observation for next step
            context = pd.concat([context, current])
        return np.array(preds), idxs

    def _evaluate_prophet(
        self, params: Dict[str, Any], df: Any, metric_name: str
    ) -> Tuple[float, Dict[str, Any]]:
        from prophet import Prophet

        metric = get_metric(metric_name)
        scores = []
        self._log(f"  Prophet trial params: {params}")
        for train_df, val_df in self.frame.rolling_origin(
            df, splits=self.config.rolling_splits, min_train_fraction=self.config.min_train_fraction
        ):
            model = Prophet(**params)
            exogenous_cols = self.config.exogenous_cols or []
            for ex_col in exogenous_cols:
                model.add_regressor(ex_col)
            if self.config.holidays_country:
                # Prophet pulls holidays automatically if country is set
                model.add_country_holidays(country_name=self.config.holidays_country)
            train = train_df.reset_index().rename(columns={self.frame.timestamp_col: "ds", self.config.target_col: "y"})
            model.fit(train)
            val = val_df.reset_index().rename(columns={self.frame.timestamp_col: "ds"})
            forecast = model.predict(val)
            preds = forecast["yhat"].values
            y_true = val_df[self.config.target_col].values
            scores.append(metric(y_true, preds))
        mean_score = float(np.mean(scores))
        result = {"model": "prophet", "params": params, "score": mean_score}
        self._log(f"  -> score {mean_score:.4f} ({metric_name})")
        return mean_score, result

    def crossval_series(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, float | None]:
        """
        Refit the best-found model across rolling-origin splits and collect
        validation predictions as a continuous time series.
        Only supports scikit-learn style regressors; returns (frame, score).
        """
        if self.best_model_name is None or not self._best_params:
            raise RuntimeError("Best model not available; fit AutoTSA first.")
        data = self.frame.load(df)
        specs = {spec.name: spec for spec in get_model_specs(self.config.model_list())}
        if self.best_model_name not in specs:
            raise RuntimeError(f"Best model spec {self.best_model_name} not found.")
        spec = specs[self.best_model_name]
        if spec.kind not in ("sklearn", "lightgbm"):
            raise RuntimeError(f"Cross-val plotting not supported for model kind {spec.kind}.")

        metric_fn = get_metric(self.config.scoring)
        rows: list[Dict[str, Any]] = []
        all_preds: list[float] = []
        all_actuals: list[float] = []
        for split_idx, (train_df, val_df) in enumerate(
            self.frame.rolling_origin(
                data, splits=self.config.rolling_splits, min_train_fraction=self.config.min_train_fraction
            )
        ):
            if val_df.empty or train_df.empty:
                continue
            fb = self._feature_builder()
            X_train, y_train = fb.fit_transform(train_df)
            model = spec.builder(self._best_params)
            model.fit(X_train, y_train)
            if self.config.holdout_stepwise:
                preds, idxs = self._stepwise_holdout_predictions(fb, model, train_df, val_df)
                y_val = val_df.loc[idxs, self.config.target_col].values
                ts_idx = idxs
            else:
                X_val, y_val, ts_idx = self._transform_with_context(fb, train_df, val_df)
                preds = model.predict(X_val)
            for ts, actual, pred in zip(ts_idx, y_val, preds):
                rows.append(
                    {
                        self.frame.timestamp_col: ts,
                        "actual": float(actual),
                        "pred": float(pred),
                        "split": split_idx,
                    }
                )
            all_actuals.extend(list(y_val))
            all_preds.extend([float(p) for p in preds])

        if not rows:
            raise RuntimeError("No validation rows produced during cross-validation.")
        frame = pd.DataFrame(rows).set_index(self.frame.timestamp_col).sort_index()
        score = metric_fn(np.array(all_actuals), np.array(all_preds)) if all_preds else None
        self.cv_prediction_frame = frame
        self.cv_score = float(score) if score is not None else None
        return frame, self.cv_score

    def _to_nf_frame(self, frame):
        return (
            frame.reset_index()
            .rename(columns={self.frame.timestamp_col: "ds", self.config.target_col: "y"})
            .assign(unique_id="ts")
        )

    # --- Simple GRU forecaster (CPU-friendly) ---
    def _gru_sequences(self, df: pd.DataFrame, window: int):
        target = df[self.config.target_col].astype("float32").values
        exog_cols = self.config.exogenous_cols or []
        exog = df[exog_cols].astype("float32").values if exog_cols else None
        X_list: list[np.ndarray] = []
        y_list: list[float] = []
        for i in range(window, len(df)):
            t_slice = target[i - window : i]
            if exog is not None:
                ex_slice = exog[i - window : i]
                seq = np.concatenate([t_slice[:, None], ex_slice], axis=1)
            else:
                seq = t_slice[:, None]
            X_list.append(seq)
            y_list.append(target[i])
        if not X_list:
            return None, None
        X = torch.tensor(np.stack(X_list), dtype=torch.float32)
        y = torch.tensor(np.array(y_list), dtype=torch.float32).unsqueeze(1)
        return X, y

    def _gru_train_model(self, train_df: pd.DataFrame, params: Dict[str, Any]):
        if torch is None or nn is None or DataLoader is None:
            raise RuntimeError("PyTorch not available for GRU model.")
        window = int(params.get("window", 16))
        hidden = int(params.get("hidden_size", 32))
        epochs = int(params.get("epochs", 15))
        lr = float(params.get("lr", 1e-3))
        X, y = self._gru_sequences(train_df, window)
        if X is None or y is None:
            raise RuntimeError("Not enough data to train GRU model.")
        target_mean = y.mean().item()
        target_std = y.std().item() or 1.0
        y_norm = (y - target_mean) / target_std
        dataset = TensorDataset(X, y_norm)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        input_size = X.shape[2]

        class GRURegressor(nn.Module):
            def __init__(self, inp, hidden_dim):
                super().__init__()
                self.gru = nn.GRU(inp, hidden_dim, batch_first=True)
                self.head = nn.Linear(hidden_dim, 1)

            def forward(self, x):
                out, _ = self.gru(x)
                return self.head(out[:, -1, :])

        model = GRURegressor(input_size, hidden)
        optim = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                optim.zero_grad()
                preds = model(xb)
                loss = loss_fn(preds, yb)
                loss.backward()
                optim.step()
        return {"model": model, "window": window, "mean": target_mean, "std": target_std}

    def _gru_forecast(self, bundle, history_df: pd.DataFrame, fut_df: pd.DataFrame):
        model = bundle["model"]
        window = bundle["window"]
        target_mean = bundle["mean"]
        target_std = bundle["std"]
        exog_cols = self.config.exogenous_cols or []

        hist = history_df[[self.config.target_col] + exog_cols].astype("float32").copy()
        if len(hist) < window:
            return np.array([])
        target_hist = hist[self.config.target_col].values
        exog_hist = hist[exog_cols].values if exog_cols else None

        def last_window_targets():
            t_slice = target_hist[-window:]
            if exog_cols:
                ex_slice = exog_hist[-window:]
                return np.concatenate([t_slice[:, None], ex_slice], axis=1)
            return t_slice[:, None]

        preds = []
        model.eval()
        with torch.no_grad():
            win = last_window_targets()
            for _, row in fut_df.iterrows():
                ex_future = row[exog_cols].values.astype("float32") if exog_cols else None
                seq = torch.tensor(win[None, :, :], dtype=torch.float32)
                pred_norm = model(seq).squeeze().item()
                pred = pred_norm * target_std + target_mean
                preds.append(pred)
                next_row = [pred] + (ex_future.tolist() if ex_future is not None else [])
                win = np.vstack([win[1:], np.array(next_row, dtype="float32")])
        return np.array(preds)

    def _tft_device(self) -> Tuple[str, int]:
        # Prefer CUDA; otherwise CPU
        if torch is not None and torch.cuda.is_available() and getattr(torch.version, "cuda", None):
            return "gpu", 1
        return "cpu", 1

    def _tft_dataloaders(self, train_df, val_df, params, horizon: int):
        if TimeSeriesDataSet is None:
            raise RuntimeError("pytorch-forecasting not available.")
        ts_col = self.frame.timestamp_col or "ds"
        target_col = self.config.target_col
        exog_cols = self.config.exogenous_cols or []
        df_train = train_df.reset_index().rename(columns={self.frame.timestamp_col: ts_col})
        df_val = val_df.reset_index().rename(columns={self.frame.timestamp_col: ts_col})
        for df_ in (df_train, df_val):
            df_["series_id"] = "ts"
        static_cats = ["series_id"]
        time_varying_known_reals = exog_cols + [ts_col]
        time_varying_unknown_reals = [target_col]
        training = TimeSeriesDataSet(
            df_train,
            time_idx=ts_col,
            target=target_col,
            group_ids=["series_id"],
            min_encoder_length=int(params.get("encoder_length", 24)),
            max_encoder_length=int(params.get("encoder_length", 24)),
            min_prediction_length=horizon,
            max_prediction_length=horizon,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            static_categorical_features=static_cats,
        )
        training.loaders = training.to_dataloader(train=True, batch_size=64, shuffle=True)
        validation = training.from_parameters(training.get_parameters(), data=df_val)
        return training.to_dataloader(train=True, batch_size=64, shuffle=True), validation.to_dataloader(train=False, batch_size=64), training

    def _build_tft_model(self, params: Dict[str, Any], horizon: int, input_cap: int, alias: str = "tft"):
        accelerator, devices = self._tft_device()
        kwargs = {
            "h": horizon,
            "input_size": min(input_cap, params.get("input_size", 24)),
            "hidden_size": max(4, int(params.get("hidden_size", 32))),
            "attention_head_size": int(params.get("attention_head_size", 1)),
            "dropout": params.get("dropout", 0.1),
            "max_epochs": int(params.get("max_epochs", 20)),
            "encoder_length": int(params.get("encoder_length", 24)),
            "alias": alias,
        }
        try:
            model = PFTFT.from_dataset(
                kwargs.pop("dataset"),  # injected in caller
                hidden_size=kwargs["hidden_size"],
                attention_head_size=kwargs["attention_head_size"],
                dropout=kwargs["dropout"],
                learning_rate=1e-3,
                log_interval=-1,
                loss=QuantileLoss([0.5]),
            )
            return model
        except TypeError:
            raise

    def _evaluate_tft(self, params: Dict[str, Any], df: Any, metric_name: str) -> Tuple[float, Dict[str, Any]]:
        if pl is None or PFTFT is None or TimeSeriesDataSet is None:
            self._log("  TFT skipped: pytorch-forecasting not available.")
            return float("inf"), {"model": "tft", "params": params, "score": float("inf")}
        metric = get_metric(metric_name)
        scores = []
        self._log(f"  TFT (pf) trial params: {params}")
        for train_df, val_df in self.frame.rolling_origin(
            df, splits=self.config.rolling_splits, min_train_fraction=self.config.min_train_fraction
        ):
            h = len(val_df)
            if h == 0 or len(train_df) <= 1:
                continue
            try:
                train_loader, pred_loader, dataset = self._tft_dataloaders(train_df, val_df, params, horizon=h)
                model = self._build_tft_model({**params, "dataset": dataset}, horizon=h, input_cap=len(train_df), alias="tft")
                trainer = pl.Trainer(
                    max_epochs=int(params.get("max_epochs", 20)),
                    accelerator="cpu",
                    devices=1,
                    logger=False,
                    enable_checkpointing=False,
                    enable_model_summary=False,
                    enable_progress_bar=False,
                )
                trainer.fit(model, train_loader)
                raw_preds = trainer.predict(model, pred_loader, mode="prediction")
                preds = torch.cat(raw_preds, dim=0).reshape(-1, h)[:, -h:].mean(dim=0).detach().cpu().numpy()
                scores.append(metric(val_df[self.config.target_col].values, preds))
            except Exception as exc:
                self._log(f"  TFT trial error: {exc}")
                continue
        if not scores:
            return float("inf"), {"model": "tft", "params": params, "score": float("inf")}
        mean_score = float(np.mean(scores))
        result = {"model": "tft", "params": params, "score": mean_score}
        self._log(f"  -> score {mean_score:.4f} ({metric_name})")
        return mean_score, result

    def _evaluate_gru(self, params: Dict[str, Any], df: Any, metric_name: str) -> Tuple[float, Dict[str, Any]]:
        if torch is None or nn is None or DataLoader is None:
            self._log("  GRU skipped: torch not available.")
            return float("inf"), {"model": "gru", "params": params, "score": float("inf")}
        metric = get_metric(metric_name)
        scores = []
        self._log(f"  GRU trial params: {params}")
        for train_df, val_df in self.frame.rolling_origin(
            df, splits=self.config.rolling_splits, min_train_fraction=self.config.min_train_fraction
        ):
            if len(train_df) < 5 or len(val_df) == 0:
                continue
            try:
                bundle = self._gru_train_model(train_df, params)
                preds = self._gru_forecast(bundle, train_df, val_df)
                if len(preds) != len(val_df):
                    continue
                scores.append(metric(val_df[self.config.target_col].values, preds))
            except Exception as exc:
                self._log(f"  GRU trial error: {exc}")
                continue
        if not scores:
            return float("inf"), {"model": "gru", "params": params, "score": float("inf")}
        mean_score = float(np.mean(scores))
        result = {"model": "gru", "params": params, "score": mean_score}
        self._log(f"  -> score {mean_score:.4f} ({metric_name})")
        return mean_score, result

    def _evaluate_sarima(
        self, params: Dict[str, Any], df: Any, metric_name: str
    ) -> Tuple[float, Dict[str, Any]]:
        if sm is None:
            raise RuntimeError("statsmodels not available for SARIMA")
        metric = get_metric(metric_name)
        seasonals = _seasonal_period_candidates(self.frame.freq, self.config.seasonal_period)
        self._log(f"  SARIMA trial params: {params} seasons={seasonals}")
        best_mean = float("inf")
        best_period = None
        for s in seasonals:
            scores = []
            for train_df, val_df in self.frame.rolling_origin(
                df, splits=self.config.rolling_splits, min_train_fraction=self.config.min_train_fraction
            ):
                order = (params["p"], params["d"], params["q"])
                seasonal_order = (params["P"], params["D"], params["Q"], s)
                try:
                    if self.frame.freq:
                        train_df = train_df.asfreq(self.frame.freq)
                        val_df = val_df.asfreq(self.frame.freq)
                    exog_train = train_df[self.config.exogenous_cols] if self.config.exogenous_cols else None
                    exog_val = val_df[self.config.exogenous_cols] if self.config.exogenous_cols else None
                    model = sm.tsa.statespace.SARIMAX(
                        train_df[self.config.target_col],
                        order=order,
                        seasonal_order=seasonal_order,
                        trend=params.get("trend", "n"),
                        exog=exog_train,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    fit_res = model.fit(disp=False, maxiter=50)
                    preds = fit_res.get_forecast(steps=len(val_df), exog=exog_val).predicted_mean
                    score_val = metric(val_df[self.config.target_col].values, preds)
                    if np.isnan(score_val):
                        score_val = np.inf
                    scores.append(score_val)
                except Exception:
                    scores.append(np.inf)
            if scores:
                mean_score = float(np.mean(scores))
                if mean_score < best_mean:
                    best_mean = mean_score
                    best_period = s
        if best_period is None or np.isinf(best_mean):
            return float("inf"), {"model": "sarima", "params": params, "score": float("inf")}
        params_with_season = {**params, "seasonal_period": best_period}
        result = {"model": "sarima", "params": params_with_season, "score": best_mean}
        self._log(f"  -> score {best_mean:.4f} ({metric_name})")
        return best_mean, result

    def _evaluate_garch(
        self, params: Dict[str, Any], df: Any, metric_name: str
    ) -> Tuple[float, Dict[str, Any]]:
        if arch_model is None:
            self._log("  GARCH skipped: arch not available.")
            return float("inf"), {"model": "garch", "params": params, "score": float("inf")}
        if self.config.exogenous_cols:
            self._log("  GARCH ignores exogenous regressors (not supported).")
        metric = get_metric(metric_name)
        scores = []
        self._log(f"  GARCH trial params: {params}")
        for train_df, val_df in self.frame.rolling_origin(
            df, splits=self.config.rolling_splits, min_train_fraction=self.config.min_train_fraction
        ):
            if len(train_df) < max(params.get("p", 1), params.get("q", 1)) + 2 or len(val_df) == 0:
                continue
            try:
                y_train = train_df[self.config.target_col].values
                mean_model = params.get("mean", "Constant")
                lags_val = int(params.get("lags", 0))
                if str(mean_model).upper() == "ARX" and lags_val <= 0:
                    lags_val = 1
                lags_param = lags_val if str(mean_model).upper() == "ARX" else None
                model = arch_model(
                    y_train,
                    mean=mean_model,
                    vol="GARCH",
                    p=int(params.get("p", 1)),
                    q=int(params.get("q", 1)),
                    power=float(params.get("power", 2.0)),
                    dist=params.get("dist", "normal"),
                    lags=lags_param,
                )
                fit_res = model.fit(disp="off", show_warning=False)
                power_val = float(params.get("power", 2.0))
                forecast = _garch_forecast(fit_res, horizon=len(val_df), power=power_val, reindex=False, align="origin")
                preds = np.asarray(forecast.mean.iloc[-1]).reshape(-1)
                if len(preds) != len(val_df):
                    continue
                score_val = metric(val_df[self.config.target_col].values, preds)
                if np.isnan(score_val):
                    score_val = np.inf
                scores.append(score_val)
            except Exception as exc:
                self._log(f"  GARCH trial error: {exc}")
                continue
        if not scores:
            return float("inf"), {"model": "garch", "params": params, "score": float("inf")}
        mean_score = float(np.mean(scores))
        result = {"model": "garch", "params": params, "score": mean_score}
        self._log(f"  -> score {mean_score:.4f} ({metric_name})")
        return mean_score, result

    def _evaluate_trial(
        self,
        model_spec: ModelSpec,
        params: Dict[str, Any],
        df: Any,
        metric_name: str,
    ) -> Tuple[float, Dict[str, Any]]:
        if model_spec.kind == "prophet":
            return self._evaluate_prophet(params, df, metric_name)
        if model_spec.kind == "sarima":
            return self._evaluate_sarima(params, df, metric_name)
        if model_spec.kind == "garch":
            return self._evaluate_garch(params, df, metric_name)
        if model_spec.kind == "tft":
            return self._evaluate_tft(params, df, metric_name)
        if model_spec.kind == "gru":
            return self._evaluate_gru(params, df, metric_name)

        metric = get_metric(metric_name)
        scores = []
        self._log(f"  Trial params: {params}")
        for train_df, val_df in self.frame.rolling_origin(
            df, splits=self.config.rolling_splits, min_train_fraction=self.config.min_train_fraction
        ):
            fb = self._feature_builder()
            try:
                X_train, y_train = fb.fit_transform(train_df)
                X_val, y_val, _ = self._transform_with_context(fb, train_df, val_df)
            except ValueError:
                continue
            if len(X_val) == 0 or len(X_train) == 0 or len(y_train) == 0 or len(y_val) == 0:
                continue
            model = model_spec.builder(params)
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                scores.append(metric(y_val, preds))
            except Exception:
                continue
        if not scores:
            return float("inf"), {"model": model_spec.name, "params": params, "score": float("inf")}
        mean_score = float(np.mean(scores))
        result = {"model": model_spec.name, "params": params, "score": mean_score}
        self._log(f"  -> score {mean_score:.4f} ({metric_name})")
        return mean_score, result

    def _suggest_params_optuna(self, trial, param_space: Dict[str, Any]) -> Dict[str, Any]:
        params = {}
        for key, space in param_space.items():
            if isinstance(space, tuple) and len(space) == 2 and all(isinstance(v, (int, float)) for v in space):
                low, high = space
                if isinstance(low, int) and isinstance(high, int):
                    params[key] = trial.suggest_int(key, low, high)
                else:
                    params[key] = trial.suggest_float(key, low, high)
            elif isinstance(space, list):
                params[key] = trial.suggest_categorical(key, space)
        return params

    def _run_trials(
        self, specs: List[ModelSpec], train_df: Any, metric_name: str
    ) -> List[Tuple[float, Dict[str, Any]]]:
        results: List[Tuple[float, Dict[str, Any]]] = []
        rng = random.Random(self.config.random_seed)
        if self.config.use_optuna and optuna is not None:
            for spec in specs:
                self._log(f"Running Optuna for {spec.name} ({self.config.optuna_trials} trials)")

                def objective(trial):
                    params = spec.default_params.copy()
                    params.update(self._suggest_params_optuna(trial, spec.param_space))
                    score, _ = self._evaluate_trial(spec, params, train_df, metric_name)
                    if np.isnan(score):
                        return float("inf")
                    return score

                study = optuna.create_study(direction="minimize")
                study.optimize(objective, n_trials=self.config.optuna_trials, n_jobs=self.config.n_jobs)
                if not study.best_trials:
                    self._log(f"No successful Optuna trials for {spec.name}; skipping.")
                    continue
                best_params = spec.default_params.copy()
                best_params.update(study.best_trial.params)
                best_score, result = self._evaluate_trial(spec, best_params, train_df, metric_name)
                results.append((best_score, result))
        elif self.config.use_optuna and optuna is None:
            self._log("Optuna not installed; falling back to random search.")
        else:
            trials: List[Tuple[ModelSpec, Dict[str, Any]]] = []
            for spec in specs:
                for _ in range(self.config.max_trials_per_model):
                    trials.append((spec, spec.sample_params(rng)))
            self._log(f"Prepared {len(trials)} trials across models {self.config.model_list()}")
            if self.config.n_jobs > 1:
                with ThreadPoolExecutor(max_workers=self.config.n_jobs) as executor:
                    future_map = {
                        executor.submit(self._evaluate_trial, spec, params, train_df, metric_name): (spec, params)
                        for spec, params in trials
                    }
                    for future in as_completed(future_map):
                        score, result = future.result()
                        results.append((score, result))
            else:
                for spec, params in trials:
                    score, result = self._evaluate_trial(spec, params, train_df, metric_name)
                    results.append((score, result))
        return results

    def fit(self, df) -> "AutoTSA":
        data = self.frame.load(df)
        max_len = len(data)
        manual_lags, manual_windows = _manual_seasonal_hints(self.config.seasonal_period, max_len)
        if manual_lags or manual_windows:
            lags_hint, windows_hint = manual_lags, manual_windows
        elif self.config.auto_seasonality:
            lags_hint, windows_hint = _auto_seasonal_hints(self.frame.freq)
        else:
            lags_hint, windows_hint = [], []
        if lags_hint or windows_hint:
            self.config.lags = tuple(
                sorted(set([l for l in list(self.config.lags) + lags_hint if l < max_len]))
            )
            self.config.windows = tuple(
                sorted(set([w for w in list(self.config.windows) + windows_hint if w < max_len]))
            )
        train_df, holdout_df = self.frame.train_validation_split(data, self.config.holdout_fraction)
        metric_name = self.config.scoring
        specs = get_model_specs(self.config.model_list())

        results = self._run_trials(specs, train_df, metric_name)
        results = [r for r in results if not (isinstance(r[0], float) and np.isnan(r[0]))]
        if not results:
            raise RuntimeError("No successful trials; try reducing lags/windows or increasing data length.")
        results.sort(key=lambda pair: pair[0])
        self.history = [r for _, r in results]
        best_score, best_result = results[0]
        best_spec = [spec for spec in specs if spec.name == best_result["model"]][0]
        self.best_model_name = best_spec.name
        self.best_score = best_score
        self._best_params = best_result["params"]

        if best_spec.kind == "prophet":
            from prophet import Prophet

            # Fit on train only for holdout assessment
            model = Prophet(**best_result["params"])
            exogenous_cols = self.config.exogenous_cols or []
            for ex_col in exogenous_cols:
                model.add_regressor(ex_col)
            if self.config.holidays_country:
                model.add_country_holidays(country_name=self.config.holidays_country)
            train_only = train_df.reset_index().rename(
                columns={self.frame.timestamp_col: "ds", self.config.target_col: "y"}
            )
            model.fit(train_only)
            holdout = holdout_df.reset_index().rename(columns={self.frame.timestamp_col: "ds"})
            preds = model.predict(holdout)["yhat"].values
            metric = get_metric(metric_name)
            self.holdout_score = metric(holdout_df[self.config.target_col].values, preds)
            self.holdout_prediction_frame = pd.DataFrame(
                {"actual": holdout_df[self.config.target_col].values, "pred": preds},
                index=holdout_df.index,
            )

            # Refit on full data for final model
            model_full = Prophet(**best_result["params"])
            for ex_col in exogenous_cols:
                model_full.add_regressor(ex_col)
            if self.config.holidays_country:
                model_full.add_country_holidays(country_name=self.config.holidays_country)
            train_full = data.reset_index().rename(
                columns={self.frame.timestamp_col: "ds", self.config.target_col: "y"}
            )
            model_full.fit(train_full)
            self.best_model = model_full
            self.feature_builder = None
            self._log(f"Best model {self.best_model_name} score={best_score:.4f} holdout={self.holdout_score:.4f}")
            return self
        if best_spec.kind == "sarima":
            s = self.config.seasonal_period or _seasonal_period_from_freq(self.frame.freq)
            order = (
                best_result["params"]["p"],
                best_result["params"]["d"],
                best_result["params"]["q"],
            )
            seasonal_order = (
                best_result["params"]["P"],
                best_result["params"]["D"],
                best_result["params"]["Q"],
                best_result["params"].get("seasonal_period", s),
            )
            if self.frame.freq:
                train_df = train_df.asfreq(self.frame.freq)
                holdout_df = holdout_df.asfreq(self.frame.freq)
                data = data.asfreq(self.frame.freq)
            exog_train = train_df[self.config.exogenous_cols] if self.config.exogenous_cols else None
            exog_holdout = holdout_df[self.config.exogenous_cols] if self.config.exogenous_cols else None
            model = sm.tsa.statespace.SARIMAX(
                train_df[self.config.target_col],
                order=order,
                seasonal_order=seasonal_order,
                trend=best_result["params"].get("trend", "n"),
                exog=exog_train,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fit_res = model.fit(disp=False, maxiter=50)
            preds = fit_res.get_forecast(steps=len(holdout_df), exog=exog_holdout).predicted_mean
            metric = get_metric(metric_name)
            self.holdout_score = metric(holdout_df[self.config.target_col].values, preds)
            self.holdout_prediction_frame = pd.DataFrame(
                {"actual": holdout_df[self.config.target_col].values, "pred": np.array(preds)},
                index=holdout_df.index,
            )
            # Refit on full data
            exog_full = data[self.config.exogenous_cols] if self.config.exogenous_cols else None
            model_full = sm.tsa.statespace.SARIMAX(
                data[self.config.target_col],
                order=order,
                seasonal_order=seasonal_order,
                trend=best_result["params"].get("trend", "n"),
                exog=exog_full,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self.best_model = model_full.fit(disp=False, maxiter=50)
            self.feature_builder = None
            self._log(f"Best model {self.best_model_name} score={best_score:.4f} holdout={self.holdout_score:.4f}")
            return self
        if best_spec.kind == "garch":
            if arch_model is None:
                raise RuntimeError("arch not available for GARCH.")
            params = best_result["params"]
            train_y = train_df[self.config.target_col].values
            holdout_y = holdout_df[self.config.target_col].values
            mean_model = params.get("mean", "Constant")
            lags_val = int(params.get("lags", 0))
            if str(mean_model).upper() == "ARX" and lags_val <= 0:
                lags_val = 1
            lags_param = lags_val if str(mean_model).upper() == "ARX" else None
            model = arch_model(
                train_y,
                mean=mean_model,
                vol="GARCH",
                p=int(params.get("p", 1)),
                q=int(params.get("q", 1)),
                power=float(params.get("power", 2.0)),
                dist=params.get("dist", "normal"),
                lags=lags_param,
            )
            fit_res = model.fit(disp="off", show_warning=False)
            power_val = float(params.get("power", 2.0))
            forecast = _garch_forecast(
                fit_res, horizon=len(holdout_df), power=power_val, reindex=False, align="origin"
            )
            preds = np.asarray(forecast.mean.iloc[-1]).reshape(-1)
            metric = get_metric(metric_name)
            if len(preds) == len(holdout_df):
                self.holdout_score = metric(holdout_y, preds)
                self.holdout_prediction_frame = pd.DataFrame(
                    {"actual": holdout_y, "pred": preds}, index=holdout_df.index
                )
            else:
                self.holdout_score = float("nan")
                self.holdout_prediction_frame = None
            lags_param_full = lags_param if str(mean_model).upper() == "ARX" else None
            model_full = arch_model(
                data[self.config.target_col].values,
                mean=mean_model,
                vol="GARCH",
                p=int(params.get("p", 1)),
                q=int(params.get("q", 1)),
                power=float(params.get("power", 2.0)),
                dist=params.get("dist", "normal"),
                lags=lags_param_full,
            )
            self.best_model = model_full.fit(disp="off", show_warning=False)
            self.feature_builder = None
            self._best_params = params
            self._log(f"Best model {self.best_model_name} score={best_score:.4f} holdout={self.holdout_score:.4f}")
            return self
        if best_spec.kind == "tft":
            if pl is None or PFTFT is None or TimeSeriesDataSet is None:
                raise RuntimeError("pytorch-forecasting not available for TFT.")
            h = len(holdout_df)
            params = best_result["params"]
            try:
                train_loader, pred_loader, dataset = self._tft_dataloaders(train_df, holdout_df, params, horizon=h)
                model = self._build_tft_model({**params, "dataset": dataset}, horizon=h, input_cap=len(train_df), alias="tft")
                trainer = pl.Trainer(
                    max_epochs=int(params.get("max_epochs", 20)),
                    accelerator="cpu",
                    devices=1,
                    logger=False,
                    enable_checkpointing=False,
                    enable_model_summary=False,
                    enable_progress_bar=False,
                )
                trainer.fit(model, train_loader)
                raw_preds = trainer.predict(model, pred_loader, mode="prediction")
                preds = torch.cat(raw_preds, dim=0).reshape(-1, h)[:, -h:].mean(dim=0).detach().cpu().numpy()
                metric = get_metric(metric_name)
                self.holdout_score = metric(holdout_df[self.config.target_col].values, preds)
                self.holdout_prediction_frame = pd.DataFrame(
                    {"actual": holdout_df[self.config.target_col].values, "pred": preds},
                    index=holdout_df.index,
                )
                # Refit for deployment with horizon=1 on full data
                train_loader_full, pred_loader_full, dataset_full = self._tft_dataloaders(data, data.iloc[-1:], params, horizon=1)
                model_full = self._build_tft_model({**params, "dataset": dataset_full}, horizon=1, input_cap=len(data), alias="tft")
                trainer_full = pl.Trainer(
                    max_epochs=int(params.get("max_epochs", 20)),
                    accelerator="cpu",
                    devices=1,
                    logger=False,
                    enable_checkpointing=False,
                    enable_model_summary=False,
                    enable_progress_bar=False,
                )
                trainer_full.fit(model_full, train_loader_full)
                self.best_model = model_full
                self._tft_dataset = dataset_full
                self._tft_trainer = trainer_full
                self.feature_builder = None
                self._best_params = params
                self._log(f"Best model {self.best_model_name} score={best_score:.4f} holdout={self.holdout_score:.4f}")
                return self
            except Exception as exc:
                raise RuntimeError(f"TFT holdout fit failed: {exc}") from exc
        if best_spec.kind == "gru":
            bundle = self._gru_train_model(train_df, best_result["params"])
            preds = self._gru_forecast(bundle, train_df, holdout_df)
            metric = get_metric(metric_name)
            if len(preds) != len(holdout_df):
                self.holdout_score = float("nan")
                self.holdout_prediction_frame = None
            else:
                self.holdout_score = metric(holdout_df[self.config.target_col].values, preds)
                self.holdout_prediction_frame = pd.DataFrame(
                    {"actual": holdout_df[self.config.target_col].values, "pred": preds},
                    index=holdout_df.index,
                )
            # Refit on full data for deployment
            self.best_model = self._gru_train_model(data, best_result["params"])
            self.feature_builder = None
            self._best_params = best_result["params"]
            self._log(f"Best model {self.best_model_name} score={best_score:.4f} holdout={self.holdout_score:.4f}")
            return self

        # Fit on train only for holdout assessment to avoid leakage
        fb_train = self._feature_builder()
        X_train, y_train = fb_train.fit_transform(train_df)
        model = best_spec.builder(best_result["params"])
        model.fit(X_train, y_train)
        try:
            # Always evaluate holdout stepwise to avoid target leakage
            holdout_preds, holdout_idx = self._stepwise_holdout_predictions(fb_train, model, train_df, holdout_df)
            y_holdout = holdout_df.loc[holdout_idx, self.config.target_col].values
            metric = get_metric(metric_name)
            self.holdout_score = metric(y_holdout, holdout_preds)
            self.holdout_prediction_frame = pd.DataFrame(
                {"actual": y_holdout, "pred": holdout_preds}, index=holdout_idx
            )
        except ValueError:
            self.holdout_score = float("nan")

        # Refit on full data for final deployment model
        fb_full = self._feature_builder()
        X_full, y_full = fb_full.fit_transform(data)
        self.feature_builder = fb_full
        self.best_model = best_spec.builder(best_result["params"])
        self.best_model.fit(X_full, y_full)

        self._log(f"Best model {self.best_model_name} score={best_score:.4f} holdout={self.holdout_score:.4f}")
        return self

    def predict(self, df) -> np.ndarray:
        if self.best_model is None:
            raise RuntimeError("Model not fit; call fit() first.")
        data = self.frame.load(df)
        if self.best_model_name == "prophet":
            frame = data.reset_index().rename(columns={self.frame.timestamp_col: "ds"})
            preds = self.best_model.predict(frame)["yhat"].values
            return preds
        if self.best_model_name == "sarima":
            frame = data
            exog = None
            if self.config.exogenous_cols:
                exog = frame[self.config.exogenous_cols]
            return self.best_model.forecast(steps=len(frame), exog=exog)
        if self.best_model_name == "garch":
            if arch_model is None:
                raise RuntimeError("arch not available for prediction.")
            params = self._best_params
            y = data[self.config.target_col].values
            mean_model = params.get("mean", "Constant")
            lags_val = int(params.get("lags", 0))
            if str(mean_model).upper() == "ARX" and lags_val <= 0:
                lags_val = 1
            lags_param = lags_val if str(mean_model).upper() == "ARX" else None
            model = arch_model(
                y,
                mean=mean_model,
                vol="GARCH",
                p=int(params.get("p", 1)),
                q=int(params.get("q", 1)),
                power=float(params.get("power", 2.0)),
                dist=params.get("dist", "normal"),
                lags=lags_param,
            )
            fit_res = model.fit(disp="off", show_warning=False)
            power_val = float(params.get("power", 2.0))
            forecast = _garch_forecast(
                fit_res, horizon=len(data), power=power_val, reindex=False, align="origin"
            )
            preds = np.asarray(forecast.mean.iloc[-1]).reshape(-1)
            return preds
        if self.best_model_name == "tft":
            if pl is None or PFTFT is None or TimeSeriesDataSet is None:
                raise RuntimeError("TFT not available for prediction.")
            horizon = 1
            _, pred_loader, _ = self._tft_dataloaders(data, data.iloc[-horizon:], self._best_params, horizon=horizon)
            trainer = pl.Trainer(accelerator="cpu", devices=1, logger=False, enable_checkpointing=False, enable_model_summary=False, enable_progress_bar=False)
            raw_preds = trainer.predict(self.best_model, pred_loader, mode="prediction")
            out = torch.cat(raw_preds, dim=0).reshape(-1, horizon)[:, -horizon:].mean(dim=0).detach().cpu().numpy()
            return out
        if self.best_model_name == "gru":
            preds = self._gru_forecast(self.best_model, data, data)
            return preds
        if self.feature_builder is None:
            raise RuntimeError("Feature builder missing; fit() not executed correctly.")
        X, _ = self.feature_builder.transform(data)
        return self.best_model.predict(X)

    def predict_next(self, history_df: Any) -> float:
        if self.best_model is None:
            raise RuntimeError("Model not fit; call fit() first.")
        data = self.frame.load(history_df)
        if self.best_model_name == "prophet":
            # use last timestamp + freq for next prediction
            if self.frame.freq is None:
                raise ValueError("Cannot predict next step without inferred frequency.")
            next_timestamp = data.index[-1] + pd.tseries.frequencies.to_offset(self.frame.freq)
            future = pd.DataFrame({"ds": [next_timestamp]})
            for ex_col in self.config.exogenous_cols or []:
                # reuse last observed exogenous value
                future[ex_col] = data[ex_col].iloc[-1]
            pred = float(self.best_model.predict(future)["yhat"].values[0])
            return pred
        if self.best_model_name == "sarima":
            steps = 1
            exog = None
            if self.config.exogenous_cols:
                exog = pd.DataFrame({col: [data[col].iloc[-1]] for col in self.config.exogenous_cols})
            forecast_res = self.best_model.get_forecast(steps=steps, exog=exog)
            predicted = forecast_res.predicted_mean
            try:
                return float(predicted.iloc[-1])
            except Exception:
                return float(predicted[-1])
        if self.best_model_name == "garch":
            if arch_model is None:
                raise RuntimeError("arch not available for prediction.")
            params = self._best_params
            y = data[self.config.target_col].values
            mean_model = params.get("mean", "Constant")
            lags_val = int(params.get("lags", 0))
            if str(mean_model).upper() == "ARX" and lags_val <= 0:
                lags_val = 1
            lags_param = lags_val if str(mean_model).upper() == "ARX" else None
            model = arch_model(
                y,
                mean=mean_model,
                vol="GARCH",
                p=int(params.get("p", 1)),
                q=int(params.get("q", 1)),
                power=float(params.get("power", 2.0)),
                dist=params.get("dist", "normal"),
                lags=lags_param,
            )
            fit_res = model.fit(disp="off", show_warning=False)
            power_val = float(params.get("power", 2.0))
            forecast = _garch_forecast(fit_res, horizon=1, power=power_val, reindex=False, align="origin")
            preds = np.asarray(forecast.mean.iloc[-1]).reshape(-1)
            return float(preds[-1]) if len(preds) else float("nan")
        if self.best_model_name == "tft":
            if pl is None or PFTFT is None or TimeSeriesDataSet is None:
                raise RuntimeError("TFT not available for prediction.")
            params = self._best_params
            horizon = 1
            _, pred_loader, _ = self._tft_dataloaders(data, data.iloc[-horizon:], params, horizon=horizon)
            trainer = pl.Trainer(accelerator="cpu", devices=1, logger=False, enable_checkpointing=False, enable_model_summary=False, enable_progress_bar=False)
            raw_preds = trainer.predict(self.best_model, pred_loader, mode="prediction")
            out = torch.cat(raw_preds, dim=0).reshape(-1, horizon)[:, -horizon:].mean(dim=0).detach().cpu().numpy()
            return float(out[-1])
        if self.best_model_name == "gru":
            exog_cols = self.config.exogenous_cols or []
            next_row = data.iloc[[-1]].copy()
            for col in exog_cols:
                next_row[col] = data[col].iloc[-1]
            preds = self._gru_forecast(self.best_model, data, next_row)
            return float(preds[-1]) if len(preds) else float("nan")
        if self.feature_builder is None:
            raise RuntimeError("Feature builder missing; fit() not executed correctly.")
        features = self.feature_builder.latest_features(data)
        pred = float(self.best_model.predict(features)[0])
        return pred
