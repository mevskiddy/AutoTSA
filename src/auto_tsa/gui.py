from __future__ import annotations

import io
import os
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor

# Allow running via `streamlit run src/auto_tsa/gui.py` without package context
if __package__ in (None, ""):
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from auto_tsa.config import AutoTSAConfig  # type: ignore
    from auto_tsa.search import AutoTSA  # type: ignore
    from auto_tsa.models import list_available_models  # type: ignore
    from auto_tsa.features import FeatureBuilder  # type: ignore
else:
    from .config import AutoTSAConfig
    from .search import AutoTSA
    from .models import list_available_models
    from .features import FeatureBuilder


st.set_page_config(
    page_title="AutoTSA Studio",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _format_model_label(model_name: str | None, labels: dict[str, str]) -> str | None:
    """
    Convert an internal model code to a friendly label for UI output.
    """
    if not model_name:
        return None
    return labels.get(model_name, model_name)


def _inject_modern_theme():
    st.markdown(
        """
        <style>
            .stApp {
                background: radial-gradient(circle at 20% 20%, #1b2b5c 0, #0b1229 35%, #060914 70%);
                color: #e6f0ff;
                font-family: "Space Grotesk", "Inter", system-ui, -apple-system, sans-serif;
            }
            .block-container { padding-top: 2rem; padding-bottom: 2rem; }
            .metric-card {
                background: linear-gradient(135deg, rgba(43,95,255,0.18), rgba(151,71,255,0.12));
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 16px;
                padding: 14px 16px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.35);
            }
            .stButton>button {
                border-radius: 12px;
                background: linear-gradient(135deg, #4f46e5, #22d3ee);
                color: white;
                font-weight: 700;
                padding: 0.75rem 1rem;
                border: none;
                box-shadow: 0 14px 30px rgba(34,211,238,0.35);
            }
            .stFileUploader, .stSelectbox, .stSlider, .stMultiSelect, .stTextInput {
                background: rgba(255,255,255,0.02);
                border-radius: 14px;
                padding: 12px;
                border: 1px solid rgba(255,255,255,0.08);
            }
            .stSlider>div>div>div>div {
                background: linear-gradient(90deg, #4f46e5, #22d3ee) !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _detect_timestamp_candidates(df: pd.DataFrame) -> list[str]:
    candidates = list(df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns)
    for col in df.columns:
        low = str(col).lower()
        if any(key in low for key in ("date", "time", "timestamp", "datetime", "ds", "period")) and col not in candidates:
            candidates.append(col)
    # try to include columns that parse as datetime reasonably well
    for col in df.columns:
        if col in candidates:
            continue
        try:
            converted = pd.to_datetime(df[col], errors="coerce")
            if converted.notna().mean() > 0.6:
                candidates.append(col)
        except Exception:
            continue
    return candidates


def _numeric_targets(df: pd.DataFrame, skip: set[str]) -> list[str]:
    numeric_cols = [c for c in df.select_dtypes(include=["number"]).columns if c not in skip]
    return numeric_cols


@st.cache_data(show_spinner=False)
def _hash_frame(df: pd.DataFrame) -> str:
    return str(pd.util.hash_pandas_object(df, index=True).sum())


def _get_cleaned_frame(df: pd.DataFrame, timestamp_col: str, base_hash: str) -> tuple[pd.DataFrame, int, int]:
    """
    Cache cleaned/sorted frame per (data, timestamp) to avoid redoing heavy datetime parsing on each rerun.
    """
    cache_key = f"{base_hash}:{timestamp_col}"
    if st.session_state.get("clean_cache_key") == cache_key and "clean_cache_value" in st.session_state:
        return st.session_state["clean_cache_value"]
    clean_df, dropped_na, dropped_dupes = _clean_time_index(df, timestamp_col)
    st.session_state["clean_cache_key"] = cache_key
    st.session_state["clean_cache_value"] = (clean_df, dropped_na, dropped_dupes)
    return clean_df, dropped_na, dropped_dupes


def _clean_time_index(df: pd.DataFrame, timestamp_col: str) -> tuple[pd.DataFrame, int, int]:
    cleaned = df.copy()
    cleaned[timestamp_col] = pd.to_datetime(cleaned[timestamp_col], errors="coerce")
    before = len(cleaned)
    cleaned = cleaned.dropna(subset=[timestamp_col])
    dropped_na = before - len(cleaned)
    cleaned = cleaned.drop_duplicates(subset=[timestamp_col], keep="last")
    cleaned = cleaned.sort_values(timestamp_col)
    dropped_dupes = before - dropped_na - len(cleaned)
    return cleaned, dropped_na, dropped_dupes


def _train_model(
    df: pd.DataFrame,
    timestamp_col: str,
    target_col: str,
    freq: str | None,
    seasonal_period: int | None,
    exog_cols: list[str],
    models: list[str] | None,
    use_optuna: bool,
    optuna_trials: int,
    add_holidays: bool,
    holiday_country: str | None,
):
    cfg = AutoTSAConfig(
        timestamp_col=timestamp_col,
        target_col=target_col,
        freq=freq or None,
        seasonal_period=seasonal_period,
        exogenous_cols=exog_cols or None,
        lags=(1, 2, 3, 6, 12),
        windows=(3, 6, 12),
        auto_seasonality=True,
        add_holiday_features=add_holidays,
        holidays_country=holiday_country if add_holidays else None,
        rolling_splits=2,
        max_trials_per_model=4,
        models=models or None,
        use_optuna=use_optuna,
        optuna_trials=optuna_trials,
        verbosity=0,
        n_jobs=2,
    )
    automl = AutoTSA(cfg).fit(df)
    return automl


def _multi_step_forecast(
    automl: AutoTSA,
    history_df: pd.DataFrame,
    timestamp_col: str,
    target_col: str,
    steps: int,
    exog_cols: list[str],
    project_exog: bool,
):
    working = history_df[[timestamp_col] + [c for c in history_df.columns if c != timestamp_col]].copy()
    working[timestamp_col] = pd.to_datetime(working[timestamp_col])
    base_frame = automl.frame.load(working)
    freq = automl.frame.freq
    if freq is None:
        raise ValueError("Could not infer a frequency; provide it explicitly to unlock multi-step forecasts.")
    offset = pd.tseries.frequencies.to_offset(freq)

    exog_projection = (
        _project_exogenous(working, timestamp_col, exog_cols, steps, offset) if project_exog else [{} for _ in range(steps)]
    )
    preds = []
    temp = working.copy()
    for step_idx in range(steps):
        next_ts = pd.to_datetime(temp[timestamp_col].iloc[-1]) + offset
        # seed the next timestep with projected exogenous values so predict_next sees them
        placeholder = temp.iloc[[-1]].copy()
        placeholder[timestamp_col] = next_ts
        placeholder[target_col] = temp[target_col].iloc[-1]
        for col in exog_cols:
            projected_val = exog_projection[step_idx].get(col, temp[col].iloc[-1])
            placeholder[col] = projected_val
        temp_with_next = pd.concat([temp, placeholder], ignore_index=True)
        next_val = automl.predict_next(temp_with_next)
        temp_with_next.loc[temp_with_next.index[-1], target_col] = next_val
        preds.append({"timestamp": next_ts, "pred": next_val})
        temp = temp_with_next

    forecast_df = pd.DataFrame(preds).set_index("timestamp")
    return base_frame, forecast_df


def _project_exogenous(
    history_df: pd.DataFrame,
    timestamp_col: str,
    exog_cols: list[str],
    steps: int,
    offset: pd.tseries.offsets.BaseOffset,
    lags: tuple[int, ...] = (1, 2, 3, 6, 12),
    windows: tuple[int, ...] = (3, 6, 12),
):
    """
    Fit a small GradientBoostingRegressor per exogenous series using lag/rolling features,
    then recursively project each exog forward.
    """
    if not exog_cols:
        return [{} for _ in range(steps)]
    projections = [{} for _ in range(steps)]
    for col in exog_cols:
        try:
            series = history_df[[timestamp_col, col]].dropna()
            series[timestamp_col] = pd.to_datetime(series[timestamp_col])
            series = series.set_index(timestamp_col).sort_index()
            fb = FeatureBuilder(
                target_col=col,
                lags=lags,
                windows=windows,
                add_datetime_features=False,
                add_holiday_features=False,
                exogenous_cols=(),
                exogenous_lags=(0,),
                scale=True,
            )
            X_train, y_train = fb.fit_transform(series)
            model = GradientBoostingRegressor(random_state=42)
            model.fit(X_train, y_train)
            temp = series.copy()
            preds: list[float] = []
            for _ in range(steps):
                next_ts = temp.index[-1] + offset
                placeholder = temp.iloc[[-1]].copy()
                placeholder.index = [next_ts]
                temp_with_next = pd.concat([temp, placeholder])
                feats = fb.latest_features(temp_with_next)
                next_val = float(model.predict(feats)[0])
                temp_with_next.iloc[-1, temp_with_next.columns.get_loc(col)] = next_val
                temp = temp_with_next
                preds.append(next_val)
            for i, val in enumerate(preds):
                projections[i][col] = val
        except Exception:
            fallback = history_df[col].dropna().iloc[-1] if not history_df[col].dropna().empty else 0.0
            for i in range(steps):
                projections[i][col] = float(fallback)
    return projections


def _plot_forecast(history: pd.DataFrame, forecast: pd.DataFrame, target_col: str, history_points: int):
    history_tail = history.tail(history_points)
    last_actual_ts = history_tail.index[-1] if len(history_tail) else None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history_tail.index,
            y=history_tail[target_col],
            mode="lines",
            name="Actuals",
            line=dict(color="#22d3ee", width=4),
            fill="tozeroy",
            fillcolor="rgba(34,211,238,0.12)",
        )
    )
    if len(forecast) > 0:
        fig.add_trace(
            go.Scatter(
                x=forecast.index,
                y=forecast["pred"],
                mode="lines+markers",
                name="Forecast",
                line=dict(color="#a855f7", width=4, dash="dash"),
                marker=dict(size=8, color="#c084fc", line=dict(width=2, color="#0b1229")),
            )
        )
        if last_actual_ts is not None:
            fig.add_vline(
                x=last_actual_ts,
                line_width=2,
                line_dash="dot",
                line_color="rgba(255,255,255,0.5)",
            )
    fig.update_layout(
        height=480,
        margin=dict(l=30, r=20, t=30, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.01)",
        legend=dict(orientation="h", y=1.08, x=0.02, font=dict(color="#e6f0ff")),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.08)",
            zeroline=False,
            showline=False,
            tickfont=dict(color="#b9c7ff"),
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.08)",
            zeroline=False,
            showline=False,
            tickfont=dict(color="#b9c7ff"),
        ),
    )
    return fig


def _plot_cv_series(actual: pd.Series, cv_df: pd.DataFrame):
    if len(cv_df) == 0:
        return None
    ts_min, ts_max = cv_df.index.min(), cv_df.index.max()
    actual_clip = actual.loc[ts_min:ts_max] if ts_min is not None and ts_max is not None else actual
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=actual_clip.index,
            y=actual_clip,
            mode="lines",
            name="Actuals",
            line=dict(color="#22d3ee", width=3),
            opacity=0.45,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cv_df.index,
            y=cv_df["pred"],
            mode="lines+markers",
            name="CV predictions",
            line=dict(color="#f472b6", width=4),
            marker=dict(size=7, color="#f9a8d4", line=dict(width=1, color="#0b1229")),
        )
    )
    fig.update_layout(
        height=360,
        margin=dict(l=30, r=20, t=30, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.01)",
        legend=dict(orientation="h", y=1.05, x=0.02, font=dict(color="#e6f0ff")),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)", tickfont=dict(color="#b9c7ff")),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", tickfont=dict(color="#b9c7ff")),
    )
    return fig


def _plot_holdout_series(holdout_df: pd.DataFrame):
    if holdout_df.empty:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=holdout_df.index,
            y=holdout_df["actual"],
            mode="lines",
            name="Holdout actuals",
            line=dict(color="#22d3ee", width=3),
            opacity=0.5,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=holdout_df.index,
            y=holdout_df["pred"],
            mode="lines+markers",
            name="Holdout predictions",
            line=dict(color="#f59e0b", width=4),
            marker=dict(size=7, color="#fcd34d", line=dict(width=1, color="#0b1229")),
        )
    )
    fig.update_layout(
        height=360,
        margin=dict(l=30, r=20, t=30, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.01)",
        legend=dict(orientation="h", y=1.05, x=0.02, font=dict(color="#e6f0ff")),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)", tickfont=dict(color="#b9c7ff")),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", tickfont=dict(color="#b9c7ff")),
    )
    return fig


def _render_metrics(best_model: str | None, holdout_score: float | None, scoring: str):
    cols = st.columns(2)
    with cols[0]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Best model", best_model or "–")
        st.markdown("</div>", unsafe_allow_html=True)
    with cols[1]:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if holdout_score is None or (isinstance(holdout_score, float) and pd.isna(holdout_score)):
            st.metric("Holdout score", "not available")
        else:
            st.metric("Holdout score", f"{holdout_score:.4f}", help=f"metric: {scoring}")
        st.markdown("</div>", unsafe_allow_html=True)


def _model_download_bytes(automl: AutoTSA) -> bytes | None:
    """
    Serialize the trained AutoTSA object so the user can download and reuse it.
    """
    try:
        buf = io.BytesIO()
        joblib.dump(automl, buf, compress=3)
        buf.seek(0)
        return buf.read()
    except Exception:
        return None


def main():
    _inject_modern_theme()
    st.title("AutoTSA Studio")
    st.caption("Upload a CSV, pick your time axis and target, choose a horizon, and see forecasts with controllable history context.")

    st.session_state.setdefault("model_state", None)
    st.session_state.setdefault("data_hash", None)

    with st.sidebar:
        st.subheader("Upload & columns")
        upload = st.file_uploader("Upload CSV", type=["csv"])
        freq_input = st.text_input("Frequency (optional)", placeholder="e.g. D, H, MS")
        seasonal_period_input = st.number_input(
            "Seasonal period (optional)",
            min_value=0,
            max_value=10000,
            value=0,
            help="Force a fixed seasonal period for SARIMA and lag/window hints (e.g., 7 or 12).",
        )
        st.markdown("---")
        st.subheader("Forecast controls")
        step_input = st.slider("Prediction steps", 1, 90, 12, help="How many steps into the future to forecast.")
        hist_window = st.slider("History points to plot", 5, 200, 10, help="How many trailing actual points to show before the forecast line.")

    if upload is None:
        st.info("Drop a CSV on the sidebar to get started.")
        return

    try:
        raw_df = pd.read_csv(upload)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        return

    if raw_df.empty:
        st.warning("Uploaded CSV has no rows to use for training.")
        return

    raw_hash = _hash_frame(raw_df)
    st.markdown("### Data preview")
    default_train_rows = int(st.session_state.get("train_rows", len(raw_df)))
    train_rows = st.number_input(
        "Rows to use for training (latest rows)",
        min_value=1,
        max_value=int(len(raw_df)),
        value=min(default_train_rows, int(len(raw_df))),
        step=1,
        help="Only the most recent rows will be used to fit models and generate forecasts.",
    )
    st.session_state["train_rows"] = int(train_rows)
    st.caption(f"Using the latest {train_rows} of {len(raw_df)} rows for training.")
    st.dataframe(raw_df.head(20), height=240, width="stretch")

    df_hash = f"{train_rows}:{raw_hash}"
    if st.session_state.get("data_hash") != df_hash:
        st.session_state["model_state"] = None
        st.session_state["data_hash"] = df_hash

    timestamp_candidates = _detect_timestamp_candidates(raw_df)
    timestamp_default = timestamp_candidates[0] if timestamp_candidates else None
    target_candidates = _numeric_targets(raw_df, skip=set())
    target_default = target_candidates[0] if target_candidates else None

    available_models = list_available_models()
    try:
        import torch

        cuda_ok = bool(torch.cuda.is_available() and torch.version.cuda)
    except Exception:
        cuda_ok = False
    if not cuda_ok and "tft" in available_models:
        available_models = [m for m in available_models if m != "tft"]
        st.info("TFT disabled: CUDA GPU not detected. Select other models or run with CUDA to enable TFT.")
    friendly_labels = {
        "gbr": "Gradient Boosting",
        "tft": "Temporal Fusion Transformer",
        "rf": "Random Forest",
        "ridge": "Ridge",
        "elasticnet": "ElasticNet",
        "gpr": "Gaussian Process",
        "lightgbm": "LightGBM",
        "prophet": "Prophet",
        "sarima": "SARIMA",
        "garch": "GARCH",
        "gru": "GRU",
    }

    with st.form("config"):
        st.markdown("#### Column & model setup")
        ts_col = st.selectbox(
            "Timestamp column",
            options=timestamp_candidates,
            index=0 if timestamp_candidates else None,
            help="We try to spot datetime-like columns automatically.",
        )
        tgt_col = st.selectbox(
            "Target column",
            options=[c for c in target_candidates if c != ts_col],
            index=0 if target_candidates else None,
        )
        available_exog = [c for c in raw_df.columns if c not in {ts_col, tgt_col}]
        exog_cols = st.multiselect(
            "Exogenous regressors (optional)",
            options=available_exog,
            help="These columns will be lagged and used as predictors alongside the target lags.",
        )
        project_exog = st.toggle(
            "Project exogenous forward",
            value=False,
            help="When enabled, each exogenous column is forecast with a small Gradient Boosting regressor on its lags/rolls; otherwise the last value is held flat.",
        )
        add_holidays = st.toggle(
            "Add holiday features",
            value=False,
            help="Adds country-specific holiday indicator features.",
        )
        holiday_country = st.text_input(
            "Holiday country (e.g., US, CA, GB)",
            value="US",
            disabled=not add_holidays,
        )
        models_selected = st.multiselect(
            "Models to include",
            options=available_models,
            default=[m for m in ["gbr", "ridge", "rf"] if m in available_models],
            format_func=lambda m: friendly_labels.get(m, m),
            help="Toggle full model names to control the search space.",
        )
        disable_optuna_for_tft = "tft" in models_selected
        use_optuna = st.toggle(
            "Use Optuna",
            value=False,
            help="If enabled, hyperparameters are tuned with Optuna. Disabled when TFT is selected.",
            disabled=disable_optuna_for_tft,
        )
        optuna_trials = st.slider(
            "Optuna trials",
            5,
            50,
            20,
            help="Trials per model when Optuna is enabled. Slider always active; only used when Optuna is on.",
            disabled=disable_optuna_for_tft,
        )
        submitted = st.form_submit_button("Train & Forecast")

    if not ts_col or not tgt_col:
        st.warning("Pick both a timestamp column and a target column to continue.")
        return

    if submitted:
        try:
            clean_df, dropped_na, dropped_dupes = _get_cleaned_frame(raw_df, ts_col, raw_hash)
            if train_rows < len(clean_df):
                clean_df = clean_df.tail(int(train_rows))
            if dropped_na > 0 or dropped_dupes > 0:
                st.warning(
                    f"Removed {dropped_na} rows with invalid timestamps and {dropped_dupes} duplicate timestamps before training."
                )
            if "tft" in models_selected and use_optuna:
                st.info("Optuna is disabled when TFT is selected; running random search instead.")
                use_optuna = False
            with st.spinner("Training AutoTSA and generating forecast..."):
                seasonal_period = int(seasonal_period_input) if seasonal_period_input > 0 else None
                automl = _train_model(
                    clean_df,
                    ts_col,
                    tgt_col,
                    freq_input or None,
                    seasonal_period,
                    exog_cols,
                    models_selected,
                    use_optuna,
                    optuna_trials,
                    add_holidays,
                    holiday_country if add_holidays else None,
                )
                history_frame, forecast_frame = _multi_step_forecast(
                    automl, clean_df[[ts_col, tgt_col] + exog_cols], ts_col, tgt_col, step_input, exog_cols, project_exog
                )
            st.session_state["model_state"] = {
                "automl": automl,
                "history_frame": history_frame,
                "raw_df": clean_df[[ts_col, tgt_col] + exog_cols],
                "ts_col": ts_col,
                "tgt_col": tgt_col,
                "exog_cols": exog_cols,
                "models": models_selected,
                "use_optuna": use_optuna,
                "optuna_trials": optuna_trials,
                "freq": freq_input or automl.frame.freq,
                "seasonal_period": seasonal_period,
                "forecast_frame": forecast_frame,
                "history_window": hist_window,
                "steps": step_input,
                "project_exog": project_exog,
                "add_holidays": add_holidays,
                "holiday_country": holiday_country if add_holidays else None,
                "train_rows": train_rows,
            }
        except Exception as exc:
            import traceback

            st.error(f"Training or forecasting failed: {exc.__class__.__name__}: {exc}")
            st.text("\n" + traceback.format_exc())
            return
    state = st.session_state.get("model_state")
    if state is None:
        st.info("Configure the columns and hit **Train & Forecast**.")
        return

    if hist_window != state.get("history_window") or step_input != state.get("steps"):
        try:
            base_df, fc_df = _multi_step_forecast(
                state["automl"],
                state["raw_df"],
                state["ts_col"],
                state["tgt_col"],
                step_input,
                state["exog_cols"],
                state.get("project_exog", False),
            )
            state["forecast_frame"] = fc_df
            state["history_frame"] = base_df
            state["history_window"] = hist_window
            state["steps"] = step_input
        except Exception as exc:
            st.error(f"Could not refresh forecast: {exc}")

    best_model_label = _format_model_label(state["automl"].best_model_name, friendly_labels)
    _render_metrics(best_model_label, state["automl"].holdout_score, state["automl"].config.scoring)
    show_holdout = st.toggle(
        "Show holdout performance",
        value=False,
        help="Plot predictions vs actuals on the holdout slice used for the reported score.",
    )
    if show_holdout:
        holdout_frame = state["automl"].holdout_prediction_frame
        if holdout_frame is not None:
            holdout_chart = _plot_holdout_series(holdout_frame)
            if holdout_chart:
                st.plotly_chart(holdout_chart, theme=None, config={"displaylogo": False, "responsive": True})
            else:
                st.info("No holdout predictions available to plot.")
        else:
            st.info("Holdout predictions not available for this model type.")

    show_cv = st.toggle(
        "Show CV performance",
        value=False,
        help="When enabled, reruns the best model across rolling splits and plots validation predictions.",
    )
    if show_cv:
        cache_key = f"{st.session_state.get('data_hash')}:{state['automl'].best_model_name}:cv"
        cache_bucket = st.session_state.setdefault("cv_cache", {})
        cached = cache_bucket.get(cache_key)
        if cached:
            cv_df, cv_score = cached
        else:
            try:
                with st.spinner("Running cross-validation for plot..."):
                    cv_df, cv_score = state["automl"].crossval_series(state["raw_df"])
                cache_bucket[cache_key] = (cv_df, cv_score)
            except Exception as exc:
                st.error(f"Could not compute CV performance: {exc}")
                cv_df = None
                cv_score = None
        if cv_df is not None:
            actual_series = state["raw_df"].set_index(state["ts_col"])[state["tgt_col"]]
            cv_chart = _plot_cv_series(actual_series, cv_df)
            st.plotly_chart(cv_chart, theme=None, config={"displaylogo": False, "responsive": True})
            if cv_score is not None:
                st.caption(f"Cross-val {state['automl'].config.scoring}: {cv_score:.4f}")
            else:
                st.caption("Cross-val metric unavailable for this run.")

    chart = _plot_forecast(state["history_frame"], state["forecast_frame"], state["tgt_col"], hist_window)
    st.plotly_chart(
        chart,
        theme=None,
        config={"displaylogo": False, "responsive": True},
    )
    st.markdown("#### Save trained model")
    download_bytes = _model_download_bytes(state["automl"])
    if download_bytes:
        model_name_for_file = best_model_label or state["automl"].best_model_name or "model"
        safe_model_name = "".join(ch if ch.isalnum() else "_" for ch in model_name_for_file).strip("_").lower() or "model"
        fname = f"auto_tsa_{safe_model_name}.joblib"
        st.download_button(
            "Save model (.joblib)",
            data=download_bytes,
            file_name=fname,
            mime="application/octet-stream",
            help="Download the fitted AutoTSA object for reuse with joblib.load().",
        )
    else:
        st.warning("Could not serialize the trained model for download.")

    st.markdown("#### Forecasted points")
    preview = state["forecast_frame"].reset_index().rename(columns={"timestamp": state["ts_col"], "pred": "forecast"})
    st.dataframe(preview, width="stretch")


if __name__ == "__main__":
    main()
