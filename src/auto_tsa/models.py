from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple
import random
import importlib.util

import numpy as np


def _optional_import(name: str):
    try:
        module = __import__(name)
        for part in name.split(".")[1:]:
            module = getattr(module, part)
        return module
    except Exception:
        return None


SklearnRegressor = Any


@dataclass
class ModelSpec:
    name: str
    builder: Callable[[Dict[str, Any]], SklearnRegressor]
    param_space: Dict[str, Any]
    default_params: Dict[str, Any]
    kind: str = "sklearn"  # or "prophet"

    def sample_params(self, rng: random.Random) -> Dict[str, Any]:
        params = self.default_params.copy()
        for key, space in self.param_space.items():
            if isinstance(space, tuple) and len(space) == 2 and all(isinstance(v, (int, float)) for v in space):
                low, high = space
                if isinstance(low, int) and isinstance(high, int):
                    params[key] = rng.randint(low, high)
                else:
                    params[key] = rng.uniform(low, high)
            elif isinstance(space, list):
                params[key] = rng.choice(space)
            elif callable(space):
                params[key] = space()
        return params


def _sklearn_specs() -> List[ModelSpec]:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import ElasticNet, Ridge
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel

    return [
        ModelSpec(
            name="gbr",
            builder=lambda params: GradientBoostingRegressor(**params),
            param_space={
                "n_estimators": (50, 350),
                "learning_rate": (0.01, 0.2),
                "max_depth": (2, 6),
                "subsample": (0.6, 1.0),
            },
            default_params={"random_state": 0},
        ),
        ModelSpec(
            name="rf",
            builder=lambda params: RandomForestRegressor(**params),
            param_space={
                "n_estimators": (100, 500),
                "max_depth": (3, 12),
                "min_samples_leaf": (1, 8),
                "max_features": ["sqrt", "log2", None],
            },
            default_params={"n_jobs": -1, "random_state": 0},
        ),
        ModelSpec(
            name="ridge",
            builder=lambda params: Ridge(**params),
            param_space={"alpha": (0.001, 100.0)},
            default_params={},
        ),
        ModelSpec(
            name="elasticnet",
            builder=lambda params: ElasticNet(**params),
            param_space={"alpha": (0.001, 1.0), "l1_ratio": (0.05, 0.95)},
            default_params={"max_iter": 5000, "random_state": 0},
        ),
        ModelSpec(
            name="gpr",
            builder=lambda params: GaussianProcessRegressor(
                kernel=RBF(length_scale=params["length_scale"])
                + WhiteKernel(noise_level=params["noise_level"]),
                alpha=params.get("alpha", 1e-6),
                random_state=0,
                normalize_y=True,
            ),
            param_space={
                "length_scale": (0.1, 10.0),
                "noise_level": (1e-5, 1.0),
                "alpha": (1e-8, 1e-3),
            },
            default_params={},
        ),
    ]


def _lightgbm_spec() -> List[ModelSpec]:
    lgb = _optional_import("lightgbm")
    if lgb is None:
        return []
    return [
        ModelSpec(
            name="lightgbm",
            builder=lambda params: lgb.LGBMRegressor(**params),
            param_space={
                "num_leaves": (16, 256),
                "learning_rate": (0.01, 0.2),
                "n_estimators": (100, 800),
                "subsample": (0.6, 1.0),
                "colsample_bytree": (0.6, 1.0),
                "min_child_samples": (10, 80),
            },
            default_params={"objective": "regression", "random_state": 0, "n_jobs": -1},
        )
    ]


def _prophet_spec() -> List[ModelSpec]:
    prophet = _optional_import("prophet")
    if prophet is None:
        return []
    return [
        ModelSpec(
            name="prophet",
            builder=lambda params: prophet.Prophet(**params),
            param_space={
                "seasonality_mode": ["additive", "multiplicative"],
                "changepoint_prior_scale": (0.001, 0.5),
                "seasonality_prior_scale": (1.0, 15.0),
                "holidays_prior_scale": (1.0, 20.0),
                "n_changepoints": (5, 50),
            },
            default_params={
                "weekly_seasonality": "auto",
                "daily_seasonality": "auto",
                "yearly_seasonality": "auto",
            },
            kind="prophet",
        )
    ]


def _sarima_spec() -> List[ModelSpec]:
    sm = _optional_import("statsmodels")
    if sm is None:
        return []
    return [
        ModelSpec(
            name="sarima",
            builder=lambda params: params,  # params consumed directly in search
            param_space={
                "p": (0, 2),
                "d": (0, 1),
                "q": (0, 2),
                "P": (0, 1),
                "D": (0, 1),
                "Q": (0, 1),
                "trend": ["n", "c"],
            },
            default_params={"p": 1, "d": 0, "q": 1, "P": 1, "D": 0, "Q": 0, "trend": "n"},
            kind="sarima",
        )
    ]


def _garch_spec() -> List[ModelSpec]:
    arch = _optional_import("arch")
    if arch is None:
        return []
    return [
        ModelSpec(
            name="garch",
            builder=lambda params: params,  # consumed in search
            param_space={
                "p": (1, 3),
                "q": (1, 3),
                "mean": ["Constant", "ARX"],
                "lags": (0, 3),
                "power": (1.0, 2.5),
                "dist": ["normal", "t"],
            },
            default_params={"p": 1, "q": 1, "mean": "Constant", "lags": 0, "power": 2.0, "dist": "normal"},
            kind="garch",
        )
    ]


def _tft_spec() -> List[ModelSpec]:
    return []  # temporarily disabled

def _gru_spec() -> List[ModelSpec]:
    torch_mod = _optional_import("torch")
    if torch_mod is None:
        return []
    return [
        ModelSpec(
            name="gru",
            builder=lambda params: params,  # consumed in search
            param_space={
                "window": (8, 48),
                "hidden_size": (16, 128),
                "epochs": (8, 40),
                "lr": (1e-4, 1e-2),
            },
            default_params={"window": 16, "hidden_size": 32, "epochs": 15, "lr": 1e-3},
            kind="gru",
        )
    ]


def list_available_models() -> List[str]:
    names = [spec.name for spec in _sklearn_specs()]
    names += [spec.name for spec in _lightgbm_spec()]
    names += [spec.name for spec in _prophet_spec()]
    names += [spec.name for spec in _sarima_spec()]
    names += [spec.name for spec in _garch_spec()]
    names += [spec.name for spec in _tft_spec()]
    names += [spec.name for spec in _gru_spec()]
    # ensure unique and stable order
    seen = set()
    ordered = []
    for name in names:
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def get_model_specs(requested: List[str]) -> List[ModelSpec]:
    specs = (
        _sklearn_specs()
        + _lightgbm_spec()
        + _prophet_spec()
        + _sarima_spec()
        + _garch_spec()
        + _tft_spec()
        + _gru_spec()
    )
    catalog = {spec.name: spec for spec in specs}
    missing_optional = [m for m in requested if m == "sarima" and "sarima" not in catalog]
    if missing_optional:
        raise ValueError("SARIMA requested but statsmodels is not installed. Install via `pip install statsmodels`.")
    missing_optional += [m for m in requested if m == "prophet" and "prophet" not in catalog]
    if missing_optional:
        raise ValueError("Prophet requested but not installed. Install via `pip install prophet`.")
    missing_optional += [m for m in requested if m == "lightgbm" and "lightgbm" not in catalog]
    if missing_optional:
        raise ValueError("LightGBM requested but not installed. Install via `pip install lightgbm`.")
    missing_optional += [m for m in requested if m == "garch" and "garch" not in catalog]
    if missing_optional:
        raise ValueError("GARCH requested but arch is not installed. Install via `pip install arch`.")
    missing_optional += [m for m in requested if m == "tft" and "tft" not in catalog]
    # For TFT, let evaluation handle dependency errors gracefully; do not hard-fail here
    unknown = []
    resolved = []
    for name in requested:
        if name in catalog:
            resolved.append(catalog[name])
        elif name == "tft":
            # Allow missing TFT (will be skipped later if deps unavailable)
            continue
        else:
            unknown.append(name)
    if unknown:
        raise ValueError(f"Unknown models requested: {unknown}")
    return resolved
