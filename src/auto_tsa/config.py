from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Sequence


@dataclass
class AutoTSAConfig:
    """
    Configuration for the AutoTSA search.
    """

    timestamp_col: Optional[str]
    target_col: str
    freq: Optional[str] = None
    lags: Sequence[int] = field(default_factory=lambda: (1, 2, 3, 6, 12, 24))
    windows: Sequence[int] = field(default_factory=lambda: (3, 6, 12, 24))
    add_datetime_features: bool = True
    add_holiday_features: bool = False
    holidays_country: Optional[str] = None
    exogenous_cols: Optional[List[str]] = None
    exogenous_lags: Sequence[int] = field(default_factory=lambda: (0, 1))
    auto_seasonality: bool = True
    holdout_fraction: float = 0.2
    rolling_splits: int = 3
    min_train_fraction: float = 0.5
    models: Optional[List[str]] = None
    max_trials_per_model: int = 8
    use_optuna: bool = False
    optuna_trials: int = 20
    holdout_stepwise: bool = True  # evaluate holdout one-step rolling (uses realized history)
    scoring: str = "smape"
    n_jobs: int = 2
    random_seed: int = 42
    scale_features: bool = True
    verbosity: int = 1

    def model_list(self) -> List[str]:
        from .models import list_available_models

        return self.models or list_available_models()
