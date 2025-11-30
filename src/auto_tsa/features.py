from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def _holiday_frame(index: pd.DatetimeIndex, country: str | None) -> pd.DataFrame:
    if country is None:
        return pd.DataFrame(index=index)
    try:
        import holidays
    except Exception:
        return pd.DataFrame(index=index)
    holiday_map = holidays.country_holidays(country)
    indicator = index.normalize().map(lambda d: 1 if d in holiday_map else 0)
    return pd.DataFrame({"is_holiday": indicator}, index=index)


@dataclass
class FeatureBuilder:
    target_col: str
    lags: Sequence[int]
    windows: Sequence[int]
    add_datetime_features: bool = True
    add_holiday_features: bool = False
    holidays_country: str | None = None
    exogenous_cols: Sequence[str] = field(default_factory=tuple)
    exogenous_lags: Sequence[int] = field(default_factory=lambda: (0, 1))
    scale: bool = True
    scaler: StandardScaler | None = field(default=None, init=False)
    feature_names_: List[str] = field(default_factory=list, init=False)

    def _generate_static_features(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        parts = {
            "month": index.month,
            "weekofyear": index.isocalendar().week.astype(int),
            "day": index.day,
            "dayofweek": index.dayofweek,
            "dayofyear": index.dayofyear,
            "hour": index.hour,
        }
        base = pd.DataFrame(parts, index=index)
        if self.add_holiday_features:
            base = pd.concat([base, _holiday_frame(index, self.holidays_country)], axis=1)
        return base

    def _build_matrix(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        feats = {}
        target = df[self.target_col]
        past_target = target.shift(1)
        n = len(df)
        for lag in self.lags:
            if lag >= n:
                continue
            feats[f"lag_{lag}"] = df[self.target_col].shift(lag)
        for window in self.windows:
            if window >= n:
                continue
            feats[f"roll_mean_{window}"] = past_target.rolling(window=window, min_periods=window).mean()
            feats[f"roll_std_{window}"] = past_target.rolling(window=window, min_periods=window).std()
        for ex_col in self.exogenous_cols:
            for lag in self.exogenous_lags:
                if lag >= n:
                    continue
                feats[f"{ex_col}_lag_{lag}"] = df[ex_col].shift(lag)
        feature_frame = pd.DataFrame(feats, index=df.index)
        if self.add_datetime_features or self.add_holiday_features:
            feature_frame = pd.concat([feature_frame, self._generate_static_features(df.index)], axis=1)
        feature_frame = feature_frame.dropna()
        if feature_frame.empty:
            raise ValueError("No rows available after feature construction; consider fewer lags/windows or more data.")
        y = df.loc[feature_frame.index, self.target_col]
        self.feature_names_ = list(feature_frame.columns)
        return feature_frame, y

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        feature_frame, y = self._build_matrix(df)
        if self.scale:
            self.scaler = StandardScaler()
            transformed = self.scaler.fit_transform(feature_frame.values)
        else:
            transformed = feature_frame.values
        return transformed, y.values

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        feature_frame, y = self._build_matrix(df)
        if self.scale and self.scaler:
            transformed = self.scaler.transform(feature_frame.values)
        else:
            transformed = feature_frame.values
        return transformed, y.values

    def latest_features(self, df: pd.DataFrame) -> np.ndarray:
        feature_frame, _ = self._build_matrix(df)
        if feature_frame.empty:
            raise ValueError("Not enough history to build features for prediction")
        latest = feature_frame.tail(1).values
        if self.scale and self.scaler:
            latest = self.scaler.transform(latest)
        return latest
