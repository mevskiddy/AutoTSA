from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Generator, Iterable, Tuple

import pandas as pd


def _infer_timestamp_column(df: pd.DataFrame) -> str:
    datetime_cols = list(df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns)
    if datetime_cols:
        return datetime_cols[0]
    for candidate in df.columns:
        lower = candidate.lower()
        if any(key in lower for key in ("date", "time", "timestamp", "datetime", "ds")):
            try:
                converted = pd.to_datetime(df[candidate])
                if converted.notna().mean() > 0.8:
                    df[candidate] = converted
                    return candidate
            except Exception:
                continue
    raise ValueError("Could not infer timestamp column; please provide timestamp_col.")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cleaned = []
    for c in df.columns:
        name = str(c).strip()
        name = name.lstrip("\ufeff")  # strip potential BOM
        cleaned.append(name)
    df.columns = cleaned
    return df


def _match_column(name: str, columns) -> str:
    if name in columns:
        return name
    lowered = {str(c).lower(): c for c in columns}
    key = name.lower()
    if key in lowered:
        return lowered[key]
    # fallback: match after stripping BOM/whitespace
    stripped = key.lstrip("\ufeff").strip()
    for col in columns:
        if str(col).lower().lstrip("\ufeff").strip() == stripped:
            return col
    available = ", ".join(map(str, columns))
    raise KeyError(f"{name} (available columns: {available})")


@dataclass
class TimeSeriesFrame:
    timestamp_col: str | None
    target_col: str
    freq: str | None = None

    def load(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = _normalize_columns(df)
        if self.timestamp_col is None:
            self.timestamp_col = _infer_timestamp_column(frame)
        else:
            self.timestamp_col = _match_column(self.timestamp_col, frame.columns)
        self.target_col = _match_column(self.target_col, frame.columns)
        frame[self.timestamp_col] = pd.to_datetime(frame[self.timestamp_col])
        frame = frame.set_index(self.timestamp_col).sort_index()
        if self.freq:
            frame = frame.asfreq(self.freq)
        else:
            inferred = pd.infer_freq(frame.index)
            if inferred:
                frame = frame.asfreq(inferred)
                self.freq = inferred
        frame[self.target_col] = frame[self.target_col].astype(float)
        frame[self.target_col] = frame[self.target_col].interpolate(limit_direction="both")
        frame = frame.dropna(subset=[self.target_col])
        return frame

    def train_validation_split(self, df: pd.DataFrame, holdout_fraction: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not 0 < holdout_fraction < 1:
            raise ValueError("holdout_fraction must be between 0 and 1")
        split_idx = int(len(df) * (1 - holdout_fraction))
        if split_idx <= 0 or split_idx >= len(df):
            raise ValueError("holdout split results in empty set")
        return df.iloc[:split_idx], df.iloc[split_idx:]

    def rolling_origin(
        self,
        df: pd.DataFrame,
        splits: int,
        min_train_fraction: float,
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
        if splits < 1:
            raise ValueError("splits must be >= 1")
        min_train = int(len(df) * min_train_fraction)
        step = max(1, math.floor((len(df) - min_train) / (splits + 1)))
        for i in range(splits):
            cutoff = min_train + i * step
            train = df.iloc[:cutoff]
            val = df.iloc[cutoff : cutoff + step]
            if len(val) == 0:
                break
            yield train, val


def load_csv(path: str, timestamp_col: str | None, target_col: str, freq: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return TimeSeriesFrame(timestamp_col=timestamp_col, target_col=target_col, freq=freq).load(frame)
