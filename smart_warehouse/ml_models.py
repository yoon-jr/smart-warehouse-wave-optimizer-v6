from __future__ import annotations

import numpy as np
import pandas as pd


class NaiveLagRegressor:
    """Pickle-friendly baseline that predicts the previous bin count."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NaiveLagRegressor":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["lag_1"].to_numpy(dtype=float)
