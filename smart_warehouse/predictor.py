from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from smart_warehouse.features import live_feature_rows


class DemandPredictor:
    def __init__(self, model_path: Path, metadata_path: Path) -> None:
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.model = joblib.load(model_path) if model_path.exists() else None
        self.metadata = {}
        if metadata_path.exists():
            self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    @property
    def available(self) -> bool:
        return self.model is not None

    @property
    def model_name(self) -> str:
        return self.metadata.get("best_model", "fallback_recent_average")

    def predict(self, current_minute: float, recent_counts: dict[str, list[int]]) -> dict[str, float]:
        features = live_feature_rows(current_minute, recent_counts)
        if self.model is None:
            values = features["rolling_3"].to_numpy(dtype=float)
        else:
            values = np.asarray(self.model.predict(features), dtype=float)
        values = np.clip(values, 0, None)
        return {zone: round(float(value), 3) for zone, value in zip(features["zone"], values)}
