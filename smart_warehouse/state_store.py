from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any


class StateStore:
    def __init__(self, state_path: Path, metrics_path: Path) -> None:
        self.state_path = state_path
        self.metrics_path = metrics_path
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    def write_state(self, state: dict[str, Any]) -> None:
        temp_path = self.state_path.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(state, file, ensure_ascii=False, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_path, self.state_path)

    def append_metrics(self, metrics: dict[str, Any]) -> None:
        file_exists = self.metrics_path.exists()
        with self.metrics_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(metrics.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)
