from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from smart_warehouse.entities import Task


REQUIRED_COLUMNS = {
    "task_id",
    "release_tick",
    "created_minute",
    "zone",
    "x",
    "y",
    "priority",
    "quantity",
    "service_ticks",
    "wave_id",
    "source_location",
    "source_sequence",
}


class FootwearTaskReplay:
    """Immutable task stream prepared from the Footwear WMS dataset."""

    def __init__(
        self,
        tasks_path: Path,
        *,
        manifest_path: Path | None = None,
        random_seed: int = 42,
    ) -> None:
        if not tasks_path.exists():
            raise FileNotFoundError(
                f"Prepared Footwear task file not found: {tasks_path}. "
                "Run `python -m scripts.prepare_footwear_data --dataset-dir <folder>` first."
            )
        frame = pd.read_csv(tasks_path, low_memory=False)
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"{tasks_path} is missing required columns: {sorted(missing)}")
        if frame.empty:
            raise ValueError(f"{tasks_path} contains no tasks")

        numeric_columns = ["release_tick", "created_minute", "x", "y", "priority", "quantity", "service_ticks"]
        for column in numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=numeric_columns).copy()
        frame["release_tick"] = frame["release_tick"].astype(int)
        frame["x"] = frame["x"].astype(int)
        frame["y"] = frame["y"].astype(int)
        frame["priority"] = frame["priority"].astype(int)
        frame["service_ticks"] = frame["service_ticks"].astype(int)
        frame["source_sequence"] = pd.to_numeric(frame["source_sequence"], errors="coerce").fillna(0).astype(int)
        frame = frame.sort_values(
            ["release_tick", "wave_id", "source_sequence", "task_id"], kind="stable"
        ).reset_index(drop=True)
        if frame.empty:
            raise ValueError("No valid tasks remained after numeric validation")

        self.tasks_path = tasks_path
        self.frame = frame
        self.rng = random.Random(random_seed)
        self._events: dict[int, list[Task]] = defaultdict(list)
        for row in frame.itertuples(index=False):
            task = Task(
                task_id=str(row.task_id),
                created_minute=float(row.created_minute),
                zone_id=str(row.zone),
                position=(int(row.x), int(row.y)),
                priority=int(row.priority),
                quantity=float(row.quantity),
                service_ticks=max(1, int(row.service_ticks)),
                wave_id=str(row.wave_id),
                source_location=str(row.source_location),
                source_sequence=int(row.source_sequence),
            )
            self._events[int(row.release_tick)].append(task)

        self.start_tick = int(frame["release_tick"].min())
        self.max_tick = int(frame["release_tick"].max())
        self.start_minute = float(frame["created_minute"].min())
        self.task_count = int(len(frame))
        self._surge_counter = 1
        self.manifest = self._load_manifest(manifest_path)

    @staticmethod
    def _load_manifest(path: Path | None) -> dict[str, Any]:
        if path is None or not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}

    @property
    def description(self) -> str:
        dataset = self.manifest.get("dataset", "Footwear Warehouse Order Picking Dataset")
        return f"{dataset} · {self.task_count:,} prepared pick-line tasks"

    @property
    def time_basis(self) -> str:
        return str(self.manifest.get("time_basis", "unknown"))

    @property
    def simulation_minutes_per_tick(self) -> float | None:
        value = self.manifest.get("simulation_minutes_per_tick")
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    def tasks_for_tick(self, tick: int) -> list[Task]:
        return list(self._events.get(int(tick), ()))

    def has_future_tasks(self, tick: int) -> bool:
        return int(tick) < self.max_tick

    def generate_surge(
        self,
        *,
        created_minute: float,
        zone_id: str,
        count: int,
    ) -> list[Task]:
        if count <= 0:
            raise ValueError("count must be positive")
        candidates = self.frame[self.frame["zone"].astype(str) == zone_id]
        if candidates.empty:
            raise ValueError(f"No Footwear task templates are available in Zone {zone_id}")

        # Sampling with replacement preserves actual Footwear locations,
        # quantities, and service-time distribution even when count exceeds the
        # number of unique rows in one zone.
        indices = [self.rng.randrange(len(candidates)) for _ in range(count)]
        rows = candidates.iloc[indices]
        tasks: list[Task] = []
        for row in rows.itertuples(index=False):
            task_id = f"FS-{self._surge_counter:06d}"
            self._surge_counter += 1
            tasks.append(
                Task(
                    task_id=task_id,
                    created_minute=float(created_minute),
                    zone_id=zone_id,
                    position=(int(row.x), int(row.y)),
                    priority=max(2, int(row.priority)),
                    quantity=float(row.quantity),
                    service_ticks=max(1, int(row.service_ticks)),
                    wave_id=f"SURGE-{row.wave_id}",
                    source_location=str(row.source_location),
                    source_sequence=int(row.source_sequence),
                )
            )
        return tasks

    def zone_template_counts(self) -> dict[str, int]:
        counts = self.frame["zone"].astype(str).value_counts()
        return {zone: int(counts.get(zone, 0)) for zone in "ABCD"}
