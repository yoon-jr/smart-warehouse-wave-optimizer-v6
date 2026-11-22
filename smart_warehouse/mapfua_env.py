from __future__ import annotations

import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from smart_warehouse.config import DATA_DIR, SimulationConfig
from smart_warehouse.entities import RobotStatus
from smart_warehouse.footwear_replay import FootwearTaskReplay
from smart_warehouse.idle_policy import ExternalJointPolicy, IdleAction
from smart_warehouse.simulation import SimulationEngine

SplitName = Literal["train", "validation", "test"]
DensityStage = Literal["low", "medium", "high", "all"]


@dataclass(frozen=True)
class TemporalSplit:
    start_tick: int
    end_tick: int

    @property
    def length(self) -> int:
        return max(0, self.end_tick - self.start_tick + 1)


def temporal_splits(frame: pd.DataFrame) -> dict[SplitName, TemporalSplit]:
    """Create chronological 70/15/15 splits by task release order."""
    ordered = frame.sort_values(["release_tick", "source_sequence", "task_id"], kind="stable")
    ticks = ordered["release_tick"].astype(int).to_numpy()
    if len(ticks) < 3:
        start, end = int(ticks.min()), int(ticks.max())
        split = TemporalSplit(start, end)
        return {"train": split, "validation": split, "test": split}
    train_index = min(len(ticks) - 1, max(0, int(len(ticks) * 0.70) - 1))
    validation_index = min(len(ticks) - 1, max(train_index + 1, int(len(ticks) * 0.85) - 1))
    start = int(ticks[0])
    train_end = int(ticks[train_index])
    validation_end = int(ticks[validation_index])
    end = int(ticks[-1])
    return {
        "train": TemporalSplit(start, train_end),
        "validation": TemporalSplit(train_end + 1, validation_end),
        "test": TemporalSplit(validation_end + 1, end),
    }


def build_agent_observations(
    engine: SimulationEngine,
    *,
    include_forecast: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Build fixed-size per-robot observations and legal-action masks."""
    queue = engine.queue_counts()
    robot_counts = engine.current_robot_counts(include_committed=True)
    density = engine.local_density_scores()
    blocking = engine.blocking_scores()
    recent = {
        zone: float((engine.recent_counts[zone][-1] if engine.recent_counts[zone] else 0) + engine.current_bin_counts[zone])
        for zone in "ABCD"
    }
    forecasts = engine.predictions if include_forecast else {zone: 0.0 for zone in "ABCD"}
    scale = max(1.0, float(engine.config.robot_count))
    task_scale = max(4.0, scale * 2.0)

    observations: list[list[float]] = []
    masks: list[np.ndarray] = []
    for robot in sorted(engine.robots, key=lambda item: item.robot_id):
        zone = engine.warehouse.zone_for(robot.position)
        zone_one_hot = [1.0 if zone == item else 0.0 for item in "ABCD"]
        status_one_hot = [
            1.0 if robot.status == RobotStatus.IDLE else 0.0,
            1.0 if robot.status == RobotStatus.REPOSITIONING else 0.0,
            1.0 if robot.status == RobotStatus.MOVING_TO_TASK else 0.0,
            1.0 if robot.status == RobotStatus.WORKING else 0.0,
        ]
        feature = [
            *zone_one_hot,
            *status_one_hot,
            float(robot.battery) / 100.0,
            float(density.get(robot.robot_id, 0.0)),
            float(blocking.get(robot.robot_id, 0.0)) / scale,
            *[min(1.0, float(queue[item]) / task_scale) for item in "ABCD"],
            *[float(robot_counts[item]) / scale for item in "ABCD"],
            *[min(1.0, recent[item] / task_scale) for item in "ABCD"],
            *[min(1.0, float(forecasts[item]) / task_scale) for item in "ABCD"],
            min(1.0, float(sum(queue.values())) / task_scale),
        ]
        observations.append(feature)
        masks.append(engine.action_mask(robot))
    return np.asarray(observations, dtype=np.float32), np.asarray(masks, dtype=np.float32)


class LifelongMAPFUAEnv:
    """High-level MARL environment for unassigned warehouse robots.

    Agents are the physical robots. A joint action is requested only every
    ``idle_decision_interval_ticks`` simulator ticks. Task assignment and
    collision-free low-level paths remain controlled by the deterministic
    allocator and reservation-table planner.
    """

    action_count = len(IdleAction)

    def __init__(
        self,
        tasks_path: Path | None = None,
        manifest_path: Path | None = None,
        *,
        config: SimulationConfig | None = None,
        split: SplitName = "train",
        episode_ticks: int = 600,
        include_forecast: bool = False,
        seed: int = 42,
    ) -> None:
        self.tasks_path = tasks_path or (DATA_DIR / "footwear_replay_tasks.csv")
        self.manifest_path = manifest_path or (DATA_DIR / "footwear_manifest.json")
        self.config = config or SimulationConfig()
        self.split_name = split
        self.episode_ticks = int(episode_ticks)
        self.include_forecast = bool(include_forecast)
        self.rng = random.Random(seed)
        self.replay = FootwearTaskReplay(
            self.tasks_path,
            manifest_path=self.manifest_path,
            random_seed=seed,
        )
        self.splits = temporal_splits(self.replay.frame)
        self.engine: SimulationEngine | None = None
        self.window_start_tick = 0
        self.local_ticks = 0
        self._last_metrics: dict[str, float] = {}
        self._obs_dim: int | None = None
        self.density_stage: DensityStage = "all"
        self.window_task_count = 0
        self._window_candidates_cache: dict[tuple[str, int], list[tuple[int, int]]] = {}

    @property
    def robot_ids(self) -> list[str]:
        if self.engine is None:
            return [f"R-{index + 1:02d}" for index in range(self.config.robot_count)]
        return [robot.robot_id for robot in sorted(self.engine.robots, key=lambda item: item.robot_id)]

    @property
    def observation_dim(self) -> int:
        if self._obs_dim is None:
            obs, _ = self.reset()
            self._obs_dim = int(obs.shape[-1])
        return self._obs_dim

    def _window_candidates(self) -> list[tuple[int, int]]:
        """Return release-aligned starts paired with task counts in each window."""
        key = (self.split_name, self.episode_ticks)
        cached = self._window_candidates_cache.get(key)
        if cached is not None:
            return cached

        split = self.splits[self.split_name]
        latest = max(split.start_tick, split.end_tick - self.episode_ticks + 1)
        frame = self.replay.frame
        ticks = frame["release_tick"].astype(int).to_numpy()
        starts = np.unique(ticks[(ticks >= split.start_tick) & (ticks <= latest)])
        if len(starts) == 0:
            starts = np.asarray([split.start_tick], dtype=int)

        # Count tasks in [start, start + episode_ticks) with binary search over
        # the sorted release-tick array. This remains fast for the full WMS CSV
        # and avoids constructing one simulator instance per candidate.
        sorted_ticks = np.sort(ticks)
        candidates: list[tuple[int, int]] = []
        for start_tick in starts.tolist():
            left = int(np.searchsorted(sorted_ticks, start_tick, side="left"))
            right = int(
                np.searchsorted(
                    sorted_ticks,
                    start_tick + self.episode_ticks,
                    side="left",
                )
            )
            candidates.append((int(start_tick), max(0, right - left)))
        self._window_candidates_cache[key] = candidates
        return candidates

    def _density_filtered_candidates(self, stage: DensityStage) -> list[tuple[int, int]]:
        candidates = self._window_candidates()
        if stage == "all" or len(candidates) < 4:
            return candidates
        loads = np.asarray([load for _, load in candidates], dtype=float)
        q35, q65 = np.quantile(loads, [0.35, 0.65])
        if stage == "low":
            selected = [item for item in candidates if item[1] <= q35]
        elif stage == "medium":
            selected = [item for item in candidates if q35 <= item[1] <= q65]
        elif stage == "high":
            selected = [item for item in candidates if item[1] >= q65]
        else:  # defensive fallback for runtime callers bypassing type checking
            selected = candidates
        return selected or candidates

    def _choose_start_tick(
        self,
        fixed_start_tick: int | None,
        density_stage: DensityStage,
    ) -> tuple[int, int]:
        split = self.splits[self.split_name]
        if fixed_start_tick is not None:
            if not split.start_tick <= fixed_start_tick <= split.end_tick:
                raise ValueError(
                    f"fixed_start_tick={fixed_start_tick} is outside {self.split_name} split "
                    f"[{split.start_tick}, {split.end_tick}]"
                )
            frame = self.replay.frame
            count = int(
                frame["release_tick"].astype(int).between(
                    fixed_start_tick,
                    fixed_start_tick + self.episode_ticks - 1,
                    inclusive="both",
                ).sum()
            )
            return int(fixed_start_tick), count
        candidates = self._density_filtered_candidates(density_stage)
        return self.rng.choice(candidates)

    def _initial_minute(self, start_tick: int) -> float:
        future = self.replay.frame[self.replay.frame["release_tick"].astype(int) >= start_tick]
        if future.empty:
            return self.replay.start_minute
        created = float(future.iloc[0]["created_minute"])
        return created - self.config.simulation_minutes_per_tick

    def reset(
        self,
        *,
        seed: int | None = None,
        fixed_start_tick: int | None = None,
        density_stage: DensityStage = "all",
    ) -> tuple[np.ndarray, np.ndarray]:
        if seed is not None:
            self.rng.seed(seed)
        if density_stage not in {"low", "medium", "high", "all"}:
            raise ValueError(f"Unknown density_stage: {density_stage}")
        self.density_stage = density_stage
        self.window_start_tick, self.window_task_count = self._choose_start_tick(
            fixed_start_tick,
            density_stage,
        )
        self.local_ticks = 0
        config = replace(
            self.config,
            snapshot_interval_ticks=self.episode_ticks + 1,
        )
        self.engine = SimulationEngine(
            config,
            mode="reactive",
            persist=False,
            initial_minute=self._initial_minute(self.window_start_tick),
            idle_policy=ExternalJointPolicy(),
        )
        if self.include_forecast:
            self.engine.refresh_predictions()
        self._last_metrics = self.engine.metrics_summary()
        obs, masks = self._observations()
        self._obs_dim = int(obs.shape[-1])
        return obs, masks

    def _observations(self) -> tuple[np.ndarray, np.ndarray]:
        if self.engine is None:
            raise RuntimeError("Environment must be reset before requesting observations")
        return build_agent_observations(self.engine, include_forecast=self.include_forecast)

    def step(
        self,
        actions: np.ndarray | list[int] | dict[str, int],
    ) -> tuple[np.ndarray, np.ndarray, float, bool, dict]:
        if self.engine is None:
            raise RuntimeError("Environment must be reset before stepping")
        engine = self.engine
        if isinstance(actions, dict):
            action_map = {str(key): int(value) for key, value in actions.items()}
        else:
            values = np.asarray(actions, dtype=int).reshape(-1)
            if len(values) != len(self.robot_ids):
                raise ValueError(f"Expected {len(self.robot_ids)} actions, got {len(values)}")
            action_map = dict(zip(self.robot_ids, values.tolist()))
        engine.apply_idle_actions(action_map)

        before = engine.metrics_summary()
        blocking_before = engine.blocking_pressure_ticks
        invalid_before = engine.invalid_idle_actions
        remaining = min(
            engine.config.idle_decision_interval_ticks,
            self.episode_ticks - self.local_ticks,
        )
        for _ in range(max(0, remaining)):
            global_tick = self.window_start_tick + self.local_ticks
            events = self.replay.tasks_for_tick(global_tick)
            engine.step(external_tasks=events)
            self.local_ticks += 1
            if self.local_ticks >= self.episode_ticks:
                break

        if self.include_forecast:
            engine.refresh_predictions()
        after = engine.metrics_summary()
        completed_delta = float(after["completed_tasks"] - before["completed_tasks"])
        distance_delta = float(after["total_distance"] - before["total_distance"])
        idle_distance_delta = float(after["idle_reposition_distance"] - before["idle_reposition_distance"])
        blocking_delta = float(engine.blocking_pressure_ticks - blocking_before)
        invalid_delta = float(engine.invalid_idle_actions - invalid_before)
        waiting = float(after["waiting_tasks"])

        # A deliberately compact reward. Every term is observable and logged so
        # ablation can reveal whether the policy is exploiting one component.
        reward = (
            2.0 * completed_delta
            - 0.05 * waiting
            - 0.02 * distance_delta
            - 0.03 * idle_distance_delta
            - 0.01 * blocking_delta
            - 0.50 * invalid_delta
        ) / max(1.0, float(engine.config.robot_count))

        done = self.local_ticks >= self.episode_ticks
        obs, masks = self._observations()
        info = {
            "window_start_tick": self.window_start_tick,
            "window_task_count": self.window_task_count,
            "density_stage": self.density_stage,
            "local_ticks": self.local_ticks,
            "completed_delta": completed_delta,
            "waiting_tasks": waiting,
            "distance_delta": distance_delta,
            "idle_distance_delta": idle_distance_delta,
            "blocking_pressure_delta": blocking_delta,
            "invalid_action_delta": invalid_delta,
            "metrics": after,
        }
        self._last_metrics = after
        return obs, masks, float(reward), done, info
