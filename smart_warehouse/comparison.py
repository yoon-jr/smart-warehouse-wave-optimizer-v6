from __future__ import annotations

from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Any

from smart_warehouse.config import DATA_DIR, RUNTIME_DIR, SimulationConfig
from smart_warehouse.control import read_command
from smart_warehouse.footwear_replay import FootwearTaskReplay
from smart_warehouse.simulation import SimulationEngine
from smart_warehouse.state_store import StateStore


def _reduction_pct(baseline: float, contender: float) -> float:
    if abs(baseline) < 1e-9:
        return 0.0
    return round(100.0 * (baseline - contender) / baseline, 2)


def _increase_pct(baseline: float, contender: float) -> float:
    if abs(baseline) < 1e-9:
        return 0.0
    return round(100.0 * (contender - baseline) / baseline, 2)


class ComparisonRunner:
    """Compare a Reactive baseline with an Adaptive Hybrid controller.

    Both engines receive independent copies of the exact same Footwear task
    stream. The hybrid controller does not replace reactive allocation. It only
    enables predictive idle-robot repositioning when operational thresholds say
    that proactive travel is likely to help.
    """

    def __init__(
        self,
        config: SimulationConfig,
        *,
        persist: bool = True,
        tasks_path: Path | None = None,
        manifest_path: Path | None = None,
    ) -> None:
        self.persist = persist
        self.tasks_path = tasks_path or (DATA_DIR / "footwear_replay_tasks.csv")
        self.manifest_path = manifest_path or (DATA_DIR / "footwear_manifest.json")
        if not self.tasks_path.exists():
            raise FileNotFoundError(
                f"Prepared Footwear tasks were not found at {self.tasks_path}. "
                "Download the Mendeley dataset and run `python -m scripts.prepare_footwear_data "
                "--dataset-dir <extracted-folder>` first."
            )
        self.replay = FootwearTaskReplay(
            self.tasks_path,
            manifest_path=self.manifest_path,
            random_seed=config.random_seed,
        )
        initial_minute = self.replay.start_minute
        prepared_tick_scale = self.replay.simulation_minutes_per_tick
        if prepared_tick_scale is not None and abs(prepared_tick_scale - config.simulation_minutes_per_tick) > 1e-9:
            config = replace(config, simulation_minutes_per_tick=prepared_tick_scale)

        self.config = config

        self.reactive = SimulationEngine(
            config,
            mode="reactive",
            persist=False,
            initial_minute=initial_minute,
        )
        self.hybrid = SimulationEngine(
            config,
            mode="hybrid",
            persist=False,
            initial_minute=initial_minute,
        )
        self.store = StateStore(
            RUNTIME_DIR / "comparison_state.json",
            RUNTIME_DIR / "comparison_metrics.csv",
        )
        self.control_path = RUNTIME_DIR / "control.json"
        existing_command = read_command(self.control_path)
        self.last_command_id: str | None = (
            str(existing_command.get("command_id"))
            if existing_command and existing_command.get("command_id")
            else None
        )
        self.scheduled_surges: list[dict[str, Any]] = []
        self.event_log: deque[dict[str, Any]] = deque(maxlen=30)
        self.paused = False
        self._last_logged_policy = self.hybrid.active_policy
        self._log(
            "SYSTEM_START",
            "Reactive baseline and Adaptive Hybrid started on one shared task stream.",
            source=self.data_source_label,
        )

    @property
    def predictive(self) -> SimulationEngine:
        """Compatibility alias for older notebooks; the contender is now Hybrid."""
        return self.hybrid

    @property
    def tick_count(self) -> int:
        return self.reactive.tick_count

    @property
    def current_minute(self) -> float:
        return self.reactive.current_minute

    @property
    def data_source_label(self) -> str:
        return self.replay.description

    @property
    def replay_finished(self) -> bool:
        return not self.replay.has_future_tasks(self.tick_count)

    def _log(self, event_type: str, message: str, **details: Any) -> None:
        self.event_log.appendleft(
            {
                "tick": self.tick_count,
                "minute": round(self.current_minute, 2),
                "type": event_type,
                "message": message,
                **details,
            }
        )

    def reset(self) -> None:
        self.__init__(
            self.config,
            persist=self.persist,
            tasks_path=self.tasks_path,
            manifest_path=self.manifest_path,
        )

    def trigger_order_surge(self, zone_id: str = "C", count: int | None = None) -> None:
        surge_count = count if count is not None else self.config.order_surge_count
        if surge_count <= 0:
            raise ValueError("surge count must be positive")
        if zone_id not in self.reactive.warehouse.zones:
            raise ValueError("zone_id must be A, B, C, or D")

        due_tick = self.tick_count + self.config.surge_lead_ticks
        due_minute = self.current_minute + self.config.surge_lead_ticks * self.config.simulation_minutes_per_tick
        tasks = self.replay.generate_surge(
            created_minute=due_minute,
            zone_id=zone_id,
            count=surge_count,
        )

        self.scheduled_surges.append(
            {
                "due_tick": due_tick,
                "zone_id": zone_id,
                "count": surge_count,
                "tasks": tasks,
            }
        )
        self._log(
            "SURGE_FORECAST",
            f"Advance notice: {surge_count} Footwear-pattern tasks will arrive in Zone {zone_id} in "
            f"{self.config.surge_lead_ticks} ticks.",
            zone_id=zone_id,
            count=surge_count,
            due_tick=due_tick,
        )
        self._apply_active_forecasts(refresh_now=True)

    def _poll_control(self) -> None:
        command = read_command(self.control_path)
        if not command:
            return
        command_id = str(command.get("command_id", ""))
        if not command_id or command_id == self.last_command_id:
            return
        self.last_command_id = command_id
        if command.get("action") == "order_surge":
            self.trigger_order_surge(
                str(command.get("zone_id", "C")),
                int(command.get("count", self.config.order_surge_count)),
            )

    def _active_forecast_boosts(self) -> dict[str, float]:
        boosts = {zone_id: 0.0 for zone_id in self.reactive.warehouse.zones}
        for surge in self.scheduled_surges:
            if int(surge["due_tick"]) > self.tick_count:
                zone_id = str(surge["zone_id"])
                boosts[zone_id] += float(surge["count"]) * self.config.surge_forecast_strength
        return boosts

    def _apply_active_forecasts(self, *, refresh_now: bool) -> None:
        boosts = self._active_forecast_boosts()
        self.hybrid.set_forecast_boosts(
            boosts,
            notice_active=any(value > 0 for value in boosts.values()),
            refresh_now=refresh_now,
        )

    def _shared_events_for_tick(self, next_tick: int, next_minute: float) -> list:
        return self.replay.tasks_for_tick(next_tick)

    def step(self) -> None:
        if self.paused:
            return
        self._poll_control()
        next_tick = self.tick_count + 1
        next_minute = self.current_minute + self.config.simulation_minutes_per_tick
        events = self._shared_events_for_tick(next_tick, next_minute)

        released_surges: list[dict[str, Any]] = []
        remaining_surges: list[dict[str, Any]] = []
        for surge in self.scheduled_surges:
            if int(surge["due_tick"]) <= next_tick:
                events.extend(surge["tasks"])
                released_surges.append(surge)
            else:
                remaining_surges.append(surge)
        self.scheduled_surges = remaining_surges

        # SimulationEngine clones every task before insertion, so the two
        # engines cannot mutate each other's Task objects.
        self.reactive.step(external_tasks=events)
        self.hybrid.step(external_tasks=events)

        if released_surges:
            for surge in released_surges:
                self._log(
                    "SURGE_RELEASED",
                    f"Released {surge['count']} identical tasks to both systems in Zone {surge['zone_id']}.",
                    zone_id=surge["zone_id"],
                    count=surge["count"],
                )
            # Remove expired notice and immediately reconsider policy using the
            # newly created queue. This is what makes overload switch safely to
            # Reactive instead of continuing unnecessary repositioning.
            self._apply_active_forecasts(refresh_now=True)

        if self.hybrid.active_policy != self._last_logged_policy:
            decision = self.hybrid.policy_decision
            self._log(
                "POLICY_SWITCH",
                f"Adaptive Hybrid switched to {self.hybrid.active_policy}. "
                f"{decision.reason if decision else ''}",
                active_policy=self.hybrid.active_policy,
                score=decision.score if decision else None,
            )
            self._last_logged_policy = self.hybrid.active_policy

        if self.persist and self.tick_count % self.config.snapshot_interval_ticks == 0:
            self.store.write_state(self.snapshot())

    def improvements(self) -> dict[str, float]:
        baseline = self.reactive.metrics_summary()
        hybrid = self.hybrid.metrics_summary()
        return {
            "wait_time_reduction_pct": _reduction_pct(
                float(baseline["average_wait_time"]), float(hybrid["average_wait_time"])
            ),
            "p95_wait_reduction_pct": _reduction_pct(
                float(baseline["p95_wait_time"]), float(hybrid["p95_wait_time"])
            ),
            "throughput_increase_pct": _increase_pct(
                float(baseline["throughput_per_hour"]), float(hybrid["throughput_per_hour"])
            ),
            "completed_increase_pct": _increase_pct(
                float(baseline["completed_tasks"]), float(hybrid["completed_tasks"])
            ),
            "distance_reduction_pct": _reduction_pct(
                float(baseline["total_distance"]), float(hybrid["total_distance"])
            ),
            "task_distance_reduction_pct": _reduction_pct(
                float(baseline["task_distance"]), float(hybrid["task_distance"])
            ),
        }

    def situation_summary(self) -> dict[str, Any]:
        metrics = self.hybrid.metrics_summary()
        queue_counts = self.hybrid.queue_counts()
        waiting = int(metrics["waiting_tasks"])
        assigned = int(metrics["assigned_tasks"])
        in_progress = int(metrics["in_progress_tasks"])
        idle = sum(robot.status.value == "IDLE" for robot in self.hybrid.robots)
        repositioning = sum(robot.status.value == "REPOSITIONING" for robot in self.hybrid.robots)
        queue_per_robot = waiting / max(1, self.config.robot_count)

        if queue_per_robot >= self.config.adaptive_overload_queue_per_robot:
            level = "OVERLOADED"
            headline = "Demand is above the current robot capacity"
        elif queue_per_robot >= 0.5 or idle == 0:
            level = "BUSY"
            headline = "All or most robots are occupied"
        else:
            level = "NORMAL"
            headline = "The warehouse is operating normally"

        if waiting > 0:
            hotspot = max(queue_counts, key=lambda zone: (queue_counts[zone], zone))
            hotspot_value = queue_counts[hotspot]
            hotspot_phrase = f"Zone {hotspot} has the largest waiting queue ({hotspot_value} tasks)."
        else:
            predictions = self.hybrid.predictions
            hotspot = max(predictions, key=lambda zone: (predictions[zone], zone))
            hotspot_value = float(predictions[hotspot])
            hotspot_phrase = f"Zone {hotspot} is the next forecast hotspot ({hotspot_value:.1f} tasks)."

        upcoming = None
        if self.scheduled_surges:
            next_surge = min(self.scheduled_surges, key=lambda item: int(item["due_tick"]))
            upcoming = {
                "zone_id": str(next_surge["zone_id"]),
                "count": int(next_surge["count"]),
                "ticks_remaining": max(0, int(next_surge["due_tick"]) - self.tick_count),
            }
            upcoming_phrase = (
                f" Advance notice is active: {upcoming['count']} additional tasks are expected in "
                f"Zone {upcoming['zone_id']} in {upcoming['ticks_remaining']} ticks."
            )
        else:
            upcoming_phrase = ""

        decision = self.hybrid.policy_decision
        reason = decision.reason if decision else "No adaptive decision has been made yet."
        if self.hybrid.active_policy == "PREDICTIVE":
            if repositioning:
                action = (
                    f"{repositioning} purple robot(s) are moving toward the forecast hotspot before "
                    "the expected tasks are released."
                )
            else:
                action = (
                    "Predictive mode is permitted, but no extra move is needed because the current "
                    "robot distribution already satisfies the target."
                )
        elif level == "OVERLOADED":
            action = (
                "The controller has disabled proactive movement and is using every available robot "
                "to clear already-released tasks."
            )
        else:
            action = "Robots remain task-driven; no proactive repositioning is allowed right now."

        waiting_phrase = f"{waiting} waiting task" if waiting == 1 else f"{waiting} waiting tasks"
        assigned_phrase = f"{assigned} task is assigned" if assigned == 1 else f"{assigned} tasks are assigned"
        progress_phrase = (
            f"{in_progress} task is being processed"
            if in_progress == 1
            else f"{in_progress} tasks are being processed"
        )

        return {
            "level": level,
            "headline": headline,
            "waiting_tasks": waiting,
            "assigned_tasks": assigned,
            "in_progress_tasks": in_progress,
            "idle_robots": idle,
            "repositioning_robots": repositioning,
            "hotspot_zone": hotspot,
            "hotspot_value": hotspot_value,
            "summary": (
                f"The red markers represent {waiting_phrase}; {assigned_phrase}, and {progress_phrase}. "
                f"{idle}/{self.config.robot_count} robots are idle. {hotspot_phrase}{upcoming_phrase}"
            ),
            "active_policy": self.hybrid.active_policy,
            "policy_reason": reason,
            "current_action": action,
            "upcoming_surge": upcoming,
            "map_legend": {
                "red_square": "released task waiting for a robot",
                "blue_circle": "idle robot",
                "green_circle_or_line": "robot moving to a released task",
                "purple_circle_or_line": "predictive repositioning before task release",
                "orange_circle": "robot processing a task",
                "dark_block": "shelf or fixed obstacle",
            },
            "replay_finished": self.replay_finished,
        }

    def snapshot(self) -> dict[str, Any]:
        scheduled = [
            {
                "due_tick": int(surge["due_tick"]),
                "zone_id": str(surge["zone_id"]),
                "count": int(surge["count"]),
                "ticks_remaining": max(0, int(surge["due_tick"]) - self.tick_count),
            }
            for surge in self.scheduled_surges
        ]
        manifest = self.replay.manifest
        return {
            "schema_version": 2,
            "tick": self.tick_count,
            "current_minute": round(self.current_minute, 2),
            "paused": self.paused,
            "fairness": {
                "same_task_stream": True,
                "same_release_times": True,
                "same_pickup_locations": True,
                "same_task_attributes": True,
                "same_robot_count_and_initial_layout": True,
            },
            "scenario": {
                "shared_task_stream": True,
                "random_seed": self.config.random_seed,
                "data_source": "footwear_wms",
                "description": self.data_source_label,
                "source_dataset": manifest.get("dataset"),
                "source_rows_used": self.replay.task_count,
                "license": manifest.get("license"),
                "doi": manifest.get("doi"),
                "time_basis": self.replay.time_basis,
                "surge_forecast_lead_ticks": self.config.surge_lead_ticks,
                "comparison": "Reactive baseline vs threshold-based Adaptive Hybrid",
            },
            "scheduled_surges": scheduled,
            "reactive": self.reactive.snapshot(),
            "hybrid": self.hybrid.snapshot(),
            "improvements": self.improvements(),
            "situation": self.situation_summary(),
            "event_log": list(self.event_log),
        }
