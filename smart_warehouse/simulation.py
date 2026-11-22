from __future__ import annotations

from collections import defaultdict
from statistics import fmean
from typing import Any
import time

import numpy as np

from smart_warehouse.adaptive_policy import AdaptivePolicyController, PolicyDecision
from smart_warehouse.allocator import AllocationDecision, ExplainableAllocator
from smart_warehouse.astar import astar
from smart_warehouse.config import MODEL_DIR, RUNTIME_DIR, SimulationConfig
from smart_warehouse.entities import Robot, RobotStatus, Task, TaskStatus
from smart_warehouse.idle_policy import (
    DensityClearPolicy,
    IdleAction,
    IdleContext,
    IdleDecision,
    IdlePolicy,
    actionable_idle_robots,
)
from smart_warehouse.placement import reposition_idle_robots
from smart_warehouse.predictor import DemandPredictor
from smart_warehouse.reservations import detect_conflicts, plan_for_robot
from smart_warehouse.state_store import StateStore
from smart_warehouse.task_utils import clone_task
from smart_warehouse.warehouse import WarehouseMap


class SimulationEngine:
    """Warehouse simulator with safe low-level planning and optional MAPFUA policy.

    v5 keeps task assignment and physical path execution deterministic. MARL or
    heuristic policies only control high-level actions of currently unassigned
    robots. Every proposed route is converted into a collision-aware space-time
    path and a synchronous safety shield remains active at execution time.
    """

    def __init__(
        self,
        config: SimulationConfig,
        mode: str = "reactive",
        persist: bool = True,
        *,
        state_filename: str = "live_state.json",
        metrics_filename: str | None = None,
        initial_minute: float | None = None,
        idle_policy: IdlePolicy | None = None,
    ) -> None:
        if mode not in {"reactive", "predictive", "hybrid"}:
            raise ValueError("mode must be 'reactive', 'predictive', or 'hybrid'.")
        self.config = config
        self.mode = mode
        self.persist = persist
        self.state_filename = state_filename
        self.metrics_filename = metrics_filename or f"metrics_{mode}.csv"
        self.idle_policy = idle_policy
        self.controller_name = idle_policy.name if idle_policy is not None else mode

        self.warehouse = WarehouseMap.create_default(config.grid_width, config.grid_height)
        self.allocator = ExplainableAllocator(
            config.min_battery_for_assignment,
            planning_horizon_steps=config.mapf_planning_horizon_steps,
            goal_hold_steps=config.mapf_goal_hold_steps,
        )
        self.predictor = DemandPredictor(
            MODEL_DIR / "best_model.joblib",
            MODEL_DIR / "model_metrics.json",
        )
        self.store = StateStore(
            RUNTIME_DIR / self.state_filename,
            RUNTIME_DIR / self.metrics_filename,
        )

        self.tick_count = 0
        self.start_minute = float(config.start_minute if initial_minute is None else initial_minute)
        self.current_minute = self.start_minute
        self.tasks: dict[str, Task] = {}
        self.robots = self._create_robots(config.robot_count)
        self.completed_tasks: list[Task] = []
        self.last_decision: AllocationDecision | None = None
        self.base_predictions = {zone_id: 0.0 for zone_id in self.warehouse.zones}
        self.predictions = {zone_id: 0.0 for zone_id in self.warehouse.zones}
        self.forecast_boosts = {zone_id: 0.0 for zone_id in self.warehouse.zones}
        self.forecast_notice_active = False
        self.target_robot_counts = {zone_id: 0 for zone_id in self.warehouse.zones}
        self.current_bin_counts: defaultdict[str, int] = defaultdict(int)
        self.recent_counts = {zone_id: [] for zone_id in self.warehouse.zones}
        self.current_bin_index = int(self.current_minute // 5)
        self.paused = False

        self.adaptive_controller = AdaptivePolicyController(config) if mode == "hybrid" else None
        self.policy_decision: PolicyDecision | None = (
            self.adaptive_controller.last_decision if self.adaptive_controller else None
        )
        self.active_policy = "PREDICTIVE" if mode == "predictive" else "REACTIVE"
        self.policy_usage_ticks = {"REACTIVE": 0, "PREDICTIVE": 0}
        self.last_repositioned_count = 0

        # Collision and MAPFUA instrumentation.
        self.vertex_conflicts_detected = 0
        self.edge_conflicts_detected = 0
        self.collision_shield_waits = 0
        self.deadlock_count = 0
        self.path_plan_failures = 0
        self.task_assignment_deferrals = 0
        self.blocking_pressure_ticks = 0.0
        self.idle_move_events = 0
        self.clear_move_events = 0
        self.preposition_move_events = 0
        self.parking_move_events = 0
        self.invalid_idle_actions = 0
        self.useful_idle_move_events = 0
        self.unnecessary_idle_move_events = 0
        self._idle_robots_moved: set[str] = set()
        self._idle_event_counter = 0
        self._idle_events: list[dict[str, Any]] = []
        self.last_idle_decisions: list[IdleDecision] = []
        self.idle_decision_calls = 0
        self._idle_decision_latencies_ms: list[float] = []

    def _create_robots(self, count: int) -> list[Robot]:
        parking = list(self.warehouse.parking_nodes())
        if count > len(parking):
            # Fallback for unusually large fleets: fill every walkable cell in
            # deterministic order. The default 20x20 map has ample capacity.
            for y in range(self.warehouse.height):
                for x in range(self.warehouse.width):
                    position = (x, y)
                    if self.warehouse.walkable(position) and position not in parking:
                        parking.append(position)
        if count > len(parking):
            raise ValueError("robot_count exceeds the number of unique walkable start cells")
        return [Robot(robot_id=f"R-{index + 1:02d}", position=parking[index]) for index in range(count)]

    def reset(self) -> None:
        self.__init__(
            self.config,
            self.mode,
            self.persist,
            state_filename=self.state_filename,
            metrics_filename=self.metrics_filename,
            initial_minute=self.start_minute,
            idle_policy=self.idle_policy,
        )

    def add_tasks(self, tasks: list[Task]) -> None:
        for source_task in tasks:
            if source_task.task_id in self.tasks:
                raise ValueError(f"Duplicate task_id: {source_task.task_id}")
            task = clone_task(source_task)
            self.tasks[task.task_id] = task
            self.current_bin_counts[task.zone_id] += 1

    def set_forecast_boosts(
        self,
        boosts: dict[str, float] | None,
        *,
        notice_active: bool,
        refresh_now: bool = False,
    ) -> None:
        self.forecast_boosts = {
            zone_id: max(0.0, float((boosts or {}).get(zone_id, 0.0)))
            for zone_id in self.warehouse.zones
        }
        self.forecast_notice_active = bool(notice_active)
        if refresh_now and (self.mode in {"predictive", "hybrid"} or self._idle_uses_forecast):
            self.refresh_forecast_and_policy()

    @property
    def _idle_uses_forecast(self) -> bool:
        return bool(self.idle_policy and self.idle_policy.uses_forecast)

    def refresh_predictions(self) -> None:
        self.base_predictions = self.predictor.predict(self.current_minute, self.recent_counts)
        self.predictions = {
            zone_id: round(
                max(0.0, float(self.base_predictions.get(zone_id, 0.0)))
                + max(0.0, float(self.forecast_boosts.get(zone_id, 0.0))),
                3,
            )
            for zone_id in self.warehouse.zones
        }

    def refresh_forecast_and_policy(self) -> None:
        self.refresh_predictions()
        before = sum(robot.status == RobotStatus.REPOSITIONING for robot in self.robots)
        if self.mode == "predictive":
            self.active_policy = "PREDICTIVE"
            self.target_robot_counts = reposition_idle_robots(
                self.robots,
                self.warehouse,
                self.predictions,
                minimum_difference=self.config.regular_reposition_min_difference,
                max_moves=self.config.max_reposition_robots_per_cycle,
                planning_horizon_steps=self.config.mapf_planning_horizon_steps,
                goal_hold_steps=self.config.mapf_goal_hold_steps,
                current_tick=self.tick_count,
            )
        elif self.mode == "hybrid":
            assert self.adaptive_controller is not None
            self.policy_decision = self.adaptive_controller.evaluate(
                predictions=self.predictions,
                queue_counts=self.queue_counts(),
                robots=self.robots,
                recent_counts=self.recent_counts,
                tick=self.tick_count,
                forecast_notice_active=self.forecast_notice_active,
            )
            self.active_policy = self.policy_decision.active_policy
            if self.active_policy == "PREDICTIVE":
                self.target_robot_counts = reposition_idle_robots(
                    self.robots,
                    self.warehouse,
                    self.predictions,
                    minimum_difference=1,
                    max_moves=self.config.max_reposition_robots_per_cycle,
                    max_target_fraction=self.config.adaptive_max_robot_share_per_zone,
                    planning_horizon_steps=self.config.mapf_planning_horizon_steps,
                    goal_hold_steps=self.config.mapf_goal_hold_steps,
                    current_tick=self.tick_count,
                )
            else:
                self._cancel_proactive_moves()
                self.target_robot_counts = self.current_robot_counts(include_committed=True)
        after = sum(robot.status == RobotStatus.REPOSITIONING for robot in self.robots)
        self.last_repositioned_count = max(0, after - before)

    def _cancel_proactive_moves(self) -> None:
        for robot in self.robots:
            if robot.status == RobotStatus.REPOSITIONING and robot.current_task_id is None:
                robot.path = []
                robot.status = RobotStatus.IDLE
                robot.target_zone_id = None
                robot.idle_action = "STAY"
                robot.idle_action_reason = "Proactive move cancelled by reactive fallback."

    def step(self, external_tasks: list[Task] | None = None) -> None:
        if self.paused:
            return
        self.tick_count += 1
        self.current_minute += self.config.simulation_minutes_per_tick
        self._roll_time_bins()

        if external_tasks:
            self.add_tasks(external_tasks)

        forecast_due = self.tick_count % self.config.prediction_interval_ticks == 0
        if forecast_due and (self.mode in {"predictive", "hybrid"} or self._idle_uses_forecast):
            self.refresh_forecast_and_policy()

        self._assign_waiting_tasks()

        if (
            self.idle_policy is not None
            and self.idle_policy.name != "external_marl"
            and self.tick_count % self.config.idle_decision_interval_ticks == 0
        ):
            self.apply_idle_policy()

        if self.tick_count % self.config.movement_interval_ticks == 0:
            self._move_robots()

        self._work_on_tasks()
        self._update_idle_event_outcomes()
        self.blocking_pressure_ticks += sum(self.blocking_scores().values())
        self.policy_usage_ticks[self.active_policy] += 1

        if self.persist and self.tick_count % self.config.snapshot_interval_ticks == 0:
            self.store.write_state(self.snapshot())

    def _roll_time_bins(self) -> None:
        bin_index = int(self.current_minute // 5)
        while self.current_bin_index < bin_index:
            for zone_id in self.recent_counts:
                self.recent_counts[zone_id].append(self.current_bin_counts[zone_id])
                self.recent_counts[zone_id] = self.recent_counts[zone_id][-12:]
                self.current_bin_counts[zone_id] = 0
            self.current_bin_index += 1

    def _assign_waiting_tasks(self) -> None:
        waiting = sorted(
            (task for task in self.tasks.values() if task.status == TaskStatus.WAITING),
            key=lambda task: (-task.priority, task.created_minute, task.task_id),
        )
        for task in waiting:
            decision = self.allocator.allocate(task, self.robots, self.warehouse, self.current_minute)
            if decision:
                self.last_decision = decision
                self._mark_preposition_useful(decision.robot_id, task.zone_id)
            elif any(robot.available for robot in self.robots):
                # No safe assignment is currently available. This often means
                # a repeated pick location is occupied by a working robot, so
                # record a temporary deferral rather than a planner failure.
                self.task_assignment_deferrals += 1

    @staticmethod
    def _movement_priority(robot: Robot, proposed: tuple[int, int]) -> tuple[int, str]:
        if proposed == robot.position:
            return (0, robot.robot_id)  # stationary occupancy wins
        if robot.status == RobotStatus.MOVING_TO_TASK:
            return (1, robot.robot_id)
        if robot.status == RobotStatus.REPOSITIONING:
            return (2, robot.robot_id)
        return (3, robot.robot_id)

    def _move_robots(self) -> None:
        current = {robot.robot_id: robot.position for robot in self.robots}
        proposed = {
            robot.robot_id: (robot.path[0] if robot.path else robot.position)
            for robot in self.robots
        }
        robot_by_id = {robot.robot_id: robot for robot in self.robots}

        if self.config.collision_shield_enabled:
            for _ in range(max(1, len(self.robots) * 2)):
                vertex, edge = detect_conflicts(current, proposed)
                if not vertex and not edge:
                    break
                self.vertex_conflicts_detected += len(vertex)
                self.edge_conflicts_detected += len(edge)
                losers: set[str] = set()
                for first, second, _ in vertex:
                    winner = min(
                        (first, second),
                        key=lambda rid: self._movement_priority(robot_by_id[rid], proposed[rid]),
                    )
                    losers.add(second if winner == first else first)
                for first, second, _, _ in edge:
                    # Letting one robot wait still blocks the other's destination,
                    # so both wait and replan at the next allocation/idle cycle.
                    losers.update((first, second))
                changed = False
                for robot_id in losers:
                    if proposed[robot_id] != current[robot_id]:
                        proposed[robot_id] = current[robot_id]
                        self.collision_shield_waits += 1
                        changed = True
                if not changed:
                    self.deadlock_count += 1
                    break

        final_vertex, final_edge = detect_conflicts(current, proposed)
        if final_vertex or final_edge:
            # Last-resort safety: freeze the entire fleet for this movement step.
            proposed = dict(current)
            self.deadlock_count += 1

        for robot in self.robots:
            next_position = proposed[robot.robot_id]
            planned_next = robot.path[0] if robot.path else robot.position
            forced_wait = next_position == robot.position and planned_next != robot.position
            if not forced_wait and robot.path:
                robot.path.pop(0)

            moved = next_position != robot.position
            if moved:
                robot.position = next_position
                robot.total_distance += 1
                robot.battery = max(0.0, robot.battery - 0.05)
                if robot.status == RobotStatus.REPOSITIONING:
                    robot.reposition_distance += 1
                elif robot.status == RobotStatus.MOVING_TO_TASK:
                    robot.task_distance += 1

            if not robot.path and not forced_wait:
                self._handle_arrival(robot)

    def _handle_arrival(self, robot: Robot) -> None:
        if robot.status == RobotStatus.REPOSITIONING:
            robot.status = RobotStatus.IDLE
            robot.target_zone_id = None
        elif robot.status == RobotStatus.MOVING_TO_TASK and robot.current_task_id:
            task = self.tasks.get(robot.current_task_id)
            if task and task.status == TaskStatus.ASSIGNED:
                task.status = TaskStatus.IN_PROGRESS
                task.started_minute = self.current_minute
                robot.status = RobotStatus.WORKING
                robot.work_remaining_ticks = max(1, task.service_ticks or self.config.work_duration_ticks)

    def _work_on_tasks(self) -> None:
        for robot in self.robots:
            if robot.status != RobotStatus.WORKING:
                continue
            robot.work_remaining_ticks -= 1
            if robot.work_remaining_ticks > 0:
                continue

            task = self.tasks.get(robot.current_task_id or "")
            if task:
                task.status = TaskStatus.COMPLETED
                task.completed_minute = self.current_minute
                self.completed_tasks.append(task)
                if self.persist:
                    self.store.append_metrics(self._completion_metric(task, robot))
            robot.completed_task_count += 1
            robot.current_task_id = None
            robot.target_zone_id = None
            robot.status = RobotStatus.IDLE
            robot.work_remaining_ticks = 0
            robot.idle_action = "STAY"
            robot.idle_action_reason = "Task completed; robot is now unassigned."

    def _completion_metric(self, task: Task, robot: Robot) -> dict:
        started = task.started_minute if task.started_minute is not None else self.current_minute
        completed = task.completed_minute if task.completed_minute is not None else self.current_minute
        return {
            "mode": self.mode,
            "controller": self.controller_name,
            "active_policy": self.active_policy,
            "task_id": task.task_id,
            "wave_id": task.wave_id,
            "source_location": task.source_location,
            "quantity": task.quantity,
            "robot_id": robot.robot_id,
            "zone": task.zone_id,
            "priority": task.priority,
            "created_minute": round(task.created_minute, 2),
            "wait_time": round(started - task.created_minute, 2),
            "completion_time": round(completed - task.created_minute, 2),
            "robot_total_distance": robot.total_distance,
            "robot_task_distance": robot.task_distance,
            "robot_reposition_distance": robot.reposition_distance,
        }

    # ------------------------------------------------------------------
    # MAPFUA observations, decisions, and metrics
    # ------------------------------------------------------------------
    def local_density_scores(self) -> dict[str, float]:
        radius = self.config.idle_local_density_radius
        scores: dict[str, float] = {}
        for robot in actionable_idle_robots(self.robots):
            cells = [
                (x, y)
                for y in range(max(0, robot.position[1] - radius), min(self.warehouse.height, robot.position[1] + radius + 1))
                for x in range(max(0, robot.position[0] - radius), min(self.warehouse.width, robot.position[0] + radius + 1))
                if self.warehouse.walkable((x, y))
                and abs(x - robot.position[0]) + abs(y - robot.position[1]) <= radius
            ]
            neighbors = sum(
                other.robot_id != robot.robot_id
                and abs(other.position[0] - robot.position[0]) + abs(other.position[1] - robot.position[1]) <= radius
                for other in self.robots
            )
            scores[robot.robot_id] = neighbors / max(1, len(cells))
        return scores

    def blocking_scores(self) -> dict[str, float]:
        idle = actionable_idle_robots(self.robots)
        scores = {robot.robot_id: 0.0 for robot in idle}
        lookahead = self.config.idle_blocking_lookahead_steps
        for active in self.robots:
            if not active.current_task_id:
                continue
            task = self.tasks.get(active.current_task_id)
            if task is None or task.status == TaskStatus.COMPLETED:
                continue
            static_path = astar(self.warehouse, active.position, task.position)[:lookahead]
            for robot in idle:
                if robot.position in static_path:
                    scores[robot.robot_id] += 1.0
        return scores

    def idle_context(self) -> IdleContext:
        return IdleContext(
            tick=self.tick_count,
            warehouse=self.warehouse,
            robots=self.robots,
            queue_counts=self.queue_counts(),
            robot_counts=self.current_robot_counts(include_committed=True),
            predictions=dict(self.predictions),
            local_density=self.local_density_scores(),
            blocking_score=self.blocking_scores(),
            parking_nodes=self.warehouse.parking_nodes(),
        )

    def _zone_target(self, robot: Robot, zone_id: str) -> tuple[int, int] | None:
        occupied = {item.position for item in self.robots if item.robot_id != robot.robot_id}
        nodes = [
            node
            for node in self.warehouse.parking_nodes()
            if self.warehouse.zone_for(node) == zone_id and node not in occupied
        ]
        if not nodes:
            target = self.warehouse.zones[zone_id].waiting_position
            return None if target in occupied else target
        return min(nodes, key=lambda node: (abs(node[0] - robot.position[0]) + abs(node[1] - robot.position[1]), node))

    def decision_from_action(self, robot: Robot, action: IdleAction, context: IdleContext | None = None) -> IdleDecision:
        context = context or self.idle_context()
        if action == IdleAction.STAY:
            return IdleDecision(robot.robot_id, action, None, "STAY", "External policy selected STAY.")
        if action == IdleAction.CLEAR:
            return DensityClearPolicy(
                self.config.idle_clear_blocking_threshold,
                self.config.idle_density_threshold,
            ).decide(robot, context)
        zone_id = action.zone_id
        if zone_id is None:
            return IdleDecision(robot.robot_id, IdleAction.STAY, None, "STAY", "Unknown action.")
        target = self._zone_target(robot, zone_id)
        if target is None:
            return IdleDecision(
                robot.robot_id,
                IdleAction.STAY,
                None,
                "STAY",
                f"No collision-free parking target is available in Zone {zone_id}.",
            )
        if self.warehouse.zone_for(robot.position) == zone_id and robot.position == target:
            return IdleDecision(
                robot.robot_id,
                IdleAction.STAY,
                None,
                "STAY",
                f"Robot is already at the selected Zone {zone_id} target.",
            )
        return IdleDecision(
            robot.robot_id,
            action,
            target,
            "PREPOSITION",
            f"External policy selected prepositioning to Zone {zone_id}.",
        )

    def action_mask(self, robot: Robot) -> np.ndarray:
        mask = np.zeros(len(IdleAction), dtype=np.float32)
        mask[IdleAction.STAY] = 1.0
        if robot.current_task_id is not None or robot.status != RobotStatus.IDLE:
            return mask
        mask[IdleAction.CLEAR] = 1.0
        for action in (IdleAction.ZONE_A, IdleAction.ZONE_B, IdleAction.ZONE_C, IdleAction.ZONE_D):
            if self._zone_target(robot, action.zone_id or "A") is not None:
                mask[action] = 1.0
        return mask

    def apply_idle_policy(self) -> list[IdleDecision]:
        if self.idle_policy is None:
            return []
        started = time.perf_counter()
        context = self.idle_context()
        decisions = [self.idle_policy.decide(robot, context) for robot in actionable_idle_robots(self.robots)]
        decisions.sort(key=lambda item: (-item.score, item.robot_id))
        self.last_idle_decisions = self._execute_idle_decisions(decisions)
        self.record_idle_decision_latency((time.perf_counter() - started) * 1000.0)
        return self.last_idle_decisions

    def record_idle_decision_latency(self, latency_ms: float) -> None:
        """Record end-to-end high-level policy latency for one joint decision."""
        value = max(0.0, float(latency_ms))
        self.idle_decision_calls += 1
        self._idle_decision_latencies_ms.append(value)

    def apply_idle_actions(self, actions: dict[str, int | IdleAction]) -> list[IdleDecision]:
        context = self.idle_context()
        decisions: list[IdleDecision] = []
        for robot in sorted(self.robots, key=lambda item: item.robot_id):
            raw = actions.get(robot.robot_id, IdleAction.STAY)
            try:
                action = raw if isinstance(raw, IdleAction) else IdleAction(int(raw))
            except (TypeError, ValueError):
                action = IdleAction.STAY
                self.invalid_idle_actions += 1
            if robot.current_task_id is not None or robot.status != RobotStatus.IDLE:
                if action != IdleAction.STAY:
                    self.invalid_idle_actions += 1
                action = IdleAction.STAY
            decisions.append(self.decision_from_action(robot, action, context))
        self.last_idle_decisions = self._execute_idle_decisions(decisions)
        return self.last_idle_decisions

    def _execute_idle_decisions(self, decisions: list[IdleDecision]) -> list[IdleDecision]:
        executed: list[IdleDecision] = []
        moves = 0
        for decision in decisions:
            robot = next((item for item in self.robots if item.robot_id == decision.robot_id), None)
            if robot is None or robot.current_task_id is not None or robot.status != RobotStatus.IDLE:
                continue
            if decision.action == IdleAction.STAY or decision.target is None:
                robot.idle_action = "STAY"
                robot.idle_action_reason = decision.reason
                executed.append(decision)
                continue
            if moves >= self.config.max_idle_moves_per_cycle:
                robot.idle_action = "STAY"
                robot.idle_action_reason = "Move budget exhausted for this decision cycle."
                continue
            path = plan_for_robot(
                self.warehouse,
                robot,
                self.robots,
                decision.target,
                horizon=self.config.mapf_planning_horizon_steps,
                goal_hold_steps=self.config.mapf_goal_hold_steps,
            )
            if robot.position != decision.target and not path:
                self.path_plan_failures += 1
                robot.idle_action = "STAY"
                robot.idle_action_reason = f"Rejected: no safe route to {decision.target}."
                continue
            if not path:
                robot.idle_action = "STAY"
                robot.idle_action_reason = "Already at target."
                continue
            robot.path = path
            robot.status = RobotStatus.REPOSITIONING
            robot.target_zone_id = self.warehouse.zone_for(decision.target)
            robot.idle_action = decision.action.name
            robot.idle_action_reason = decision.reason
            robot.idle_move_started_tick = self.tick_count
            robot.idle_move_origin = robot.position
            robot.idle_move_purpose = decision.purpose
            self._record_idle_move(robot, decision)
            moves += 1
            executed.append(decision)
        return executed

    def _record_idle_move(self, robot: Robot, decision: IdleDecision) -> None:
        self._idle_event_counter += 1
        blocking_before = float(self.blocking_scores().get(robot.robot_id, 0.0))
        record = {
            "event_id": self._idle_event_counter,
            "robot_id": robot.robot_id,
            "start_tick": self.tick_count,
            "purpose": decision.purpose,
            "target_zone": robot.target_zone_id,
            "blocking_before": blocking_before,
            "resolved": False,
            "useful": decision.purpose == "CLEAR" and blocking_before > 0,
        }
        self._idle_events.append(record)
        self.idle_move_events += 1
        self._idle_robots_moved.add(robot.robot_id)
        if decision.purpose == "CLEAR":
            self.clear_move_events += 1
        elif decision.purpose == "PREPOSITION":
            self.preposition_move_events += 1
        elif decision.purpose == "PARK":
            self.parking_move_events += 1

    def _mark_preposition_useful(self, robot_id: str, task_zone: str) -> None:
        for event in reversed(self._idle_events):
            if event["resolved"] or event["robot_id"] != robot_id:
                continue
            if self.tick_count - int(event["start_tick"]) > self.config.idle_usefulness_window_ticks:
                continue
            if event["purpose"] == "PREPOSITION" and event["target_zone"] == task_zone:
                event["useful"] = True
                event["resolved"] = True
                self.useful_idle_move_events += 1
                return

    def _update_idle_event_outcomes(self) -> None:
        for event in self._idle_events:
            if event["resolved"]:
                continue
            age = self.tick_count - int(event["start_tick"])
            if event["purpose"] == "CLEAR" and event["useful"]:
                event["resolved"] = True
                self.useful_idle_move_events += 1
            elif age > self.config.idle_usefulness_window_ticks:
                event["resolved"] = True
                if event["useful"]:
                    self.useful_idle_move_events += 1
                else:
                    self.unnecessary_idle_move_events += 1

    # ------------------------------------------------------------------
    # Common state and output
    # ------------------------------------------------------------------
    def queue_counts(self) -> dict[str, int]:
        return {
            zone_id: sum(
                task.zone_id == zone_id and task.status == TaskStatus.WAITING
                for task in self.tasks.values()
            )
            for zone_id in self.warehouse.zones
        }

    def current_robot_counts(self, *, include_committed: bool = False) -> dict[str, int]:
        counts = {zone_id: 0 for zone_id in self.warehouse.zones}
        for robot in self.robots:
            if include_committed and robot.status == RobotStatus.REPOSITIONING and robot.target_zone_id:
                counts[robot.target_zone_id] += 1
            else:
                counts[self.warehouse.zone_for(robot.position)] += 1
        return counts

    def demand_heatmap_values(self) -> dict[str, float]:
        if (self.mode in {"predictive", "hybrid"} or self._idle_uses_forecast) and any(
            value > 0 for value in self.predictions.values()
        ):
            return {zone_id: float(value) for zone_id, value in self.predictions.items()}
        queue = self.queue_counts()
        return {
            zone_id: float(queue[zone_id] + self.current_bin_counts[zone_id])
            for zone_id in self.warehouse.zones
        }

    def metrics_summary(self) -> dict:
        waits = [
            (task.started_minute if task.started_minute is not None else self.current_minute) - task.created_minute
            for task in self.completed_tasks
        ]
        completions = [
            (task.completed_minute if task.completed_minute is not None else self.current_minute)
            - task.created_minute
            for task in self.completed_tasks
        ]
        elapsed_hours = max((self.current_minute - self.start_minute) / 60.0, 1e-9)
        generated = len(self.tasks)
        completed = len(self.completed_tasks)
        total_policy_ticks = max(1, sum(self.policy_usage_ticks.values()))
        estimated_useful_rate = (
            100.0 * self.useful_idle_move_events / self.idle_move_events
            if self.idle_move_events
            else 0.0
        )
        pending_idle_events = sum(not bool(event["resolved"]) for event in self._idle_events)
        latency_mean = (
            fmean(self._idle_decision_latencies_ms)
            if self._idle_decision_latencies_ms
            else 0.0
        )
        latency_p95 = (
            float(np.percentile(self._idle_decision_latencies_ms, 95))
            if self._idle_decision_latencies_ms
            else 0.0
        )
        return {
            "generated_tasks": generated,
            "completed_tasks": completed,
            "completion_rate_pct": round(100.0 * completed / generated, 2) if generated else 0.0,
            "throughput_per_hour": round(completed / elapsed_hours, 2),
            "waiting_tasks": sum(task.status == TaskStatus.WAITING for task in self.tasks.values()),
            "assigned_tasks": sum(task.status == TaskStatus.ASSIGNED for task in self.tasks.values()),
            "in_progress_tasks": sum(task.status == TaskStatus.IN_PROGRESS for task in self.tasks.values()),
            "average_wait_time": round(fmean(waits), 2) if waits else 0.0,
            "p95_wait_time": round(float(np.percentile(waits, 95)), 2) if waits else 0.0,
            "average_completion_time": round(fmean(completions), 2) if completions else 0.0,
            "total_distance": sum(robot.total_distance for robot in self.robots),
            "task_distance": sum(robot.task_distance for robot in self.robots),
            "reposition_distance": sum(robot.reposition_distance for robot in self.robots),
            "idle_reposition_distance": sum(robot.reposition_distance for robot in self.robots),
            "idle_move_events": self.idle_move_events,
            "idle_robots_moved": len(self._idle_robots_moved),
            "clear_move_events": self.clear_move_events,
            "preposition_move_events": self.preposition_move_events,
            "parking_move_events": self.parking_move_events,
            "estimated_useful_idle_move_rate_pct": round(estimated_useful_rate, 2),
            "unnecessary_idle_move_events": self.unnecessary_idle_move_events,
            "pending_idle_move_events": pending_idle_events,
            "blocking_pressure_ticks": round(self.blocking_pressure_ticks, 2),
            "vertex_conflicts_detected": self.vertex_conflicts_detected,
            "edge_conflicts_detected": self.edge_conflicts_detected,
            "collision_shield_waits": self.collision_shield_waits,
            "deadlock_count": self.deadlock_count,
            "path_plan_failures": self.path_plan_failures,
            "task_assignment_deferrals": self.task_assignment_deferrals,
            "invalid_idle_actions": self.invalid_idle_actions,
            "idle_decision_calls": self.idle_decision_calls,
            "idle_decision_latency_mean_ms": round(latency_mean, 4),
            "idle_decision_latency_p95_ms": round(latency_p95, 4),
            "predictive_policy_pct": round(100.0 * self.policy_usage_ticks["PREDICTIVE"] / total_policy_ticks, 2),
            "reactive_policy_pct": round(100.0 * self.policy_usage_ticks["REACTIVE"] / total_policy_ticks, 2),
        }

    def snapshot(self) -> dict:
        allocation = None
        if self.last_decision:
            allocation = {
                "task_id": self.last_decision.task_id,
                "robot_id": self.last_decision.robot_id,
                "score": round(self.last_decision.score, 3),
                "reason": self.last_decision.reason,
                "candidates": self.last_decision.candidates,
            }

        controller = {
            "mode": self.mode,
            "controller_name": self.controller_name,
            "active_policy": self.active_policy,
            "policy_usage_ticks": dict(self.policy_usage_ticks),
            "last_repositioned_count": self.last_repositioned_count,
            "forecast_notice_active": self.forecast_notice_active,
            "decision": self.policy_decision.to_dict() if self.policy_decision else None,
            "idle_decisions": [
                {
                    "robot_id": item.robot_id,
                    "action": item.action.name,
                    "purpose": item.purpose,
                    "target": list(item.target) if item.target else None,
                    "reason": item.reason,
                    "score": item.score,
                }
                for item in self.last_idle_decisions
            ],
        }
        return {
            "mode": self.mode,
            "tick": self.tick_count,
            "current_minute": round(self.current_minute, 2),
            "paused": self.paused,
            "predictor": {
                "available": self.predictor.available,
                "model_name": self.predictor.model_name,
            },
            "controller": controller,
            "warehouse": self.warehouse.to_dict(),
            "robots": [robot.to_dict() for robot in self.robots],
            "tasks": [task.to_dict() for task in self.tasks.values() if task.status != TaskStatus.COMPLETED],
            "predictions": self.predictions,
            "demand_heatmap": self.demand_heatmap_values(),
            "target_robot_counts": self.target_robot_counts,
            "current_robot_counts": self.current_robot_counts(),
            "queue_counts": self.queue_counts(),
            "blocking_scores": self.blocking_scores(),
            "local_density": self.local_density_scores(),
            "last_allocation": allocation,
            "metrics": self.metrics_summary(),
        }
