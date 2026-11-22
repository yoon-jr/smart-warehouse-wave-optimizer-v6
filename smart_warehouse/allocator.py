from __future__ import annotations

from dataclasses import dataclass

from smart_warehouse.entities import Robot, RobotStatus, Task, TaskStatus
from smart_warehouse.reservations import plan_for_robot
from smart_warehouse.warehouse import WarehouseMap


@dataclass(frozen=True)
class AllocationDecision:
    task_id: str
    robot_id: str
    score: float
    reason: str
    candidates: list[dict]


class ExplainableAllocator:
    """Task allocator with collision-aware space-time path planning.

    The allocation score remains deliberately interpretable. The main v5
    change is that candidate distances come from a reservation-table planner,
    so a path may include waits and cannot create vertex or edge-swap conflicts
    with routes already committed by the other robots.
    """

    def __init__(
        self,
        min_battery: float = 10.0,
        *,
        planning_horizon_steps: int = 80,
        goal_hold_steps: int = 12,
    ) -> None:
        self.min_battery = min_battery
        self.planning_horizon_steps = planning_horizon_steps
        self.goal_hold_steps = goal_hold_steps

    def allocate(
        self,
        task: Task,
        robots: list[Robot],
        warehouse: WarehouseMap,
        current_minute: float,
    ) -> AllocationDecision | None:
        candidates: list[tuple[float, Robot, list[tuple[int, int]], dict]] = []

        for robot in robots:
            if not robot.available or robot.battery < self.min_battery:
                continue
            path = plan_for_robot(
                warehouse,
                robot,
                robots,
                task.position,
                horizon=self.planning_horizon_steps,
                goal_hold_steps=self.goal_hold_steps,
            )
            if robot.position != task.position and not path:
                continue

            movement_distance = sum(
                first != second
                for first, second in zip([robot.position, *path[:-1]], path)
            )
            wait_steps = len(path) - movement_distance
            battery_penalty = (100.0 - robot.battery) / 100.0
            reposition_penalty = 0.25 if robot.status == RobotStatus.REPOSITIONING else 0.0
            distance_weight = 0.70 if task.priority >= 2 else 0.60
            score = (
                distance_weight * movement_distance
                + 0.15 * wait_steps
                + 2.0 * battery_penalty
                + reposition_penalty
            )
            detail = {
                "robot_id": robot.robot_id,
                "distance": movement_distance,
                "reserved_wait_steps": wait_steps,
                "battery": round(robot.battery, 1),
                "repositioning": robot.status == RobotStatus.REPOSITIONING,
                "score": round(score, 3),
            }
            candidates.append((score, robot, path, detail))

        if not candidates:
            return None

        candidates.sort(key=lambda item: (item[0], item[1].robot_id))
        score, robot, path, selected_detail = candidates[0]
        robot.path = path
        robot.status = RobotStatus.MOVING_TO_TASK
        robot.current_task_id = task.task_id
        robot.target_zone_id = task.zone_id
        robot.idle_action = "SERVE_TASK"
        robot.idle_action_reason = "Assigned to a released task."
        task.status = TaskStatus.ASSIGNED
        task.assigned_robot_id = robot.robot_id
        task.assigned_minute = current_minute

        reason = (
            f"{robot.robot_id} selected: {selected_detail['distance']} movement cells, "
            f"{selected_detail['reserved_wait_steps']} reserved waits, battery {robot.battery:.0f}%, "
            f"allocation score {score:.2f}."
        )
        return AllocationDecision(
            task_id=task.task_id,
            robot_id=robot.robot_id,
            score=score,
            reason=reason,
            candidates=[item[3] for item in candidates],
        )
