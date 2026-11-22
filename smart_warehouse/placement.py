from __future__ import annotations

import math

from smart_warehouse.astar import manhattan
from smart_warehouse.entities import Robot, RobotStatus
from smart_warehouse.reservations import plan_for_robot
from smart_warehouse.warehouse import WarehouseMap


def proportional_targets(
    predictions: dict[str, float],
    robot_count: int,
    *,
    max_per_zone: int | None = None,
) -> dict[str, int]:
    zones = sorted(predictions)
    total = sum(max(0.0, value) for value in predictions.values())
    if robot_count <= 0:
        return {zone: 0 for zone in zones}
    if total <= 0:
        base = robot_count // len(zones)
        result = {zone: base for zone in zones}
        for zone in zones[: robot_count - base * len(zones)]:
            result[zone] += 1
        return result

    raw = {zone: robot_count * max(0.0, predictions[zone]) / total for zone in zones}
    targets = {zone: math.floor(value) for zone, value in raw.items()}
    remaining = robot_count - sum(targets.values())
    remainders = sorted(zones, key=lambda zone: raw[zone] - targets[zone], reverse=True)
    for zone in remainders[:remaining]:
        targets[zone] += 1

    if max_per_zone is not None and zones:
        effective_cap = max(int(max_per_zone), math.ceil(robot_count / len(zones)))
        excess = 0
        for zone in zones:
            if targets[zone] > effective_cap:
                excess += targets[zone] - effective_cap
                targets[zone] = effective_cap
        allocation_order = sorted(
            zones,
            key=lambda zone: (max(0.0, predictions[zone]), zone),
            reverse=True,
        )
        while excess > 0:
            eligible = [zone for zone in allocation_order if targets[zone] < effective_cap]
            if not eligible:
                break
            for zone in eligible:
                if excess <= 0:
                    break
                targets[zone] += 1
                excess -= 1
    return targets


def reposition_idle_robots(
    robots: list[Robot],
    warehouse: WarehouseMap,
    predictions: dict[str, float],
    minimum_difference: int = 2,
    max_moves: int | None = None,
    max_target_fraction: float | None = None,
    *,
    planning_horizon_steps: int = 80,
    goal_hold_steps: int = 12,
    current_tick: int | None = None,
) -> dict[str, int]:
    """Reposition idle robots using collision-aware space-time paths."""
    unassigned = [robot for robot in robots if robot.current_task_id is None]
    max_per_zone = None
    if max_target_fraction is not None and unassigned:
        if not 0 < max_target_fraction <= 1:
            raise ValueError("max_target_fraction must be in (0, 1]")
        max_per_zone = max(1, math.ceil(len(unassigned) * max_target_fraction))
    targets = proportional_targets(predictions, len(unassigned), max_per_zone=max_per_zone)

    effective_counts = {zone_id: 0 for zone_id in warehouse.zones}
    movable: list[Robot] = []
    for robot in unassigned:
        if robot.status == RobotStatus.REPOSITIONING and robot.target_zone_id:
            effective_counts[robot.target_zone_id] += 1
        else:
            effective_counts[warehouse.zone_for(robot.position)] += 1
            if robot.status == RobotStatus.IDLE:
                movable.append(robot)

    deficits = {
        zone_id: targets[zone_id] - effective_counts[zone_id]
        for zone_id in targets
        if targets[zone_id] - effective_counts[zone_id] >= minimum_difference
    }
    reserved: set[str] = set()
    moves_started = 0

    for destination_zone, deficit in sorted(deficits.items(), key=lambda item: item[1], reverse=True):
        destination = warehouse.zones[destination_zone].waiting_position
        for _ in range(deficit):
            if max_moves is not None and moves_started >= max_moves:
                break
            candidates = [
                robot
                for robot in movable
                if robot.robot_id not in reserved
                and warehouse.zone_for(robot.position) != destination_zone
                and effective_counts[warehouse.zone_for(robot.position)] > targets[warehouse.zone_for(robot.position)]
            ]
            if not candidates:
                break
            candidates.sort(key=lambda item: (manhattan(item.position, destination), item.robot_id))
            selected: tuple[Robot, list[tuple[int, int]]] | None = None
            for robot in candidates:
                path = plan_for_robot(
                    warehouse,
                    robot,
                    robots,
                    destination,
                    horizon=planning_horizon_steps,
                    goal_hold_steps=goal_hold_steps,
                )
                if robot.position == destination or path:
                    selected = (robot, path)
                    break
            if selected is None:
                break
            robot, path = selected
            source_zone = warehouse.zone_for(robot.position)
            robot.path = path
            robot.status = RobotStatus.REPOSITIONING if path else RobotStatus.IDLE
            robot.target_zone_id = destination_zone
            robot.idle_action = f"ZONE_{destination_zone}"
            robot.idle_action_reason = f"v4 forecast-rule prepositioning to Zone {destination_zone}."
            robot.idle_move_started_tick = current_tick
            robot.idle_move_origin = robot.position
            robot.idle_move_purpose = "PREPOSITION"
            effective_counts[source_zone] -= 1
            effective_counts[destination_zone] += 1
            reserved.add(robot.robot_id)
            moves_started += 1

    return targets
