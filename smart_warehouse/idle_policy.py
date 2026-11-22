from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum

from smart_warehouse.astar import manhattan
from smart_warehouse.entities import Position, Robot, RobotStatus
from smart_warehouse.warehouse import WarehouseMap


class IdleAction(IntEnum):
    STAY = 0
    CLEAR = 1
    ZONE_A = 2
    ZONE_B = 3
    ZONE_C = 4
    ZONE_D = 5

    @property
    def zone_id(self) -> str | None:
        mapping = {
            IdleAction.ZONE_A: "A",
            IdleAction.ZONE_B: "B",
            IdleAction.ZONE_C: "C",
            IdleAction.ZONE_D: "D",
        }
        return mapping.get(self)


@dataclass(frozen=True)
class IdleContext:
    tick: int
    warehouse: WarehouseMap
    robots: list[Robot]
    queue_counts: dict[str, int]
    robot_counts: dict[str, int]
    predictions: dict[str, float]
    local_density: dict[str, float]
    blocking_score: dict[str, float]
    parking_nodes: tuple[Position, ...]


@dataclass(frozen=True)
class IdleDecision:
    robot_id: str
    action: IdleAction
    target: Position | None
    purpose: str
    reason: str
    score: float = 0.0


class IdlePolicy(ABC):
    name = "abstract"
    uses_forecast = False

    @abstractmethod
    def decide(self, robot: Robot, context: IdleContext) -> IdleDecision:
        raise NotImplementedError


class IdleStayPolicy(IdlePolicy):
    name = "idle_stay"

    def decide(self, robot: Robot, context: IdleContext) -> IdleDecision:
        return IdleDecision(
            robot_id=robot.robot_id,
            action=IdleAction.STAY,
            target=None,
            purpose="STAY",
            reason="Unassigned robot remains stationary.",
        )


class NearestParkingPolicy(IdlePolicy):
    name = "nearest_parking"

    def decide(self, robot: Robot, context: IdleContext) -> IdleDecision:
        occupied = {item.position for item in context.robots if item.robot_id != robot.robot_id}
        candidates = [node for node in context.parking_nodes if node not in occupied]
        if not candidates:
            return IdleStayPolicy().decide(robot, context)
        target = min(candidates, key=lambda node: (manhattan(robot.position, node), node))
        if target == robot.position:
            return IdleDecision(
                robot.robot_id,
                IdleAction.STAY,
                None,
                "STAY",
                "Robot is already at the nearest available parking node.",
            )
        return IdleDecision(
            robot.robot_id,
            IdleAction.CLEAR,
            target,
            "PARK",
            f"Move to nearest available parking node {target}.",
            score=-float(manhattan(robot.position, target)),
        )


class DensityClearPolicy(IdlePolicy):
    """Move only idle robots that are likely to obstruct active traffic."""

    name = "density_clear"

    def __init__(self, blocking_threshold: float = 1.0, density_threshold: float = 0.28) -> None:
        self.blocking_threshold = blocking_threshold
        self.density_threshold = density_threshold

    def decide(self, robot: Robot, context: IdleContext) -> IdleDecision:
        blocking = float(context.blocking_score.get(robot.robot_id, 0.0))
        density = float(context.local_density.get(robot.robot_id, 0.0))
        if blocking < self.blocking_threshold and density < self.density_threshold:
            return IdleDecision(
                robot.robot_id,
                IdleAction.STAY,
                None,
                "STAY",
                f"Low obstruction risk: blocking={blocking:.1f}, local density={density:.2f}.",
                score=-(blocking + density),
            )

        occupied = {item.position for item in context.robots if item.robot_id != robot.robot_id}
        candidates = [node for node in context.parking_nodes if node not in occupied]
        if not candidates:
            return IdleDecision(
                robot.robot_id,
                IdleAction.STAY,
                None,
                "STAY",
                "No unoccupied parking node is available.",
                score=-(blocking + density),
            )

        def target_score(node: Position) -> tuple[float, int, Position]:
            # Prefer nodes away from current active paths and close enough to
            # avoid moving every idle robot across the warehouse.
            path_pressure = 0.0
            for active in context.robots:
                if active.robot_id == robot.robot_id or not active.path:
                    continue
                path_pressure += sum(node == step for step in active.path[:12])
            return (path_pressure, manhattan(robot.position, node), node)

        target = min(candidates, key=target_score)
        if target == robot.position:
            return IdleDecision(
                robot.robot_id,
                IdleAction.STAY,
                None,
                "STAY",
                "Current location is already the safest available parking node.",
                score=blocking + density,
            )
        return IdleDecision(
            robot.robot_id,
            IdleAction.CLEAR,
            target,
            "CLEAR",
            (
                f"Clear probable bottleneck: blocking={blocking:.1f}, "
                f"local density={density:.2f}; move to {target}."
            ),
            score=blocking + density,
        )


class ForecastRulePolicy(IdlePolicy):
    """Simple forecast baseline retained from v4 as an ablation."""

    name = "forecast_rule"
    uses_forecast = True

    def __init__(self, minimum_gap: float = 0.75) -> None:
        self.minimum_gap = minimum_gap

    def decide(self, robot: Robot, context: IdleContext) -> IdleDecision:
        ordered = sorted(
            context.predictions,
            key=lambda zone: (float(context.predictions[zone]), zone),
            reverse=True,
        )
        if not ordered:
            return IdleStayPolicy().decide(robot, context)
        top = ordered[0]
        second = ordered[1] if len(ordered) > 1 else top
        gap = float(context.predictions[top]) - float(context.predictions[second])
        if gap < self.minimum_gap:
            return IdleDecision(
                robot.robot_id,
                IdleAction.STAY,
                None,
                "STAY",
                f"Forecast is diffuse; top-zone gap is only {gap:.2f}.",
            )
        current_zone = context.warehouse.zone_for(robot.position)
        if current_zone == top:
            return IdleDecision(
                robot.robot_id,
                IdleAction.STAY,
                None,
                "STAY",
                f"Robot is already in forecast hotspot Zone {top}.",
            )
        target = context.warehouse.zones[top].waiting_position
        action = IdleAction(2 + "ABCD".index(top))
        return IdleDecision(
            robot.robot_id,
            action,
            target,
            "PREPOSITION",
            f"Preposition to forecast hotspot Zone {top}; gap={gap:.2f}.",
            score=gap,
        )


class ExternalJointPolicy(IdlePolicy):
    """Marker policy used by the MARL environment; actions are supplied externally."""

    name = "external_marl"

    def decide(self, robot: Robot, context: IdleContext) -> IdleDecision:
        return IdleStayPolicy().decide(robot, context)


def policy_from_name(name: str) -> IdlePolicy:
    normalized = name.strip().lower().replace("-", "_")
    if normalized in {"idle_stay", "stay", "none"}:
        return IdleStayPolicy()
    if normalized in {"nearest_parking", "parking"}:
        return NearestParkingPolicy()
    if normalized in {"density_clear", "clear"}:
        return DensityClearPolicy()
    if normalized in {"forecast_rule", "forecast"}:
        return ForecastRulePolicy()
    if normalized in {"external_marl", "marl", "external"}:
        return ExternalJointPolicy()
    raise ValueError(f"Unknown idle policy: {name}")


def actionable_idle_robots(robots: list[Robot]) -> list[Robot]:
    return [
        robot
        for robot in robots
        if robot.current_task_id is None and robot.status == RobotStatus.IDLE
    ]
