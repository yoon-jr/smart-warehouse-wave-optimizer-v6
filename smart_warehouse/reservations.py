from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from itertools import count
from typing import Iterable

from smart_warehouse.astar import manhattan
from smart_warehouse.entities import Position, Robot
from smart_warehouse.warehouse import WarehouseMap


@dataclass
class ReservationTable:
    """Discrete-time reservations for vertex and opposite-edge conflicts.

    Time is measured in robot movement steps rather than simulator ticks. A
    path contains the positions occupied at times 1..T; time 0 is the current
    position. Stationary robots reserve their current cell for the full horizon.

    Lists of sets are used instead of dictionaries because collision queries are
    on the critical path of space-time A* and time indices are dense.
    """

    horizon: int
    vertex: list[set[Position]] = field(default_factory=list)
    edge: list[set[tuple[Position, Position]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.horizon < 0:
            raise ValueError("horizon must be non-negative")
        if not self.vertex:
            self.vertex = [set() for _ in range(self.horizon + 1)]
        if not self.edge:
            self.edge = [set() for _ in range(self.horizon + 1)]
        if len(self.vertex) < self.horizon + 1 or len(self.edge) < self.horizon + 1:
            raise ValueError("reservation arrays must cover 0..horizon")

    def reserve_vertex(self, position: Position, time_step: int) -> None:
        if 0 <= int(time_step) <= self.horizon:
            self.vertex[int(time_step)].add(position)

    def reserve_edge(self, start: Position, end: Position, time_step: int) -> None:
        if 0 <= int(time_step) <= self.horizon:
            self.edge[int(time_step)].add((start, end))

    def vertex_reserved(self, position: Position, time_step: int) -> bool:
        time_step = int(time_step)
        return not (0 <= time_step <= self.horizon) or position in self.vertex[time_step]

    def opposite_edge_reserved(self, start: Position, end: Position, time_step: int) -> bool:
        time_step = int(time_step)
        return not (0 <= time_step <= self.horizon) or (end, start) in self.edge[time_step]

    def reserve_path(
        self,
        start: Position,
        path: Iterable[Position],
        *,
        hold_steps: int = 0,
    ) -> None:
        previous = start
        last = start
        path_list = list(path)[: self.horizon]
        self.reserve_vertex(start, 0)
        for time_step, position in enumerate(path_list, start=1):
            self.reserve_vertex(position, time_step)
            self.reserve_edge(previous, position, time_step)
            previous = position
            last = position
        finish_time = len(path_list)
        for extra in range(1, hold_steps + 1):
            time_step = finish_time + extra
            if time_step > self.horizon:
                break
            self.reserve_vertex(last, time_step)
            self.reserve_edge(last, last, time_step)


def build_reservation_table(
    robots: Iterable[Robot],
    *,
    exclude_robot_id: str | None,
    horizon: int,
    hold_steps: int,
) -> ReservationTable:
    table = ReservationTable(horizon=horizon)
    for robot in robots:
        if robot.robot_id == exclude_robot_id:
            continue
        path = list(robot.path[:horizon])
        if path:
            # A robot remains physically present after reaching the end of its
            # committed route (it may work at a pick location or wait at an idle
            # target). Reserve that terminal vertex through the rolling horizon.
            terminal_hold = max(0, horizon - len(path))
            table.reserve_path(robot.position, path, hold_steps=terminal_hold)
        else:
            # An idle or working robot is a dynamic obstacle until replanning.
            table.reserve_path(robot.position, [], hold_steps=horizon)
    return table


def space_time_astar(
    warehouse: WarehouseMap,
    start: Position,
    goal: Position,
    reservations: ReservationTable,
    *,
    max_time: int,
    goal_hold_steps: int = 0,
) -> list[Position]:
    """Find a collision-free path with wait actions.

    The returned path excludes ``start`` and includes the goal. Consecutive
    duplicate positions are legal waits. The planner checks vertex conflicts,
    opposite-edge swaps, and whether the goal remains free for a short hold
    interval. An empty list means either no movement is needed or no safe path
    was found within ``max_time``.
    """

    max_time = min(int(max_time), reservations.horizon)
    if not warehouse.walkable(start) or not warehouse.walkable(goal):
        return []
    if start == goal:
        return []
    minimum_arrival = manhattan(start, goal)
    if minimum_arrival > max_time:
        return []

    def goal_available(arrival_time: int) -> bool:
        final_time = min(max_time, arrival_time + goal_hold_steps)
        return all(
            not reservations.vertex_reserved(goal, future)
            for future in range(arrival_time, final_time + 1)
        )

    # A common warehouse case is a pick cell occupied by a robot that is still
    # working. Without this check, A* explores the entire space-time graph only
    # to discover that no legal arrival time exists.
    if not any(goal_available(time_step) for time_step in range(minimum_arrival, max_time + 1)):
        return []

    State = tuple[Position, int]
    start_state: State = (start, 0)
    tie = count()
    frontier: list[tuple[int, int, int, State]] = [
        (minimum_arrival, 0, next(tie), start_state)
    ]
    came_from: dict[State, State] = {}
    cost_so_far: dict[State, int] = {start_state: 0}
    option_cache: dict[Position, tuple[Position, ...]] = {}

    while frontier:
        _, cost, _, (current, time_step) = heapq.heappop(frontier)
        if current == goal and goal_available(time_step):
            states: list[State] = [(current, time_step)]
            cursor = (current, time_step)
            while cursor in came_from:
                cursor = came_from[cursor]
                states.append(cursor)
            states.reverse()
            return [position for position, _ in states[1:]]

        if time_step >= max_time:
            continue

        next_time = time_step + 1
        candidates = option_cache.get(current)
        if candidates is None:
            candidates = (current, *warehouse.neighbors(current))
            option_cache[current] = candidates
        for neighbor in candidates:
            if reservations.vertex_reserved(neighbor, next_time):
                continue
            if reservations.opposite_edge_reserved(current, neighbor, next_time):
                continue
            # Even under ideal movement, the goal must remain reachable within
            # the remaining planning horizon.
            if next_time + manhattan(neighbor, goal) > max_time:
                continue
            next_state: State = (neighbor, next_time)
            new_cost = cost + 1
            if new_cost >= cost_so_far.get(next_state, 10**12):
                continue
            cost_so_far[next_state] = new_cost
            came_from[next_state] = (current, time_step)
            priority = new_cost + manhattan(neighbor, goal)
            heapq.heappush(frontier, (priority, new_cost, next(tie), next_state))

    return []


def plan_for_robot(
    warehouse: WarehouseMap,
    robot: Robot,
    robots: Iterable[Robot],
    goal: Position,
    *,
    horizon: int,
    goal_hold_steps: int,
) -> list[Position]:
    reservations = build_reservation_table(
        robots,
        exclude_robot_id=robot.robot_id,
        horizon=horizon,
        hold_steps=goal_hold_steps,
    )
    return space_time_astar(
        warehouse,
        robot.position,
        goal,
        reservations,
        max_time=horizon,
        goal_hold_steps=goal_hold_steps,
    )


def detect_conflicts(
    current_positions: dict[str, Position],
    proposed_positions: dict[str, Position],
) -> tuple[list[tuple[str, str, Position]], list[tuple[str, str, Position, Position]]]:
    """Return vertex and edge-swap conflicts for one synchronous move."""

    vertex_conflicts: list[tuple[str, str, Position]] = []
    edge_conflicts: list[tuple[str, str, Position, Position]] = []
    robot_ids = sorted(proposed_positions)
    for index, first in enumerate(robot_ids):
        for second in robot_ids[index + 1 :]:
            first_next = proposed_positions[first]
            second_next = proposed_positions[second]
            if first_next == second_next:
                vertex_conflicts.append((first, second, first_next))
            if (
                current_positions[first] == second_next
                and current_positions[second] == first_next
                and current_positions[first] != current_positions[second]
            ):
                edge_conflicts.append(
                    (first, second, current_positions[first], current_positions[second])
                )
    return vertex_conflicts, edge_conflicts
