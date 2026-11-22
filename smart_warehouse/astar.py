from __future__ import annotations

import heapq
from itertools import count

from smart_warehouse.entities import Position
from smart_warehouse.warehouse import WarehouseMap


def manhattan(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def reconstruct_path(came_from: dict[Position, Position], current: Position) -> list[Position]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def astar(warehouse: WarehouseMap, start: Position, goal: Position) -> list[Position]:
    """Return a path excluding start and including goal. Empty means no movement or no path."""
    if not warehouse.walkable(start) or not warehouse.walkable(goal):
        return []
    if start == goal:
        return []

    tie_breaker = count()
    frontier: list[tuple[int, int, Position]] = [(manhattan(start, goal), next(tie_breaker), start)]
    came_from: dict[Position, Position] = {}
    cost_so_far: dict[Position, int] = {start: 0}

    while frontier:
        _, _, current = heapq.heappop(frontier)
        if current == goal:
            return reconstruct_path(came_from, current)[1:]

        for neighbor in warehouse.neighbors(current):
            new_cost = cost_so_far[current] + 1
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + manhattan(neighbor, goal)
                heapq.heappush(frontier, (priority, next(tie_breaker), neighbor))
                came_from[neighbor] = current

    return []
