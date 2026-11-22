from __future__ import annotations

import random
from dataclasses import dataclass

from smart_warehouse.entities import Position, Zone


@dataclass
class WarehouseMap:
    width: int
    height: int
    obstacles: set[Position]
    zones: dict[str, Zone]

    @classmethod
    def create_default(cls, width: int = 20, height: int = 20) -> "WarehouseMap":
        if width < 12 or height < 12:
            raise ValueError("Default warehouse requires at least a 12x12 grid.")

        obstacles: set[Position] = set()
        # Short shelf rows. Gaps are deliberately left so A* always has corridors.
        for x in (4, 5, 9, 10, 14, 15):
            for y in range(2, height - 2):
                if y not in {5, 10, 15}:
                    obstacles.add((x, y))

        mid_x = width // 2
        mid_y = height // 2
        zones = {
            "A": Zone("A", 0, mid_x - 1, 0, mid_y - 1, (2, 2)),
            "B": Zone("B", mid_x, width - 1, 0, mid_y - 1, (width - 3, 2)),
            "C": Zone("C", 0, mid_x - 1, mid_y, height - 1, (2, height - 3)),
            "D": Zone("D", mid_x, width - 1, mid_y, height - 1, (width - 3, height - 3)),
        }
        warehouse = cls(width, height, obstacles, zones)
        for zone in zones.values():
            if zone.waiting_position in obstacles:
                obstacles.remove(zone.waiting_position)
        return warehouse

    def in_bounds(self, position: Position) -> bool:
        x, y = position
        return 0 <= x < self.width and 0 <= y < self.height

    def walkable(self, position: Position) -> bool:
        return self.in_bounds(position) and position not in self.obstacles

    def neighbors(self, position: Position) -> list[Position]:
        x, y = position
        candidates = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [candidate for candidate in candidates if self.walkable(candidate)]

    def zone_for(self, position: Position) -> str:
        for zone_id, zone in self.zones.items():
            if zone.contains(position):
                return zone_id
        raise ValueError(f"Position {position} does not belong to a zone.")

    def random_walkable_position(self, zone_id: str, rng: random.Random) -> Position:
        zone = self.zones[zone_id]
        candidates = [
            (x, y)
            for x in range(zone.x_min, zone.x_max + 1)
            for y in range(zone.y_min, zone.y_max + 1)
            if self.walkable((x, y))
        ]
        if not candidates:
            raise RuntimeError(f"No walkable cells in zone {zone_id}.")
        return rng.choice(candidates)

    def random_walkable_position_near(
        self,
        zone_id: str,
        center: Position,
        radius: int,
        rng: random.Random,
    ) -> Position:
        if radius < 0:
            raise ValueError("radius must be non-negative.")
        zone = self.zones[zone_id]
        candidates = [
            (x, y)
            for x in range(zone.x_min, zone.x_max + 1)
            for y in range(zone.y_min, zone.y_max + 1)
            if self.walkable((x, y)) and abs(x - center[0]) + abs(y - center[1]) <= radius
        ]
        if not candidates:
            return self.random_walkable_position(zone_id, rng)
        return rng.choice(candidates)


    def nearby_walkable_positions(self, center: Position, limit: int) -> list[Position]:
        """Return unique walkable cells in increasing Manhattan distance."""
        if limit <= 0:
            return []
        candidates = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if self.walkable((x, y))
        ]
        candidates.sort(key=lambda point: (abs(point[0] - center[0]) + abs(point[1] - center[1]), point[1], point[0]))
        return candidates[:limit]

    def parking_nodes(self) -> tuple[Position, ...]:
        """Low-risk waiting nodes used by MAPFUA idle-robot baselines.

        The four zone waiting positions are augmented with nearby walkable
        cells so multiple idle robots never need to occupy the same vertex.
        """
        nodes: list[Position] = []
        for zone_id in sorted(self.zones):
            center = self.zones[zone_id].waiting_position
            for position in self.nearby_walkable_positions(center, 4):
                if position not in nodes:
                    nodes.append(position)
        return tuple(nodes)

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "obstacles": [list(point) for point in sorted(self.obstacles)],
            "zones": {zone_id: zone.to_dict() for zone_id, zone in self.zones.items()},
        }
