from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from smart_warehouse.astar import astar
from smart_warehouse.entities import Position
from smart_warehouse.warehouse import WarehouseMap


def canonical_wave_id(value: object) -> str:
    """Return a stable string key for wave identifiers read from CSV files.

    Pandas may infer an otherwise-integer wave number as a float when a CSV
    column contains missing values, producing values such as ``33182.0`` in one
    artifact and ``33182`` in another.  The optimizer treats those two textual
    forms as the same identifier while preserving non-numeric IDs such as
    ``W-003`` exactly.
    """
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    head, separator, tail = text.partition(".")
    if (
        separator
        and head.lstrip("+-").isdigit()
        and tail
        and set(tail) == {"0"}
    ):
        return head
    return text


REQUIRED_TASK_COLUMNS = {
    "wave_id",
    "source_sequence",
    "x",
    "y",
    "zone",
    "quantity",
}

# Only columns available before route optimization are allowed as model inputs.
# Post-optimization columns such as optimized_distance and saving_pct are targets
# and must never be included here.
MODEL_FEATURE_COLUMNS = [
    "task_count",
    "total_quantity",
    "unique_sku_count",
    "unique_location_count",
    "unique_grid_stop_count",
    "operator_count",
    "repeated_location_lines",
    "active_zone_count",
    "dominant_zone_ratio",
    "zone_entropy",
    "zone_a_ratio",
    "zone_b_ratio",
    "zone_c_ratio",
    "zone_d_ratio",
    "mean_x",
    "mean_y",
    "std_x",
    "std_y",
    "bounding_box_area",
    "mean_pairwise_distance",
    "max_pairwise_distance",
    "original_route_distance",
    "distance_per_stop",
    "mean_route_jump",
    "max_route_jump",
    "zone_transition_count",
]


@dataclass(frozen=True)
class RouteOptimizationConfig:
    depot: Position = (1, 1)
    two_opt_max_passes: int = 8
    two_opt_candidate_span: int = 60
    seconds_per_grid_step: float = 2.0
    seconds_per_pick_line: float = 6.0

    def __post_init__(self) -> None:
        if self.two_opt_max_passes < 0:
            raise ValueError("two_opt_max_passes must be non-negative")
        if self.two_opt_candidate_span < 2:
            raise ValueError("two_opt_candidate_span must be at least 2")
        if self.seconds_per_grid_step <= 0:
            raise ValueError("seconds_per_grid_step must be positive")
        if self.seconds_per_pick_line < 0:
            raise ValueError("seconds_per_pick_line must be non-negative")


@dataclass(frozen=True)
class WaveRouteResult:
    wave_id: str
    original_stops: tuple[Position, ...]
    nearest_neighbor_stops: tuple[Position, ...]
    optimized_stops: tuple[Position, ...]
    original_distance: int
    nearest_neighbor_distance: int
    optimized_distance: int

    @property
    def distance_saved(self) -> int:
        return max(0, self.original_distance - self.optimized_distance)

    @property
    def saving_pct(self) -> float:
        if self.original_distance <= 0:
            return 0.0
        return 100.0 * self.distance_saved / self.original_distance


class GridDistanceOracle:
    """All-pairs shortest-path distance cache for the small simulator grid.

    The warehouse grid is at most a few hundred walkable cells. Precomputing one
    BFS per walkable cell makes thousands of wave evaluations substantially
    faster and keeps every route metric consistent with the simulator's shelves
    and corridors.
    """

    def __init__(self, warehouse: WarehouseMap) -> None:
        self.warehouse = warehouse
        self.positions = tuple(
            (x, y)
            for y in range(warehouse.height)
            for x in range(warehouse.width)
            if warehouse.walkable((x, y))
        )
        if not self.positions:
            raise ValueError("warehouse has no walkable positions")
        self.index = {position: idx for idx, position in enumerate(self.positions)}
        self._distances = np.full(
            (len(self.positions), len(self.positions)),
            fill_value=np.iinfo(np.int32).max,
            dtype=np.int32,
        )
        self._precompute()

    def _precompute(self) -> None:
        for source_idx, source in enumerate(self.positions):
            queue: deque[Position] = deque([source])
            distances: dict[Position, int] = {source: 0}
            while queue:
                current = queue.popleft()
                next_distance = distances[current] + 1
                for neighbor in self.warehouse.neighbors(current):
                    if neighbor in distances:
                        continue
                    distances[neighbor] = next_distance
                    queue.append(neighbor)
            for target, distance in distances.items():
                self._distances[source_idx, self.index[target]] = distance

    def distance(self, start: Position, goal: Position) -> int:
        if start not in self.index:
            raise ValueError(f"start position is not walkable: {start}")
        if goal not in self.index:
            raise ValueError(f"goal position is not walkable: {goal}")
        value = int(self._distances[self.index[start], self.index[goal]])
        if value == np.iinfo(np.int32).max:
            raise ValueError(f"no path between {start} and {goal}")
        return value

    def shortest_path(self, start: Position, goal: Position) -> list[Position]:
        if start == goal:
            return [start]
        path = astar(self.warehouse, start, goal)
        if not path:
            raise ValueError(f"no path between {start} and {goal}")
        return [start, *path]


def resolve_depot(warehouse: WarehouseMap, requested: Position) -> Position:
    if warehouse.walkable(requested):
        return requested
    candidates = [
        (x, y)
        for y in range(warehouse.height)
        for x in range(warehouse.width)
        if warehouse.walkable((x, y))
    ]
    if not candidates:
        raise ValueError("warehouse has no walkable depot candidate")
    return min(
        candidates,
        key=lambda point: (
            abs(point[0] - requested[0]) + abs(point[1] - requested[1]),
            point[1],
            point[0],
        ),
    )


def _position_tuple(x: object, y: object) -> Position:
    return int(float(x)), int(float(y))


def unique_stops_in_sequence(wave: pd.DataFrame) -> tuple[Position, ...]:
    """Return unique physical grid stops in the recorded first-visit order.

    Both the original and optimized routes use the same unique stop set. This
    ensures that the measured gain comes from reordering rather than unfairly
    deleting duplicate work only from the optimized route.
    """
    ordered = wave.sort_values(["source_sequence"], kind="stable")
    seen: set[Position] = set()
    stops: list[Position] = []
    for row in ordered.itertuples(index=False):
        position = _position_tuple(row.x, row.y)
        if position in seen:
            continue
        seen.add(position)
        stops.append(position)
    return tuple(stops)


def route_distance(
    stops: Sequence[Position],
    *,
    oracle: GridDistanceOracle,
    depot: Position,
) -> int:
    if not stops:
        return 0
    total = oracle.distance(depot, stops[0])
    for start, goal in zip(stops, stops[1:]):
        total += oracle.distance(start, goal)
    total += oracle.distance(stops[-1], depot)
    return int(total)


def nearest_neighbor_route(
    stops: Sequence[Position],
    *,
    oracle: GridDistanceOracle,
    depot: Position,
) -> tuple[Position, ...]:
    remaining = list(dict.fromkeys(stops))
    result: list[Position] = []
    current = depot
    while remaining:
        best_index = min(
            range(len(remaining)),
            key=lambda idx: (
                oracle.distance(current, remaining[idx]),
                remaining[idx][1],
                remaining[idx][0],
                idx,
            ),
        )
        current = remaining.pop(best_index)
        result.append(current)
    return tuple(result)


def two_opt_route(
    stops: Sequence[Position],
    *,
    oracle: GridDistanceOracle,
    depot: Position,
    max_passes: int = 8,
    candidate_span: int = 60,
) -> tuple[Position, ...]:
    """Deterministic first-improvement 2-opt with a bounded search budget."""
    route = list(stops)
    n = len(route)
    if n < 3 or max_passes == 0:
        return tuple(route)

    for _ in range(max_passes):
        improved = False
        for i in range(n - 1):
            a = depot if i == 0 else route[i - 1]
            b = route[i]
            max_j = min(n - 1, i + candidate_span)
            for j in range(i + 1, max_j + 1):
                c = route[j]
                d = depot if j == n - 1 else route[j + 1]
                current_cost = oracle.distance(a, b) + oracle.distance(c, d)
                swapped_cost = oracle.distance(a, c) + oracle.distance(b, d)
                if swapped_cost < current_cost:
                    route[i : j + 1] = reversed(route[i : j + 1])
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return tuple(route)


def optimize_wave_route(
    wave_id: str,
    stops: Sequence[Position],
    *,
    oracle: GridDistanceOracle,
    config: RouteOptimizationConfig,
) -> WaveRouteResult:
    original = tuple(stops)
    original_distance = route_distance(original, oracle=oracle, depot=config.depot)
    nn_route = nearest_neighbor_route(original, oracle=oracle, depot=config.depot)
    nn_distance = route_distance(nn_route, oracle=oracle, depot=config.depot)
    two_opt = two_opt_route(
        nn_route,
        oracle=oracle,
        depot=config.depot,
        max_passes=config.two_opt_max_passes,
        candidate_span=config.two_opt_candidate_span,
    )
    two_opt_distance = route_distance(two_opt, oracle=oracle, depot=config.depot)

    # A production optimizer must never make a wave worse. Keep the best route
    # among the recorded order, nearest-neighbor, and 2-opt refinement.
    candidates = [
        (original_distance, 0, original),
        (nn_distance, 1, nn_route),
        (two_opt_distance, 2, two_opt),
    ]
    optimized_distance, _, optimized = min(candidates, key=lambda item: (item[0], item[1]))
    return WaveRouteResult(
        wave_id=str(wave_id),
        original_stops=original,
        nearest_neighbor_stops=nn_route,
        optimized_stops=optimized,
        original_distance=original_distance,
        nearest_neighbor_distance=nn_distance,
        optimized_distance=optimized_distance,
    )


def _safe_nunique(series: pd.Series | None) -> int:
    if series is None:
        return 0
    values = series.astype("string").str.strip()
    values = values[values.notna() & values.ne("") & values.str.lower().ne("none")]
    return int(values.nunique())


def _zone_entropy(zone_ratios: Iterable[float]) -> float:
    values = np.asarray(list(zone_ratios), dtype=float)
    values = values[values > 0]
    if values.size == 0:
        return 0.0
    return float(-(values * np.log2(values)).sum())


def _pairwise_distance_stats(
    stops: Sequence[Position],
    *,
    oracle: GridDistanceOracle,
) -> tuple[float, float]:
    if len(stops) < 2:
        return 0.0, 0.0
    distances = [
        oracle.distance(stops[i], stops[j])
        for i in range(len(stops) - 1)
        for j in range(i + 1, len(stops))
    ]
    return float(np.mean(distances)), float(np.max(distances))


def _route_jump_stats(
    stops: Sequence[Position],
    *,
    oracle: GridDistanceOracle,
    depot: Position,
) -> tuple[float, float]:
    if not stops:
        return 0.0, 0.0
    full = [depot, *stops, depot]
    jumps = [oracle.distance(a, b) for a, b in zip(full, full[1:])]
    return float(np.mean(jumps)), float(np.max(jumps))


def _zone_transition_count(wave: pd.DataFrame) -> int:
    ordered = wave.sort_values(["source_sequence"], kind="stable")
    seen_positions: set[Position] = set()
    zones: list[str] = []
    for row in ordered.itertuples(index=False):
        position = _position_tuple(row.x, row.y)
        if position in seen_positions:
            continue
        seen_positions.add(position)
        zones.append(str(row.zone))
    return int(sum(a != b for a, b in zip(zones, zones[1:])))


def validate_task_frame(tasks: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_TASK_COLUMNS - set(tasks.columns)
    if missing:
        raise ValueError(f"task frame is missing required columns: {sorted(missing)}")
    if tasks.empty:
        raise ValueError("task frame is empty")

    frame = tasks.copy()
    for column in ("source_sequence", "x", "y", "quantity"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["wave_id", "source_sequence", "x", "y", "zone", "quantity"]).copy()
    if frame.empty:
        raise ValueError("no valid rows remain after wave optimization validation")
    frame["wave_id"] = frame["wave_id"].map(canonical_wave_id).astype("string")
    frame = frame[frame["wave_id"].ne("")].copy()
    frame["source_sequence"] = frame["source_sequence"].astype(int)
    frame["x"] = frame["x"].astype(int)
    frame["y"] = frame["y"].astype(int)
    frame["quantity"] = frame["quantity"].clip(lower=0).astype(float)
    if "release_tick" not in frame.columns:
        frame["release_tick"] = frame.groupby("wave_id", sort=False).ngroup()
    frame["release_tick"] = pd.to_numeric(frame["release_tick"], errors="coerce").fillna(0).astype(int)
    if "task_id" not in frame.columns:
        frame["task_id"] = [f"ROW-{index:07d}" for index in range(len(frame))]
    return frame.sort_values(
        ["release_tick", "wave_id", "source_sequence", "task_id"], kind="stable"
    ).reset_index(drop=True)


def build_wave_optimization_dataset(
    tasks: pd.DataFrame,
    *,
    warehouse: WarehouseMap | None = None,
    config: RouteOptimizationConfig | None = None,
    max_waves: int | None = None,
) -> pd.DataFrame:
    """Create one leakage-safe ML row per WMS picking wave."""
    frame = validate_task_frame(tasks)
    warehouse = warehouse or WarehouseMap.create_default()
    requested_config = config or RouteOptimizationConfig()
    depot = resolve_depot(warehouse, requested_config.depot)
    config = RouteOptimizationConfig(
        depot=depot,
        two_opt_max_passes=requested_config.two_opt_max_passes,
        two_opt_candidate_span=requested_config.two_opt_candidate_span,
        seconds_per_grid_step=requested_config.seconds_per_grid_step,
        seconds_per_pick_line=requested_config.seconds_per_pick_line,
    )
    oracle = GridDistanceOracle(warehouse)

    wave_order = (
        frame.groupby("wave_id", sort=False)["release_tick"]
        .min()
        .sort_values(kind="stable")
        .index.astype(str)
        .tolist()
    )
    if max_waves is not None:
        if max_waves <= 0:
            raise ValueError("max_waves must be positive")
        wave_order = wave_order[:max_waves]

    rows: list[dict[str, object]] = []
    grouped = {str(key): group for key, group in frame.groupby("wave_id", sort=False)}
    for order_index, wave_id in enumerate(wave_order):
        wave = grouped[wave_id].sort_values("source_sequence", kind="stable")
        stops = unique_stops_in_sequence(wave)
        route = optimize_wave_route(
            wave_id,
            stops,
            oracle=oracle,
            config=config,
        )

        task_count = int(len(wave))
        source_location = wave["source_location"] if "source_location" in wave.columns else None
        unique_location_count = _safe_nunique(source_location)
        if unique_location_count == 0:
            unique_location_count = int(len(stops))
        unique_grid_stop_count = int(len(stops))
        product_reference = wave["product_reference"] if "product_reference" in wave.columns else None
        operator = wave["operator"] if "operator" in wave.columns else None

        zone_counts = wave["zone"].astype(str).value_counts().reindex(list("ABCD"), fill_value=0)
        zone_ratios = zone_counts.astype(float) / max(task_count, 1)
        coords = np.asarray(stops, dtype=float) if stops else np.empty((0, 2), dtype=float)
        if len(coords):
            mean_x, mean_y = coords.mean(axis=0)
            std_x, std_y = coords.std(axis=0, ddof=0)
            bbox_area = float((coords[:, 0].max() - coords[:, 0].min() + 1) * (coords[:, 1].max() - coords[:, 1].min() + 1))
        else:
            mean_x = mean_y = std_x = std_y = bbox_area = 0.0

        pair_mean, pair_max = _pairwise_distance_stats(stops, oracle=oracle)
        jump_mean, jump_max = _route_jump_stats(stops, oracle=oracle, depot=depot)
        service_seconds = task_count * config.seconds_per_pick_line
        original_estimated_seconds = route.original_distance * config.seconds_per_grid_step + service_seconds
        optimized_estimated_seconds = route.optimized_distance * config.seconds_per_grid_step + service_seconds
        time_saved_seconds = max(0.0, original_estimated_seconds - optimized_estimated_seconds)
        time_saving_pct = (
            100.0 * time_saved_seconds / original_estimated_seconds
            if original_estimated_seconds > 0
            else 0.0
        )

        rows.append(
            {
                "wave_id": wave_id,
                "wave_order_index": order_index,
                "wave_start_tick": int(wave["release_tick"].min()),
                "task_count": task_count,
                "total_quantity": float(wave["quantity"].sum()),
                "unique_sku_count": _safe_nunique(product_reference),
                "unique_location_count": unique_location_count,
                "unique_grid_stop_count": unique_grid_stop_count,
                "operator_count": _safe_nunique(operator),
                "repeated_location_lines": max(0, task_count - unique_location_count),
                "active_zone_count": int((zone_counts > 0).sum()),
                "dominant_zone_ratio": float(zone_ratios.max()),
                "zone_entropy": _zone_entropy(zone_ratios.tolist()),
                "zone_a_ratio": float(zone_ratios.loc["A"]),
                "zone_b_ratio": float(zone_ratios.loc["B"]),
                "zone_c_ratio": float(zone_ratios.loc["C"]),
                "zone_d_ratio": float(zone_ratios.loc["D"]),
                "mean_x": float(mean_x),
                "mean_y": float(mean_y),
                "std_x": float(std_x),
                "std_y": float(std_y),
                "bounding_box_area": bbox_area,
                "mean_pairwise_distance": pair_mean,
                "max_pairwise_distance": pair_max,
                "original_route_distance": route.original_distance,
                "distance_per_stop": (
                    float(route.original_distance) / unique_grid_stop_count
                    if unique_grid_stop_count > 0
                    else 0.0
                ),
                "mean_route_jump": jump_mean,
                "max_route_jump": jump_max,
                "zone_transition_count": _zone_transition_count(wave),
                "nearest_neighbor_distance": route.nearest_neighbor_distance,
                "optimized_route_distance": route.optimized_distance,
                "distance_saved": route.distance_saved,
                "saving_pct": route.saving_pct,
                "original_estimated_seconds": original_estimated_seconds,
                "optimized_estimated_seconds": optimized_estimated_seconds,
                "estimated_time_saved_seconds": time_saved_seconds,
                "estimated_time_saving_pct": time_saving_pct,
                "optimization_changed_route": int(route.optimized_stops != route.original_stops),
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("no wave-level rows were generated")
    missing_features = set(MODEL_FEATURE_COLUMNS) - set(result.columns)
    if missing_features:
        raise RuntimeError(f"internal feature generation error: {sorted(missing_features)}")
    return result.sort_values("wave_order_index", kind="stable").reset_index(drop=True)


def route_for_wave(
    tasks: pd.DataFrame,
    wave_id: str,
    *,
    warehouse: WarehouseMap | None = None,
    config: RouteOptimizationConfig | None = None,
) -> WaveRouteResult:
    frame = validate_task_frame(tasks)
    target_wave_id = canonical_wave_id(wave_id)
    wave = frame[frame["wave_id"].map(canonical_wave_id) == target_wave_id]
    if wave.empty:
        raise KeyError(f"wave_id not found: {wave_id}")
    warehouse = warehouse or WarehouseMap.create_default()
    requested = config or RouteOptimizationConfig()
    config = RouteOptimizationConfig(
        depot=resolve_depot(warehouse, requested.depot),
        two_opt_max_passes=requested.two_opt_max_passes,
        two_opt_candidate_span=requested.two_opt_candidate_span,
        seconds_per_grid_step=requested.seconds_per_grid_step,
        seconds_per_pick_line=requested.seconds_per_pick_line,
    )
    oracle = GridDistanceOracle(warehouse)
    return optimize_wave_route(
        target_wave_id,
        unique_stops_in_sequence(wave),
        oracle=oracle,
        config=config,
    )


def expand_route_path(
    stops: Sequence[Position],
    *,
    warehouse: WarehouseMap,
    depot: Position,
) -> list[Position]:
    """Expand stop-to-stop route into actual grid cells for visualization."""
    oracle = GridDistanceOracle(warehouse)
    full_stops = [depot, *stops, depot]
    if len(full_stops) == 2 and full_stops[0] == full_stops[1]:
        return [depot]
    expanded: list[Position] = []
    for start, goal in zip(full_stops, full_stops[1:]):
        segment = oracle.shortest_path(start, goal)
        if expanded and segment and segment[0] == expanded[-1]:
            segment = segment[1:]
        expanded.extend(segment)
    return expanded or [depot]
