from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
RUNTIME_DIR = ROOT_DIR / "runtime"

for directory in (DATA_DIR, MODEL_DIR, RUNTIME_DIR):
    directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class SimulationConfig:
    """Runtime configuration for the warehouse simulator.

    The adaptive controller uses two thresholds (enter/exit) rather than one
    threshold. This hysteresis prevents rapid mode switching when the score is
    close to the boundary.
    """

    grid_width: int = 20
    grid_height: int = 20
    robot_count: int = 8
    cell_size: int = 30
    panel_width: int = 430
    fps: int = 30
    movement_interval_ticks: int = 2
    prediction_interval_ticks: int = 25
    work_duration_ticks: int = 18
    simulation_minutes_per_tick: float = 0.5
    start_minute: float = 8 * 60.0
    random_seed: int = 42
    snapshot_interval_ticks: int = 5
    min_battery_for_assignment: float = 10.0

    # Collision-aware rolling-horizon planning.
    mapf_planning_horizon_steps: int = 80
    mapf_goal_hold_steps: int = 12
    collision_shield_enabled: bool = True

    # Lifelong MAPF with unassigned agents (MAPFUA).
    idle_decision_interval_ticks: int = 25
    max_idle_moves_per_cycle: int = 2
    idle_local_density_radius: int = 2
    idle_blocking_lookahead_steps: int = 12
    idle_usefulness_window_ticks: int = 100
    idle_clear_blocking_threshold: float = 1.0
    idle_density_threshold: float = 0.28

    # Interactive stress test. Surge locations and quantities are sampled from
    # the prepared Footwear dataset.
    order_surge_count: int = 20
    surge_lead_ticks: int = 40
    surge_forecast_strength: float = 0.30

    # Pure predictive placement policy.
    regular_reposition_min_difference: int = 2
    max_reposition_robots_per_cycle: int = 2
    adaptive_max_robot_share_per_zone: float = 0.50

    # Adaptive Reactive/Predictive policy.
    adaptive_enter_score: float = 0.62
    adaptive_regular_enter_score: float = 0.76
    adaptive_regular_min_top_forecast: float = 3.5
    adaptive_exit_score: float = 0.48
    adaptive_overload_queue_per_robot: float = 2.0
    adaptive_min_idle_ratio: float = 0.25
    adaptive_min_forecast_total: float = 2.0
    adaptive_min_top_zone_share: float = 0.40
    adaptive_min_forecast_gap: float = 0.75
    adaptive_min_policy_hold_ticks: int = 25

    def __post_init__(self) -> None:
        if self.grid_width < 12 or self.grid_height < 12:
            raise ValueError("grid dimensions must be at least 12x12")
        if self.robot_count <= 0:
            raise ValueError("robot_count must be positive")
        if self.simulation_minutes_per_tick <= 0:
            raise ValueError("simulation_minutes_per_tick must be positive")
        if not 0 <= self.adaptive_exit_score <= self.adaptive_enter_score <= 1:
            raise ValueError("adaptive thresholds must satisfy 0 <= exit <= enter <= 1")
        if not self.adaptive_enter_score <= self.adaptive_regular_enter_score <= 1:
            raise ValueError("adaptive_regular_enter_score must be between enter threshold and 1")
        if self.adaptive_regular_min_top_forecast < 0:
            raise ValueError("adaptive_regular_min_top_forecast cannot be negative")
        if not 0 <= self.adaptive_min_idle_ratio <= 1:
            raise ValueError("adaptive_min_idle_ratio must be between 0 and 1")
        if self.max_reposition_robots_per_cycle < 0:
            raise ValueError("max_reposition_robots_per_cycle cannot be negative")
        if not 0 < self.adaptive_max_robot_share_per_zone <= 1:
            raise ValueError("adaptive_max_robot_share_per_zone must be in (0, 1]")
        if self.mapf_planning_horizon_steps < 4:
            raise ValueError("mapf_planning_horizon_steps must be at least 4")
        if self.mapf_goal_hold_steps < 0:
            raise ValueError("mapf_goal_hold_steps cannot be negative")
        if self.idle_decision_interval_ticks <= 0:
            raise ValueError("idle_decision_interval_ticks must be positive")
        if self.max_idle_moves_per_cycle < 0:
            raise ValueError("max_idle_moves_per_cycle cannot be negative")
        if self.idle_local_density_radius < 1:
            raise ValueError("idle_local_density_radius must be positive")
        if self.idle_blocking_lookahead_steps < 1:
            raise ValueError("idle_blocking_lookahead_steps must be positive")
