from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from smart_warehouse.comparison import ComparisonRunner
from smart_warehouse.config import SimulationConfig
from smart_warehouse.entities import RobotStatus


def small_config() -> SimulationConfig:
    return replace(
        SimulationConfig(),
        robot_count=4,
        prediction_interval_ticks=2,
        movement_interval_ticks=1,
        work_duration_ticks=2,
        simulation_minutes_per_tick=1.0,
        snapshot_interval_ticks=10_000,
        random_seed=7,
        surge_lead_ticks=2,
        surge_forecast_strength=1.0,
        max_reposition_robots_per_cycle=2,
        adaptive_min_policy_hold_ticks=0,
    )


def runner(paths: dict[str, Path]) -> ComparisonRunner:
    return ComparisonRunner(
        small_config(),
        persist=False,
        tasks_path=paths["tasks"],
        manifest_path=paths["manifest"],
    )


def test_comparison_uses_identical_footwear_task_stream(
    prepared_footwear_paths: dict[str, Path],
) -> None:
    comparison = runner(prepared_footwear_paths)
    for _ in range(12):
        comparison.step()

    reactive = comparison.reactive.tasks
    hybrid = comparison.hybrid.tasks
    assert reactive.keys() == hybrid.keys()
    assert len(reactive) > 0
    for task_id in reactive:
        assert reactive[task_id].created_minute == hybrid[task_id].created_minute
        assert reactive[task_id].zone_id == hybrid[task_id].zone_id
        assert reactive[task_id].position == hybrid[task_id].position
        assert reactive[task_id] is not hybrid[task_id]


def test_footwear_pattern_surge_is_injected_into_both_modes(
    prepared_footwear_paths: dict[str, Path],
) -> None:
    comparison = runner(prepared_footwear_paths)
    comparison.trigger_order_surge("C", 6)
    comparison.step()
    assert not {task_id for task_id in comparison.reactive.tasks if task_id.startswith("FS-")}
    comparison.step()

    reactive_surge = {task_id for task_id in comparison.reactive.tasks if task_id.startswith("FS-")}
    hybrid_surge = {task_id for task_id in comparison.hybrid.tasks if task_id.startswith("FS-")}
    assert reactive_surge == hybrid_surge
    assert len(reactive_surge) == 6
    assert all(comparison.reactive.tasks[task_id].zone_id == "C" for task_id in reactive_surge)


def test_snapshot_explains_policy_and_fairness(prepared_footwear_paths: dict[str, Path]) -> None:
    comparison = runner(prepared_footwear_paths)
    comparison.step()
    snapshot = comparison.snapshot()
    assert snapshot["scenario"]["shared_task_stream"] is True
    assert snapshot["scenario"]["data_source"] == "footwear_wms"
    assert snapshot["scenario"]["comparison"].startswith("Reactive baseline")
    assert snapshot["situation"]["level"] in {"NORMAL", "BUSY", "OVERLOADED"}
    assert snapshot["situation"]["active_policy"] in {"REACTIVE", "PREDICTIVE"}
    assert snapshot["situation"]["policy_reason"]
    assert set(snapshot["improvements"]) >= {
        "wait_time_reduction_pct",
        "throughput_increase_pct",
        "distance_reduction_pct",
    }


def test_surge_repositions_at_most_configured_number_of_robots(
    prepared_footwear_paths: dict[str, Path],
) -> None:
    comparison = runner(prepared_footwear_paths)
    comparison.trigger_order_surge("C", 8)
    moving = sum(robot.status == RobotStatus.REPOSITIONING for robot in comparison.hybrid.robots)
    assert moving <= comparison.config.max_reposition_robots_per_cycle
    assert comparison.hybrid.target_robot_counts["C"] <= 2
    assert comparison.hybrid.active_policy == "PREDICTIVE"
