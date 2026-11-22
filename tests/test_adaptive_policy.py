from __future__ import annotations

from smart_warehouse.adaptive_policy import AdaptivePolicyController
from smart_warehouse.config import SimulationConfig
from smart_warehouse.entities import Robot
from smart_warehouse.warehouse import WarehouseMap


def robots() -> list[Robot]:
    warehouse = WarehouseMap.create_default()
    positions = [warehouse.zones[zone].waiting_position for zone in "ABCD"] * 2
    return [Robot(f"R-{index:02d}", position) for index, position in enumerate(positions, start=1)]


def stable_history() -> dict[str, list[int]]:
    return {zone: [3, 3, 4, 3, 4, 3] for zone in "ABCD"}


def test_strong_spatial_signal_selects_predictive() -> None:
    controller = AdaptivePolicyController(SimulationConfig(adaptive_min_policy_hold_ticks=0))
    decision = controller.evaluate(
        predictions={"A": 12.0, "B": 1.0, "C": 1.0, "D": 1.0},
        queue_counts={zone: 0 for zone in "ABCD"},
        robots=robots(),
        recent_counts=stable_history(),
        tick=1,
    )
    assert decision.active_policy == "PREDICTIVE"
    assert decision.score >= controller.config.adaptive_enter_score
    assert decision.top_zone == "A"


def test_backlog_forces_reactive_even_with_strong_forecast() -> None:
    controller = AdaptivePolicyController(SimulationConfig(adaptive_min_policy_hold_ticks=0))
    decision = controller.evaluate(
        predictions={"A": 20.0, "B": 1.0, "C": 1.0, "D": 1.0},
        queue_counts={"A": 16, "B": 0, "C": 0, "D": 0},
        robots=robots(),
        recent_counts=stable_history(),
        tick=1,
    )
    assert decision.active_policy == "REACTIVE"
    assert "backlog" in decision.reason.lower()


def test_flat_forecast_stays_reactive() -> None:
    controller = AdaptivePolicyController(SimulationConfig(adaptive_min_policy_hold_ticks=0))
    decision = controller.evaluate(
        predictions={zone: 4.0 for zone in "ABCD"},
        queue_counts={zone: 0 for zone in "ABCD"},
        robots=robots(),
        recent_counts=stable_history(),
        tick=1,
    )
    assert decision.active_policy == "REACTIVE"
    assert "spread" in decision.reason.lower() or "top two" in decision.reason.lower()


def test_moderate_regular_signal_requires_stricter_threshold_but_notice_can_enter() -> None:
    config = SimulationConfig(adaptive_min_policy_hold_ticks=0)
    regular = AdaptivePolicyController(config).evaluate(
        predictions={"A": 3.0, "B": 0.0, "C": 0.0, "D": 0.0},
        queue_counts={zone: 0 for zone in "ABCD"},
        robots=robots(),
        recent_counts=stable_history(),
        tick=1,
        forecast_notice_active=False,
    )
    notice = AdaptivePolicyController(config).evaluate(
        predictions={"A": 3.0, "B": 0.0, "C": 0.0, "D": 0.0},
        queue_counts={zone: 0 for zone in "ABCD"},
        robots=robots(),
        recent_counts=stable_history(),
        tick=1,
        forecast_notice_active=True,
    )
    assert regular.active_policy == "REACTIVE"
    assert notice.active_policy == "PREDICTIVE"
