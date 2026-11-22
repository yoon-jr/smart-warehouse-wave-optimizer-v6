from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from smart_warehouse.wave_benefit_models import (
    build_adaptive_policy_leaderboard,
    chronological_split,
    evaluate_predictions,
    train_wave_benefit_pipeline,
)
from smart_warehouse.wave_optimization import (
    MODEL_FEATURE_COLUMNS,
    GridDistanceOracle,
    RouteOptimizationConfig,
    build_wave_optimization_dataset,
    nearest_neighbor_route,
    optimize_wave_route,
    route_distance,
    two_opt_route,
)
from smart_warehouse.warehouse import WarehouseMap


def make_task_frame(waves: int = 30) -> pd.DataFrame:
    warehouse = WarehouseMap.create_default()
    candidate_stops = [
        position
        for position in [
            (2, 2),
            (7, 2),
            (12, 2),
            (17, 2),
            (2, 7),
            (7, 7),
            (12, 7),
            (17, 7),
            (2, 12),
            (7, 12),
            (12, 12),
            (17, 12),
            (2, 17),
            (7, 17),
            (12, 17),
            (17, 17),
        ]
        if warehouse.walkable(position)
    ]
    rows: list[dict[str, object]] = []
    for wave_index in range(waves):
        stop_count = 4 + wave_index % 8
        # Alternate between a coherent order and a deliberately zig-zag order
        # so the target has meaningful variance.
        chosen = candidate_stops[:stop_count]
        if wave_index % 2:
            chosen = chosen[::2] + chosen[1::2][::-1]
        for sequence, (x, y) in enumerate(chosen, start=1):
            rows.append(
                {
                    "task_id": f"T-{wave_index}-{sequence}",
                    "release_tick": wave_index * 100 + sequence,
                    "wave_id": f"W-{wave_index:03d}",
                    "source_sequence": sequence,
                    "x": x,
                    "y": y,
                    "zone": warehouse.zone_for((x, y)),
                    "quantity": 1 + (sequence % 3),
                    "source_location": f"L-{x}-{y}",
                    "product_reference": f"SKU-{sequence % 5}",
                    "operator": f"OP-{wave_index % 3}",
                }
            )
    return pd.DataFrame(rows)


def test_two_opt_never_worsens_nearest_neighbor() -> None:
    warehouse = WarehouseMap.create_default()
    oracle = GridDistanceOracle(warehouse)
    depot = (1, 1)
    stops = [(2, 2), (17, 2), (2, 17), (17, 17), (7, 7), (12, 12)]
    nn = nearest_neighbor_route(stops, oracle=oracle, depot=depot)
    improved = two_opt_route(nn, oracle=oracle, depot=depot, max_passes=10)
    assert route_distance(improved, oracle=oracle, depot=depot) <= route_distance(
        nn, oracle=oracle, depot=depot
    )


def test_optimizer_never_worsens_recorded_route() -> None:
    warehouse = WarehouseMap.create_default()
    oracle = GridDistanceOracle(warehouse)
    stops = [(2, 2), (17, 17), (17, 2), (2, 17)]
    result = optimize_wave_route(
        "W",
        stops,
        oracle=oracle,
        config=RouteOptimizationConfig(),
    )
    assert result.optimized_distance <= result.original_distance
    assert result.distance_saved == result.original_distance - result.optimized_distance


def test_wave_dataset_has_leakage_safe_features_and_targets() -> None:
    dataset = build_wave_optimization_dataset(make_task_frame(20))
    assert len(dataset) == 20
    assert set(MODEL_FEATURE_COLUMNS).issubset(dataset.columns)
    assert "saving_pct" not in MODEL_FEATURE_COLUMNS
    assert "optimized_route_distance" not in MODEL_FEATURE_COLUMNS
    assert (dataset["optimized_route_distance"] <= dataset["original_route_distance"]).all()
    assert dataset["saving_pct"].between(0, 100).all()


def test_chronological_split_preserves_order() -> None:
    dataset = build_wave_optimization_dataset(make_task_frame(30))
    split = chronological_split(dataset)
    assert split.train["wave_order_index"].max() < split.validation["wave_order_index"].min()
    assert split.validation["wave_order_index"].max() < split.test["wave_order_index"].min()
    assert len(split.train) + len(split.validation) + len(split.test) == len(dataset)


def test_prediction_policy_metrics_are_bounded() -> None:
    dataset = build_wave_optimization_dataset(make_task_frame(30))
    split = chronological_split(dataset)
    predictions = split.validation["saving_pct"].to_numpy() + 0.1
    evaluation = evaluate_predictions(
        split.validation,
        predictions,
        model_name="perfectish",
        prediction_seconds=0.001,
        coverage=0.20,
    )
    assert evaluation.mae < 0.2
    assert 0 <= evaluation.top20_recall_pct <= 100
    assert 0 <= evaluation.captured_savings_pct_at_20 <= 100


def test_end_to_end_training_outputs_artifacts(tmp_path: Path) -> None:
    dataset = build_wave_optimization_dataset(make_task_frame(40))
    report = train_wave_benefit_pipeline(
        dataset,
        output_dir=tmp_path / "runtime",
        model_dir=tmp_path / "models",
        coverage=0.20,
        random_seed=7,
    )
    assert report["selected_model"]
    assert (tmp_path / "models" / "wave_benefit_model.joblib").exists()
    assert (tmp_path / "runtime" / "wave_model_leaderboard.csv").exists()
    assert (tmp_path / "runtime" / "wave_adaptive_policy_leaderboard.csv").exists()
    assert (tmp_path / "runtime" / "wave_test_predictions.csv").exists()
