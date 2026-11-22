from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from smart_warehouse.config import SimulationConfig
from smart_warehouse.footwear_data import prepare_footwear_dataset


def test_footwear_preparation_joins_coordinates_and_builds_training_data(
    footwear_source_dir: Path,
) -> None:
    prepared = prepare_footwear_dataset(
        footwear_source_dir,
        config=SimulationConfig(simulation_minutes_per_tick=1.0),
        minimum_join_rate=0.95,
    )

    assert len(prepared.tasks) == 90
    assert prepared.manifest["join_rate"] == 1.0
    assert prepared.manifest["time_basis"] == "wave_duration_sequence"
    assert set(prepared.tasks["zone"]) == {"A", "B", "C", "D"}
    assert prepared.tasks["release_tick"].min() == 1
    assert prepared.tasks["service_ticks"].min() >= 1
    assert not prepared.training_table.empty
    assert set(["zone", "lag_1", "rolling_3", "target_next_count"]).issubset(
        prepared.training_table.columns
    )


def test_footwear_manifest_records_fallback_assumption(prepared_footwear_paths: dict[str, Path]) -> None:
    manifest = json.loads(prepared_footwear_paths["manifest"].read_text(encoding="utf-8"))
    tasks = pd.read_csv(prepared_footwear_paths["tasks"])

    assert manifest["dataset"].startswith("Order Picking Dataset")
    assert manifest["license"] == "CC BY 4.0"
    assert "fallback" in manifest["time_basis_note"].lower()
    assert tasks["data_source"].eq("footwear_wms").all()


def test_point_parser_handles_comma_separated_coordinates() -> None:
    from smart_warehouse.footwear_data import parse_point

    assert parse_point("(8, 1, 0)") == (8.0, 1.0, 0.0)
    assert parse_point("8,5; 1,25; 0") == (8.5, 1.25, 0.0)
