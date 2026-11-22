from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from smart_warehouse.config import SimulationConfig
from smart_warehouse.footwear_data import prepare_footwear_dataset, save_prepared_footwear_data


@pytest.fixture
def footwear_source_dir(tmp_path: Path) -> Path:
    source = tmp_path / "footwear_source"
    source.mkdir()

    storage = pd.DataFrame(
        {
            "Location": ["A-01-11", "A-02-11", "B-01-11", "B-02-11", "C-01-11", "D-01-11"],
            "Points": [
                "(0.0, 0.0, 0.0)",
                "(2.0, 1.0, 0.0)",
                "(8.0, 1.0, 0.0)",
                "(9.0, 3.0, 0.0)",
                "(1.0, 9.0, 0.0)",
                "(9.0, 9.0, 0.0)",
            ],
        }
    )
    storage.to_csv(source / "Storage_Location.csv", index=False)

    locations = ["A-01-11", "A-02-11", "B-01-11", "B-02-11", "C-01-11", "D-01-11"]
    rows: list[dict] = []
    for wave_number in range(1, 31):
        for sequence in range(1, 4):
            location = locations[(wave_number + sequence) % len(locations)]
            rows.append(
                {
                    "Picking Wave": f"W-{wave_number:03d}",
                    "Duration": 8 + (wave_number % 4),
                    "Reference": f"SKU-{sequence:03d}",
                    "QuantityToPick": 1 + ((wave_number + sequence) % 5),
                    "OriginalLocation": location,
                    "Operator": f"OP-{1 + wave_number % 3}",
                    "Pick Sequence": sequence,
                }
            )
    pd.DataFrame(rows).to_csv(source / "Picking_Wave.csv", index=False)
    return source


@pytest.fixture
def prepared_footwear_paths(tmp_path: Path, footwear_source_dir: Path) -> dict[str, Path]:
    config = SimulationConfig(simulation_minutes_per_tick=1.0, start_minute=480)
    prepared = prepare_footwear_dataset(
        footwear_source_dir,
        config=config,
        minutes_per_wave=10,
        minimum_join_rate=0.95,
    )
    return save_prepared_footwear_data(prepared, output_dir=tmp_path / "prepared")
