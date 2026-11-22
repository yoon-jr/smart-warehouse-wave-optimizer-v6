from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from smart_warehouse.config import DATA_DIR, SimulationConfig
from smart_warehouse.footwear_data import prepare_footwear_dataset, save_prepared_footwear_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the public Footwear Warehouse Order Picking dataset for replay, "
            "training, and four-zone visualization."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Folder containing extracted Picking_Wave.csv and Storage_Location.csv.",
    )
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--minutes-per-wave", type=float, default=15.0)
    parser.add_argument("--simulation-minutes-per-tick", type=float, default=0.5)
    parser.add_argument("--minimum-join-rate", type=float, default=0.80)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = replace(
        SimulationConfig(),
        simulation_minutes_per_tick=args.simulation_minutes_per_tick,
    )
    prepared = prepare_footwear_dataset(
        args.dataset_dir,
        config=config,
        minutes_per_wave=args.minutes_per_wave,
        minimum_join_rate=args.minimum_join_rate,
        max_rows=args.max_rows,
    )
    paths = save_prepared_footwear_data(prepared, output_dir=args.output_dir)

    print(f"Prepared tasks: {len(prepared.tasks):,}")
    print(f"Training rows: {len(prepared.training_table):,}")
    print(f"Location join rate: {prepared.manifest['join_rate']:.1%}")
    print(f"Time basis: {prepared.manifest['time_basis']}")
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
