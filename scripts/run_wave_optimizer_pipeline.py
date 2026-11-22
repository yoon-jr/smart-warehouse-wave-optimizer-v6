from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from smart_warehouse.wave_benefit_models import train_wave_benefit_pipeline
from smart_warehouse.wave_optimization import RouteOptimizationConfig, build_wave_optimization_dataset
from smart_warehouse.warehouse import WarehouseMap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the complete WMS wave optimizer pipeline.")
    parser.add_argument("--tasks", type=Path, default=Path("data/footwear_replay_tasks.csv"))
    parser.add_argument(
        "--dataset-output",
        type=Path,
        default=Path("data/wave_optimization_dataset.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runtime/wave_optimizer"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--coverage", type=float, default=0.20)
    parser.add_argument("--max-waves", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--two-opt-passes", type=int, default=8)
    parser.add_argument("--two-opt-span", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.tasks.exists():
        raise FileNotFoundError(
            f"Prepared task file not found: {args.tasks}. Run the Footwear preprocessing step first."
        )
    started = time.perf_counter()
    tasks = pd.read_csv(args.tasks, low_memory=False)
    dataset = build_wave_optimization_dataset(
        tasks,
        warehouse=WarehouseMap.create_default(),
        config=RouteOptimizationConfig(
            two_opt_max_passes=args.two_opt_passes,
            two_opt_candidate_span=args.two_opt_span,
        ),
        max_waves=args.max_waves,
    )
    args.dataset_output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.dataset_output, index=False)
    report = train_wave_benefit_pipeline(
        dataset,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        coverage=args.coverage,
        random_seed=args.seed,
    )
    report["total_pipeline_seconds"] = round(time.perf_counter() - started, 3)
    report["dataset_output"] = str(args.dataset_output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
