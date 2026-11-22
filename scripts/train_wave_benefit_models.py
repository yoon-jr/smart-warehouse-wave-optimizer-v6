from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from smart_warehouse.wave_benefit_models import train_wave_benefit_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Dummy, Ridge, Random Forest, HistGradientBoosting and XGBoost "
            "on chronological validation data, then evaluate the selected wave-benefit model once on test."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/wave_optimization_dataset.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runtime/wave_optimizer"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--coverage",
        type=float,
        default=0.20,
        help="Fraction of test waves sent to full NN + 2-opt optimization.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset.exists():
        raise FileNotFoundError(
            f"Wave optimization dataset not found: {args.dataset}. "
            "Run `python -m scripts.build_wave_optimization_dataset` first."
        )
    dataset = pd.read_csv(args.dataset, low_memory=False)
    report = train_wave_benefit_pipeline(
        dataset,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        coverage=args.coverage,
        random_seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nValidation model leaderboard")
    print(pd.read_csv(args.output_dir / "wave_model_leaderboard.csv").to_string(index=False))
    print("\nHeld-out test adaptive-policy leaderboard")
    print(pd.read_csv(args.output_dir / "wave_adaptive_policy_leaderboard.csv").to_string(index=False))


if __name__ == "__main__":
    main()
