from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.run_mapfua_experiment import _reduction, choose_start_ticks, run_policy
from smart_warehouse.config import DATA_DIR, MODEL_DIR, RUNTIME_DIR, SimulationConfig
from smart_warehouse.footwear_replay import FootwearTaskReplay
from smart_warehouse.marl_vdn import VDNLearner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the main Lifelong MAPFUA ablation: heuristics, VDN without "
            "forecast, and VDN with forecast on matched WMS task windows."
        )
    )
    parser.add_argument("--tasks", type=Path, default=DATA_DIR / "footwear_replay_tasks.csv")
    parser.add_argument("--manifest", type=Path, default=DATA_DIR / "footwear_manifest.json")
    parser.add_argument("--ticks", type=int, default=1200)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--start-tick", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--robot-count", type=int, default=8)
    parser.add_argument(
        "--baseline-policies",
        nargs="+",
        default=["idle_stay", "nearest_parking", "density_clear", "forecast_rule"],
    )
    parser.add_argument(
        "--no-forecast-model",
        type=Path,
        default=MODEL_DIR / "mapfua_vdn_no_forecast.pt",
    )
    parser.add_argument(
        "--forecast-model",
        type=Path,
        default=MODEL_DIR / "mapfua_vdn_with_forecast.pt",
    )
    return parser.parse_args()


def load_model(path: Path, robot_count: int, expected_forecast: bool) -> VDNLearner:
    if not path.exists():
        raise FileNotFoundError(f"Required ablation model does not exist: {path}")
    learner = VDNLearner.load(path)
    if learner.robot_count != robot_count:
        raise ValueError(
            f"{path.name} expects {learner.robot_count} robots, but --robot-count={robot_count}"
        )
    metadata_value = learner.metadata.get("include_forecast")
    if metadata_value is not None and bool(metadata_value) != expected_forecast:
        raise ValueError(
            f"{path.name} metadata include_forecast={metadata_value}, "
            f"expected {expected_forecast}"
        )
    return learner


def main() -> None:
    args = parse_args()
    if not args.tasks.exists():
        raise FileNotFoundError(f"Prepared task replay not found: {args.tasks}")
    if args.ticks <= 0 or args.robot_count <= 0:
        raise ValueError("--ticks and --robot-count must be positive")

    replay = FootwearTaskReplay(args.tasks, manifest_path=args.manifest, random_seed=args.seeds[0])
    starts = choose_start_ticks(
        replay,
        split_name=args.split,
        ticks=args.ticks,
        seeds=[int(seed) for seed in args.seeds],
        fixed_start_tick=args.start_tick,
    )
    config = replace(SimulationConfig(), robot_count=args.robot_count)
    no_forecast = load_model(args.no_forecast_model, args.robot_count, False)
    with_forecast = load_model(args.forecast_model, args.robot_count, True)

    rows: list[dict] = []
    baseline_policies = [item.strip().lower().replace("-", "_") for item in args.baseline_policies]
    for seed in args.seeds:
        start_tick = starts[int(seed)]
        for policy in baseline_policies:
            rows.append(
                run_policy(
                    policy_name=policy,
                    seed=int(seed),
                    start_tick=start_tick,
                    ticks=args.ticks,
                    replay=replay,
                    config=config,
                    learner=None,
                    learner_name="unused",
                    learner_include_forecast=False,
                )
            )
        rows.append(
            run_policy(
                policy_name="vdn_no_forecast",
                seed=int(seed),
                start_tick=start_tick,
                ticks=args.ticks,
                replay=replay,
                config=config,
                learner=no_forecast,
                learner_name="vdn_no_forecast",
                learner_include_forecast=False,
            )
        )
        rows.append(
            run_policy(
                policy_name="vdn_with_forecast",
                seed=int(seed),
                start_tick=start_tick,
                ticks=args.ticks,
                replay=replay,
                config=config,
                learner=with_forecast,
                learner_name="vdn_with_forecast",
                learner_include_forecast=True,
            )
        )

    detail = pd.DataFrame(rows)
    numeric = [column for column in detail.columns if column not in {"policy", "seed"}]
    summary = detail.groupby("policy", as_index=False)[numeric].mean(numeric_only=True).round(4)

    indexed = summary.set_index("policy")
    comparisons: dict[str, dict] = {}
    if "idle_stay" in indexed.index:
        baseline = indexed.loc["idle_stay"]
        for policy, row in indexed.iterrows():
            comparisons[str(policy)] = {
                "wait_reduction_pct_vs_idle_stay": _reduction(
                    float(baseline["average_wait_time"]), float(row["average_wait_time"])
                ),
                "p95_reduction_pct_vs_idle_stay": _reduction(
                    float(baseline["p95_wait_time"]), float(row["p95_wait_time"])
                ),
                "distance_reduction_pct_vs_idle_stay": _reduction(
                    float(baseline["total_distance"]), float(row["total_distance"])
                ),
                "blocking_reduction_pct_vs_idle_stay": _reduction(
                    float(baseline["blocking_pressure_ticks"]),
                    float(row["blocking_pressure_ticks"]),
                ),
            }

    forecast_value: dict[str, float] = {}
    if {"vdn_no_forecast", "vdn_with_forecast"}.issubset(indexed.index):
        without = indexed.loc["vdn_no_forecast"]
        with_ = indexed.loc["vdn_with_forecast"]
        forecast_value = {
            "wait_reduction_pct_with_vs_without_forecast": _reduction(
                float(without["average_wait_time"]), float(with_["average_wait_time"])
            ),
            "p95_reduction_pct_with_vs_without_forecast": _reduction(
                float(without["p95_wait_time"]), float(with_["p95_wait_time"])
            ),
            "distance_reduction_pct_with_vs_without_forecast": _reduction(
                float(without["total_distance"]), float(with_["total_distance"])
            ),
            "blocking_reduction_pct_with_vs_without_forecast": _reduction(
                float(without["blocking_pressure_ticks"]),
                float(with_["blocking_pressure_ticks"]),
            ),
        }

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    detail_path = RUNTIME_DIR / "mapfua_ablation_detail.csv"
    summary_path = RUNTIME_DIR / "mapfua_ablation_summary.csv"
    result_path = RUNTIME_DIR / "mapfua_ablation_results.json"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    result_path.write_text(
        json.dumps(
            {
                "same_wms_task_window_per_seed": True,
                "split": args.split,
                "start_ticks_by_seed": {str(key): value for key, value in starts.items()},
                "runs_per_policy": len(args.seeds),
                "comparisons_vs_idle_stay": comparisons,
                "forecast_value": forecast_value,
                "interpretation": (
                    "Positive reduction values favor the named contender. Forecast value must "
                    "be interpreted jointly with idle movement, blocking pressure, completion, "
                    "and decision latency; a positive forecast effect is not assumed."
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\nMean results")
    print(summary.to_string(index=False))
    print("\nForecast ablation")
    print(json.dumps(forecast_value, ensure_ascii=False, indent=2))
    print(f"\nSaved: {detail_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {result_path}")


if __name__ == "__main__":
    main()
