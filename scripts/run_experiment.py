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

from smart_warehouse.comparison import ComparisonRunner
from smart_warehouse.config import DATA_DIR, RUNTIME_DIR, SimulationConfig


def run_seed(
    seed: int,
    ticks: int,
    surge_tick: int,
    surge_zone: str,
    surge_count: int,
    *,
    tasks_path: Path,
    manifest_path: Path,
) -> list[dict]:
    config = replace(
        SimulationConfig(),
        random_seed=seed,
        snapshot_interval_ticks=ticks + 1,
    )
    runner = ComparisonRunner(
        config,
        persist=False,
        tasks_path=tasks_path,
        manifest_path=manifest_path,
    )
    for tick in range(1, ticks + 1):
        if surge_tick > 0 and tick == surge_tick:
            runner.trigger_order_surge(surge_zone, surge_count)
        runner.step()

    rows: list[dict] = []
    for mode, engine in (("reactive", runner.reactive), ("adaptive_hybrid", runner.hybrid)):
        rows.append(
            {
                "mode": mode,
                "seed": seed,
                "ticks": ticks,
                "surge_tick": surge_tick,
                "surge_zone": surge_zone if surge_tick > 0 else None,
                "surge_count": surge_count if surge_tick > 0 else 0,
                **engine.metrics_summary(),
            }
        )
    return rows


def _pct(baseline: float, contender: float, direction: str) -> float:
    if abs(baseline) < 1e-9:
        return 0.0
    if direction == "reduction":
        return round(100.0 * (baseline - contender) / baseline, 3)
    return round(100.0 * (contender - baseline) / baseline, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Average Reactive baseline and threshold-based Adaptive Hybrid results "
            "on the same prepared Footwear task replay."
        )
    )
    parser.add_argument("--tasks", type=Path, default=DATA_DIR / "footwear_replay_tasks.csv")
    parser.add_argument("--manifest", type=Path, default=DATA_DIR / "footwear_manifest.json")
    parser.add_argument(
        "--ticks",
        type=int,
        default=0,
        help="Run length. Zero means last replay tick plus --drain-ticks.",
    )
    parser.add_argument("--drain-ticks", type=int, default=600)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    parser.add_argument("--surge-tick", type=int, default=300)
    parser.add_argument("--surge-zone", choices=["A", "B", "C", "D"], default="C")
    parser.add_argument("--surge-count", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.tasks.exists():
        raise FileNotFoundError(
            f"Prepared Footwear replay not found: {args.tasks}. Run "
            "`python -m scripts.prepare_footwear_data --dataset-dir <folder>` first."
        )
    if args.drain_ticks < 0:
        raise ValueError("--drain-ticks cannot be negative")
    if not args.seeds:
        raise ValueError("--seeds must contain at least one seed")

    probe = ComparisonRunner(
        replace(SimulationConfig(), random_seed=args.seeds[0]),
        persist=False,
        tasks_path=args.tasks,
        manifest_path=args.manifest,
    )
    assert probe.replay is not None
    ticks = args.ticks if args.ticks > 0 else probe.replay.max_tick + args.drain_ticks
    if ticks <= 0:
        raise ValueError("experiment ticks must be positive")
    if args.surge_tick > ticks:
        raise ValueError("--surge-tick cannot exceed the resolved run length")

    rows = [
        row
        for seed in args.seeds
        for row in run_seed(
            seed,
            ticks,
            args.surge_tick,
            args.surge_zone,
            args.surge_count,
            tasks_path=args.tasks,
            manifest_path=args.manifest,
        )
    ]
    detail = pd.DataFrame(rows)
    metric_columns = [
        "generated_tasks",
        "completed_tasks",
        "completion_rate_pct",
        "throughput_per_hour",
        "waiting_tasks",
        "assigned_tasks",
        "in_progress_tasks",
        "average_wait_time",
        "p95_wait_time",
        "average_completion_time",
        "total_distance",
        "task_distance",
        "reposition_distance",
        "predictive_policy_pct",
        "reactive_policy_pct",
    ]
    summary = detail.groupby("mode", as_index=False)[metric_columns].mean().round(3)

    reactive = summary.loc[summary["mode"] == "reactive"].iloc[0]
    hybrid = summary.loc[summary["mode"] == "adaptive_hybrid"].iloc[0]
    improvements = {
        "runs": len(args.seeds),
        "shared_footwear_task_stream_per_seed": True,
        "ticks": ticks,
        "drain_ticks": args.drain_ticks if args.ticks == 0 else 0,
        "surge_tick": args.surge_tick,
        "surge_zone": args.surge_zone,
        "surge_count": args.surge_count,
        "wait_time_reduction_pct": _pct(
            float(reactive["average_wait_time"]), float(hybrid["average_wait_time"]), "reduction"
        ),
        "p95_wait_reduction_pct": _pct(
            float(reactive["p95_wait_time"]), float(hybrid["p95_wait_time"]), "reduction"
        ),
        "throughput_increase_pct": _pct(
            float(reactive["throughput_per_hour"]), float(hybrid["throughput_per_hour"]), "increase"
        ),
        "distance_reduction_pct": _pct(
            float(reactive["total_distance"]), float(hybrid["total_distance"]), "reduction"
        ),
        "task_distance_reduction_pct": _pct(
            float(reactive["task_distance"]), float(hybrid["task_distance"]), "reduction"
        ),
        "hybrid_predictive_policy_pct": float(hybrid["predictive_policy_pct"]),
        "hybrid_reactive_policy_pct": float(hybrid["reactive_policy_pct"]),
        "interpretation": (
            "Positive reduction/increase values favor Adaptive Hybrid. The Hybrid may still increase "
            "total distance if proactive repositioning costs more than the task travel it saves."
        ),
    }

    detail_path = RUNTIME_DIR / "experiment_detail.csv"
    summary_path = RUNTIME_DIR / "experiment_summary.csv"
    improvements_path = RUNTIME_DIR / "experiment_improvements.json"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    improvements_path.write_text(json.dumps(improvements, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nMean results")
    print(summary.to_string(index=False))
    print("\nImprovements")
    print(json.dumps(improvements, ensure_ascii=False, indent=2))
    print(f"\nSaved: {detail_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {improvements_path}")


if __name__ == "__main__":
    main()
