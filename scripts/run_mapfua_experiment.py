from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from smart_warehouse.config import DATA_DIR, MODEL_DIR, RUNTIME_DIR, SimulationConfig
from smart_warehouse.footwear_replay import FootwearTaskReplay
from smart_warehouse.idle_policy import ExternalJointPolicy, policy_from_name
from smart_warehouse.mapfua_env import SplitName, build_agent_observations, temporal_splits
from smart_warehouse.marl_vdn import VDNLearner
from smart_warehouse.simulation import SimulationEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Lifelong MAPFUA idle-robot policies on matched, "
            "chronological WMS-derived task windows."
        )
    )
    parser.add_argument("--tasks", type=Path, default=DATA_DIR / "footwear_replay_tasks.csv")
    parser.add_argument("--manifest", type=Path, default=DATA_DIR / "footwear_manifest.json")
    parser.add_argument("--ticks", type=int, default=1200)
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="test",
        help="Chronological split from which seeded evaluation windows are sampled.",
    )
    parser.add_argument(
        "--start-tick",
        type=int,
        default=None,
        help=(
            "Use one fixed window for every run. When omitted, each seed chooses "
            "one release-tick-aligned window from --split; all policies receive "
            "the same window for that seed."
        ),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--robot-count", type=int, default=8)
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["idle_stay", "nearest_parking", "density_clear", "forecast_rule"],
        help=(
            "Policies to evaluate. Add vdn_marl after training to include the learned policy."
        ),
    )
    parser.add_argument("--vdn-model", type=Path, default=MODEL_DIR / "mapfua_vdn.pt")
    parser.add_argument("--vdn-name", default="vdn_marl")
    parser.add_argument(
        "--vdn-include-forecast",
        action="store_true",
        help=(
            "Force forecast observations for a legacy model without embedded metadata. "
            "New v5 models infer this setting automatically."
        ),
    )
    return parser.parse_args()


def _initial_minute(replay: FootwearTaskReplay, start_tick: int, tick_minutes: float) -> float:
    future = replay.frame[replay.frame["release_tick"].astype(int) >= start_tick]
    if future.empty:
        return replay.start_minute
    return float(future.iloc[0]["created_minute"]) - tick_minutes


def choose_start_ticks(
    replay: FootwearTaskReplay,
    *,
    split_name: SplitName,
    ticks: int,
    seeds: list[int],
    fixed_start_tick: int | None,
) -> dict[int, int]:
    """Choose matched task windows for fair policy comparisons.

    A seed denotes a workload-window seed, not injected simulator noise. For a
    given seed, every policy receives the exact same release ticks and task
    rows. Window starts are chosen from observed release ticks so an episode is
    not accidentally dominated by a long empty prefix.
    """

    if fixed_start_tick is not None:
        if not replay.start_tick <= fixed_start_tick <= replay.max_tick:
            raise ValueError(
                f"--start-tick={fixed_start_tick} is outside replay range "
                f"[{replay.start_tick}, {replay.max_tick}]"
            )
        return {int(seed): int(fixed_start_tick) for seed in seeds}

    split = temporal_splits(replay.frame)[split_name]
    latest_start = max(split.start_tick, split.end_tick - ticks + 1)
    release_ticks = sorted(
        {
            int(value)
            for value in replay.frame["release_tick"].astype(int)
            if split.start_tick <= int(value) <= latest_start
        }
    )
    candidates = release_ticks or [split.start_tick]
    return {
        int(seed): int(random.Random(int(seed)).choice(candidates))
        for seed in seeds
    }


def run_policy(
    *,
    policy_name: str,
    seed: int,
    start_tick: int,
    ticks: int,
    replay: FootwearTaskReplay,
    config: SimulationConfig,
    learner: VDNLearner | None,
    learner_name: str,
    learner_include_forecast: bool,
) -> dict:
    is_vdn = policy_name == learner_name
    policy = ExternalJointPolicy() if is_vdn else policy_from_name(policy_name)
    engine = SimulationEngine(
        replace(config, random_seed=seed, snapshot_interval_ticks=ticks + 1),
        mode="reactive",
        persist=False,
        initial_minute=_initial_minute(replay, start_tick, config.simulation_minutes_per_tick),
        idle_policy=policy,
    )

    wall_start = time.perf_counter()
    for local_tick in range(1, ticks + 1):
        global_tick = start_tick + local_tick - 1
        events = replay.tasks_for_tick(global_tick)
        if is_vdn and (
            local_tick == 1
            or (local_tick - 1) % config.idle_decision_interval_ticks == 0
        ):
            assert learner is not None
            decision_started = time.perf_counter()
            if learner_include_forecast:
                engine.refresh_predictions()
            observations, masks = build_agent_observations(
                engine,
                include_forecast=learner_include_forecast,
            )
            actions = learner.select_actions(observations, masks, explore=False)
            engine.apply_idle_actions(
                dict(
                    zip(
                        sorted(robot.robot_id for robot in engine.robots),
                        actions.tolist(),
                    )
                )
            )
            engine.record_idle_decision_latency(
                (time.perf_counter() - decision_started) * 1000.0
            )
        engine.step(external_tasks=events)

    wall_seconds = time.perf_counter() - wall_start
    metrics = engine.metrics_summary()
    return {
        "policy": policy_name,
        "seed": seed,
        "start_tick": start_tick,
        "ticks": ticks,
        "wall_time_seconds": wall_seconds,
        **metrics,
    }


def _reduction(baseline: float, contender: float) -> float:
    if abs(baseline) < 1e-9:
        return 0.0
    return round(100.0 * (baseline - contender) / baseline, 3)


def main() -> None:
    args = parse_args()
    if not args.tasks.exists():
        raise FileNotFoundError(f"Prepared task replay not found: {args.tasks}")
    if args.ticks <= 0 or args.robot_count <= 0:
        raise ValueError("--ticks and --robot-count must be positive")
    if not args.seeds:
        raise ValueError("--seeds must contain at least one integer")

    replay = FootwearTaskReplay(args.tasks, manifest_path=args.manifest, random_seed=args.seeds[0])
    policies = [item.strip().lower().replace("-", "_") for item in args.policies]
    if len(policies) != len(set(policies)):
        raise ValueError("--policies contains duplicates")

    learner: VDNLearner | None = None
    learner_include_forecast = bool(args.vdn_include_forecast)
    if args.vdn_name in policies:
        if not args.vdn_model.exists():
            raise FileNotFoundError(
                f"VDN policy requested but model does not exist: {args.vdn_model}"
            )
        learner = VDNLearner.load(args.vdn_model)
        if learner.robot_count != args.robot_count:
            raise ValueError(
                f"VDN model expects {learner.robot_count} robots, "
                f"but --robot-count={args.robot_count}"
            )
        learner_include_forecast = bool(
            learner.metadata.get("include_forecast", learner_include_forecast)
        )

    starts = choose_start_ticks(
        replay,
        split_name=args.split,
        ticks=args.ticks,
        seeds=[int(seed) for seed in args.seeds],
        fixed_start_tick=args.start_tick,
    )
    config = replace(SimulationConfig(), robot_count=args.robot_count)
    rows = [
        run_policy(
            policy_name=policy,
            seed=int(seed),
            start_tick=starts[int(seed)],
            ticks=args.ticks,
            replay=replay,
            config=config,
            learner=learner,
            learner_name=args.vdn_name,
            learner_include_forecast=learner_include_forecast,
        )
        for seed in args.seeds
        for policy in policies
    ]
    detail = pd.DataFrame(rows)
    numeric = [column for column in detail.columns if column not in {"policy", "seed"}]
    summary = detail.groupby("policy", as_index=False)[numeric].mean(numeric_only=True).round(4)

    baseline_name = "idle_stay"
    if baseline_name not in set(summary["policy"]):
        baseline_name = str(summary.iloc[0]["policy"])
    baseline = summary.loc[summary["policy"] == baseline_name].iloc[0]
    comparisons: dict[str, dict] = {}
    for _, row in summary.iterrows():
        name = str(row["policy"])
        comparisons[name] = {
            "wait_time_reduction_pct_vs_idle_stay": _reduction(
                float(baseline["average_wait_time"]), float(row["average_wait_time"])
            ),
            "p95_wait_reduction_pct_vs_idle_stay": _reduction(
                float(baseline["p95_wait_time"]), float(row["p95_wait_time"])
            ),
            "distance_reduction_pct_vs_idle_stay": _reduction(
                float(baseline["total_distance"]), float(row["total_distance"])
            ),
            "blocking_pressure_reduction_pct_vs_idle_stay": _reduction(
                float(baseline["blocking_pressure_ticks"]),
                float(row["blocking_pressure_ticks"]),
            ),
        }

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    detail_path = RUNTIME_DIR / "mapfua_experiment_detail.csv"
    summary_path = RUNTIME_DIR / "mapfua_experiment_summary.csv"
    comparison_path = RUNTIME_DIR / "mapfua_experiment_comparisons.json"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    comparison_path.write_text(
        json.dumps(
            {
                "runs_per_policy": len(args.seeds),
                "same_wms_task_window_per_seed": True,
                "window_sampling": (
                    "fixed start tick" if args.start_tick is not None
                    else "seeded release-tick-aligned windows"
                ),
                "split": args.split,
                "start_ticks_by_seed": {str(key): value for key, value in starts.items()},
                "ticks": args.ticks,
                "policies": policies,
                "baseline": baseline_name,
                "vdn_include_forecast": learner_include_forecast if learner else None,
                "comparisons": comparisons,
                "metric_notes": {
                    "blocking_pressure_ticks": (
                        "Cumulative count of unassigned robots lying on active robots' "
                        "static shortest-path lookahead. It is a pressure proxy, not "
                        "measured physical delay."
                    ),
                    "estimated_useful_idle_move_rate_pct": (
                        "Heuristic attribution: CLEAR is useful when triggered by positive "
                        "blocking pressure; PREPOSITION is useful when the same robot receives "
                        "a task in the target zone within the configured window."
                    ),
                    "seed": (
                        "A workload-window seed. Policies are deterministic at evaluation, "
                        "so multiple seeds sample different chronological WMS windows."
                    ),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\nWindow starts")
    print(json.dumps(starts, ensure_ascii=False, indent=2))
    print("\nMean results")
    print(summary.to_string(index=False))
    print("\nComparisons")
    print(json.dumps(comparisons, ensure_ascii=False, indent=2))
    print(f"\nSaved: {detail_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {comparison_path}")


if __name__ == "__main__":
    main()
