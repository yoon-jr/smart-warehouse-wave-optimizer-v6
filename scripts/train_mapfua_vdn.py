from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from statistics import fmean

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from smart_warehouse.config import DATA_DIR, MODEL_DIR, RUNTIME_DIR, SimulationConfig
from smart_warehouse.mapfua_env import LifelongMAPFUAEnv
from smart_warehouse.marl_vdn import VDNConfig, VDNLearner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a parameter-sharing VDN policy for high-level actions of "
            "unassigned warehouse robots. Low-level paths remain collision-aware."
        )
    )
    parser.add_argument("--tasks", type=Path, default=DATA_DIR / "footwear_replay_tasks.csv")
    parser.add_argument("--manifest", type=Path, default=DATA_DIR / "footwear_manifest.json")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--episode-ticks", type=int, default=600)
    parser.add_argument("--robot-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-forecast", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup-transitions", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--validation-interval",
        type=int,
        default=25,
        help="Evaluate on chronological validation windows every N episodes.",
    )
    parser.add_argument(
        "--no-density-curriculum",
        action="store_true",
        help=(
            "Disable the WMS-window workload-density curriculum. By default, "
            "training progresses from low- to medium- to high-load windows."
        ),
    )
    parser.add_argument("--model-out", type=Path, default=MODEL_DIR / "mapfua_vdn.pt")
    parser.add_argument("--metadata-out", type=Path, default=MODEL_DIR / "mapfua_vdn_metrics.json")
    return parser.parse_args()



def curriculum_stage(episode: int, total_episodes: int, enabled: bool) -> str:
    if not enabled:
        return "all"
    progress = (episode - 1) / max(1, total_episodes)
    if progress < 0.25:
        return "low"
    if progress < 0.60:
        return "medium"
    return "high"

def evaluate(
    learner: VDNLearner,
    *,
    tasks: Path,
    manifest: Path,
    config: SimulationConfig,
    episode_ticks: int,
    include_forecast: bool,
    seeds: list[int],
) -> dict[str, float]:
    rewards: list[float] = []
    completed: list[float] = []
    waits: list[float] = []
    distances: list[float] = []
    for seed in seeds:
        env = LifelongMAPFUAEnv(
            tasks,
            manifest,
            config=config,
            split="validation",
            episode_ticks=episode_ticks,
            include_forecast=include_forecast,
            seed=seed,
        )
        observations, masks = env.reset(seed=seed)
        done = False
        episode_reward = 0.0
        info: dict = {}
        while not done:
            actions = learner.select_actions(observations, masks, explore=False)
            observations, masks, reward, done, info = env.step(actions)
            episode_reward += reward
        metrics = info["metrics"]
        rewards.append(episode_reward)
        completed.append(float(metrics["completed_tasks"]))
        waits.append(float(metrics["average_wait_time"]))
        distances.append(float(metrics["total_distance"]))
    return {
        "validation_reward_mean": round(fmean(rewards), 6),
        "validation_completed_mean": round(fmean(completed), 3),
        "validation_wait_mean": round(fmean(waits), 3),
        "validation_distance_mean": round(fmean(distances), 3),
    }


def main() -> None:
    args = parse_args()
    if not args.tasks.exists():
        raise FileNotFoundError(
            f"Prepared task replay not found: {args.tasks}. Run prepare_footwear_data first."
        )
    if args.episodes <= 0 or args.episode_ticks <= 0:
        raise ValueError("--episodes and --episode-ticks must be positive")
    if args.validation_interval <= 0:
        raise ValueError("--validation-interval must be positive")

    sim_config = replace(
        SimulationConfig(),
        robot_count=args.robot_count,
        random_seed=args.seed,
        snapshot_interval_ticks=args.episode_ticks + 1,
    )
    env = LifelongMAPFUAEnv(
        args.tasks,
        args.manifest,
        config=sim_config,
        split="train",
        episode_ticks=args.episode_ticks,
        include_forecast=args.include_forecast,
        seed=args.seed,
    )
    observations, masks = env.reset(seed=args.seed)
    learner_config = VDNConfig(
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        warmup_transitions=args.warmup_transitions,
    )
    learner = VDNLearner(
        observation_dim=observations.shape[-1],
        action_count=env.action_count,
        robot_count=args.robot_count,
        config=learner_config,
        seed=args.seed,
    )

    history: list[dict] = []
    validation_history: list[dict] = []
    best_validation_reward = float("-inf")
    best_validation_episode = 0
    best_state_dict = copy.deepcopy(learner.online.state_dict())
    density_curriculum_enabled = not args.no_density_curriculum
    for episode in range(1, args.episodes + 1):
        stage = curriculum_stage(episode, args.episodes, density_curriculum_enabled)
        observations, masks = env.reset(
            seed=args.seed + episode,
            density_stage=stage,
        )
        done = False
        episode_reward = 0.0
        losses: list[float] = []
        final_info: dict = {}
        while not done:
            actions = learner.select_actions(observations, masks, explore=True)
            next_observations, next_masks, reward, done, info = env.step(actions)
            learner.remember(
                observations,
                masks,
                actions,
                reward,
                next_observations,
                next_masks,
                done,
            )
            update = learner.update()
            if update is not None:
                losses.append(update["loss"])
            observations, masks = next_observations, next_masks
            episode_reward += reward
            final_info = info

        record = {
            "episode": episode,
            "reward": round(episode_reward, 6),
            "loss": round(fmean(losses), 6) if losses else None,
            "epsilon": round(learner.epsilon(), 5),
            "density_stage": stage,
            "window_start_tick": final_info.get("window_start_tick", 0),
            "window_task_count": final_info.get("window_task_count", 0),
            "completed_tasks": final_info.get("metrics", {}).get("completed_tasks", 0),
            "average_wait_time": final_info.get("metrics", {}).get("average_wait_time", 0),
            "total_distance": final_info.get("metrics", {}).get("total_distance", 0),
        }
        history.append(record)
        validation_due = (
            episode % args.validation_interval == 0
            or episode == args.episodes
        )
        if validation_due:
            validation_snapshot = evaluate(
                learner,
                tasks=args.tasks,
                manifest=args.manifest,
                config=sim_config,
                episode_ticks=args.episode_ticks,
                include_forecast=args.include_forecast,
                seeds=[1001, 1002, 1003],
            )
            validation_record = {"episode": episode, **validation_snapshot}
            validation_history.append(validation_record)
            if validation_snapshot["validation_reward_mean"] > best_validation_reward:
                best_validation_reward = float(validation_snapshot["validation_reward_mean"])
                best_validation_episode = episode
                best_state_dict = copy.deepcopy(learner.online.state_dict())
            print({**record, "validation": validation_snapshot})
        elif episode == 1 or episode % 10 == 0:
            print(record)

    # Export the checkpoint selected only by validation reward, not the final
    # training episode. This prevents the test split from influencing model selection.
    learner.online.load_state_dict(best_state_dict)
    learner.target.load_state_dict(best_state_dict)
    validation = evaluate(
        learner,
        tasks=args.tasks,
        manifest=args.manifest,
        config=sim_config,
        episode_ticks=args.episode_ticks,
        include_forecast=args.include_forecast,
        seeds=[1001, 1002, 1003],
    )
    learner.save(
        args.model_out,
        args.metadata_out,
        include_forecast=args.include_forecast,
        train_episodes=args.episodes,
        episode_ticks=args.episode_ticks,
        temporal_split="70/15/15 chronological by task release order",
        density_curriculum=(
            "low(0-25% episodes) -> medium(25-60%) -> high(60-100%)"
            if density_curriculum_enabled
            else "disabled"
        ),
        validation=validation,
        best_validation_episode=best_validation_episode,
        validation_interval=args.validation_interval,
        reward_definition=(
            "2*completed - 0.05*waiting - 0.02*distance - 0.03*idle_distance "
            "- 0.01*blocking_pressure - 0.50*invalid_action, divided by robot_count"
        ),
    )
    history_path = RUNTIME_DIR / "mapfua_training_history.json"
    history_path.write_text(
        json.dumps(
            {
                "learner_config": asdict(learner_config),
                "include_forecast": args.include_forecast,
                "density_curriculum_enabled": density_curriculum_enabled,
                "history": history,
                "validation_history": validation_history,
                "best_validation_episode": best_validation_episode,
                "validation": validation,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\nValidation")
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    print(f"Saved model: {args.model_out}")
    print(f"Saved metadata: {args.metadata_out}")
    print(f"Saved history: {history_path}")


if __name__ == "__main__":
    main()
