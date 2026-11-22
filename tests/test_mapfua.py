from dataclasses import replace

import numpy as np
import pytest

from smart_warehouse.config import SimulationConfig
from smart_warehouse.idle_policy import DensityClearPolicy, IdleAction
from smart_warehouse.mapfua_env import LifelongMAPFUAEnv
from smart_warehouse.marl_vdn import VDNConfig, VDNLearner
from smart_warehouse.simulation import SimulationEngine


def test_external_actions_are_masked_for_busy_robot() -> None:
    engine = SimulationEngine(
        replace(SimulationConfig(), robot_count=4),
        persist=False,
    )
    robot = engine.robots[0]
    mask = engine.action_mask(robot)
    assert mask[IdleAction.STAY] == 1
    assert mask.sum() > 1

    robot.current_task_id = "T"
    mask = engine.action_mask(robot)
    assert mask[IdleAction.STAY] == 1
    assert mask.sum() == 1


def test_density_policy_does_not_move_low_risk_idle_robot() -> None:
    engine = SimulationEngine(
        replace(SimulationConfig(), robot_count=4),
        persist=False,
        idle_policy=DensityClearPolicy(),
    )
    decisions = engine.apply_idle_policy()
    assert decisions
    assert all(decision.action == IdleAction.STAY for decision in decisions)


def test_mapfua_environment_replays_prepared_csv(prepared_footwear_paths) -> None:
    config = replace(
        SimulationConfig(),
        robot_count=4,
        idle_decision_interval_ticks=2,
        movement_interval_ticks=1,
        simulation_minutes_per_tick=1.0,
    )
    env = LifelongMAPFUAEnv(
        prepared_footwear_paths["tasks"],
        prepared_footwear_paths["manifest"],
        config=config,
        episode_ticks=12,
        seed=5,
    )
    observations, masks = env.reset(seed=5)
    assert observations.shape[0] == 4
    assert masks.shape == (4, len(IdleAction))
    done = False
    info = {}
    while not done:
        actions = np.zeros(4, dtype=np.int64)
        observations, masks, reward, done, info = env.step(actions)
        assert np.isfinite(reward)
    assert info["metrics"]["generated_tasks"] >= 0


def test_vdn_smoke_update(prepared_footwear_paths) -> None:
    pytest.importorskip(
        "torch",
        reason="Legacy MARL test requires optional PyTorch dependency.",
    )
    config = replace(
        SimulationConfig(),
        robot_count=4,
        idle_decision_interval_ticks=2,
        movement_interval_ticks=1,
        simulation_minutes_per_tick=1.0,
    )
    env = LifelongMAPFUAEnv(
        prepared_footwear_paths["tasks"],
        prepared_footwear_paths["manifest"],
        config=config,
        episode_ticks=8,
        seed=9,
    )
    observations, masks = env.reset(seed=9)
    learner = VDNLearner(
        observations.shape[-1],
        env.action_count,
        robot_count=4,
        config=VDNConfig(
            hidden_dim=32,
            batch_size=2,
            warmup_transitions=2,
            replay_capacity=20,
            target_update_interval=2,
        ),
        seed=9,
        device="cpu",
    )
    update = None
    done = False
    while not done:
        actions = learner.select_actions(observations, masks, explore=True)
        next_observations, next_masks, reward, done, _ = env.step(actions)
        learner.remember(
            observations,
            masks,
            actions,
            reward,
            next_observations,
            next_masks,
            done,
        )
        update = learner.update() or update
        observations, masks = next_observations, next_masks
    assert update is not None
    assert np.isfinite(update["loss"])


def test_mapfua_metrics_label_usefulness_as_estimate() -> None:
    engine = SimulationEngine(replace(SimulationConfig(), robot_count=4), persist=False)
    metrics = engine.metrics_summary()
    assert "estimated_useful_idle_move_rate_pct" in metrics
    assert "pending_idle_move_events" in metrics
    assert "idle_decision_latency_mean_ms" in metrics


def test_density_curriculum_selects_release_aligned_windows(prepared_footwear_paths) -> None:
    config = replace(
        SimulationConfig(),
        robot_count=4,
        idle_decision_interval_ticks=2,
        movement_interval_ticks=1,
        simulation_minutes_per_tick=1.0,
    )
    env = LifelongMAPFUAEnv(
        prepared_footwear_paths["tasks"],
        prepared_footwear_paths["manifest"],
        config=config,
        episode_ticks=8,
        split="train",
        seed=17,
    )
    env.reset(seed=17, density_stage="low")
    low_count = env.window_task_count
    low_start = env.window_start_tick
    env.reset(seed=17, density_stage="high")
    high_count = env.window_task_count
    high_start = env.window_start_tick
    release_ticks = set(env.replay.frame["release_tick"].astype(int).tolist())
    assert low_start in release_ticks
    assert high_start in release_ticks
    assert low_count <= high_count
