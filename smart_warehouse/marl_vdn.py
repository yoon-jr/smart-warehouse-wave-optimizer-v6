from __future__ import annotations

import json
import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - depends on optional dependency
    torch = None
    nn = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


@dataclass(frozen=True)
class VDNConfig:
    hidden_dim: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.99
    batch_size: int = 64
    replay_capacity: int = 50_000
    warmup_transitions: int = 512
    target_update_interval: int = 250
    gradient_clip_norm: float = 10.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 25_000


@dataclass
class JointTransition:
    observations: np.ndarray
    action_masks: np.ndarray
    actions: np.ndarray
    reward: float
    next_observations: np.ndarray
    next_action_masks: np.ndarray
    done: bool


class JointReplayBuffer:
    def __init__(self, capacity: int, seed: int = 42) -> None:
        self.data: deque[JointTransition] = deque(maxlen=int(capacity))
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.data)

    def append(self, transition: JointTransition) -> None:
        self.data.append(transition)

    def sample(self, batch_size: int) -> list[JointTransition]:
        if batch_size > len(self.data):
            raise ValueError("batch_size exceeds replay buffer size")
        return self.rng.sample(list(self.data), batch_size)


if nn is not None:

    class SharedQNetwork(nn.Module):
        """One parameter-shared local utility network for homogeneous robots."""

        def __init__(self, observation_dim: int, action_count: int, hidden_dim: int) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(observation_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_count),
            )

        def forward(self, observations: torch.Tensor) -> torch.Tensor:
            return self.network(observations)


class VDNLearner:
    """Parameter-sharing VDN learner for fixed-size homogeneous robot teams.

    During training, individual selected Q-values are summed into Q_tot and
    optimized against the shared team return. At execution, each robot only
    evaluates its own observation and action mask.
    """

    def __init__(
        self,
        observation_dim: int,
        action_count: int,
        robot_count: int,
        *,
        config: VDNConfig | None = None,
        seed: int = 42,
        device: str | None = None,
    ) -> None:
        if torch is None or nn is None:
            raise RuntimeError(
                "PyTorch is required for VDN training. Install requirements-marl.txt."
            ) from _TORCH_IMPORT_ERROR
        self.config = config or VDNConfig()
        self.observation_dim = int(observation_dim)
        self.action_count = int(action_count)
        self.robot_count = int(robot_count)
        self.seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.online = SharedQNetwork(
            self.observation_dim,
            self.action_count,
            self.config.hidden_dim,
        ).to(self.device)
        self.target = SharedQNetwork(
            self.observation_dim,
            self.action_count,
            self.config.hidden_dim,
        ).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.config.learning_rate)
        self.buffer = JointReplayBuffer(self.config.replay_capacity, seed=seed)
        self.train_steps = 0
        self.environment_steps = 0
        self.metadata: dict[str, Any] = {}

    def epsilon(self) -> float:
        fraction = min(1.0, self.environment_steps / max(1, self.config.epsilon_decay_steps))
        return self.config.epsilon_start + fraction * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    def select_actions(
        self,
        observations: np.ndarray,
        action_masks: np.ndarray,
        *,
        explore: bool = True,
    ) -> np.ndarray:
        observations = np.asarray(observations, dtype=np.float32)
        action_masks = np.asarray(action_masks, dtype=np.float32)
        if observations.shape != (self.robot_count, self.observation_dim):
            raise ValueError(
                f"Expected observations {(self.robot_count, self.observation_dim)}, "
                f"got {observations.shape}"
            )
        if action_masks.shape != (self.robot_count, self.action_count):
            raise ValueError(
                f"Expected masks {(self.robot_count, self.action_count)}, got {action_masks.shape}"
            )
        with torch.no_grad():
            obs_tensor = torch.as_tensor(observations, device=self.device)
            q_values = self.online(obs_tensor).cpu().numpy()
        actions: list[int] = []
        epsilon = self.epsilon() if explore else 0.0
        for agent_index in range(self.robot_count):
            valid = np.flatnonzero(action_masks[agent_index] > 0.5)
            if len(valid) == 0:
                valid = np.asarray([0], dtype=int)
            if explore and random.random() < epsilon:
                action = int(random.choice(valid.tolist()))
            else:
                masked = np.full(self.action_count, -np.inf, dtype=np.float32)
                masked[valid] = q_values[agent_index, valid]
                action = int(np.argmax(masked))
            actions.append(action)
        if explore:
            self.environment_steps += 1
        return np.asarray(actions, dtype=np.int64)

    def remember(
        self,
        observations: np.ndarray,
        action_masks: np.ndarray,
        actions: np.ndarray,
        reward: float,
        next_observations: np.ndarray,
        next_action_masks: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append(
            JointTransition(
                observations=np.asarray(observations, dtype=np.float32).copy(),
                action_masks=np.asarray(action_masks, dtype=np.float32).copy(),
                actions=np.asarray(actions, dtype=np.int64).copy(),
                reward=float(reward),
                next_observations=np.asarray(next_observations, dtype=np.float32).copy(),
                next_action_masks=np.asarray(next_action_masks, dtype=np.float32).copy(),
                done=bool(done),
            )
        )

    def update(self) -> dict[str, float] | None:
        if len(self.buffer) < max(self.config.batch_size, self.config.warmup_transitions):
            return None
        batch = self.buffer.sample(self.config.batch_size)
        observations = torch.as_tensor(
            np.stack([item.observations for item in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.as_tensor(
            np.stack([item.actions for item in batch]),
            dtype=torch.int64,
            device=self.device,
        )
        rewards = torch.as_tensor(
            [item.reward for item in batch],
            dtype=torch.float32,
            device=self.device,
        )
        next_observations = torch.as_tensor(
            np.stack([item.next_observations for item in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        next_masks = torch.as_tensor(
            np.stack([item.next_action_masks for item in batch]),
            dtype=torch.bool,
            device=self.device,
        )
        dones = torch.as_tensor(
            [item.done for item in batch],
            dtype=torch.float32,
            device=self.device,
        )

        batch_size, robot_count, observation_dim = observations.shape
        flat_obs = observations.reshape(batch_size * robot_count, observation_dim)
        q_all = self.online(flat_obs).reshape(batch_size, robot_count, self.action_count)
        chosen = q_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        q_total = chosen.sum(dim=1)

        with torch.no_grad():
            flat_next = next_observations.reshape(batch_size * robot_count, observation_dim)
            next_q_online = self.online(flat_next).reshape(
                batch_size, robot_count, self.action_count
            )
            next_q_target = self.target(flat_next).reshape(
                batch_size, robot_count, self.action_count
            )
            masked_online = next_q_online.masked_fill(~next_masks, -1e9)
            next_actions = masked_online.argmax(dim=-1, keepdim=True)
            next_selected = next_q_target.gather(-1, next_actions).squeeze(-1)
            next_total = next_selected.sum(dim=1)
            target_total = rewards + self.config.gamma * (1.0 - dones) * next_total

        loss = torch.nn.functional.smooth_l1_loss(q_total, target_total)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.online.parameters(), self.config.gradient_clip_norm
        )
        self.optimizer.step()
        self.train_steps += 1
        if self.train_steps % self.config.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())

        return {
            "loss": float(loss.detach().cpu()),
            "q_total_mean": float(q_total.detach().mean().cpu()),
            "target_total_mean": float(target_total.detach().mean().cpu()),
            "gradient_norm": float(grad_norm.detach().cpu()),
            "epsilon": float(self.epsilon()),
            "buffer_size": float(len(self.buffer)),
        }

    def save(self, model_path: Path, metadata_path: Path, **metadata: Any) -> None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata = dict(metadata)
        payload = {
            "state_dict": self.online.state_dict(),
            "observation_dim": self.observation_dim,
            "action_count": self.action_count,
            "robot_count": self.robot_count,
            "vdn_config": asdict(self.config),
            "seed": self.seed,
            "train_steps": self.train_steps,
            "environment_steps": self.environment_steps,
            "metadata": self.metadata,
        }
        torch.save(payload, model_path)
        metadata_payload = {
            "algorithm": "parameter_sharing_vdn",
            "model_path": str(model_path),
            "observation_dim": self.observation_dim,
            "action_count": self.action_count,
            "robot_count": self.robot_count,
            "vdn_config": asdict(self.config),
            "seed": self.seed,
            "train_steps": self.train_steps,
            "environment_steps": self.environment_steps,
            **metadata,
        }
        metadata_path.write_text(
            json.dumps(metadata_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, model_path: Path, *, device: str | None = None) -> "VDNLearner":
        if torch is None:
            raise RuntimeError("PyTorch is required to load a VDN model") from _TORCH_IMPORT_ERROR
        payload = torch.load(model_path, map_location=device or "cpu", weights_only=False)
        config = VDNConfig(**payload["vdn_config"])
        learner = cls(
            int(payload["observation_dim"]),
            int(payload["action_count"]),
            int(payload["robot_count"]),
            config=config,
            seed=int(payload.get("seed", 42)),
            device=device,
        )
        learner.online.load_state_dict(payload["state_dict"])
        learner.target.load_state_dict(payload["state_dict"])
        learner.train_steps = int(payload.get("train_steps", 0))
        learner.environment_steps = int(payload.get("environment_steps", 0))
        learner.metadata = dict(payload.get("metadata", {}))
        return learner
