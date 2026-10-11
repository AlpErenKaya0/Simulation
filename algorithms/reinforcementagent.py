from __future__ import annotations

import copy
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from algorithms.q_network import Linear_QNet
from simulation_core.algorithm_api import SimulationContext


# --- Replay memory ---
MAX_MEMORY = 100_000
BATCH_SIZE = 64
MIN_REPLAY_SIZE = 5_000            # eğitime başlamadan önce ~80 batch'lik çeşitlilik
TRAIN_EVERY_TRANSITIONS = 4

# --- Optimizasyon ---
LR = 0.0003
GAMMA = 0.99                       # karar başına; SMDP'de gamma ** k kullanılır

# --- Keşif (epsilon-greedy) ---
EPSILON = 0.0                      # değerlendirme: tamamen greedy
EPSILON_START = 1.0                # eğitim başlangıç epsilon'u
EPSILON_MIN = 0.05                 # eğitimde alt sınır
EPSILON_DECAY_DECISIONS = 50_000   # 1.0 -> 0.05 lineer

# Geriye dönük uyumluluk: model.py güncellenince kaldırılacak.
TRAINING_EPSILON = EPSILON_START

# --- Ağ ---
TARGET_UPDATE_INTERVAL = 1000
HIDDEN_SIZE = 256

# --- Ödül ---
REWARD_DECAY_HALF_LIFE = 2000      # ortalama bölüm uzunluğuna göre ayarla
REWARD_TARGET_VISITED = 1.0
REWARD_SUCCESS = 10.0
REWARD_CONNECTION_LOST = -20.0
REWARD_STEP_PENALTY = -0.001       # her fiziksel adımda; bölüm uzunluğuna göre kalibre et

OBSERVATION_VERSION = 3
TRAINING_CONFIG_VERSION = 4        # ödül (adım cezası) ve Transition (steps) değişti

assert 0.0 <= EPSILON <= 1.0
assert 0.0 <= EPSILON_MIN <= EPSILON_START <= 1.0
assert EPSILON_DECAY_DECISIONS > 0
assert 0.0 < GAMMA <= 1.0
assert MIN_REPLAY_SIZE >= BATCH_SIZE
assert MAX_MEMORY >= MIN_REPLAY_SIZE
assert REWARD_DECAY_HALF_LIFE > 0
assert REWARD_STEP_PENALTY <= 0.0
assert REWARD_CONNECTION_LOST < 0.0 < REWARD_SUCCESS

ROLE_NAMES = (
    "Free",
    "Leader",
    "Slave",
    "Ground_Leader",
    "Relay",
)

GOAL_STATE_NAMES = (
    "Free",
    "Assigned",
    "Visited",
)


@dataclass(frozen=True)
class DroneAction:
    kind: str
    uav_id: Optional[int] = None
    goal_id: Optional[int] = None
    leader_id: Optional[int] = None


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    next_action_mask: np.ndarray
    # Bu kararın kaç fiziksel adım sürdüğü (SMDP indirimi için).
    steps: int = 1


def reward_decay_factor(
    episode_steps: int,
) -> float:
    if episode_steps < 0:
        raise ValueError(
            "episode_steps cannot be negative."
        )

    return float(
        0.5 ** (
            episode_steps / REWARD_DECAY_HALF_LIFE
        )
    )


def calculate_reward(
    newly_visited_count: int,
    connection_lost: bool,
    success: bool,
    goal_reward_factor: float = 1.0,
) -> float:
    """
    Tek bir fiziksel adımın ödülü.

    - Yeni ziyaret edilen hedef: decay faktörüyle çarpılır
    - Her adım: küçük sabit ceza (hızlı bitirmeyi teşvik eder)
    - Bağlantı koptuysa: büyük ceza (başarı bonusunu iptal eder)
    - Başarı: sabit bonus
    """
    if newly_visited_count < 0:
        raise ValueError(
            "newly_visited_count cannot be negative."
        )

    if (
        not np.isfinite(goal_reward_factor)
        or goal_reward_factor < 0
    ):
        raise ValueError(
            "Invalid goal reward factor."
        )

    reward = (
        newly_visited_count
        * REWARD_TARGET_VISITED
        * goal_reward_factor
    ) + REWARD_STEP_PENALTY

    if connection_lost:
        reward += REWARD_CONNECTION_LOST
    elif success:
        reward += REWARD_SUCCESS

    return float(reward)
def motion_between(
    previous_context: SimulationContext,
    context: SimulationContext,
) -> dict[int, tuple[float, float]]:
    """
    İki ardışık durum arasında her UAV'nin yer değiştirmesini döndürür:
    {uav_id: (dx, dy)}. Yalnızca iki durumda da bulunan UAV'ler dahildir.
    """
    previous = {
        uav.id: uav
        for uav in previous_context.uavs
    }

    return {
        uav.id: (
            uav.position.x - previous[uav.id].position.x,
            uav.position.y - previous[uav.id].position.y,
        )
        for uav in context.uavs
        if uav.id in previous
    }


class ReplayBuffer:
    """Sabit kapasiteli halka tampon; rastgele örnekleme O(k)."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive.")

        self.capacity = int(capacity)
        self._data: list = []
        self._next = 0

    def __len__(self) -> int:
        return len(self._data)

    def append(self, item) -> None:
        if len(self._data) < self.capacity:
            self._data.append(item)
        else:
            self._data[self._next] = item

        self._next = (self._next + 1) % self.capacity

    def sample(self, k: int, rng: random.Random) -> list:
        if not 0 < k <= len(self._data):
            raise ValueError("Invalid sample size.")

        return rng.sample(self._data, k)

    def clear(self) -> None:
        self._data.clear()
        self._next = 0


class Agent:
    def __init__(
        self,
        initial_context: SimulationContext,
        canvas_width: float,
        canvas_height: float,
        epsilon: float = EPSILON,
        seed: Optional[int] = None,
    ):
        if (
            not np.isfinite(canvas_width)
            or not np.isfinite(canvas_height)
            or min(canvas_width, canvas_height) <= 0
        ):
            raise ValueError(
                "Canvas dimensions must be finite and positive."
            )

        if not 0 <= epsilon <= 1:
            raise ValueError(
                "epsilon must be between 0 and 1."
            )

        self.uav_ids = tuple(
            sorted(u.id for u in initial_context.uavs)
        )
        self.goal_ids = tuple(
            sorted(g.id for g in initial_context.goals)
        )

        if not self.uav_ids or not self.goal_ids:
            raise ValueError(
                "At least one UAV and one goal are required."
            )

        if len(set(self.uav_ids)) != len(self.uav_ids):
            raise ValueError("UAV IDs must be unique.")

        if len(set(self.goal_ids)) != len(self.goal_ids):
            raise ValueError("Goal IDs must be unique.")

        self.uav_count = len(self.uav_ids)
        self.goal_count = len(self.goal_ids)

        self.width = float(canvas_width)
        self.height = float(canvas_height)
        self.distance_scale = float(
            np.hypot(self.width, self.height)
        )

        self.gamma = GAMMA
        self.epsilon = float(epsilon)
        self.exploration_start = float(epsilon)

        self.n_games = 0
        self.decision_steps = 0
        self.transition_steps = 0
        self.training_steps = 0
        self.last_loss = None
        self._last_scheduled_transition = -1

        self.random = random.Random(seed)

        if seed is not None:
            torch.manual_seed(seed)

        self.memory = ReplayBuffer(MAX_MEMORY)
        self.actions = self._build_actions()

        expected_actions = 1 + self.uav_count * (
            self.goal_count + 2 * (self.uav_count - 1)
        )

        if len(self.actions) != expected_actions:
            raise RuntimeError(
                "Action space size mismatch."
            )

        self.input_size = len(
            self.get_state(initial_context)
        )
        self.output_size = len(self.actions)

        self.device = torch.device("cpu")

        self.model = Linear_QNet(
            self.input_size,
            HIDDEN_SIZE,
            self.output_size,
        ).to(self.device)

        self.target_model = copy.deepcopy(self.model)
        self.target_model.eval()

        for parameter in self.target_model.parameters():
            parameter.requires_grad_(False)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=LR,
        )
        self.criterion = nn.SmoothL1Loss()

    def _build_actions(self):
        actions = [
            DroneAction(kind="keep")
        ]

        for uav_id in self.uav_ids:
            for goal_id in self.goal_ids:
                actions.append(
                    DroneAction(
                        "assign_goal",
                        uav_id,
                        goal_id=goal_id,
                    )
                )

            for leader_id in self.uav_ids:
                if leader_id == uav_id:
                    continue

                for kind in (
                    "assign_slave",
                    "assign_relay",
                ):
                    actions.append(
                        DroneAction(
                            kind,
                            uav_id,
                            leader_id=leader_id,
                        )
                    )

        return tuple(actions)
    def _check_context(self, context):
        if tuple(sorted(u.id for u in context.uavs)) != self.uav_ids:
            raise ValueError(
                "UAV configuration changed."
            )

        if tuple(sorted(g.id for g in context.goals)) != self.goal_ids:
            raise ValueError(
                "Goal configuration changed."
            )

        if (
            not np.isfinite(context.comm_thr)
            or context.comm_thr <= 0
        ):
            raise ValueError(
                "Communication threshold must be positive."
            )

    def _ordered_entities(self, context):
        self._check_context(context)

        uav_by_id = {
            u.id: u for u in context.uavs
        }
        goal_by_id = {
            g.id: g for g in context.goals
        }

        return (
            [uav_by_id[i] for i in self.uav_ids],
            [goal_by_id[i] for i in self.goal_ids],
        )

    @staticmethod
    def _pairwise_distances(a, b):
        return np.linalg.norm(
            a[:, None, :] - b[None, :, :],
            axis=2,
        )

    def get_uav_distance_matrix(self, context):
        uavs, _ = self._ordered_entities(context)

        positions = np.asarray(
            [[u.position.x, u.position.y] for u in uavs],
            dtype=np.float32,
        )

        return self._pairwise_distances(
            positions,
            positions,
        )

    def get_uav_goal_distance_matrix(self, context):
        uavs, goals = self._ordered_entities(context)

        positions = np.asarray(
            [[u.position.x, u.position.y] for u in uavs],
            dtype=np.float32,
        )
        targets = np.asarray(
            [[g.position.x, g.position.y] for g in goals],
            dtype=np.float32,
        )

        return self._pairwise_distances(
            positions,
            targets,
        )

    def get_state(
        self,
        context: SimulationContext,
        episode_steps: int = 0,
        motion: Optional[dict] = None,   # API uyumu için duruyor, kullanılmıyor
    ) -> np.ndarray:
        uavs, goals = self._ordered_entities(context)

        if episode_steps < 0:
            raise ValueError(
                "episode_steps cannot be negative."
            )

        threshold = float(context.comm_thr)
        factor = reward_decay_factor(episode_steps)
        n = self.uav_count

        positions = np.asarray(
            [[u.position.x, u.position.y] for u in uavs],
            dtype=np.float32,
        )
        targets = np.asarray(
            [[g.position.x, g.position.y] for g in goals],
            dtype=np.float32,
        )

        distances = self._pairwise_distances(positions, positions)
        goal_distances = self._pairwise_distances(positions, targets)

        def scaled(values):
            # 1.0 = bağlantı sınırı; çok uzağı 3.0'da kırp.
            return np.clip(values / threshold, 0.0, 3.0)

        # --- Genel ---
        features = [
            sum(g.state == "Visited" for g in goals)
            / self.goal_count,
            threshold / self.distance_scale,
            float(context.ground is not None),
            factor,
        ]

        # --- Ground konumu ---
        if context.ground is not None:
            features.extend([
                context.ground.position.x / self.width,
                context.ground.position.y / self.height,
            ])
        else:
            features.extend([0.0, 0.0])

        # --- Drone başına: konum + roller ---
        for uav in uavs:
            features.extend([
                uav.position.x / self.width,
                uav.position.y / self.height,
            ])
            features.extend(
                float(uav.state == role)
                for role in ROLE_NAMES
            )
            features.extend(
                float(uav.target_goal_id == goal_id)
                for goal_id in self.goal_ids
            )
            features.extend(
                float(uav.leader_id == leader_id)
                for leader_id in self.uav_ids
            )

        # --- Hedef başına: konum + durum ---
        for goal in goals:
            features.extend([
                goal.position.x / self.width,
                goal.position.y / self.height,
            ])
            features.extend(
                float(goal.state == state)
                for state in GOAL_STATE_NAMES
            )

        # --- Drone-drone mesafeleri (üst üçgen, comm_thr birimiyle) ---
        upper = np.triu_indices(n, k=1)
        features.extend(scaled(distances)[upper].tolist())

        # --- Drone-hedef mesafeleri (tam matris) ---
        features.extend(scaled(goal_distances).ravel().tolist())

        # --- Drone-ground mesafeleri ---
        if context.ground is not None:
            ground_position = np.asarray(
                [[
                    context.ground.position.x,
                    context.ground.position.y,
                ]],
                dtype=np.float32,
            )
            ground_distances = self._pairwise_distances(
                positions,
                ground_position,
            )[:, 0]
            features.extend(scaled(ground_distances).tolist())
        else:
            features.extend([0.0] * n)

        state = np.asarray(features, dtype=np.float32)

        if not np.isfinite(state).all():
            raise ValueError(
                "Observation contains non-finite values."
            )

        return state
    def _validate_state(self, state):
        state = np.asarray(
            state,
            dtype=np.float32,
        )

        if state.shape != (self.input_size,):
            raise ValueError(
                "Observation has the wrong shape."
            )

        if not np.isfinite(state).all():
            raise ValueError(
                "Observation contains non-finite values."
            )

        return state

    def _validate_mask(
        self,
        action_mask,
        require_valid_action,
    ):
        mask = np.asarray(
            action_mask,
            dtype=bool,
        )

        if mask.shape != (self.output_size,):
            raise ValueError(
                "Action mask has the wrong shape."
            )

        if require_valid_action and not mask.any():
            raise ValueError(
                "No valid action is available."
            )

        return mask

    def _update_epsilon(self):
        minimum = min(
            EPSILON_MIN,
            self.exploration_start,
        )
        progress = min(
            1.0,
            self.decision_steps / EPSILON_DECAY_DECISIONS,
        )

        self.epsilon = float(
            self.exploration_start
            + progress * (
                minimum - self.exploration_start
            )
        )

    def get_action(
        self,
        state,
        action_mask,
        training=True,
    ):
        state = self._validate_state(state)
        mask = self._validate_mask(
            action_mask,
            require_valid_action=True,
        )

        if training:
            self._update_epsilon()
            self.decision_steps += 1

            if self.random.random() < self.epsilon:
                return int(
                    self.random.choice(
                        np.flatnonzero(mask).tolist()
                    )
                )

        was_training = self.model.training
        self.model.eval()

        try:
            with torch.no_grad():
                values = self.model(
                    torch.as_tensor(
                        state,
                        dtype=torch.float32,
                        device=self.device,
                    )
                )

                if not torch.isfinite(values).all().item():
                    raise RuntimeError(
                        "Non-finite Q-values."
                    )

                valid = torch.as_tensor(
                    mask,
                    dtype=torch.bool,
                    device=self.device,
                )

                return int(
                    values.masked_fill(
                        ~valid,
                        float("-inf"),
                    ).argmax().item()
                )
        finally:
            self.model.train(was_training)

    def decode_action(self, action_index):
        if not 0 <= action_index < self.output_size:
            raise ValueError("Invalid action index.")

        return self.actions[action_index]

    def remember_q(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        next_action_mask,
        steps=1,
    ):
        """
        Bir karar geçişini kaydeder.

        reward: karar başladığından bu yana biriken toplam ödül
        steps:  kararın sürdüğü fiziksel adım sayısı (SMDP indirimi için)
        """
        if not 0 <= action < self.output_size:
            raise ValueError("Invalid action index.")

        if not np.isfinite(reward):
            raise ValueError("Reward must be finite.")

        if int(steps) < 1:
            raise ValueError("steps must be at least 1.")

        mask = self._validate_mask(
            next_action_mask,
            require_valid_action=not done,
        )

        self.memory.append(
            Transition(
                state=self._validate_state(state).copy(),
                action=int(action),
                reward=float(reward),
                next_state=self._validate_state(
                    next_state
                ).copy(),
                done=bool(done),
                next_action_mask=mask.copy(),
                steps=int(steps),
            )
        )
        self.transition_steps += 1

    def _train_batch(self, transitions):
        self.model.train()

        def tensor(values, dtype):
            return torch.as_tensor(
                values,
                dtype=dtype,
                device=self.device,
            )

        states = tensor(
            np.stack([t.state for t in transitions]),
            torch.float32,
        )
        next_states = tensor(
            np.stack([t.next_state for t in transitions]),
            torch.float32,
        )
        actions = tensor(
            [t.action for t in transitions],
            torch.long,
        )
        rewards = tensor(
            [t.reward for t in transitions],
            torch.float32,
        )
        dones = tensor(
            [t.done for t in transitions],
            torch.bool,
        )
        masks = tensor(
            np.stack([
                t.next_action_mask
                for t in transitions
            ]),
            torch.bool,
        )
        steps = tensor(
            [t.steps for t in transitions],
            torch.float32,
        )

        predicted = self.model(states).gather(
            1,
            actions[:, None],
        ).squeeze(1)

        with torch.no_grad():
            next_values = torch.zeros_like(rewards)
            active = ~dones

            if active.any().item():
                valid = masks[active]

                if not valid.any(dim=1).all().item():
                    raise ValueError(
                        "Nonterminal state has no valid action."
                    )

                # Double DQN: online ağ seçer, target ağ değerlendirir.
                self.model.eval()

                try:
                    selected = self.model(
                        next_states[active]
                    ).masked_fill(
                        ~valid,
                        float("-inf"),
                    ).argmax(dim=1)
                finally:
                    self.model.train()

                next_values[active] = self.target_model(
                    next_states[active]
                ).gather(
                    1,
                    selected[:, None],
                ).squeeze(1)

            # SMDP: karar k fiziksel adım sürdüyse indirim gamma ** k.
            discounts = self.gamma ** steps
            expected = rewards + discounts * next_values

        loss = self.criterion(
            predicted,
            expected,
        )

        if not torch.isfinite(loss).item():
            raise RuntimeError(
                "DQN loss became non-finite."
            )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=10.0,
        )
        self.optimizer.step()

        self.training_steps += 1

        if self.training_steps % TARGET_UPDATE_INTERVAL == 0:
            self.target_model.load_state_dict(
                self.model.state_dict()
            )

        self.last_loss = float(loss.item())
        return self.last_loss

    def train_long_memory(self):
        """Replay'den bir mini-batch örnekleyip tek güncelleme yapar."""
        if len(self.memory) < max(MIN_REPLAY_SIZE, BATCH_SIZE):
            return None

        transitions = self.memory.sample(
            BATCH_SIZE,
            self.random,
        )
        return self._train_batch(transitions)

    def maybe_train(self):
        """
        remember_q() sonrasında çağrılır. Replay yeterince doluysa ve
        TRAIN_EVERY_TRANSITIONS takvimi geldiyse bir güncelleme yapar.
        """
        if len(self.memory) < MIN_REPLAY_SIZE:
            return None

        if self.transition_steps % TRAIN_EVERY_TRANSITIONS:
            return None

        if self._last_scheduled_transition == self.transition_steps:
            return None

        self._last_scheduled_transition = self.transition_steps

        return self.train_long_memory()
    def finish_episode(self):
        self.n_games += 1

    def save(self, file_name="drone_dqn.pth"):
        directory = Path("model")
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._update_epsilon()

        target = directory / file_name
        temporary = target.with_suffix(target.suffix + ".tmp")

        torch.save(
            {
                "observation_version": OBSERVATION_VERSION,
                "training_config_version": TRAINING_CONFIG_VERSION,
                "model_state_dict": self.model.state_dict(),
                "target_model_state_dict": (
                    self.target_model.state_dict()
                ),
                "optimizer_state_dict": (
                    self.optimizer.state_dict()
                ),
                "uav_ids": self.uav_ids,
                "goal_ids": self.goal_ids,
                "input_size": self.input_size,
                "output_size": self.output_size,
                "canvas_width": self.width,
                "canvas_height": self.height,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "exploration_start": self.exploration_start,
                "decision_steps": self.decision_steps,
                "transition_steps": self.transition_steps,
                "training_steps": self.training_steps,
                "n_games": self.n_games,
                "random_state": self.random.getstate(),
            },
            temporary,
        )

        # Atomik değiştirme: yazma yarıda kesilse bile eski dosya bozulmaz.
        temporary.replace(target)

    def load(
        self,
        file_name="drone_dqn_best.pth",
        *,
        strict_ids=True,
    ):
        path = Path(file_name)

        if not path.is_absolute() and not path.is_file():
            path = Path("model") / path

        if not path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {path}"
            )

        try:
            data = torch.load(
                path,
                map_location=self.device,
                weights_only=True,
            )
        except Exception as exc:
            raise ValueError(
                f"Unreadable checkpoint: {path.name}"
            ) from exc

        if not isinstance(data, dict):
            raise ValueError(
                "Incompatible checkpoint: not a dictionary"
            )

        required = (
            "model_state_dict",
            "canvas_width",
            "canvas_height",
            "gamma",
            "exploration_start",
        )
        missing = [key for key in required if key not in data]

        if missing:
            raise ValueError(
                f"Incompatible checkpoint: missing {missing}"
            )

        expected = {
            "observation_version": OBSERVATION_VERSION,
            "training_config_version": TRAINING_CONFIG_VERSION,
            "input_size": self.input_size,
            "output_size": self.output_size,
        }

        for key, value in expected.items():
            if data.get(key) != value:
                raise ValueError(
                    f"Incompatible checkpoint: {key}"
                )

        for key, current in (
            ("uav_ids", self.uav_ids),
            ("goal_ids", self.goal_ids),
        ):
            saved = tuple(data.get(key, ()))

            matches = (
                saved == current
                if strict_ids
                else len(saved) == len(current)
            )

            if not matches:
                raise ValueError(
                    f"Incompatible checkpoint: {key}"
                )

        try:
            width = float(data["canvas_width"])
            height = float(data["canvas_height"])
            gamma = float(data["gamma"])
            start = float(data["exploration_start"])
            counters = {
                key: int(data.get(key, 0))
                for key in (
                    "decision_steps",
                    "transition_steps",
                    "training_steps",
                    "n_games",
                )
            }
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Invalid checkpoint values."
            ) from exc

        if (
            not np.isfinite(width)
            or not np.isfinite(height)
            or min(width, height) <= 0
        ):
            raise ValueError(
                "Invalid checkpoint dimensions."
            )

        if not np.isfinite(start) or not 0 <= start <= 1:
            raise ValueError(
                "Invalid checkpoint training parameters."
            )

        # gamma sabitle uyuşmuyorsa model farklı bir yapılandırmayla
        # eğitilmiştir; sessizce ezmek yerine reddedilir.
        if not np.isclose(gamma, self.gamma):
            raise ValueError(
                "Incompatible checkpoint: gamma"
            )

        if any(value < 0 for value in counters.values()):
            raise ValueError(
                "Invalid checkpoint counters."
            )

        self.model.load_state_dict(
            data["model_state_dict"]
        )
        self.target_model.load_state_dict(
            data.get(
                "target_model_state_dict",
                data["model_state_dict"],
            )
        )

        if "optimizer_state_dict" in data:
            self.optimizer.load_state_dict(
                data["optimizer_state_dict"]
            )

        self.width = width
        self.height = height
        self.distance_scale = float(
            np.hypot(width, height)
        )

        self.exploration_start = start

        for key, value in counters.items():
            setattr(self, key, value)

        self._update_epsilon()

        if "random_state" in data:
            try:
                self.random.setstate(data["random_state"])
            except (TypeError, ValueError):
                pass

        self.target_model.eval()
        self.memory.clear()
        self.last_loss = None
        self._last_scheduled_transition = -1


def train(
    environment,
    max_games: int = 10_000,
    epsilon: float = EPSILON_START,
    seed: Optional[int] = None,
    best_window: int = 50,
) -> Agent:
    """
    environment.reset() -> SimulationContext
    environment.action_mask(actions) -> bool dizisi
    environment.step(action) -> context, reward, done, info

    Her step fiziksel hareket üretmiyorsa info içinde:
        episode_steps: fiziksel adım sayacı
        physical_step: bu çağrıda fiziksel hareket yapıldı mı?
    Zaman sınırıyla biten bölümde info["truncated"] = True olmalı.

    Ödül hesabı ve zaman sınırı ortamın sorumluluğundadır.
    """
    if max_games <= 0:
        raise ValueError("max_games must be positive.")

    context = environment.reset()

    agent = Agent(
        context,
        environment.canvas_width,
        environment.canvas_height,
        epsilon=epsilon,
        seed=seed,
    )

    # model.py ile aynı checkpoint adları.
    scoped_prefix = (
        f"drone_dqn_{agent.uav_count}uav_"
        f"{agent.goal_count}goals"
    )
    best_name = f"{scoped_prefix}_best.pth"
    last_name = f"{scoped_prefix}_last.pth"

    steps = 0
    motion = {}
    state = agent.get_state(context, steps, motion)
    mask = environment.action_mask(agent.actions)

    episode_return = 0.0

    # En iyi model: tek bölüm yerine hareketli ortalama.
    window = max(1, min(best_window, max_games))
    recent_returns = deque(maxlen=window)
    best_average = float("-inf")

    try:
        while agent.n_games < max_games:
            index = agent.get_action(
                state,
                mask,
                training=True,
            )

            next_context, reward, done, info = environment.step(
                agent.decode_action(index)
            )
            info = info or {}

            physical_step = bool(
                info.get("physical_step", True)
            )
            next_steps = int(
                info.get(
                    "episode_steps",
                    steps + int(physical_step),
                )
            )

            if next_steps < steps:
                raise ValueError(
                    "Episode steps decreased before reset."
                )

            # Zaman sınırıyla biten bölüm gerçek terminal durum değildir:
            # bootstrap devam eder, ama bölüm yine de sıfırlanır.
            truncated = done and bool(
                info.get("truncated", False)
            )
            terminal = done and not truncated

            # Bu kararın sürdüğü fiziksel adım sayısı (SMDP indirimi).
            elapsed = max(1, next_steps - steps)

            if physical_step:
                motion = motion_between(
                    context,
                    next_context,
                )

            next_state = agent.get_state(
                next_context,
                next_steps,
                motion,
            )

            next_mask = (
                np.zeros(agent.output_size, dtype=bool)
                if terminal
                else environment.action_mask(agent.actions)
            )

            agent.remember_q(
                state,
                index,
                reward,
                next_state,
                terminal,
                next_mask,
                steps=elapsed,
            )
            agent.maybe_train()

            episode_return += float(reward)

            if done:
                agent.finish_episode()
                recent_returns.append(episode_return)

                average = (
                    sum(recent_returns) / len(recent_returns)
                )

                if (
                    len(recent_returns) == recent_returns.maxlen
                    and average > best_average
                ):
                    best_average = average
                    agent.save(best_name)

                print(
                    f"Episode={agent.n_games} | "
                    f"Reward={episode_return:.3f} | "
                    f"Avg{len(recent_returns)}={average:.3f} | "
                    f"BestAvg={best_average:.3f} | "
                    f"Steps={next_steps} | "
                    f"End={'truncated' if truncated else 'terminal'} | "
                    f"Epsilon={agent.epsilon:.4f} | "
                    f"Updates={agent.training_steps}"
                )

                if agent.n_games >= max_games:
                    break

                context = environment.reset()
                steps = 0
                motion = {}
                episode_return = 0.0

                state = agent.get_state(
                    context,
                    steps,
                    motion,
                )
                mask = environment.action_mask(
                    agent.actions
                )

            else:
                context = next_context
                steps = next_steps
                state = next_state
                mask = next_mask
    finally:
        agent.save(last_name)

    return agent