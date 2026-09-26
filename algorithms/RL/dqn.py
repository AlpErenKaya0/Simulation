#hedefe giderken iki leader de farklı yöne gitmek isterse donup kalıyor, çözmüyor, yol üstünde birsürü hedefe gidebilecekken
#berbat hedefler seçiyor
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np

from algorithms.current import ConnectivityAwareAlgorithm
from algorithms.base import Position, SimulationContext, UavView
from simulation_core.topology import build_distance_matrix, distance_between, find_connected_components


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class _QNetwork:
    """Small dependency-free MLP used by the DQN policy."""

    def __init__(self, input_size: int, output_size: int, seed: int = 7) -> None:
        rng = np.random.default_rng(seed)
        self.weights_1 = rng.normal(0.0, 0.05, (input_size, 32))
        self.bias_1 = np.zeros(32)
        self.weights_2 = rng.normal(0.0, 0.05, (32, output_size))
        self.bias_2 = np.array([0.0, 1.0], dtype=float)

    def predict(self, state: np.ndarray) -> np.ndarray:
        hidden = np.maximum(0.0, state @ self.weights_1 + self.bias_1)
        return hidden @ self.weights_2 + self.bias_2

    def copy_from(self, other: "_QNetwork") -> None:
        self.weights_1 = other.weights_1.copy()
        self.bias_1 = other.bias_1.copy()
        self.weights_2 = other.weights_2.copy()
        self.bias_2 = other.bias_2.copy()

    def train(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        targets: np.ndarray,
        learning_rate: float,
    ) -> None:
        hidden_pre_activation = states @ self.weights_1 + self.bias_1
        hidden = np.maximum(0.0, hidden_pre_activation)
        predictions = hidden @ self.weights_2 + self.bias_2
        errors = predictions[np.arange(len(actions)), actions] - targets

        output_gradient = np.zeros_like(predictions)
        output_gradient[np.arange(len(actions)), actions] = errors
        gradient_weights_2 = hidden.T @ output_gradient / len(actions)
        gradient_bias_2 = output_gradient.mean(axis=0)

        hidden_gradient = (output_gradient @ self.weights_2.T) * (
            hidden_pre_activation > 0
        )
        gradient_weights_1 = states.T @ hidden_gradient / len(actions)
        gradient_bias_1 = hidden_gradient.mean(axis=0)

        self.weights_2 -= learning_rate * gradient_weights_2
        self.bias_2 -= learning_rate * gradient_bias_2
        self.weights_1 -= learning_rate * gradient_weights_1
        self.bias_1 -= learning_rate * gradient_bias_1


class DeepQNetworkAlgorithm(ConnectivityAwareAlgorithm):
    name = "deep_q_network"
    display_name = "Deep Q-Network"
    description = "DQN ile hedef seçen, bağlantı kopmasını sert kısıt olarak koruyan algoritma."

    # Rewards are deliberately asymmetric: losing the network is never worth
    # completing a nearby goal.
    GOAL_REWARD = 100.0
    CONNECTIVITY_REWARD = 8.0
    PROGRESS_SCALE = 2.0
    STEP_PENALTY = 0.5
    ENERGY_SCALE = 0.02
    DISCONNECTION_PENALTY = -200.0

    def __init__(self) -> None:
        super().__init__()
        self._network = _QNetwork(input_size=10, output_size=2)
        self._target_network = _QNetwork(input_size=10, output_size=2)
        self._target_network.copy_from(self._network)
        self._replay: Deque[Transition] = deque(maxlen=2_000)
        self._previous_context: Optional[SimulationContext] = None
        self._pending_actions: list[np.ndarray] = []
        self._epsilon = 0.25
        self._epsilon_min = 0.05
        self._epsilon_decay = 0.995
        self._gamma = 0.9
        self._learning_rate = 0.01
        self._batch_size = 32
        self._training_steps = 0

    def reset(self) -> None:
        self._network = _QNetwork(input_size=10, output_size=2)
        self._target_network = _QNetwork(input_size=10, output_size=2)
        self._target_network.copy_from(self._network)
        self._replay.clear()
        self._previous_context = None
        self._pending_actions.clear()
        self._epsilon = 0.25
        self._training_steps = 0

    def build_plan(self, context: SimulationContext) -> AlgorithmPlan:
        self._learn_from_previous_step(context)
        plan = super().build_plan(context)
        self._previous_context = context
        return plan

    def assign_uavs_to_goals(self, context, uavs, goals) -> bool:
        free_uavs = [uav for uav in uavs.values() if uav.state == "Free"]
        free_goals = [goal for goal in goals.values() if goal.state == "Free"]
        if not free_uavs or not free_goals:
            self._pending_actions = []
            return False

        candidates: list[tuple[float, int, int, np.ndarray]] = []
        for uav in free_uavs:
            for goal in free_goals:
                state = self._state_vector(context, uav, goal.position, goal.state)
                q_values = self._network.predict(state)
                candidates.append((float(q_values[1] - q_values[0]), uav.id, goal.id, state))

        selected: list[tuple[int, int, np.ndarray]] = []
        assigned_uavs: set[int] = set()
        assigned_goals: set[int] = set()
        while len(assigned_uavs) < len(free_uavs) and len(assigned_goals) < len(free_goals):
            available = [
                candidate
                for candidate in candidates
                if candidate[1] not in assigned_uavs and candidate[2] not in assigned_goals
            ]
            if not available:
                break
            if np.random.random() < self._epsilon:
                score, uav_id, goal_id, state = available[
                    np.random.randint(len(available))
                ]
            else:
                score, uav_id, goal_id, state = max(available, key=lambda item: item[0])
            if score < 0.0 and selected:
                break
            if not self.become_leader(uavs, goals, uav_id, goal_id):
                break
            selected.append((uav_id, goal_id, state))
            assigned_uavs.add(uav_id)
            assigned_goals.add(goal_id)

        self._pending_actions = [state for _, _, state in selected]
        self._epsilon = max(self._epsilon_min, self._epsilon * self._epsilon_decay)
        return bool(selected)

    def allow_movement(
        self,
        context: SimulationContext,
        uav: UavView,
        target_position: Position,
    ) -> bool:
        next_position = self.get_next_position(uav.position, target_position, uav.speed)
        positions = [
            next_position if candidate.id == uav.id else candidate.position
            for candidate in context.uavs
        ]
        if context.ground is not None:
            positions.append(context.ground.position)
        matrix = build_distance_matrix(positions)
        components = find_connected_components(matrix, context.comm_thr)
        return len(components) <= 1

    def _state_vector(
        self,
        context: SimulationContext,
        uav,
        goal_position: Position,
        goal_state: str,
    ) -> np.ndarray:
        scale = max(context.comm_thr, 1.0)
        distance = distance_between(uav.position, goal_position)
        dx = (goal_position.x - uav.position.x) / scale
        dy = (goal_position.y - uav.position.y) / scale
        connectivity = self._connectivity_margin(context, uav)
        speed = float(getattr(uav, "speed", context.config.uav_speed))
        return np.array(
            [
                dx,
                dy,
                distance / scale,
                min(speed / scale, 1.0),
                connectivity,
                1.0 if goal_state == "Free" else 0.0,
                1.0 if uav.state == "Free" else 0.0,
                len(context.uavs) / 100.0,
                len(context.goals) / 100.0,
                context.current_time / max(context.config.simulation_time, 1.0),
            ],
            dtype=float,
        )

    def _connectivity_margin(self, context: SimulationContext, uav: UavView) -> float:
        neighbors = [
            distance_between(uav.position, other.position)
            for other in context.uavs
            if other.id != uav.id
        ]
        if context.ground is not None:
            neighbors.append(distance_between(uav.position, context.ground.position))
        if not neighbors:
            return 1.0
        return float(np.clip(1.0 - min(neighbors) / max(context.comm_thr, 1.0), -1.0, 1.0))

    def _learn_from_previous_step(self, context: SimulationContext) -> None:
        if self._previous_context is None or not self._pending_actions:
            return
        reward = self._reward(self._previous_context, context)
        next_state = self._next_state(context)
        done = not any(goal.state != "Visited" for goal in context.goals)
        for state in self._pending_actions:
            self._replay.append(
                Transition(state, 1, reward, next_state, done)
            )
        self._train_from_replay()
        self._pending_actions = []

    def _next_state(self, context: SimulationContext) -> np.ndarray:
        if context.uavs and context.goals:
            uav = context.uavs[0]
            goal = context.goals[0]
            return self._state_vector(context, uav, goal.position, goal.state)
        return np.zeros(10, dtype=float)

    def _reward(
        self,
        previous: SimulationContext,
        current: SimulationContext,
    ) -> float:
        if not self._is_connected(current):
            return self.DISCONNECTION_PENALTY

        reward = self.CONNECTIVITY_REWARD - self.STEP_PENALTY
        previous_goals = {goal.id: goal for goal in previous.goals}
        current_goals = {goal.id: goal for goal in current.goals}
        for goal_id, goal in current_goals.items():
            if goal.state == "Visited" and previous_goals.get(goal_id, goal).state != "Visited":
                reward += self.GOAL_REWARD

        for previous_uav in previous.uavs:
            current_uav = current.uav_by_id(previous_uav.id)
            if current_uav is None:
                continue
            if previous_uav.target_goal_id is not None:
                goal = previous.goal_by_id(previous_uav.target_goal_id)
                if goal is not None:
                    before = distance_between(previous_uav.position, goal.position)
                    after = distance_between(current_uav.position, goal.position)
                    reward += self.PROGRESS_SCALE * (before - after)
            reward -= self.ENERGY_SCALE * distance_between(
                previous_uav.position, current_uav.position
            )
        return reward

    def _is_connected(self, context: SimulationContext) -> bool:
        positions = [uav.position for uav in context.uavs]
        if context.ground is not None:
            positions.append(context.ground.position)
        matrix = build_distance_matrix(positions)
        return len(find_connected_components(matrix, context.comm_thr)) <= 1

    def _train_from_replay(self) -> None:
        if not self._replay:
            return
        batch_size = min(self._batch_size, len(self._replay))
        indices = np.random.choice(len(self._replay), batch_size, replace=False)
        batch = [self._replay[index] for index in indices]
        states = np.vstack([item.state for item in batch])
        next_states = np.vstack([item.next_state for item in batch])
        next_values = np.max(
            np.vstack([self._target_network.predict(state) for state in next_states]),
            axis=1,
        )
        targets = np.array(
            [
                item.reward if item.done else item.reward + self._gamma * next_value
                for item, next_value in zip(batch, next_values)
            ]
        )
        actions = np.array([item.action for item in batch], dtype=int)
        self._network.train(states, actions, targets, self._learning_rate)
        self._training_steps += 1
        if self._training_steps % 10 == 0:
            self._target_network.copy_from(self._network)


ALGORITHMS = [DeepQNetworkAlgorithm]
