from __future__ import annotations

import warnings
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

from algorithms.base import (
    AlgorithmPlan,
    GoalAssignment,
    Position,
    RelayAssignment,
    SimulationAlgorithm,
    SlaveAssignment,
)
from simulation_core.algorithm_api import SimulationContext
from simulation_core.topology import (
    build_distance_matrix,
    find_connected_components,
)


class DroneDQNAlgorithm(SimulationAlgorithm):
    name = "drone_dqn"
    display_name = "Drone DQN"

    description = (
        "DQN ile hedef ve rol ataması yapar. "
        "Hareketi mevcut simülasyon motoru uygular."
    )

    # En iyi model, bu kadar bölümün hareketli ortalamasına göre seçilir.
    BEST_WINDOW = 50

    # None: zaman sınırı yok. Sayı verilirse bölüm bu adımda kesilir
    # (terminal sayılmaz, bootstrap devam eder).
    max_episode_steps: Optional[int] = None

    def __init__(self) -> None:
        self.agent = None
        self._agent_signature = None
        self.training_enabled = False

        self._pending_decision = None
        self._pending_reward = 0.0
        self._pending_steps = 0

        self._motion = {}
        self._episode_finished = False
        self._terminal_reason = None

        self.episode_steps = 0
        self.episode_reward = 0.0

        self.last_episode_reward = 0.0
        self.best_episode_reward = float("-inf")

        self._recent_returns = deque(maxlen=self.BEST_WINDOW)
        self._best_average = float("-inf")

        self._best_checkpoint_name = "drone_dqn_best.pth"
        self._last_checkpoint_name = "drone_dqn_last.pth"

    def enable_training(self, enabled: bool) -> None:
        enabled = bool(enabled)

        if enabled == self.training_enabled:
            return

        self.training_enabled = enabled
        self.reset()

        if (
            enabled
            and self.agent is not None
            and self.agent.decision_steps == 0
            and self.agent.transition_steps == 0
            and self.agent.training_steps == 0
        ):
            from algorithms.reinforcementagent import (
                EPSILON_START,
            )

            self.agent.exploration_start = EPSILON_START
            self.agent.epsilon = EPSILON_START

    def reset(self) -> None:
        self._pending_decision = None
        self._pending_reward = 0.0
        self._pending_steps = 0
        self._motion = {}

        self._episode_finished = False
        self._terminal_reason = None

        self.episode_steps = 0
        self.episode_reward = 0.0

    def _get_state(self, context: SimulationContext):
        return self.agent.get_state(
            context,
            episode_steps=self.episode_steps,
            motion=self._motion,
        )

    def build_plan(
        self,
        context: SimulationContext,
    ) -> AlgorithmPlan:
        if (
            self._episode_finished
            or not context.uavs
            or not context.goals
        ):
            return AlgorithmPlan.empty()

        if all(g.state == "Visited" for g in context.goals):
            return AlgorithmPlan.empty()

        agent = self._get_agent(context)

        if (
            self.training_enabled
            and self._pending_decision is not None
        ):
            self._complete_pending_decision(
                context,
                done=False,
            )

        state = self._get_state(context)
        mask = self._build_action_mask(context, agent)

        if not mask.any():
            raise RuntimeError(
                "DQN has no valid action in a nonterminal state."
            )

        action_index = agent.get_action(
            state,
            mask,
            training=self.training_enabled,
        )
        action = agent.decode_action(action_index)

        if self.training_enabled:
            self._pending_decision = (
                state.copy(),
                action_index,
            )
            self._pending_reward = 0.0
            self._pending_steps = 0

        if action.kind == "keep":
            return AlgorithmPlan.empty()

        uav = context.uav_by_id(action.uav_id)

        if uav is None:
            raise ValueError("DQN selected an unknown UAV.")

        if action.kind == "assign_goal":
            goal = context.goal_by_id(action.goal_id)

            if goal is None or goal.state != "Free":
                raise ValueError(
                    "DQN selected an unavailable goal."
                )

            return AlgorithmPlan(
                goal_assignments=(
                    GoalAssignment(
                        uav_id=uav.id,
                        goal_id=goal.id,
                        role="Leader",
                    ),
                )
            )

        if action.kind in {"assign_slave", "assign_relay"}:
            leader = context.uav_by_id(action.leader_id)

            if (
                leader is None
                or leader.id == uav.id
                or leader.state not in {
                    "Leader",
                    "Ground_Leader",
                }
                or leader.target_goal_id is None
            ):
                raise ValueError(
                    "DQN selected an unavailable leader."
                )

            if action.kind == "assign_slave":
                return AlgorithmPlan(
                    slave_assignments=(
                        SlaveAssignment(
                            uav_id=uav.id,
                            leader_id=leader.id,
                        ),
                    )
                )

            return AlgorithmPlan(
                relay_assignments=(
                    RelayAssignment(
                        uav_id=uav.id,
                        leader_id=leader.id,
                        target_position=Position(
                            x=(
                                uav.position.x
                                + leader.position.x
                            ) / 2,
                            y=(
                                uav.position.y
                                + leader.position.y
                            ) / 2,
                        ),
                    ),
                )
            )

        raise ValueError(
            f"Unsupported action: {action.kind}"
        )

    def allow_movement(
        self,
        context: SimulationContext,
        uav,
        target_position: Position,
    ) -> bool:
        return True

    def finish_step(
        self,
        previous_context: SimulationContext,
        context: SimulationContext,
    ):
        if self._episode_finished:
            return self._terminal_reason

        if not context.uavs or not context.goals:
            return None

        agent = self._get_agent(context)

        previous_uavs = {
            uav.id: uav
            for uav in previous_context.uavs
        }

        self._motion = {}

        for uav in context.uavs:
            previous = previous_uavs.get(uav.id)

            if previous is not None:
                self._motion[uav.id] = (
                    uav.position.x - previous.position.x,
                    uav.position.y - previous.position.y,
                )

        disconnected = self._has_disconnected_network(
            context
        )
        success = all(
            goal.state == "Visited"
            for goal in context.goals
        )

        visited_before = {
            goal.id
            for goal in previous_context.goals
            if goal.state == "Visited"
        }
        visited_after = {
            goal.id
            for goal in context.goals
            if goal.state == "Visited"
        }

        from algorithms.reinforcementagent import (
            calculate_reward,
            reward_decay_factor,
        )

        reward = calculate_reward(
            newly_visited_count=len(
                visited_after - visited_before
            ),
            connection_lost=disconnected,
            success=success and not disconnected,
            goal_reward_factor=reward_decay_factor(
                self.episode_steps
            ),
        )

        self.episode_steps += 1
        self.episode_reward += reward

        if (
            self.training_enabled
            and self._pending_decision is not None
        ):
            # Karar içindeki ödül gamma ** i ile indirimli toplanır;
            # bootstrap indirimi (gamma ** k) Agent tarafında uygulanır.
            self._pending_reward += (
                agent.gamma ** self._pending_steps
            ) * reward
            self._pending_steps += 1

        terminal = disconnected or success
        truncated = (
            not terminal
            and self.max_episode_steps is not None
            and self.episode_steps >= self.max_episode_steps
        )

        if not terminal and not truncated:
            return None

        if truncated:
            self._terminal_reason = "time_limit"
        elif disconnected:
            self._terminal_reason = "network_disconnected"
        else:
            self._terminal_reason = "all_goals_visited"

        if self.training_enabled:
            # Zaman aşımı gerçek terminal durum değildir: done=False.
            self._complete_pending_decision(
                context,
                done=terminal,
            )

            agent.finish_episode()
            self._record_episode(agent)

        self.last_episode_reward = self.episode_reward
        self._episode_finished = True

        return self._terminal_reason

    def _record_episode(self, agent) -> None:
        self._recent_returns.append(self.episode_reward)

        average = (
            sum(self._recent_returns)
            / len(self._recent_returns)
        )

        if self.episode_reward > self.best_episode_reward:
            self.best_episode_reward = self.episode_reward

        # En iyi model: tek bölüm yerine hareketli ortalama.
        if (
            len(self._recent_returns)
            == self._recent_returns.maxlen
            and average > self._best_average
        ):
            self._best_average = average
            agent.save(self._best_checkpoint_name)

        agent.save(self._last_checkpoint_name)

        print(
            f"Episode={agent.n_games} | "
            f"Reward={self.episode_reward:.3f} | "
            f"Avg{len(self._recent_returns)}={average:.3f} | "
            f"BestAvg={self._best_average:.3f} | "
            f"Steps={self.episode_steps} | "
            f"End={self._terminal_reason} | "
            f"Epsilon={agent.epsilon:.4f} | "
            f"Updates={agent.training_steps}"
        )

    def _complete_pending_decision(
        self,
        context: SimulationContext,
        done: bool,
    ) -> None:
        if self._pending_decision is None:
            return

        agent = self._get_agent(context)
        state, action_index = self._pending_decision

        if done:
            next_mask = np.zeros(
                agent.output_size,
                dtype=bool,
            )
        else:
            next_mask = self._build_action_mask(
                context,
                agent,
            )

            # Geçerli aksiyon kalmadıysa geçişi terminal say (güvenlik).
            if not next_mask.any():
                done = True

        next_state = self._get_state(context)

        agent.remember_q(
            state,
            action_index,
            self._pending_reward,
            next_state,
            done,
            next_mask,
            steps=max(1, self._pending_steps),
        )
        agent.maybe_train()

        self._pending_decision = None
        self._pending_reward = 0.0
        self._pending_steps = 0

    @staticmethod
    def _has_disconnected_network(
        context: SimulationContext,
    ) -> bool:
        positions = [
            uav.position for uav in context.uavs
        ]

        if context.ground is not None:
            positions.append(context.ground.position)

        if len(positions) <= 1:
            return False

        components = find_connected_components(
            build_distance_matrix(positions),
            context.comm_thr,
            threshold_ratio=1.0,
        )
        return len(components) > 1

    @staticmethod
    def _canvas_size(
        context: SimulationContext,
    ) -> tuple[float, float]:
        # Simülatör gerçek sahne boyutunu sağlıyorsa onu kullan.
        width = getattr(context, "canvas_width", None)
        height = getattr(context, "canvas_height", None)

        if (
            width is not None
            and height is not None
            and np.isfinite(width)
            and np.isfinite(height)
            and min(width, height) > 0
        ):
            return float(width), float(height)

        # Yedek: ilk bağlamdaki en büyük koordinat.
        coordinates = [
            (u.position.x, u.position.y)
            for u in context.uavs
        ]
        coordinates.extend(
            (g.position.x, g.position.y)
            for g in context.goals
        )

        if context.ground is not None:
            coordinates.append(
                (
                    context.ground.position.x,
                    context.ground.position.y,
                )
            )

        return (
            max(1.0, max(abs(x) for x, _ in coordinates)),
            max(1.0, max(abs(y) for _, y in coordinates)),
        )

    def _get_agent(
        self,
        context: SimulationContext,
    ):
        signature = (
            tuple(sorted(u.id for u in context.uavs)),
            tuple(sorted(g.id for g in context.goals)),
        )

        if (
            self.agent is not None
            and signature == self._agent_signature
        ):
            return self.agent

        try:
            from algorithms.reinforcementagent import (
                Agent,
                EPSILON,
                EPSILON_START,
            )
        except ModuleNotFoundError as exc:
            if exc.name == "torch":
                raise RuntimeError(
                    "PyTorch gerekli: "
                    "python -m pip install torch"
                ) from exc
            raise

        width, height = self._canvas_size(context)

        def create_agent():
            return Agent(
                initial_context=context,
                canvas_width=width,
                canvas_height=height,
                epsilon=(
                    EPSILON_START
                    if self.training_enabled
                    else EPSILON
                ),
            )

        scoped_prefix = (
            f"drone_dqn_{len(context.uavs)}uav_"
            f"{len(context.goals)}goals"
        )

        candidates = (
            Path("model") / f"{scoped_prefix}_best.pth",
            Path("model/drone_dqn_best.pth"),
        )

        new_agent = create_agent()
        loaded = False
        errors = []

        for checkpoint in candidates:
            if not checkpoint.is_file():
                continue

            try:
                new_agent.load(
                    checkpoint,
                    strict_ids=False,
                )
                loaded = True
                break
            except (ValueError, RuntimeError, KeyError) as exc:
                errors.append(
                    f"{checkpoint.name}: {exc}"
                )
                new_agent = create_agent()

        if not loaded and errors:
            warnings.warn(
                "Uyumlu checkpoint bulunamadı; "
                "yeni ağ oluşturuldu. "
                + " | ".join(errors),
                RuntimeWarning,
                stacklevel=2,
            )
        elif not loaded and not self.training_enabled:
            warnings.warn(
                "Checkpoint bulunamadı. "
                "Eğitim kapalı: eğitilmemiş ağ kullanılıyor.",
                RuntimeWarning,
                stacklevel=2,
            )

        if (
            self._agent_signature is not None
            and signature != self._agent_signature
        ):
            self.reset()
            self.last_episode_reward = 0.0

        self._best_checkpoint_name = (
            f"{scoped_prefix}_best.pth"
        )
        self._last_checkpoint_name = (
            f"{scoped_prefix}_last.pth"
        )

        self.best_episode_reward = float("-inf")
        self._best_average = float("-inf")
        self._recent_returns.clear()

        self.agent = new_agent
        self._agent_signature = signature

        return self.agent

    @staticmethod
    def _build_action_mask(
        context: SimulationContext,
        agent,
    ) -> np.ndarray:
        uavs = {
            uav.id: uav
            for uav in context.uavs
        }
        goals = {
            goal.id: goal
            for goal in context.goals
        }

        leaders = {
            uav.id
            for uav in context.uavs
            if (
                uav.state in {
                    "Leader",
                    "Ground_Leader",
                }
                and uav.target_goal_id is not None
            )
        }

        mask = np.zeros(
            agent.output_size,
            dtype=bool,
        )

        for index, action in enumerate(agent.actions):
            if action.kind == "keep":
                mask[index] = any(
                    uav.state != "Free"
                    for uav in context.uavs
                )
                continue

            uav = uavs.get(action.uav_id)

            if uav is None:
                continue

            if action.kind == "assign_goal":
                goal = goals.get(action.goal_id)
                mask[index] = (
                    goal is not None
                    and goal.state == "Free"
                )

            elif action.kind in {
                "assign_slave",
                "assign_relay",
            }:
                mask[index] = (
                    action.leader_id in leaders
                    and action.leader_id != uav.id
                )

        return mask


ALGORITHMS = [DroneDQNAlgorithm]