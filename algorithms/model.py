from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from simulation_core.algorithm_api import (
    AlgorithmPlan,
    GoalAssignment,
    Position,
    RelayAssignment,
    SimulationAlgorithm,
    SimulationContext,
    SlaveAssignment,
    UavView,
)
from simulation_core.topology import (
    build_distance_matrix,
    distance_between,
    find_connected_components,
)


class Linear_QNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name: str = "drone_dqn_weights.pth") -> None:
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_path = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_path)

    def load(self, file_name: str = "drone_dqn_weights.pth") -> None:
        model_folder_path = "./model"
        file_path = os.path.join(model_folder_path, file_name)
        if os.path.exists(file_path):
            self.load_state_dict(torch.load(file_path))


class QTrainer:
    def __init__(self, model: nn.Module, lr: float, gamma: float):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float32)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float32)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float32)

        if state.ndim == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done,)

        pred = self.model(state)

        with torch.no_grad():
            target = pred.clone()

            for idx in range(len(done)):
                Q_new = reward[idx]

                if not done[idx]:
                    Q_new = (
                        reward[idx]
                        + self.gamma
                        * torch.max(self.model(next_state[idx]))
                    )

                action_index = torch.argmax(action[idx]).item()
                target[idx, action_index] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()


class DroneDQNAlgorithm(SimulationAlgorithm):
    name = "drone_dqn"
    display_name = "Drone DQN"
    description = "DQN tabanli pekistirmeli ogrenme algoritmasi."

    def __init__(self) -> None:
        self.training_enabled: bool = False
        self._agents: dict = {}
        self._prev_states: dict = {}
        self._prev_actions: dict = {}
        self._prev_visited: int = 0
        self.episode_steps: int = 0
        self.episode_reward: float = 0.0
        self._last_context_sig: tuple = ()

    # ------------------------------------------------------------------ #
    # Public interface                                                   #
    # ------------------------------------------------------------------ #

    def enable_training(self, enabled: bool) -> None:
        self.training_enabled = bool(enabled)

    def reset(self) -> None:
        self._prev_states.clear()
        self._prev_actions.clear()
        self._prev_visited = 0
        self.episode_steps = 0
        self.episode_reward = 0.0

    def _get_agent(self, context: SimulationContext, uav_id: int):
        from algorithms.reinforcementagent import Agent

        sig = (
            tuple(sorted(u.id for u in context.uavs)),
            tuple(sorted(g.id for g in context.goals)),
        )
        if sig != self._last_context_sig:
            self._agents.clear()
            self._last_context_sig = sig

        if uav_id not in self._agents:
            self._agents[uav_id] = Agent(context, uav_id)
        return self._agents[uav_id]

    # ------------------------------------------------------------------ #
    # Core algorithm hooks                                               #
    # ------------------------------------------------------------------ #

    def build_plan(self, context: SimulationContext) -> AlgorithmPlan:
        if not context.uavs or not context.goals:
            return AlgorithmPlan.empty()

        if all(g.state == "Visited" for g in context.goals):
            return AlgorithmPlan.empty()

        actions: dict[int, list[int]] = {}
        for uav in context.uavs:
            agent = self._get_agent(context, uav.id)
            state = agent.get_state(context)
            action = agent.get_action(state)
            self._prev_states[uav.id] = state
            self._prev_actions[uav.id] = action
            actions[uav.id] = action

        return self._build_assignments(context, actions)

    def finish_step(
        self, context: SimulationContext, next_context: SimulationContext
    ) -> Optional[str]:
        from algorithms.reinforcementagent import (
            calculate_reward,
            reward_decay_factor,
        )

        self.episode_steps += 1

        visited_now = sum(1 for g in next_context.goals if g.state == "Visited")
        new_visits = max(0, visited_now - self._prev_visited)
        self._prev_visited = visited_now

        positions = [uav.position for uav in next_context.uavs]
        if next_context.ground is not None:
            positions.append(next_context.ground.position)
        adj = build_distance_matrix(positions)
        # Gercek kopma kontrolu: threshold_ratio=1.0
        # %90-100 arasi sadece alert (risky) durumudur, episod sonlandirilmaz, devam eder.
        components = find_connected_components(
            adj, next_context.comm_thr, threshold_ratio=1.0
        )
        disconnected = len(components) > 1

        all_visited = all(g.state == "Visited" for g in next_context.goals)
        done = disconnected or all_visited

        step_reward = calculate_reward(
            newly_visited_count=new_visits,
            connection_lost=disconnected,
            success=all_visited,
            goal_reward_factor=reward_decay_factor(self.episode_steps - 1),
        )
        self.episode_reward += step_reward

        if self.training_enabled:
            for uav in context.uavs:
                agent = self._agents.get(uav.id)
                prev_state = self._prev_states.get(uav.id)
                prev_action = self._prev_actions.get(uav.id)
                if agent is None or prev_state is None or prev_action is None:
                    continue

                next_state = agent.get_state(next_context)
                agent.train_short_memory(
                    prev_state, prev_action, step_reward, next_state, done
                )
                agent.remember(
                    prev_state, prev_action, step_reward, next_state, done
                )

            if done:
                for agent in self._agents.values():
                    agent.train_long_memory()
                    agent.n_games += 1
                    try:
                        agent.model.save("drone_dqn_weights.pth")
                    except Exception:
                        pass

        if disconnected:
            return "network_disconnected"
        if all_visited:
            return "all_goals_visited"
        return None

    def allow_movement(
        self,
        context: SimulationContext,
        uav: UavView,
        target_position: Position,
    ) -> bool:
        if len(context.uavs) <= 1:
            return True

        next_pos = self._next_position(uav.position, target_position, uav.speed)
        positions = [
            next_pos if candidate.id == uav.id else candidate.position
            for candidate in context.uavs
        ]
        if context.ground is not None:
            positions.append(context.ground.position)
        adj = build_distance_matrix(positions)
        components = find_connected_components(
            adj, context.comm_thr, threshold_ratio=1.0
        )
        if len(components) <= 1:
            return True

        # Ag zaten kopuksa durumun daha da kotulesmesine engel ol
        curr_positions = [candidate.position for candidate in context.uavs]
        if context.ground is not None:
            curr_positions.append(context.ground.position)
        curr_adj = build_distance_matrix(curr_positions)
        curr_components = find_connected_components(
            curr_adj, context.comm_thr, threshold_ratio=1.0
        )
        return len(components) <= len(curr_components)

    # ------------------------------------------------------------------ #
    # Assignment helpers                                                 #
    # ------------------------------------------------------------------ #

    def _build_assignments(
        self, context: SimulationContext, actions: dict[int, list[int]]
    ) -> AlgorithmPlan:
        unvisited_goals = [g for g in context.goals if g.state != "Visited"]
        assigned_goal_ids: set[int] = set()

        leader_uav_ids = [
            uid for uid, act in actions.items() if act[0] == 1
        ]

        # Hicbir UAV lider secmediyse ve ziyaret edilmemis hedef varsa,
        # en yakin UAV'yi lider olarak ata
        if not leader_uav_ids and unvisited_goals and context.uavs:
            best_uav = min(
                context.uavs,
                key=lambda u: min(
                    distance_between(u.position, g.position)
                    for g in unvisited_goals
                ),
            )
            leader_uav_ids.append(best_uav.id)
            actions[best_uav.id] = [1, 0, 0]

        goal_assignments: list[GoalAssignment] = []
        relay_assignments: list[RelayAssignment] = []
        slave_assignments: list[SlaveAssignment] = []

        for uav in context.uavs:
            # Eger UAV zaten aktif bir hedefe lider olarak gidiyorsa ve o hedef hala Visited degilse, devam et
            if uav.state in {"Leader", "Ground_Leader"} and uav.target_goal_id is not None:
                current_target = context.goal_by_id(uav.target_goal_id)
                if current_target is not None and current_target.state != "Visited":
                    goal_assignments.append(
                        GoalAssignment(
                            uav_id=uav.id,
                            goal_id=current_target.id,
                            role="Leader",
                        )
                    )
                    assigned_goal_ids.add(current_target.id)
                    continue

            action = actions.get(uav.id, [1, 0, 0])
            role_idx = action.index(1) if 1 in action else 0

            if role_idx == 0:  # Leader -> en yakin unvisited hedefe git
                available = [
                    g for g in unvisited_goals if g.id not in assigned_goal_ids
                ]
                if available:
                    target = min(
                        available,
                        key=lambda g: distance_between(uav.position, g.position),
                    )
                    goal_assignments.append(
                        GoalAssignment(
                            uav_id=uav.id, goal_id=target.id, role="Leader"
                        )
                    )
                    assigned_goal_ids.add(target.id)
                elif leader_uav_ids and any(lid != uav.id for lid in leader_uav_ids):
                    other_leader = min(
                        (
                            u
                            for u in context.uavs
                            if u.id in leader_uav_ids and u.id != uav.id
                        ),
                        key=lambda l: distance_between(uav.position, l.position),
                    )
                    slave_assignments.append(
                        SlaveAssignment(
                            uav_id=uav.id, leader_id=other_leader.id
                        )
                    )

            elif role_idx == 1:  # Slave -> en yakin lidere eslik et
                other_leaders = [
                    u
                    for u in context.uavs
                    if u.id in leader_uav_ids and u.id != uav.id
                ]
                if other_leaders:
                    leader_uav = min(
                        other_leaders,
                        key=lambda l: distance_between(uav.position, l.position),
                    )
                    slave_assignments.append(
                        SlaveAssignment(
                            uav_id=uav.id, leader_id=leader_uav.id
                        )
                    )
                else:
                    available = [
                        g for g in unvisited_goals if g.id not in assigned_goal_ids
                    ]
                    if available:
                        target = min(
                            available,
                            key=lambda g: distance_between(uav.position, g.position),
                        )
                        goal_assignments.append(
                            GoalAssignment(
                                uav_id=uav.id, goal_id=target.id, role="Leader"
                            )
                        )
                        assigned_goal_ids.add(target.id)

            elif role_idx == 2:  # Relay -> ground ile lider arasina konumlan
                other_leaders = [
                    u
                    for u in context.uavs
                    if u.id in leader_uav_ids and u.id != uav.id
                ]
                if other_leaders:
                    leader_uav = min(
                        other_leaders,
                        key=lambda l: distance_between(uav.position, l.position),
                    )
                    if context.ground is not None:
                        anchor_pos = context.ground.position
                    else:
                        others = [
                            u for u in context.uavs if u.id != leader_uav.id
                        ]
                        anchor_pos = Position(
                            x=sum(u.position.x for u in others) / max(1, len(others)),
                            y=sum(u.position.y for u in others) / max(1, len(others)),
                        )
                    mid = Position(
                        x=(anchor_pos.x + leader_uav.position.x) / 2,
                        y=(anchor_pos.y + leader_uav.position.y) / 2,
                    )
                    relay_assignments.append(
                        RelayAssignment(
                            uav_id=uav.id,
                            leader_id=leader_uav.id,
                            target_position=mid,
                        )
                    )
                else:
                    available = [
                        g for g in unvisited_goals if g.id not in assigned_goal_ids
                    ]
                    if available:
                        target = min(
                            available,
                            key=lambda g: distance_between(uav.position, g.position),
                        )
                        goal_assignments.append(
                            GoalAssignment(
                                uav_id=uav.id, goal_id=target.id, role="Leader"
                            )
                        )
                        assigned_goal_ids.add(target.id)

        return AlgorithmPlan(
            clear_uav_roles=True,
            clear_goal_assignments=True,
            goal_assignments=tuple(goal_assignments),
            relay_assignments=tuple(relay_assignments),
            slave_assignments=tuple(slave_assignments),
        )

    @staticmethod
    def _next_position(
        current: Position, target: Position, speed: float
    ) -> Position:
        import math

        dx = target.x - current.x
        dy = target.y - current.y
        dist = math.hypot(dx, dy)
        if dist == 0 or dist <= speed:
            return target
        return Position(
            x=current.x + (dx / dist) * speed,
            y=current.y + (dy / dist) * speed,
        )