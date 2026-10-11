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
        has_unvisited = any(g.state != "Visited" for g in next_context.goals)
        decay_factor = reward_decay_factor(self.episode_steps - 1)

        # UAV basina cezalari ve odulleri hesapla
        step_rewards: list[float] = []

        for uav in context.uavs:
            prev_uav = context.uav_by_id(uav.id)
            next_uav = next_context.uav_by_id(uav.id)
            if prev_uav is None or next_uav is None:
                continue

            prev_near = self._is_uav_near_boundary(context, prev_uav)
            next_near = self._is_uav_near_boundary(next_context, next_uav)

            # Hedefin bu adimda tamamlanip tamamlanmadigini kontrol et
            visited_its_goal = (
                prev_uav.target_goal_id is not None
                and next_context.goal_by_id(prev_uav.target_goal_id) is not None
                and next_context.goal_by_id(prev_uav.target_goal_id).state == "Visited"
            )

            # 1. Surekli hedef degistirme cezasi:
            # Drone guvenli bolgedeyken (%90 alti) hedefi henuz ziyaret EDILMEMISKEN farkli bir hedefe gecmesi
            target_changed = False
            if not prev_near and prev_uav.target_goal_id is not None:
                prev_goal = next_context.goal_by_id(prev_uav.target_goal_id)
                if prev_goal is not None and prev_goal.state != "Visited":
                    if next_uav.target_goal_id is not None and next_uav.target_goal_id != prev_uav.target_goal_id:
                        target_changed = True

            # 2. %90 baglanti kopma ihtimali olmamasina ragmen gorev degistirme cezasi:
            # Drone guvenli bolgedeydi (%90 alti), ama hedefini tamamlamadan sebepsiz yere gorev/rol degistirdi
            unnecessary_role_changed = False
            if not prev_near and prev_uav.state in {"Leader", "Slave", "Relay"}:
                if next_uav.state != prev_uav.state and not visited_its_goal:
                    unnecessary_role_changed = True

            # 3. %90 baglanti kopus riski olmamasina ragmen hedefsiz / bos bekleme cezasi:
            # Ortamda unvisited hedef var, drone %90 alti guvenli bolgede, ama hedefsiz/gorevsiz
            # (Hedef ziyaret edildiyse, gorev degistirme cezasi alindiysa veya drone hedefini tamamladiysa hedefsiz sayilmaz)
            is_aimless = False
            if (
                has_unvisited
                and not next_near
                and not visited_its_goal
                and not unnecessary_role_changed
                and new_visits == 0
            ):
                if (
                    next_uav.state == "Free"
                    or (next_uav.state in {"Leader", "Ground_Leader"} and next_uav.target_goal_id is None)
                    or (next_uav.state == "Slave" and next_uav.leader_id is None)
                    or (next_uav.state == "Relay" and next_uav.leader_id is None and next_uav.relay_target is None)
                ):
                    is_aimless = True

            uav_reward = calculate_reward(
                newly_visited_count=new_visits,
                connection_lost=disconnected,
                success=all_visited,
                goal_reward_factor=decay_factor,
                target_changed=target_changed,
                unnecessary_role_changed=unnecessary_role_changed,
                is_aimless=is_aimless,
            )
            step_rewards.append(uav_reward)

            if self.training_enabled:
                agent = self._agents.get(uav.id)
                prev_state = self._prev_states.get(uav.id)
                prev_action = self._prev_actions.get(uav.id)
                if agent is not None and prev_state is not None and prev_action is not None:
                    next_state = agent.get_state(next_context)
                    agent.train_short_memory(
                        prev_state, prev_action, uav_reward, next_state, done
                    )
                    agent.remember(
                        prev_state, prev_action, uav_reward, next_state, done
                    )

        if self.training_enabled and done:
            for agent in self._agents.values():
                agent.train_long_memory()
                agent.n_games += 1
                try:
                    agent.model.save("drone_dqn_weights.pth")
                except Exception:
                    pass

        avg_reward = (
            sum(step_rewards) / len(step_rewards)
            if step_rewards
            else calculate_reward(new_visits, disconnected, all_visited, decay_factor)
        )
        self.episode_reward += avg_reward

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

    @staticmethod
    def _is_uav_near_boundary(context: SimulationContext, uav: UavView) -> bool:
        """UAV'nin diger ag dugumlerine gore %90 sinirini asmis olup olmadigini kontrol eder."""
        other_positions = [
            other.position for other in context.uavs if other.id != uav.id
        ]
        if context.ground is not None:
            other_positions.append(context.ground.position)

        if not other_positions:
            return False

        min_dist = min(distance_between(uav.position, pos) for pos in other_positions)
        return min_dist > context.comm_thr * 0.9

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
        # ag sinirinda olmayan (%90 alti) ya da hedefe en yakin UAV'yi lider yap
        if not leader_uav_ids and unvisited_goals and context.uavs:
            safe_uavs = [
                u for u in context.uavs if not self._is_uav_near_boundary(context, u)
            ]
            candidates = safe_uavs if safe_uavs else list(context.uavs)
            best_uav = min(
                candidates,
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
            near_boundary = self._is_uav_near_boundary(context, uav)
            action = actions.get(uav.id, [1, 0, 0])
            role_idx = action.index(1) if 1 in action else 0

            # Eger UAV aktif bir lidere atanmissa:
            if uav.state in {"Leader", "Ground_Leader"} and uav.target_goal_id is not None:
                current_target = context.goal_by_id(uav.target_goal_id)
                target_valid = current_target is not None and current_target.state != "Visited"
                can_move = target_valid and self.allow_movement(context, uav, current_target.position)

                # %90 siniri ASILMAMISSA ve hedefe hareket serbestse ve ajan lider kalmak istiyorsa rotayi koru
                if target_valid and not near_boundary and can_move and role_idx == 0:
                    goal_assignments.append(
                        GoalAssignment(
                            uav_id=uav.id,
                            goal_id=current_target.id,
                            role="Leader",
                        )
                    )
                    assigned_goal_ids.add(current_target.id)
                    continue

                # EGER %90 ASILDIYSA VEYA HAREKET ENGELLENDIYSE VEYA AJAN BASKA ROL SECTIYSE:
                # Kilit kalkar! Ajanin secimine gore Relay, Slave veya farkli bir hedef tanimlanir.

            if role_idx == 0:  # Leader -> hedefe git
                available = [
                    g for g in unvisited_goals if g.id not in assigned_goal_ids
                ]
                # Eger sinirdaysak, ag baglantisini koparmadan gidilebilecek hedefleri onceliklendir
                reachable_goals = [
                    g for g in available if self.allow_movement(context, uav, g.position)
                ]
                candidates = reachable_goals if reachable_goals else available

                if candidates:
                    target = min(
                        candidates,
                        key=lambda g: distance_between(uav.position, g.position),
                    )
                    # Eger bu hedefe hareket hala engelleniyorsa ve baska drone varsa donmamak icin relay'e gec
                    if not self.allow_movement(context, uav, target.position) and len(context.uavs) > 1:
                        role_idx = 2  # Relay'e donuserek baglantiyi destekle
                    else:
                        goal_assignments.append(
                            GoalAssignment(
                                uav_id=uav.id, goal_id=target.id, role="Leader"
                            )
                        )
                        assigned_goal_ids.add(target.id)

            if role_idx == 1:  # Slave -> en yakin lidere / diger drone'a eslik et
                other_candidates = [
                    u for u in context.uavs if u.id != uav.id
                ]
                if other_candidates:
                    other_leaders = [u for u in other_candidates if u.id in leader_uav_ids]
                    target_lead = other_leaders[0] if other_leaders else min(
                        other_candidates,
                        key=lambda l: distance_between(uav.position, l.position),
                    )
                    slave_assignments.append(
                        SlaveAssignment(
                            uav_id=uav.id, leader_id=target_lead.id
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

            elif role_idx == 2:  # Relay -> ag ile lider arasina konumlan
                other_uavs = [u for u in context.uavs if u.id != uav.id]
                if other_uavs:
                    other_leaders = [u for u in other_uavs if u.id in leader_uav_ids]
                    target_leader = other_leaders[0] if other_leaders else min(
                        other_uavs,
                        key=lambda l: distance_between(uav.position, l.position),
                    )
                    if context.ground is not None:
                        anchor_pos = context.ground.position
                    else:
                        others = [
                            u for u in context.uavs if u.id != target_leader.id and u.id != uav.id
                        ]
                        if others:
                            anchor_pos = Position(
                                x=sum(u.position.x for u in others) / len(others),
                                y=sum(u.position.y for u in others) / len(others),
                            )
                        else:
                            anchor_pos = target_leader.position

                    mid = Position(
                        x=(anchor_pos.x + target_leader.position.x) / 2,
                        y=(anchor_pos.y + target_leader.position.y) / 2,
                    )
                    relay_assignments.append(
                        RelayAssignment(
                            uav_id=uav.id,
                            leader_id=target_leader.id,
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