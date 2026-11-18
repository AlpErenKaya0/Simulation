"""
DroneDQNAlgorithm: motor ile paylasilan ajan arasindaki kopru.

Karar zamanlamasi olay tabanlidir. Karar turu su durumlarda acilir:
bolum basi, hedef ziyareti, motorun rolleri kendisinin degistirmesi
(ekip dagilmasi vb.), en zayif halkanin %90 esigini yukari yonde gecmesi
ve MAX_DECISION_INTERVAL dolmasi. Turda tum drone'lar sirayla karar verir.
Ayni turda once karar verenin secimi sonrakinin gozleminde gorunur.

Motor (engine.py) ile dogrulanan davranis:
- _apply_plan sirasi: free_uav_ids -> goal -> relay -> slave. Her atama
  once _free_uav cagirir (ekibini dagitir, hedefini birakir).
- Relay de liderin my_slave_list'ine eklenir; lider serbest kalinca dagilir.
- Ayni lidere yeniden RelayAssignment yalnizca target_position'i yeniler
  (_free_uav + yeniden ekleme). Relay'in kendi takipcisi varsa onlar
  dagilacagi icin o Relay guncellenmez.
"""

from __future__ import annotations

import os
import random
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from simulation_core.algorithm_api import (
    AlgorithmPlan,
    GoalAssignment,
    Position,
    RelayAssignment,
    SimulationAlgorithm,
    SimulationContext,
    SlaveAssignment,
)
from simulation_core.topology import (
    build_distance_matrix,
    find_connected_components,
)
from algorithms.reinforcementagent import (
    GAMMA,
    MARGIN_ALARM_RATIO,
    MAX_DECISION_INTERVAL,
    MAX_EPISODE_STEPS,
    SharedAgent,
    Transition,
    WorldView,
    bottleneck_edge,
    compute_relay_point,
    normalize_role,
    pairwise_distances,
    step_reward,
)


CHECKPOINT_DIR = "./model"
CHECKPOINT_LAST = os.path.join(CHECKPOINT_DIR, "drone_dqn_shared_last.pth")
CHECKPOINT_BEST = os.path.join(CHECKPOINT_DIR, "drone_dqn_shared_best.pth")
SAVE_EVERY_EPISODES = 10
AVG_WINDOW = 50

DYNAMIC_RELAY = True
RELAY_UPDATE_MIN_SHIFT = 10.0

RoleKey = Tuple[str, Optional[int], Optional[int]]


def _role_key(uav) -> RoleKey:
    role = normalize_role(uav.state)
    return (
        role,
        uav.target_goal_id if role == "Leader" else None,
        uav.leader_id if role in ("Slave", "Relay") else None,
    )


class DroneDQNAlgorithm(SimulationAlgorithm):
    name = "drone_dqn"
    display_name = "Drone DQN"
    description = "Paylasilan agli, egosentrik gozlemli, olay tabanli DQN."

    def __init__(
        self,
        max_episode_steps: Optional[int] = MAX_EPISODE_STEPS,
        seed: Optional[int] = None,
        verbose: bool = True,
        auto_save: bool = True,
    ) -> None:
        self.training_enabled: bool = False
        self.verbose = verbose
        self.auto_save = auto_save
        self.last_episode: Optional[dict] = None
        self.max_episode_steps = max_episode_steps
        self.seed = seed
        self.agent: Optional[SharedAgent] = None
        self._rng = random.Random(seed)
        self._episode_counter = 0
        self._returns: deque = deque(maxlen=AVG_WINDOW)
        self._disconnects: deque = deque(maxlen=AVG_WINDOW)
        self._reset_episode()

    # ------------------------------------------------------------------ #
    # Disa acik arayuz                                                   #
    # ------------------------------------------------------------------ #

    def enable_training(self, enabled: bool) -> None:
        self.training_enabled = bool(enabled)

    def reset(self) -> None:
        self._reset_episode()

    def _reset_episode(self) -> None:
        self.episode_steps = 0
        self.episode_reward = 0.0
        self._pending: Dict[int, Tuple[np.ndarray, np.ndarray, int]] = {}
        self._acc_reward = 0.0
        self._acc_discount = 1.0
        self._force_decision = True
        self._event = False
        self._steps_since_decision = 0
        self._motions: Dict[int, Tuple[float, float]] = {}
        self._role_since: Dict[int, int] = {}
        self._last_roles: Dict[int, RoleKey] = {}
        self._prev_visited = 0
        self._prev_ratio = 0.0
        self._decision_rounds = 0
        self._role_changes = 0

    def _ensure_agent(self) -> None:
        if self.agent is not None:
            return
        self.agent = SharedAgent(seed=self.seed)
        if self.training_enabled:
            paths = [CHECKPOINT_LAST]
        else:
            paths = [CHECKPOINT_BEST, CHECKPOINT_LAST]
        for path in paths:
            if not os.path.exists(path):
                continue
            try:
                self.agent.load(path)
                print(f"[DroneDQN] Yuklendi: {path}")
                return
            except ValueError as exc:
                print(f"[DroneDQN] Yuklenemedi ({path}): {exc}")
        print("[DroneDQN] Yeni ag ile baslaniyor.")

    # ------------------------------------------------------------------ #
    # Plan                                                               #
    # ------------------------------------------------------------------ #

    def build_plan(self, context: SimulationContext) -> AlgorithmPlan:
        if not context.uavs or not context.goals:
            return AlgorithmPlan.empty()

        visited_now = sum(1 for g in context.goals if g.state == "Visited")
        single_visit = (
            getattr(context.config, "target_eval_mode", "single_visit")
            == "single_visit"
        )
        if single_visit and visited_now < self._prev_visited:
            # GUI reset() cagirmadan yeni senaryo baslatmis.
            self._reset_episode()

        if all(g.state == "Visited" for g in context.goals):
            return AlgorithmPlan.empty()

        self._ensure_agent()

        relay_updates = (
            self._dynamic_relay_updates(context) if DYNAMIC_RELAY else []
        )

        due = (
            self._force_decision
            or self._event
            or self._steps_since_decision >= MAX_DECISION_INTERVAL
        )
        if not due:
            if relay_updates:
                return AlgorithmPlan(relay_assignments=tuple(relay_updates))
            return AlgorithmPlan.empty()

        return self._decision_round(context, relay_updates)

    def _decision_round(
        self, context: SimulationContext, relay_updates: List[RelayAssignment]
    ) -> AlgorithmPlan:
        world = WorldView(
            context,
            self._motions,
            self._role_since,
            self.episode_steps,
            self.max_episode_steps,
        )
        order = list(world.uav_ids)
        if self.training_enabled:
            self._rng.shuffle(order)

        # (tur, referans, relay noktasi). Relay noktasi atama aninda saklanir:
        # ayni turda lideri sonradan degisirse takipci sanal olarak dagilir,
        # ama motor da atamayi isleyip ayni sekilde dagitacagi icin plana yazilir.
        decided: Dict[int, Tuple[str, Optional[int], Optional[np.ndarray]]] = {}

        for uid in order:
            obs = world.observation(uid)
            cands = world.candidates(uid)
            feats = np.stack([world.candidate_features(uid, c) for c in cands])

            if self.training_enabled and uid in self._pending:
                p_obs, p_feats, p_action = self._pending.pop(uid)
                self.agent.push(
                    Transition(
                        obs=p_obs,
                        cand_feats=p_feats,
                        action=p_action,
                        reward=self._acc_reward,
                        discount=self._acc_discount,
                        done=False,
                        next_obs=obs,
                        next_cand_feats=feats,
                    )
                )

            keep_index = 0 if cands[0][0] == "keep" else None
            action = self.agent.select_action(
                obs, feats, self.training_enabled, keep_index
            )
            cand = cands[action]

            if self.training_enabled:
                self._pending[uid] = (obs, feats, action)

            if world.apply(uid, cand):
                point = world.relay_point[uid]
                decided[uid] = (
                    cand[0], cand[1], None if point is None else point.copy()
                )
                self._role_changes += 1
                self._role_since[uid] = self.episode_steps

        goal_as: List[GoalAssignment] = []
        slave_as: List[SlaveAssignment] = []
        relay_as: List[RelayAssignment] = []
        free_ids: List[int] = []

        for uid, (kind, ref, point) in decided.items():
            if kind == "free":
                free_ids.append(uid)
            elif kind == "leader":
                goal_as.append(GoalAssignment(uav_id=uid, goal_id=ref, role="Leader"))
            elif kind == "slave":
                slave_as.append(SlaveAssignment(uav_id=uid, leader_id=ref))
            elif kind == "relay":
                relay_as.append(
                    RelayAssignment(
                        uav_id=uid,
                        leader_id=ref,
                        target_position=Position(float(point[0]), float(point[1])),
                    )
                )

        # Dinamik relay guncellemesi de _free_uav cagirir. Bu turda bu Relay'e
        # yeni takipci atandiysa ya da lideri degistiyse guncelleme atlanir.
        new_leaders = {
            ref for (kind, ref, _) in decided.values() if kind in ("slave", "relay")
        }
        for upd in relay_updates:
            if (
                upd.uav_id not in decided
                and upd.leader_id not in decided
                and upd.uav_id not in new_leaders
            ):
                relay_as.append(upd)

        # Beklenen roller; motor farkli uygularsa finish_step olay olarak yakalar.
        for uid in world.uav_ids:
            self._last_roles[uid] = (
                world.role[uid], world.target[uid], world.leader[uid]
            )

        self._acc_reward = 0.0
        self._acc_discount = 1.0
        self._force_decision = False
        self._event = False
        self._steps_since_decision = 0
        self._decision_rounds += 1

        if not (goal_as or slave_as or relay_as or free_ids):
            return AlgorithmPlan.empty()
        return AlgorithmPlan(
            goal_assignments=tuple(goal_as),
            relay_assignments=tuple(relay_as),
            slave_assignments=tuple(slave_as),
            free_uav_ids=tuple(free_ids),
        )

    def _dynamic_relay_updates(
        self, context: SimulationContext
    ) -> List[RelayAssignment]:
        uavs = list(context.uavs)
        index = {u.id: i for i, u in enumerate(uavs)}
        pts = [[u.position.x, u.position.y] for u in uavs]
        if context.ground is not None:
            pts.append([context.ground.position.x, context.ground.position.y])
        node_pos = np.array(pts, dtype=np.float64).reshape(-1, 2)

        out: List[RelayAssignment] = []
        for u in uavs:
            if u.state != "Relay" or u.leader_id not in index:
                continue
            if any(
                o.state in ("Slave", "Relay") and o.leader_id == u.id for o in uavs
            ):
                continue
            leader_id = u.leader_id
            exclude = {index[u.id]} | {
                index[o.id]
                for o in uavs
                if o.state in ("Slave", "Relay") and o.leader_id == leader_id
            }
            point = compute_relay_point(node_pos, index[leader_id], exclude)
            if u.relay_target is not None:
                shift = float(
                    np.hypot(point[0] - u.relay_target.x, point[1] - u.relay_target.y)
                )
                if shift < RELAY_UPDATE_MIN_SHIFT:
                    continue
            out.append(
                RelayAssignment(
                    uav_id=u.id,
                    leader_id=leader_id,
                    target_position=Position(float(point[0]), float(point[1])),
                )
            )
        return out

    # ------------------------------------------------------------------ #
    # Adim sonu                                                          #
    # ------------------------------------------------------------------ #

    def finish_step(
        self, context: SimulationContext, next_context: SimulationContext
    ) -> Optional[str]:
        self.episode_steps += 1
        self._steps_since_decision += 1

        for u in next_context.uavs:
            prev = context.uav_by_id(u.id)
            if prev is None:
                self._motions[u.id] = (0.0, 0.0)
            else:
                self._motions[u.id] = (
                    u.position.x - prev.position.x,
                    u.position.y - prev.position.y,
                )

        visited_now = sum(1 for g in next_context.goals if g.state == "Visited")
        new_visits = max(0, visited_now - self._prev_visited)
        self._prev_visited = visited_now
        remaining = len(next_context.goals) - visited_now

        positions = [u.position for u in next_context.uavs]
        if next_context.ground is not None:
            positions.append(next_context.ground.position)
        components = find_connected_components(
            build_distance_matrix(positions),
            next_context.comm_thr,
            threshold_ratio=1.0,
        )
        disconnected = len(components) > 1
        all_visited = remaining == 0

        node_pos = np.array([[p.x, p.y] for p in positions], dtype=np.float64)
        thr = next_context.comm_thr if next_context.comm_thr > 0 else 1.0
        ratio = bottleneck_edge(pairwise_distances(node_pos))[0] / thr

        reward = step_reward(new_visits, remaining, disconnected, all_visited, ratio)
        self.episode_reward += reward
        self._acc_reward += self._acc_discount * reward
        self._acc_discount *= GAMMA

        if new_visits > 0:
            self._event = True
        if ratio > MARGIN_ALARM_RATIO >= self._prev_ratio:
            self._event = True
        self._prev_ratio = ratio

        for u in next_context.uavs:
            key = _role_key(u)
            old = self._last_roles.get(u.id)
            if old is not None and old != key:
                self._event = True
                self._role_since[u.id] = self.episode_steps
            self._last_roles[u.id] = key

        stop: Optional[str] = None
        terminal = False
        if disconnected:
            stop, terminal = "network_disconnected", True
        elif all_visited:
            stop, terminal = "all_goals_visited", True
        elif (
            self.max_episode_steps is not None
            and self.episode_steps >= self.max_episode_steps
        ):
            # GUI yalnizca bu ucunu taniyor; "time_limit" dongusu kilitler.
            stop, terminal = "time_elapsed", False

        if stop is not None:
            self._end_episode(next_context, stop, terminal, visited_now)
        return stop

    def _end_episode(
        self,
        next_context: SimulationContext,
        stop: str,
        terminal: bool,
        visited_now: int,
    ) -> None:
        agent = self.agent
        if self.training_enabled and agent is not None and self._pending:
            if terminal:
                for p_obs, p_feats, p_action in self._pending.values():
                    agent.push(
                        Transition(
                            obs=p_obs,
                            cand_feats=p_feats,
                            action=p_action,
                            reward=self._acc_reward,
                            discount=self._acc_discount,
                            done=True,
                            next_obs=None,
                            next_cand_feats=None,
                        )
                    )
            else:
                # Truncation: terminal degil, sonraki durumdan bootstrap yapilir.
                world = WorldView(
                    next_context,
                    self._motions,
                    self._role_since,
                    self.episode_steps,
                    self.max_episode_steps,
                )
                for uid, (p_obs, p_feats, p_action) in self._pending.items():
                    if uid not in world.index:
                        continue
                    cands = world.candidates(uid)
                    feats = np.stack(
                        [world.candidate_features(uid, c) for c in cands]
                    )
                    agent.push(
                        Transition(
                            obs=p_obs,
                            cand_feats=p_feats,
                            action=p_action,
                            reward=self._acc_reward,
                            discount=self._acc_discount,
                            done=False,
                            next_obs=world.observation(uid),
                            next_cand_feats=feats,
                        )
                    )

        self._episode_counter += 1
        self._returns.append(self.episode_reward)
        self._disconnects.append(stop == "network_disconnected")
        avg = float(np.mean(self._returns))
        disconnect_rate = float(np.mean(self._disconnects))

        self.last_episode = {
            "reward": self.episode_reward,
            "visited": visited_now,
            "goals": len(next_context.goals),
            "steps": self.episode_steps,
            "end": stop,
            "role_changes": self._role_changes,
            "rounds": self._decision_rounds,
        }

        if self.training_enabled and agent is not None:
            agent.episodes += 1
            if self.auto_save:
                if len(self._returns) >= AVG_WINDOW and avg > agent.best_avg:
                    agent.best_avg = avg
                    self._safe_save(CHECKPOINT_BEST)
                if agent.episodes % SAVE_EVERY_EPISODES == 0:
                    self._safe_save(CHECKPOINT_LAST)

        if agent is not None and self.verbose:
            loss = f"{agent.last_loss:.4f}" if agent.last_loss is not None else "n/a"
            best = f"{agent.best_avg:.2f}" if agent.best_avg > float("-inf") else "n/a"
            print(
                f"Episode {self._episode_counter} | "
                f"Reward {self.episode_reward:.2f} | Avg {avg:.2f} | BestAvg {best} | "
                f"Visited {visited_now}/{len(next_context.goals)} | "
                f"RoleChanges {self._role_changes} | Rounds {self._decision_rounds} | "
                f"DisconnectRate {disconnect_rate:.2f} | Steps {self.episode_steps} | "
                f"End {stop} | Epsilon {agent.epsilon if self.training_enabled else 0.0:.3f} | "
                f"Loss {loss} | Updates {agent.updates} | Replay {len(agent.memory)}"
            )

        self._reset_episode()

    def _safe_save(self, path: str) -> None:
        try:
            self.agent.save(path)
        except Exception as exc:
            print(f"[DroneDQN] Kayit basarisiz ({path}): {exc}")


ALGORITHMS = [DroneDQNAlgorithm]
