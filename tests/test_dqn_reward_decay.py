import unittest
from dataclasses import replace

import numpy as np

from algorithms.model import DroneDQNAlgorithm
from algorithms.reinforcementagent import Agent
from algorithms.reinforcementagent import (
    PENALTY_AIMLESS,
    PENALTY_TARGET_CHANGE,
    PENALTY_UNNECESSARY_ROLE_CHANGE,
    REWARD_DECAY_HALF_LIFE,
    calculate_reward,
    reward_decay_factor,
)
from simulation_core.config import SimulationConfig
from simulation_core.engine import SimulationEngine


class DQNRewardDecayTests(unittest.TestCase):
    @staticmethod
    def make_context():
        engine = SimulationEngine(algorithm_name="drone_dqn")
        engine.set_config_provider(lambda: SimulationConfig())
        engine.add_uav(0, 0)
        for index in range(3):
            engine.add_goal(10 + index * 5, 0)
        return engine._build_context(0.0, SimulationConfig())

    def test_uav_goal_distance_matrix_is_given_to_the_agent(self):
        context = self.make_context()
        context = replace(context, comm_thr=10.0)
        agent = Agent(
            context,
            canvas_width=100,
            canvas_height=100,
            seed=1,
        )

        distances = agent.get_uav_goal_distance_matrix(context)
        state = agent.get_state(context)

        self.assertEqual((1, 3), distances.shape)
        np.testing.assert_allclose(
            distances,
            [[10.0, 15.0, 20.0]],
        )
        matrix_end = state.size - len(context.uavs)
        np.testing.assert_allclose(
            state[matrix_end - distances.size:matrix_end],
            (distances / context.comm_thr).reshape(-1),
        )

    def test_goal_reward_decays_within_an_episode(self):
        context = self.make_context()
        algorithm = DroneDQNAlgorithm()
        algorithm.training_enabled = True
        algorithm._get_agent = lambda _: type(
            "AgentStub",
            (),
            {"training_steps": 0},
        )()

        first_goal_context = replace(
            context,
            goals=(
                replace(context.goals[0], state="Visited"),
                *context.goals[1:],
            ),
        )
        second_goal_context = replace(
            first_goal_context,
            goals=(
                first_goal_context.goals[0],
                replace(first_goal_context.goals[1], state="Visited"),
                first_goal_context.goals[2],
            ),
        )

        algorithm.finish_step(context, first_goal_context)
        first_step_reward = algorithm.episode_reward
        self.assertEqual(1, algorithm.episode_steps)
        algorithm.finish_step(first_goal_context, second_goal_context)
        second_step_reward = algorithm.episode_reward - first_step_reward
        self.assertEqual(2, algorithm.episode_steps)

        self.assertEqual(1.0, first_step_reward)
        self.assertAlmostEqual(
            reward_decay_factor(1),
            second_step_reward,
        )

    def test_goal_reward_resets_to_one_for_the_next_episode(self):
        context = self.make_context()
        algorithm = DroneDQNAlgorithm()
        algorithm.training_enabled = True
        algorithm.episode_steps = 500
        algorithm.reset()

        self.assertEqual(0, algorithm.episode_steps)
        self.assertEqual(1.0, reward_decay_factor(algorithm.episode_steps))

    def test_terminal_rewards_are_not_decayed(self):
        self.assertEqual(
            -10.0,
            calculate_reward(0, connection_lost=True, success=False),
        )
        self.assertEqual(
            10.0,
            calculate_reward(
                0,
                connection_lost=False,
                success=True,
                goal_reward_factor=0.01,
            ),
        )
        self.assertAlmostEqual(
            10.5,
            calculate_reward(
                1,
                connection_lost=False,
                success=True,
                goal_reward_factor=0.5,
            ),
        )

    def test_reward_halves_after_1000_simulation_steps(self):
        self.assertAlmostEqual(
            0.5,
            reward_decay_factor(REWARD_DECAY_HALF_LIFE),
        )

    def test_penalties_in_calculate_reward(self):
        # Target change penalty
        self.assertEqual(
            -PENALTY_TARGET_CHANGE,
            calculate_reward(0, False, False, target_changed=True),
        )
        # Unnecessary role change penalty
        self.assertEqual(
            -PENALTY_UNNECESSARY_ROLE_CHANGE,
            calculate_reward(0, False, False, unnecessary_role_changed=True),
        )
        # Aimless penalty
        self.assertEqual(
            -PENALTY_AIMLESS,
            calculate_reward(0, False, False, is_aimless=True),
        )

    def test_penalize_target_change_when_safe(self):
        context = self.make_context()
        algorithm = DroneDQNAlgorithm()
        algorithm.training_enabled = False

        # UAV has target goal 0
        uav_with_goal0 = replace(context.uavs[0], state="Leader", target_goal_id=context.goals[0].id)
        ctx = replace(context, uavs=(uav_with_goal0,))

        # In next context, switches target to goal 1 while goal 0 is still unvisited and safe
        uav_with_goal1 = replace(uav_with_goal0, target_goal_id=context.goals[1].id)
        next_ctx = replace(ctx, uavs=(uav_with_goal1,))

        algorithm.finish_step(ctx, next_ctx)
        self.assertEqual(-PENALTY_TARGET_CHANGE, algorithm.episode_reward)

    def test_penalize_unnecessary_role_change_when_safe(self):
        context = self.make_context()
        algorithm = DroneDQNAlgorithm()
        algorithm.training_enabled = False

        # UAV is Leader with target goal 0
        uav_leader = replace(context.uavs[0], state="Leader", target_goal_id=context.goals[0].id)
        ctx = replace(context, uavs=(uav_leader,))

        # In next context, switches to Relay without completing goal 0, while safe (<90%)
        uav_relay = replace(uav_leader, state="Relay", target_goal_id=None)
        next_ctx = replace(ctx, uavs=(uav_relay,))

        algorithm.finish_step(ctx, next_ctx)
        # In this transition, role changed unnecessarily
        self.assertEqual(-PENALTY_UNNECESSARY_ROLE_CHANGE, algorithm.episode_reward)

    def test_no_penalty_for_role_or_target_change_when_near_boundary(self):
        engine = SimulationEngine(algorithm_name="drone_dqn")
        engine.set_config_provider(lambda: SimulationConfig())
        engine.environment.set_comm_thr(100.0)
        engine.add_uav(0, 0)
        engine.add_uav(95, 0)  # comm_thr is 100, distance is 95 (>90% threshold)
        engine.add_goal(50, 0)
        context = engine._build_context(0.0, SimulationConfig())

        algorithm = DroneDQNAlgorithm()
        algorithm.training_enabled = False

        # UAV 1 is at x=0 (Leader), UAV 2 is at x=95 (near boundary >90%, also Leader)
        uav1 = replace(context.uavs[0], state="Leader", target_goal_id=context.goals[0].id)
        uav2_leader = replace(context.uavs[1], state="Leader", target_goal_id=context.goals[0].id)
        ctx = replace(context, uavs=(uav1, uav2_leader))

        # UAV 2 switches role to Relay to preserve connection (near boundary >90%)
        uav2_relay = replace(uav2_leader, state="Relay", target_goal_id=None, leader_id=uav1.id)
        next_ctx = replace(ctx, uavs=(uav1, uav2_relay))

        algorithm.finish_step(ctx, next_ctx)
        # Since UAV 2 was near boundary (>90%), changing role is NOT penalized as unnecessary
        self.assertEqual(0.0, algorithm.episode_reward)

    def test_penalize_aimless_uav_when_safe(self):
        context = self.make_context()
        algorithm = DroneDQNAlgorithm()
        algorithm.training_enabled = False

        # UAV is Free / aimless while goals exist and it is safe
        ctx = replace(context, uavs=(replace(context.uavs[0], state="Free", target_goal_id=None),))
        next_ctx = replace(ctx)

        algorithm.finish_step(ctx, next_ctx)
        self.assertEqual(-PENALTY_AIMLESS, algorithm.episode_reward)


if __name__ == "__main__":
    unittest.main()
