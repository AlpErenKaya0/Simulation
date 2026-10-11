import unittest
from dataclasses import replace

import numpy as np

from algorithms.model import DroneDQNAlgorithm
from algorithms.reinforcementagent import Agent
from algorithms.reinforcementagent import (
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


if __name__ == "__main__":
    unittest.main()
