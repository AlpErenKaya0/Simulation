import torch
import random
import numpy as np
from collections import deque

from algorithms.model import Linear_QNet, QTrainer
from simulation_core.topology import (
    build_distance_matrix,
    build_uav_goal_distance_matrix,
    find_connected_components,
)
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
REWARD_DECAY_HALF_LIFE = 1000


def reward_decay_factor(episode_steps: int) -> float:
    if episode_steps < 0:
        raise ValueError("episode_steps cannot be negative.")
    return float(0.5 ** (episode_steps / REWARD_DECAY_HALF_LIFE))


def calculate_reward(
    newly_visited_count: int,
    connection_lost: bool,
    success: bool,
    goal_reward_factor: float = 1.0,
) -> float:
    if newly_visited_count < 0:
        raise ValueError("newly_visited_count cannot be negative.")
    if connection_lost:
        return -10.0
    reward = newly_visited_count * goal_reward_factor
    if success:
        reward += 10.0
    return float(reward)


class Agent:

    def __init__(self, initial_context, uav_id=None, **kwargs):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()

        if uav_id is None and initial_context.uavs:
            uav_id = initial_context.uavs[0].id
        self.uav_id = uav_id
        self.uav_ids = tuple(
            sorted(uav.id for uav in initial_context.uavs)
        )
        self.goal_ids = tuple(
            sorted(goal.id for goal in initial_context.goals)
        )

        input_size = len(self.get_state(initial_context))

        self.model = Linear_QNet(input_size, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_uav_goal_distance_matrix(self, context):
        uavs = [context.uav_by_id(uav_id) for uav_id in self.uav_ids]
        goals = [context.goal_by_id(goal_id) for goal_id in self.goal_ids]
        return build_uav_goal_distance_matrix(
            [uav.position for uav in uavs],
            [goal.position for goal in goals],
        )

    def get_state(self, context):
        uavs = [
            context.uav_by_id(uav_id)
            for uav_id in self.uav_ids
        ]
        goals = [
            context.goal_by_id(goal_id)
            for goal_id in self.goal_ids
        ]

        uav_positions = [uav.position for uav in uavs]
        goal_positions = [goal.position for goal in goals]

        drone_distances = build_distance_matrix(uav_positions)

        drone_goal_distances = build_uav_goal_distance_matrix(
            uav_positions,
            goal_positions,
        )

        network_positions = list(uav_positions)

        if context.ground is not None:
            network_positions.append(context.ground.position)

        network_distances = build_distance_matrix(network_positions)

        components = find_connected_components(
            network_distances,
            context.comm_thr,
            threshold_ratio=1.0,
        )

        controlled_index = self.uav_ids.index(self.uav_id)

        distances_to_others = np.delete(
            network_distances[controlled_index],
            controlled_index,
        )

        state = [
            # Ajanın kontrol ettiği drone
            *[
                float(uav_id == self.uav_id)
                for uav_id in self.uav_ids
            ],

            # Ağ kopuk mu?
            float(len(components) > 1),

            # Kontrol edilen drone'un bağlantısı sınır yakınında mı?
            float(np.any(
                (distances_to_others > context.comm_thr * 0.9)
                & (distances_to_others <= context.comm_thr)
            )),

            # Ground var mı ve konumu
            float(context.ground is not None),
            context.ground.position.x / context.comm_thr if context.ground is not None else 0.0,
            context.ground.position.y / context.comm_thr if context.ground is not None else 0.0,
        ]

        for uav in uavs:
            # Drone konumu ve hızı
            state.extend([
                uav.position.x / context.comm_thr,
                uav.position.y / context.comm_thr,
                uav.speed / context.comm_thr,
            ])

            # Drone'un mevcut görevi
            state.extend(
                float(uav.state == role)
                for role in (
                    "Free",
                    "Leader",
                    "Slave",
                    "Ground_Leader",
                    "Relay",
                )
            )

            # Atandığı hedef
            state.extend(
                float(uav.target_goal_id == goal_id)
                for goal_id in self.goal_ids
            )

            # Takip ettiği lider
            state.extend(
                float(uav.leader_id == leader_id)
                for leader_id in self.uav_ids
            )

            # Relay hedef konumu
            relay = uav.relay_target

            state.extend([
                float(relay is not None),
                relay.x / context.comm_thr if relay is not None else 0.0,
                relay.y / context.comm_thr if relay is not None else 0.0,
            ])

        for goal in goals:
            # Hedef konumu
            state.extend([
                goal.position.x / context.comm_thr,
                goal.position.y / context.comm_thr,
            ])

            # Hedefin mevcut durumu
            state.extend(
                float(goal.state == status)
                for status in ("Free", "Assigned", "Visited")
            )

        # Drone–drone uzaklıkları
        state.extend(
            (drone_distances / context.comm_thr).ravel()
        )

        # Drone–hedef uzaklıkları
        state.extend(
            (drone_goal_distances / context.comm_thr).ravel()
        )

        # Drone–ground uzaklıkları
        if context.ground is not None:
            state.extend(
                network_distances[:len(uavs), -1] / context.comm_thr
            )
        else:
            state.extend([0.0] * len(uavs))

        return np.array(state, dtype=np.float32)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if not self.memory:
            return

        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(list(self.memory), BATCH_SIZE)
        else:
            mini_sample = list(self.memory)

        states, actions, rewards, next_states, dones = zip(*mini_sample)

        self.trainer.train_step(
            states, actions, rewards, next_states, dones
        )

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(
            state, action, reward, next_state, done
        )

    def get_action(self, state):
        self.epsilon = max(0, 80 - self.n_games)

        # [1, 0, 0] → Leader
        # [0, 1, 0] → Slave
        # [0, 0, 1] → Relay
        final_action = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            action_index = random.randint(0, 2)
        else:
            state0 = torch.tensor(state, dtype=torch.float32)

            with torch.no_grad():
                prediction = self.model(state0)

            action_index = torch.argmax(prediction).item()

        final_action[action_index] = 1

        return final_action

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()