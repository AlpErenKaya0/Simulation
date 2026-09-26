import torch
import random
import numpy as np
#from game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from collections import deque
from model import Linear_QNet, QTrainer, SarsaTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    def __init__(self, agent_id, algo_type="q"):
        """
        algo_type: "q" -> Q-learning, "sarsa" -> SARSA
        """
        self.agent_id = agent_id
        self.algo_type = algo_type
        self.n_games = 0
        self.epsilon = 0.01
        self.gamma = 0.9

        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 3)

        if self.algo_type == "q":
            self.trainer = QTrainer(self.model, LR, self.gamma)
        else:
            self.trainer = SarsaTrainer(self.model, LR, self.gamma)

    def get_state(self, game):
        # 3 yılan: snake1, snake2, snake3 ve direction1,2,3
        if self.agent_id == 1:
            head = game.snake1[0]
            direction = game.direction1
            snake_id = 1
        elif self.agent_id == 2:
            head = game.snake2[0]
            direction = game.direction2
            snake_id = 2
        else:
            head = game.snake3[0]
            direction = game.direction3
            snake_id = 3

        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = direction == Direction.LEFT
        dir_r = direction == Direction.RIGHT
        dir_u = direction == Direction.UP
        dir_d = direction == Direction.DOWN

        state = [
            # danger straight
            (dir_r and game.is_collision(point_r, snake_id)) or
            (dir_l and game.is_collision(point_l, snake_id)) or
            (dir_u and game.is_collision(point_u, snake_id)) or
            (dir_d and game.is_collision(point_d, snake_id)),

            # danger right
            (dir_u and game.is_collision(point_r, snake_id)) or
            (dir_d and game.is_collision(point_l, snake_id)) or
            (dir_l and game.is_collision(point_u, snake_id)) or
            (dir_r and game.is_collision(point_d, snake_id)),

            # danger left
            (dir_d and game.is_collision(point_r, snake_id)) or
            (dir_u and game.is_collision(point_l, snake_id)) or
            (dir_r and game.is_collision(point_u, snake_id)) or
            (dir_l and game.is_collision(point_d, snake_id)),

            # move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # food location
            game.food.x < head.x,
            game.food.x > head.x,
            game.food.y < head.y,
            game.food.y > head.y,
        ]

        return np.array(state, dtype=int)

    # --- Bellek ve eğitim ---

    def remember_q(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def remember_sarsa(self, state, action, reward, next_state, next_action, done):
        self.memory.append((state, action, reward, next_state, next_action, done))

    def train_long_memory(self):
        # BOŞ BELLEK KORUMASI
        if len(self.memory) == 0:
            return

        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = list(self.memory)

        if self.algo_type == "q":
            states, actions, rewards, next_states, dones = zip(*mini_sample)
            self.trainer.train_step(states, actions, rewards, next_states, dones)
        else:
            states, actions, rewards, next_states, next_actions, dones = zip(*mini_sample)
            self.trainer.train_step(states, actions, rewards, next_states, next_actions, dones)

    def train_short_memory_q(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory_sarsa(self, state, action, reward, next_state, next_action, done):
        self.trainer.train_step(state, action, reward, next_state, next_action, done)

    # --- Aksiyon seçimi ---

    def get_action(self, state):
        self.epsilon = max(0.01, 80 - self.n_games)
        final_move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # uyarıyı azaltmak için state'i np.array'e çeviriyoruz
            state0 = torch.tensor(np.array(state), dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    # Agent 1: Q-learning normal ajan
    agent1 = Agent(agent_id=1, algo_type="q")
    # Agent 2: SARSA normal ajan
    agent2 = Agent(agent_id=2, algo_type="sarsa")
    # Agent 3: SARSA killer ajan
    agent3 = Agent(agent_id=3, algo_type="sarsa")

    game = SnakeGameAI()

    plot_scores_1, plot_mean_scores_1 = [], []
    plot_scores_2, plot_mean_scores_2 = [], []
    plot_scores_3, plot_mean_scores_3 = [], []

    total_score_1 = total_score_2 = total_score_3 = 0
    record_1 = record_2 = record_3 = 0

    # Maksimum oyun sayısı
    MAX_GAMES = 10000

    while agent1.n_games < MAX_GAMES:
        # --- Eski state'ler ---
        state_old_1 = agent1.get_state(game)
        state_old_2 = agent2.get_state(game)
        state_old_3 = agent3.get_state(game)

        # --- Aksiyon seçimi ---
        action_1 = agent1.get_action(state_old_1)
        action_2 = agent2.get_action(state_old_2)
        action_3 = agent3.get_action(state_old_3)

        # Oyun adımı
        reward1, reward2, reward3, done, score1, score2, score3 = game.play_step(
            action_1, action_2, action_3
        )

        # --- Yeni state'ler ---
        state_new_1 = agent1.get_state(game)
        state_new_2 = agent2.get_state(game)
        state_new_3 = agent3.get_state(game)

        # --- SARSA için next_action ---
        next_action_2 = agent2.get_action(state_new_2)
        next_action_3 = agent3.get_action(state_new_3)

        # --- Kısa bellek eğitimi ---
        agent1.train_short_memory_q(state_old_1, action_1, reward1, state_new_1, done)
        agent1.remember_q(state_old_1, action_1, reward1, state_new_1, done)

        agent2.train_short_memory_sarsa(state_old_2, action_2, reward2, state_new_2, next_action_2, done)
        agent2.remember_sarsa(state_old_2, action_2, reward2, state_new_2, next_action_2, done)

        agent3.train_short_memory_sarsa(state_old_3, action_3, reward3, state_new_3, next_action_3, done)
        agent3.remember_sarsa(state_old_3, action_3, reward3, state_new_3, next_action_3, done)

        if done:
            game.reset()

            agent1.n_games += 1
            agent2.n_games += 1
            agent3.n_games += 1

            agent1.train_long_memory()
            agent2.train_long_memory()
            agent3.train_long_memory()

            if score1 > record_1:
                record_1 = score1
                agent1.model.save('model_agent1_q.pth')

            if score2 > record_2:
                record_2 = score2
                agent2.model.save('model_agent2_sarsa.pth')

            if score3 > record_3:
                record_3 = score3
                agent3.model.save('model_agent3_sarsa_killer.pth')

            print(
                'Game', agent1.n_games,
                'Score1', score1, 'Record1:', record_1,
                'Score2', score2, 'Record2:', record_2,
                'Score3', score3, 'Record3:', record_3,
            )

            plot_scores_1.append(score1)
            total_score_1 += score1
            mean_score_1 = total_score_1 / agent1.n_games
            plot_mean_scores_1.append(mean_score_1)

            plot_scores_2.append(score2)
            total_score_2 += score2
            mean_score_2 = total_score_2 / agent2.n_games
            plot_mean_scores_2.append(mean_score_2)

            plot_scores_3.append(score3)
            total_score_3 += score3
            mean_score_3 = total_score_3 / agent3.n_games
            plot_mean_scores_3.append(mean_score_3)

            plot(
                plot_scores_1, plot_mean_scores_1,
                plot_scores_2, plot_mean_scores_2,
                plot_scores_3, plot_mean_scores_3,
            )

    print("Training finished after", agent1.n_games, "games.")


if __name__ == '__main__':
    train()