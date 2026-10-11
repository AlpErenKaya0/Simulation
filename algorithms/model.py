from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
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


from simulation_core.algorithm_api import AlgorithmPlan, SimulationAlgorithm, SimulationContext
from simulation_core.algorithm_api import UavView, Position


class DroneDQNAlgorithm(SimulationAlgorithm):
    name = "drone_dqn"
    display_name = "Drone DQN"
    description = "DQN tabanli pekistirmeli ogrenme algoritmasi."

    def __init__(self) -> None:
        self._training = False
        self._agents: dict = {}
        # Gercek DQN implement edilene kadar mevcut baglanti algoritmasini kullan
        from algorithms.current import ConnectivityAwareAlgorithm
        self._delegate = ConnectivityAwareAlgorithm()

    def enable_training(self, enabled: bool) -> None:
        self._training = enabled
        if not enabled:
            self._agents.clear()

    def reset(self) -> None:
        self._agents.clear()
        self._delegate.reset()

    def build_plan(self, context: SimulationContext) -> AlgorithmPlan:
        # DQN mantigi ilerleyen asamada buraya eklenecek.
        # Simdilik ConnectivityAwareAlgorithm mantigi ile dronelar hareket eder.
        return self._delegate.build_plan(context)

    def allow_movement(
        self,
        context: SimulationContext,
        uav: UavView,
        target_position: Position,
    ) -> bool:
        return self._delegate.allow_movement(context, uav, target_position)