import torch
import torch.nn as nn


class DuelingDQNModel(nn.Module):
    """
    Dueling Deep Q-Network Model. Approximates a state-value function in a Q-Learning
    framework with a neural network.

    Dueling DQN is an extension of DQN that separates the learning of the state value
    and the advantages of each action, and then combines them to produce Q-values.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_1_size: int = 128,
        hidden_2_size: int = 128,
    ):
        super(DuelingDQNModel, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_1_size)
        self.fc2 = nn.Linear(hidden_1_size, hidden_2_size)
        self.value_stream = nn.Linear(hidden_2_size, 1)
        self.advantage_stream = nn.Linear(hidden_2_size, action_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Called with either one element to determine next action, or a batch
        during optimization.
        Args:
            state: The current state of the game.
        Returns:
            A tensor representing the chosen action.
        """
        action = torch.relu(self.fc1(state))
        action = torch.relu(self.fc2(action))
        value = self.value_stream(action)
        advantage = self.advantage_stream(action)
        return value + advantage - advantage.mean()
