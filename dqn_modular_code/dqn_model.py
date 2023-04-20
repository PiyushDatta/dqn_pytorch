import torch
import torch.nn as nn


class DQNModel(nn.Module):
    """
    Deep Q-Network Model. Approximates a state-value function in a Q-Learning
    framework with a neural network.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_1_size: int = 128,
        hidden_2_size: int = 128,
    ):
        super(DQNModel, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_1_size)
        self.fc2 = nn.Linear(hidden_1_size, hidden_2_size)
        self.fc3 = nn.Linear(hidden_2_size, action_size)

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
        action = self.fc3(action)
        return action
