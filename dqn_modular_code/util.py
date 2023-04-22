from enum import Enum
from collections import namedtuple


# Named tuple for storing experience steps gathered in training
Experience = namedtuple(
    "Experience",
    field_names=["state", "action", "reward", "next_state", "done"],
)


class MetricsEnum(str, Enum):
    DurationsMetric = "Duration"
    RewardsMetric = "Rewards"

    def __str__(self) -> str:
        return self.value


class AgentsEnum(str, Enum):
    DQNAgent = "DQNAgent"

    def __str__(self) -> str:
        return self.value


class AgentConfig:
    def __init__(
        self,
        hidden_1_size: int = 128,
        hidden_2_size: int = 128,
        max_grad_norm: float = 100.0,
        gamma: float = 0.99,
        tau: float = 0.005,
        lr: float = 0.001,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: int = 1000,
        num_update_target: int = 1,
        num_save_weights: int = 50,
        batch_size: int = 128,
        replay_mem_size: int = 10_000,
    ):
        self.hidden_nodes_1: int = hidden_1_size
        self.hidden_nodes_2: int = hidden_2_size
        self.max_grad_norm: int = max_grad_norm
        self.gamma: float = gamma
        self.tau: float = tau
        self.lr: float = lr
        self.epsilon_start: float = epsilon_start
        self.epsilon_min: float = epsilon_min
        self.epsilon_decay: int = epsilon_decay
        self.num_update_target: int = num_update_target
        self.num_save_weights: int = num_save_weights
        self.batch_size: int = batch_size
        self.replay_mem_size: int = replay_mem_size
