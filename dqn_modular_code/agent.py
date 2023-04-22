from abc import ABCMeta, abstractmethod
from torch import Tensor


class Agent(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, device) -> None:
        """
        Args:
            device (torch.device): CUDA GPU or CPU.
        """
        self.device = device
        pass

    @abstractmethod
    def take_action(self) -> Tensor:
        """
        Agent making an action in the environment.

        Args: Up to implementation class.

        Returns: A tensor representing the chosen action.
        """
        raise NotImplementedError()

    @abstractmethod
    def learn(self) -> None:
        """
        Agent takes the attributes of the environment and tries to learn
        based on the experiences gained so far.

        Args: Up to implementation class.

        Returns: None.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_eps_threshold(self) -> float:
        """
        Agent return current epsilon value. If not using epsilon greedy then return 0.0.

        Returns: Float.
        """
        return 0.0
