import random
from collections import deque

from dqn_modular_code.util import Experience


class ReplayMemory(object):
    """
    A cyclic buffer of bounded size that holds the experiences observed recently.

    Methods:
    push: Adds a new experience to the memory.
    sample: Retrieves several random experiences from the memory.
    """

    def __init__(self, capacity: int) -> None:
        self.memory = deque([], maxlen=capacity)

    def push(self, experience: Experience) -> None:
        """
        Save an experience. The deque will automatically remove items when
        full, so no need to check capacity.
        Args:
          experience: Experience. Tuple of arguments that should fit into an experience.
        """
        self.memory.append(experience)

    def sample(self, batch_size: int) -> Experience:
        """
        Retrieve batch_size number of random experiences from the memory.
        Args:
          batch_size: Number of experiences to retrieve
        """
        x = random.sample(self.memory, batch_size)
        return Experience(*zip(*(x)))

    def __len__(self) -> int:
        return len(self.memory)
