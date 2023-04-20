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
