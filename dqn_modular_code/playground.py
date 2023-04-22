import logging
from typing import List
from tqdm import tqdm
from itertools import count

import matplotlib
import matplotlib.pyplot as plt

import gymnasium as gym
import torch

from dqn_modular_code.agent import Agent
from dqn_modular_code.util import AgentConfig, AgentsEnum, MetricsEnum
from dqn_modular_code.dqn.dqn_agent import DQNAgent
from dqn_modular_code.dueling_dqn.dueling_dqn_agent import DuelingDQNAgent

# Set up matplotlib and check if we're in an IPython environment.
is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display
matplotlib.use("TkAgg")
plt.ion()


class Playground:
    """
    Playground is an entity that will create agent(s) and model(s) and have
    them interact with the selected environment.

    Args:
        gym_env (gym.Env): Gym environment.
        weights_file (str): Path to the file where the weights will be saved/loaded.
        agent_type (AgentsEnum): Type of agent we should create.
        config (AgentConfig): Configuration/Hyperparameters of the agent and model.

    Methods:
        train(self, episodes: int, should_plot: bool) -> None:
        Trains the agent and network through all the episodes.

        validate(self, episodes: int, should_plot: bool) -> None:
        Validate the agent and network against the set episodes.

        clear_metrics(self) -> None:
        Clear metrics before starting.
    """

    def __init__(
        self,
        gym_env: gym.Env,
        weights_file: str,
        agent_type: AgentsEnum,
        config: AgentConfig,
    ):
        self.env = gym_env
        # Number of state observations.
        self.state_size = self.env.observation_space.shape[0]
        # Number of actions from gym action space.
        self.action_size = self.env.action_space.n
        self.weights_file = weights_file
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent: Agent = self._get_agent(
            agent_type=agent_type,
            state_size=self.state_size,
            action_size=self.action_size,
            weights_file=weights_file,
            config=config,
        )
        # Metrics
        self.metric_log = {
            MetricsEnum.DurationsMetric: [],
            MetricsEnum.RewardsMetric: [],
        }

    def train(self, episodes: int, should_plot: bool) -> None:
        """
        Trains the agent and network through all the episodes. We train the
        network on a batch_size number of states, rather than train it on the
        most recent one. Saves the weights and copies to the target network
        after each episode.

        Args:
          episodes: The number of episodes we play through.
          should_plot: Use matplotlib to plot the graph.
        """
        self._play_episodes(episodes=episodes, should_plot=should_plot, train=True)

    def validate(self, episodes: int, should_plot: bool) -> None:
        """
        Validate the agent and network against the set episodes.

        Args:
          episodes: The number of episodes we play through.
          should_plot: Use matplotlib to plot the graph.
        """
        self._play_episodes(episodes=episodes, should_plot=should_plot, train=False)

    def clear_metrics(self) -> None:
        """
        Clear metrics before starting.
        """
        for key in self.metric_log:
            self.metric_log[key] = []

    def _get_agent(
        self,
        agent_type: AgentsEnum,
        state_size: int,
        action_size: int,
        weights_file: str,
        config: AgentConfig,
    ) -> Agent:
        """
        Returns an deep learning agent.
        """
        if agent_type == AgentsEnum.DQN_AGENT:
            return DQNAgent(
                device=self.device,
                state_size=state_size,
                action_size=action_size,
                weights_file=weights_file,
                config=config,
            )
        elif agent_type == AgentsEnum.DUELING_DQN_AGENT:
            return DuelingDQNAgent(
                device=self.device,
                state_size=state_size,
                action_size=action_size,
                weights_file=weights_file,
                config=config,
            )
        raise ValueError(f"No agent of type {agent_type} found")

    def _play_episodes(self, episodes: int, should_plot: bool, train: bool) -> None:
        """
        Have the agent play through x amount of episodes in the environment.

        Args:
          episodes: The number of episodes we play through.
          should_plot: Use matplotlib to plot the graph.
          train: Whether the agent in training or not.
        """
        self.steps_done = 0
        highest_reward = 0
        for episode in tqdm(range(episodes)):
            total_reward = 0
            # Initialize the environment and get it's state.
            state, _ = self.env.reset()
            state = torch.tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            # Keep interacting with the environment until done.
            for t in count():
                # Have the agent take an action.
                action: torch.Tensor = self.agent.take_action(
                    self.env, state, steps_done=self.steps_done, train=train
                )

                # Environment step.
                observation, reward, terminated, truncated, _ = self.env.step(
                    action.item()
                )

                self.steps_done += 1
                total_reward += reward
                reward = torch.tensor([reward], device=self.device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(
                        observation, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)

                # If training, have the agent learn from this experience.
                if train:
                    self.agent.learn(
                        state=state,
                        action=action,
                        reward=reward,
                        done=done,
                        next_state=next_state,
                        steps_done=self.steps_done,
                    )

                # Move to the next state.
                state = next_state

                # Episode finished.
                if done:
                    # Records metrics.
                    self.metric_log[MetricsEnum.DurationsMetric].append(t + 1)
                    self.metric_log[MetricsEnum.RewardsMetric].append(total_reward)
                    logging.debug(
                        "Episode: {eps}, Score: {scre}, Epsilon: {epsilon}".format(
                            scre=highest_reward,
                            eps=episode,
                            epsilon=self.agent.get_eps_threshold(),
                        )
                    )
                    # Plot current results.
                    if should_plot:
                        self._plot_graph(
                            scores=self.metric_log[MetricsEnum.DurationsMetric],
                            is_ipython=is_ipython,
                        )
                    if total_reward > highest_reward:
                        highest_reward = total_reward
                        logging.info(
                            "New high score: {scre} at episode {eps} with epsilon {epsilon}".format(
                                scre=highest_reward,
                                eps=episode,
                                epsilon=self.agent.get_eps_threshold(),
                            )
                        )
                    break

        # Plot final results.
        if should_plot:
            self._plot_graph(
                scores=self.metric_log[MetricsEnum.DurationsMetric],
                is_ipython=is_ipython,
                show_result=True,
            )
        plt.ioff()
        plt.show()

    def _plot_graph(
        self, scores: List[int], is_ipython: bool, show_result: bool = False
    ) -> None:
        """
        Plot the matplotlib graph of our results/scores.
        """
        plt.figure(1)
        scores_t = torch.tensor(scores, dtype=torch.float)
        if show_result:
            plt.title("Result")
        else:
            plt.clf()
            plt.title("Running...")

        plt.xlabel("Episode")
        plt.grid(True)

        # Plot score.
        (score_plot,) = plt.plot(scores_t.numpy(), label="Score", color="r")

        # Show legends.
        plt.legend(handles=[score_plot])

        # Take 100 episode averages and plot the avg score.
        if len(scores_t) >= 100:
            means = scores_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy(), label="Average score")

        # Pause a bit so that plots are updated.
        plt.pause(0.001)

        # Only if IPython environment.
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())
