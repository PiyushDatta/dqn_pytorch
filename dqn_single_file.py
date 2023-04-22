import os
import random
import math
import logging
from enum import Enum
from typing import List
from tqdm import tqdm
from collections import deque
from collections import namedtuple
from itertools import count

import matplotlib
import matplotlib.pyplot as plt

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

import hydra
from omegaconf import DictConfig

# Set up matplotlib and check if we're in an IPython environment.
is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display
matplotlib.use("TkAgg")
plt.ion()

# Named tuple for storing experience steps gathered in training
Experience = namedtuple(
    "Experience",
    field_names=["state", "action", "reward", "next_state", "done"],
)

WEIGHTS_FILE_NAME = "weights/weights.pt"


class MetricsEnum(str, Enum):
    DurationsMetric = "Duration"
    RewardsMetric = "Rewards"

    def __str__(self) -> str:
        return self.value


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


class DQNAgent:
    """
        Deep Q-Network (DQN) agent that uses a neural network to approximate Q-values
        and trains the network using experience replay and a target network.

        Args:
          state_size (int): size of the state space
          action_size (int): size of the action space
          capacity (int): maximum size of the replay memory
          weights_file (str): path to the file where the weights will be saved/loaded
          lr (float): learning rate of the optimizer
          gamma (float): discount factor used in the Bellman equation
          epsilon_start (float): initial/max value of the exploration parameter
          epsilon_min (float): final/min value of the exploration parameter
          epsilon_decay (float): decay rate of the exploration parameter
          tau (float): update rate of the target network
          num_update_target (int): update target network at every num_update_target episode
          num_save_weights (int): save target network weights at every num_save_weights episode
          max_grad_norm (float): maximum norm of the gradients used to clip the gradients
          hidden_1_size (int): number of neuron nodes in first hidden layer
          hidden_2_size (int): number of neuron nodes in second hidden layer

    Methods:
          take_action(env, state, eps_threshold) -> torch.Tensor:
            Choose an action using the epsilon-greedy policy.

          store_experience(state, action, reward, next_state, done) -> None:
            Stores the experience (state, action, reward, next_state, done) in the
            memory buffer.

          save_network_weights() -> None:
            Saves the weights of the target network to the specified file.

          load_network_weights() -> None:
            Loads the weights of the current and target networks from the specified file.

          update_target_network() -> None:
            Updates the weights of the target network by copying the weights of
            the main network.

          optimize_model(batch_size: int) -> None:
            Performs a replay of the experiences stored in the memory buffer and
            optimizes the model based of this batch of experiences.

          train(self, env: gym.Env, episodes: int, batch_size: int, should_plot: bool, is_ipython: bool) -> None:
            Trains the agent and network through all the episodes. We train the
            network on a batch_size number of states, rather than train it on the
            most recent one. Saves the weights and copies to the target network
            after each episode.

          validate(self, env: gym.Env, episodes: int, should_plot: bool, is_ipython: bool) -> int:
            Validate the agent and network against the set episodes.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        capacity: int,
        weights_file: str,
        lr: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: int = 1000,
        tau: float = 0.005,
        num_update_target: int = 1,
        num_save_weights: int = 50,
        max_grad_norm: float = 100.0,
        hidden_1_size: int = 128,
        hidden_2_size: int = 128,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.capacity = capacity
        self.lr = lr
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.num_update_target = num_update_target
        self.num_save_weights = num_save_weights
        self.max_grad_norm = max_grad_norm
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = DQNModel(
            state_size,
            action_size,
            hidden_1_size=hidden_1_size,
            hidden_2_size=hidden_2_size,
        ).to(self.device)
        self.target_network = DQNModel(
            state_size,
            action_size,
            hidden_1_size=hidden_1_size,
            hidden_2_size=hidden_2_size,
        ).to(self.device)
        self.weights_file = weights_file
        self.load_network_weights()
        self.optimizer = optim.AdamW(
            self.network.parameters(), lr=self.lr, amsgrad=True
        )
        self.loss_fn = nn.MSELoss()
        self.memory = ReplayMemory(self.capacity)
        self.metric_log = {
            MetricsEnum.DurationsMetric: [],
            MetricsEnum.RewardsMetric: [],
        }
        self.steps_done = 0

    def take_action(
        self, env: gym.Env, state: torch.Tensor, eps_threshold: float
    ) -> torch.Tensor:
        """
        Choose an action using the epsilon-greedy policy.

        Args:
            env: The game environment.
            state: The current state of the game.
            eps_threshold: The exploration rate parameter used for the
                           epsilon-greedy policy.

        Returns: A tensor representing the chosen action.
        """
        # Have the network pick an action.
        if random.random() > eps_threshold:
            with torch.no_grad():
                return self.network(state).max(1)[1].view(1, 1)

        # Choose a random action.
        return torch.tensor([[env.action_space.sample()]]).long().to(self.device)

    def store_experience(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ):
        """
        Stores the experience (state, action, reward, next_state, done) in the
        memory buffer.

        Args:
            state: The current state of the game.
            action: The action taken at the current state.
            reward: The reward obtained for taking the action.
            next_state: The next state of the game.
            done: A flag indicating whether the episode is finished or not.
        """
        self.memory.push(
            Experience(
                state=state,
                action=action,
                reward=reward,
                done=done,
                next_state=next_state,
            )
        )

    def save_network_weights(self) -> None:
        # Create a weights file if it doesn't exist.
        if not os.path.exists(self.weights_file):
            logging.info(
                f"Weight file not found: {self.weights_file}. Creating it now."
            )
            with open(self.weights_file, "w") as _:
                pass
        torch.save(self.target_network.state_dict(), self.weights_file)

    def load_network_weights(self) -> None:
        try:
            # Load the state_dict from the weights file.
            state_dict = torch.load(self.weights_file, map_location=self.device)

            # Map the state_dict keys to the current model's keys.
            new_state_dict = {}
            if state_dict.keys() == self.target_network.state_dict().keys():
                new_state_dict = state_dict
            else:
                # Transfer learning.
                # If we ever need to load weights trained somewhere else.
                linear_one_key = "q_net._fc.0"
                linear_two_key = "q_net._fc.2"
                linear_three_key = "q_net._fc.4"
                for key, value in state_dict.items():
                    if linear_one_key in key:
                        new_state_dict[key.replace(linear_one_key, "fc1")] = value
                    elif linear_two_key in key:
                        new_state_dict[key.replace(linear_two_key, "fc2")] = value
                    elif linear_three_key in key:
                        new_state_dict[key.replace(linear_three_key, "fc3")] = value

            # Load the mapped state_dict into the models
            strict = True
            self.network.load_state_dict(new_state_dict, strict=strict)
            self.target_network.load_state_dict(new_state_dict, strict=strict)
            logging.info(f"Loaded weights from {self.weights_file}")
        except FileNotFoundError:
            logging.info(
                f"No weights file found at {self.weights_file}, not loading any weights."
            )

    def update_target_network(self) -> None:
        """
        Soft update of the target network's weights. Weights are copied from
        the main network at a slower (target_tau) rate.

        θ′ ← τ θ + (1 −τ )θ′
        """
        target_network_state_dict = self.target_network.state_dict()
        network_state_dict = self.network.state_dict()
        for key in network_state_dict:
            target_network_state_dict[key] = network_state_dict[
                key
            ] * self.tau + target_network_state_dict[key] * (1 - self.tau)
        self.target_network.load_state_dict(target_network_state_dict)

    def optimize_model(self, batch_size: int) -> None:
        """
        Performs a replay of the experiences stored in the memory buffer.

        Args:
            batch_size: The size of the mini-batch used for training the network.
        """
        if len(self.memory) < batch_size:
            return

        batch: Experience = self.memory.sample(batch_size)

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        next_state_batch = torch.cat([s for s in batch.next_state if s is not None]).to(
            self.device
        )
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)

        # Compute the Q-values for the current state-action pairs.
        current_q_values = self.network(state_batch).gather(1, action_batch)

        # Compute the Q-values for the next state-actions pairs.
        next_state_values = torch.zeros(batch_size).to(self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_network(
                next_state_batch
            ).max(1)[0]

        # Compute the expected Q-values using the Bellman equation.
        expected_q_values = reward_batch + (next_state_values * self.gamma)

        # Compute the loss between the predicted and expected Q-values.
        loss = self.loss_fn(current_q_values, expected_q_values.unsqueeze(1))

        # Reset the gradients weights & biases before back propagation.
        self.optimizer.zero_grad()

        # Calculate the gradients of the loss.
        loss.backward()

        # In-place gradient clipping.
        nn.utils.clip_grad_value_(self.network.parameters(), self.max_grad_norm)

        # Update the network with the gradients.
        self.optimizer.step()

    def train(
        self,
        env: gym.Env,
        episodes: int,
        batch_size: int,
        should_plot: bool,
        is_ipython: bool,
    ) -> None:
        """
        Trains the agent and network through all the episodes. We train the
        network on a batch_size number of states, rather than train it on the
        most recent one. Saves the weights and copies to the target network
        after each episode.

        Args:
          env: The game environment.
          episodes: The number of episodes we play through.
          batch_size: The size of the mini-batch used for training the network.
          should_plot: Use matplotlib to plot the graph.
          is_ipython: True, if its IPython environment.
        """
        # Clear metric log before starting.
        for key in self.metric_log:
            self.metric_log[key] = []

        highest_reward = 0
        for episode in tqdm(range(episodes)):
            total_reward = 0
            # Initialize the environment and get it's state.
            state, _ = env.reset()
            state = torch.tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            # Keep interacting with the environment until done.
            for t in count():
                # Take an action.
                eps_threshold = self.epsilon_min + (
                    self.epsilon_start - self.epsilon_min
                ) * math.exp(-1.0 * self.steps_done / self.epsilon_decay)
                self.steps_done += 1
                action = self.take_action(env, state, eps_threshold=eps_threshold)

                # Environment step.
                observation, reward, terminated, truncated, _ = env.step(action.item())

                total_reward += reward
                reward = torch.tensor([reward], device=self.device)

                done = terminated or truncated
                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(
                        observation, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)

                # Store the transition in memory.
                self.store_experience(
                    state=state,
                    action=action,
                    reward=reward,
                    done=done,
                    next_state=next_state,
                )

                # Move to the next state.
                state = next_state

                # Perform one step of the optimization (on the network).
                self.optimize_model(batch_size)

                # Update the target network's weights based off the current network.
                if self.steps_done % self.num_update_target == 0:
                    self.update_target_network()

                # Save the target network's weights.
                if self.steps_done % self.num_save_weights == 0:
                    self.save_network_weights()

                # Episode finished.
                if done:
                    self.metric_log[MetricsEnum.DurationsMetric].append(t + 1)
                    self.metric_log[MetricsEnum.RewardsMetric].append(total_reward)
                    logging.debug(
                        f"Episode: {episode+1}, Score: {total_reward}, Epsilon: {eps_threshold:.2f}"
                    )
                    if should_plot:
                        plot_graph(
                            scores=self.metric_log[MetricsEnum.DurationsMetric],
                            is_ipython=is_ipython,
                        )
                    if total_reward > highest_reward:
                        highest_reward = total_reward
                        logging.info(
                            "New high score: {scre} at episode {eps} with epsilon {epsilon}".format(
                                scre=highest_reward, eps=episode, epsilon=eps_threshold
                            )
                        )
                    break

        if should_plot:
            # plot graph
            plot_graph(
                scores=self.metric_log[MetricsEnum.DurationsMetric],
                is_ipython=is_ipython,
                show_result=True,
            )
        plt.ioff()
        plt.show()

    def validate(
        self, env: gym.Env, episodes: int, should_plot: bool, is_ipython: bool
    ) -> int:
        """
        Validate the agent and network against the set episodes.

        Args:
          env: The game environment.
          episodes: The number of episodes we play through.
          should_plot: Use matplotlib to plot the graph.
          is_ipython: True, if its IPython environment.
        """

        # Clear metric log before starting.
        for key in self.metric_log:
            self.metric_log[key] = []
        highest_reward = 0
        for episode in tqdm(range(episodes)):
            total_reward = 0
            # Initialize the environment and get it's state
            state, _ = env.reset()
            state = torch.tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            for t in count():
                # Always have the network take an action
                action = self.take_action(env, state, eps_threshold=-1)

                # Environment step
                observation, reward, terminated, truncated, _ = env.step(action.item())

                total_reward += reward
                done = terminated or truncated
                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(
                        observation, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)

                # Move to the next state
                state = next_state

                if done:
                    self.metric_log[MetricsEnum.DurationsMetric].append(t + 1)
                    self.metric_log[MetricsEnum.RewardsMetric].append(total_reward)
                    logging.debug(f"Episode: {episode+1}, Score: {total_reward}")
                    if should_plot:
                        plot_graph(
                            scores=self.metric_log[MetricsEnum.DurationsMetric],
                            is_ipython=is_ipython,
                        )
                    if total_reward > highest_reward:
                        highest_reward = total_reward
                        logging.info(
                            "New high score: {scre} at episode {eps}".format(
                                scre=highest_reward, eps=episode
                            )
                        )
                    break

        self.steps_done = 0
        if should_plot:
            # plot graph
            plot_graph(
                scores=self.metric_log[MetricsEnum.DurationsMetric],
                is_ipython=is_ipython,
                show_result=True,
            )
        plt.ioff()
        plt.show()


def plot_graph(scores: List[int], is_ipython: bool, show_result: bool = False) -> None:
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


def generate_env(env_name: str) -> gym.Env:
    """
    Generates the specified environment
    """
    if env_name == "CartPole-v1":
        return gym.make("CartPole-v1")
    else:
        raise ValueError("Unsupported environment: {env}".format(env=env_name))


@hydra.main(version_base="1.2", config_path="configs", config_name="dqn")
def main(cfg: DictConfig):
    # Setup logging.
    logging.getLogger().setLevel(level=logging.getLevelName(str(cfg.env.logging_level)))

    # Game: Gym Environment.
    env = generate_env(str(cfg.env.env_name))

    # Get number of actions from gym action space.
    action_size = env.action_space.n

    # Get the number of state observations.
    state_size = env.observation_space.shape[0]

    # Setup DQN Agent with ReplayMemory.
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        capacity=cfg.agent.replay_mem_size,
        weights_file=WEIGHTS_FILE_NAME,
        lr=cfg.agent.lr,
        gamma=cfg.agent.gamma,
        epsilon_start=cfg.agent.epsilon_start,
        epsilon_min=cfg.agent.epsilon_min,
        epsilon_decay=cfg.agent.epsilon_decay,
        tau=cfg.agent.tau,
        num_update_target=cfg.env.num_update_target,
        num_save_weights=cfg.env.num_save_weights,
        max_grad_norm=cfg.agent.max_grad_norm,
        hidden_1_size=cfg.neural_net.hidden_nodes_1,
        hidden_2_size=cfg.neural_net.hidden_nodes_2,
    )

    # Train.
    if cfg.env.train:
        logging.info(
            "Starting training for {eps} episodes.".format(
                eps=cfg.env.training_episodes
            )
        )
        try:
            agent.train(
                env=env,
                episodes=cfg.env.training_episodes,
                batch_size=cfg.env.batch_size,
                should_plot=cfg.env.plot_training,
                is_ipython=is_ipython,
            )
            logging.info("Finished training!\n")
        except KeyboardInterrupt:
            logging.error("Got interrupted by user, stopping training.")

    # Validate/Test.
    if cfg.env.validate:
        logging.info(
            "Starting validation for {eps} episodes.".format(
                eps=cfg.env.validating_episodes
            )
        )
        try:
            agent.validate(
                env=env,
                episodes=cfg.env.validating_episodes,
                should_plot=cfg.env.plot_validation,
                is_ipython=is_ipython,
            )
            logging.info("Finished validating!\n")
        except KeyboardInterrupt:
            logging.error("Got interrupted by user, stopping validation.")

    # Close our env, since we're done training/validating.
    env.close()


if __name__ == "__main__":
    main()
