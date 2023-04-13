from collections import deque
from collections import namedtuple

import os
import random
import math
from typing import List
import numpy as np

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
plt.ion()

# Named tuple for storing experience steps gathered in training
Experience = namedtuple(
    "Experience",
    field_names=["state", "action", "reward", "next_state", "done"],
)

WEIGHTS_DIR_NAME = "weights"
WEIGHTS_FILE_NAME = "weights.pt"
WEIGHTS_FILE = f"{WEIGHTS_DIR_NAME}/{WEIGHTS_FILE_NAME}"


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
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        return Experience(*zip(*(self.memory[idx] for idx in indices)))

    def __len__(self) -> None:
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
        during optimization. Returns tensor([[left0exp,right0exp]...]).
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
      epsilon_max (float): initial/max value of the exploration parameter
      epsilon_min (float): final/min value of the exploration parameter
      epsilon_decay (float): decay rate of the exploration parameter
      tau (float): update rate of the target network
      eps_num_update_target (int): update target network at every eps_num_update_target episode
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

      replay(batch_size: int) -> None:
        Performs a replay of the experiences stored in the memory buffer.

      train(env, episodes, batch_size, is_ipython):
        Trains the agent and network through all the episodes. We train the
        network on a batch_size number of states, rather than train it on the
        most recent one. Saves the weights and copies to the target network
        after each episode.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        capacity: int,
        weights_file: str,
        lr: float = 0.001,
        gamma: float = 0.99,
        epsilon_max: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: int = 1000,
        tau: float = 0.005,
        eps_num_update_target: int = 1,
        max_grad_norm: float = 100.0,
        hidden_1_size: int = 128,
        hidden_2_size: int = 128,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.capacity = capacity
        self.lr = lr
        self.gamma = gamma
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.eps_num_update_target = eps_num_update_target
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
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

    def take_action(self, env: gym.Env, state: torch.Tensor, eps_threshold: float) -> torch.Tensor:
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

    def store_experience(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_state: torch.Tensor, done: torch.Tensor):
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
        self.memory.push(Experience(state=state, action=action,
                         reward=reward, done=done, next_state=next_state))

    def save_network_weights(self) -> None:
        # Create a weights file if it doesn't exist.
        if not os.path.exists(self.weights_file):
            print(
                f"Weight file not found: {self.weights_file}. Creating it now.")
            with open(self.weights_file, "w") as _:
                pass
        torch.save(self.target_network.state_dict(), self.weights_file)

    def load_network_weights(self) -> None:
        try:
            # Load the state_dict from the weights file
            state_dict = torch.load(
                self.weights_file, map_location=self.device)

            # Map the state_dict keys to the current model's keys
            new_state_dict = {}
            if (state_dict.keys() == self.target_network.state_dict().keys()):
                new_state_dict = state_dict
            else:
                linear_one_key = "q_net._fc.0"
                linear_two_key = "q_net._fc.2"
                linear_three_key = "q_net._fc.4"
                for key, value in state_dict.items():
                    if linear_one_key in key:
                        new_state_dict[key.replace(
                            linear_one_key, "fc1")] = value
                    elif linear_two_key in key:
                        new_state_dict[key.replace(
                            linear_two_key, "fc2")] = value
                    elif linear_three_key in key:
                        new_state_dict[key.replace(
                            linear_three_key, "fc3")] = value

            # Load the mapped state_dict into the models
            strict = True
            self.network.load_state_dict(new_state_dict, strict=strict)
            self.target_network.load_state_dict(new_state_dict, strict=strict)
            print(f"Loaded weights from {self.weights_file}")
        except FileNotFoundError:
            print(
                f"No weights file found at {self.weights_file}, not loading any weights."
            )

    def update_target_network(self) -> None:
        """
        Soft update of the target network's weights. Weights are copied from
        the main network at a slower (target_tau) rate.

        θ′ ← τ θ + (1 −τ )θ′
        """
        network_sd = self.target_network.state_dict()
        target_network_sd = self.target_network.state_dict()
        for key in network_sd:
            target_network_sd[key] = network_sd[key] * self.tau + target_network_sd[
                key
            ] * (1 - self.tau)
        self.target_network.load_state_dict(target_network_sd)
        # Save the weights as well.
        self.save_network_weights()

    def replay(self, batch_size: int) -> None:
        """
        Performs a replay of the experiences stored in the memory buffer.

        Args:
            batch_size: The size of the mini-batch used for training the network.
        """
        if len(self.memory) < batch_size:
            return

        batch: Experience = self.memory.sample(batch_size)
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        next_state_batch = torch.cat([s for s in batch.next_state if s is not None]).to(
            self.device
        )

        # Compute the Q-values for the current state-action pairs.
        current_q_values = self.network(state_batch).gather(1, action_batch)

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )

        # Compute the Q-values for the next state-actions pairs.
        next_state_values = torch.zeros(batch_size).to(self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_network(
                next_state_batch
            ).max(1)[0]

        # Compute the expected Q-values using the Bellman equation.
        expected_q_values = reward_batch + (next_state_values * self.gamma)

        # Compute the loss between the predicted and expected Q-values.
        loss = self.loss_fn(current_q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm:
            nn.utils.clip_grad_norm_(
                self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()

    def train(self, env: gym.Env, episodes: int, batch_size: int, should_plot: bool, is_ipython: bool) -> None:
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
        steps_done = 0
        highest_score = 0
        scores = []
        episode_durations = []
        eps_threshold = self.epsilon_min
        for episode in range(episodes):
            done = False
            score = 0
            # the _ var is info
            state, _ = env.reset()
            state = torch.from_numpy(
                state).float().unsqueeze(0).to(self.device)

            # Run through the entire episode
            while not done:
                eps_threshold = self.epsilon_min + (
                    self.epsilon_max - self.epsilon_min
                ) * math.exp(-1.0 * steps_done / self.epsilon_decay)
                action = self.take_action(env, state, eps_threshold)
                # the _ var is info
                next_state, reward, done, truncated, _ = env.step(
                    action.item())
                reward = torch.tensor([reward]).to(self.device)
                done = done or truncated

                if done:
                    next_state = None
                else:
                    next_state = (
                        torch.from_numpy(next_state)
                        .float()
                        .unsqueeze(0)
                        .to(self.device)
                    )

                done = torch.tensor([done]).to(self.device)
                self.store_experience(
                    state=state,
                    action=action,
                    reward=reward,
                    done=done,
                    next_state=next_state,
                )
                state = next_state
                score += reward.item()
                # Train the agent on a batch of experiences.
                self.replay(batch_size)
                steps_done += 1

            # Save and update the target network's weights based off
            # the current network.
            if (episode % self.eps_num_update_target == 0):
                self.update_target_network()

            episode_durations.append(episode)
            scores.append(score)
            if should_plot:
                plot_durations(scores=scores, is_ipython=is_ipython)
            print(
                f"Episode: {episode+1}, Score: {score}, Epsilon: {eps_threshold:.2f}")
            if score > highest_score:
                highest_score = score
                print(
                    "\nNew high score: {scre} at episode {eps}\n".format(
                        scre=highest_score, eps=episode
                    )
                )

        if should_plot:
            # plot graph
            plot_durations(scores=scores, is_ipython=is_ipython,
                           show_result=True)
        plt.ioff()
        plt.show()

    def test_network(self, env: gym.Env, episodes: int) -> int:
        scores = []
        for episode in range(episodes):
            state, _ = env.reset()
            state_tensor = torch.from_numpy(
                state).float().unsqueeze(0).to(self.device)
            done = False
            score = 0
            for t_test in range(210):
                # Always use the network.
                eps_threshold = -1
                action = self.take_action(
                    env, state_tensor, eps_threshold).item()
                next_state, reward, done, truncated, info = env.step(action)
                done = done or truncated
                score += reward
                state = next_state
                if done or t_test == 209:
                    scores.append(score)
                    print(
                        "episode: {}/{}, score: {}, e: {}".format(
                            episode, episodes, score, 0
                        )
                    )
                    break


def plot_durations(scores: List[int], is_ipython: bool, show_result: bool = False) -> None:
    plt.figure(1)
    scores_t = torch.tensor(scores, dtype=torch.float)
    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training...")

    plt.xlabel("Episode")
    plt.ylabel("Score/Duration")

    # Plot score.
    (score_plot,) = plt.plot(scores_t, label="Score")

    # Take 100 episode averages and plot the avg score.
    if len(scores_t) >= 100:
        means = scores_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy(), label="Average score")

    # Show legends
    plt.legend(handles=[score_plot])

    # Pause a bit so that plots are updated.
    plt.pause(0.001)

    # Only if IPython environment.
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


@hydra.main(version_base="1.2", config_path=".", config_name="dqn_single_file")
def main(cfg: DictConfig):
    # Game: CartPole-v1.
    env = gym.make("CartPole-v1")

    # Get number of actions from gym action space.
    action_size = env.action_space.n

    # Get the number of state observations.
    state_size = env.observation_space.shape[0]

    # Setup DQN Agent with ReplayMemory.
    agent = DQNAgent(
        state_size,
        action_size,
        cfg.training.replay_mem_size,
        weights_file=WEIGHTS_FILE,
        lr=cfg.optimizer.lr,
        gamma=cfg.optimizer.gamma,
        epsilon_max=cfg.optimizer.epsilon_max,
        epsilon_min=cfg.optimizer.epsilon_min,
        epsilon_decay=cfg.optimizer.epsilon_decay,
        tau=cfg.optimizer.tau,
        eps_num_update_target=cfg.training.eps_num_update_target,
        max_grad_norm=cfg.optimizer.max_grad_norm,
        hidden_1_size=cfg.model.hidden_nodes_1,
        hidden_2_size=cfg.model.hidden_nodes_2,
    )

    # Train.
    if cfg.training.train:
        try:
            agent.train(
                env=env,
                episodes=cfg.training.episodes,
                batch_size=cfg.training.batch_size,
                should_plot=cfg.training.plot,
                is_ipython=is_ipython,
            )
        except KeyboardInterrupt:
            print("Got interrupted by user, stopping training.")

    # Validate/Test.
    if cfg.training.validate:
        agent.test_network(env=env, episodes=cfg.training.episodes)

    # Close our env, since we're done training.
    env.close()


if __name__ == "__main__":
    main()
