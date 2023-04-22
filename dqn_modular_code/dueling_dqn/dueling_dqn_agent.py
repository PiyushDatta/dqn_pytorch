import logging
import math
import os
import random
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

from dqn_modular_code.agent import Agent
from dqn_modular_code.dueling_dqn.dueling_dqn_model import DuelingDQNModel
from dqn_modular_code.replay_memory import ReplayMemory
from dqn_modular_code.util import AgentConfig, Experience


# Note we could just use DQNAgent as base class, but want to test if this
# codebase is modular and clean.
class DuelingDQNAgent(Agent):
    """
    Dueling Deep Q-Network (Dueling DQN) agent that uses a neural network to
    approximate Q-values and trains the network using experience replay and
    a target network.

    Args:
        state_size (int): Size of the state space.
        action_size (int): Size of the action space.
        weights_file (str): Path to the file where the weights will be saved/loaded.
        config (AgentConfig): Configuration/Hyperparameters of the agent and model.
        device (torch.device): CUDA GPU or CPU.


    Methods:
        take_action(
            env: gym.Env, state: torch.Tensor, steps_done: int, train: bool
        ) -> torch.Tensor:
        Choose an action using the epsilon-greedy policy.

        learn(self, state, action, reward, done, next_state, steps_done) -> None:
        DQN Agent learning. Store the transition/step of the environment in our memory.
        If we have atleast batch_size amount of transitions, then we can have the DQN
        learn and train. Trains the agent and network through all the episodes.
        We train the network on a batch_size number of states, rather than train it
        on the most recent one. Saves the weights and copies to the target network.

        get_eps_threshold(self) -> float:
        Agent return current epsilon value. If not using epsilon greedy then return 0.0.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        weights_file: str,
        config: AgentConfig,
        device: torch.device,
    ):
        super().__init__(device=device)
        self.state_size = state_size
        self.action_size = action_size
        self.weights_file = weights_file
        self.config = config
        self.memory = ReplayMemory(config.replay_mem_size)

        # Policy network
        self.network = DuelingDQNModel(
            state_size,
            action_size,
            hidden_1_size=config.hidden_nodes_1,
            hidden_2_size=config.hidden_nodes_2,
        ).to(self.device)

        # Target network
        self.target_network = DuelingDQNModel(
            state_size,
            action_size,
            hidden_1_size=config.hidden_nodes_1,
            hidden_2_size=config.hidden_nodes_2,
        ).to(self.device)

        self._load_network_weights()
        self.optimizer = optim.AdamW(
            self.network.parameters(), lr=self.config.lr, amsgrad=True
        )
        self.loss_fn = nn.MSELoss()
        self.eps_threshold = self.config.epsilon_start

    def take_action(
        self, env: gym.Env, state: torch.Tensor, steps_done: int, train: bool
    ) -> torch.Tensor:
        """
        Choose an action using the epsilon-greedy policy.
        Args:
            env: The game environment.
            state: The current state of the game.
            steps_done: The number of steps this Agent has made in total.
            train: True, if the Agent is trying to learn and adjust it's weights.

        Returns: A tensor representing the chosen action.
        """
        # If not learning, use the network right away
        if not train:
            with torch.no_grad():
                return self.network(state).max(1)[1].view(1, 1)

        # Calculate epsilon
        self.eps_threshold = self.config.epsilon_min + (
            self.config.epsilon_start - self.config.epsilon_min
        ) * math.exp(-1.0 * steps_done / self.config.epsilon_decay)

        # Have the network pick an action.
        if random.random() > self.eps_threshold:
            with torch.no_grad():
                return self.network(state).max(1)[1].view(1, 1)

        # Choose a random action.
        return torch.tensor([[env.action_space.sample()]]).long().to(self.device)

    def learn(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
        steps_done: int,
    ) -> None:
        """
        DQN Agent learning. Store the transition/step of the environment in our memory.

        If we have atleast batch_size amount of transitions, then we can have the DQN
        learn and train.

        Trains the agent and network through all the episodes. We train the
        network on a batch_size number of states, rather than train it on the
        most recent one. Saves the weights and copies to the target network.

        Args:
            state: The current state of the game.
            action: The action taken at the current state.
            reward: The reward obtained for taking the action.
            next_state: The next state of the game.
            done: A flag indicating whether the episode is finished or not.
            steps_done: The number of steps this Agent has made in total.
        """
        # Store the transition in memory.
        self._store_experience(
            state=state,
            action=action,
            reward=reward,
            done=done,
            next_state=next_state,
        )

        # Perform one step of the optimization (on the network).
        self._optimize_model()

        # Update the target network's weights based off the current network.
        if steps_done % self.config.num_update_target == 0:
            self._update_target_network()

        # Save the target network's weights.
        if steps_done % self.config.num_save_weights == 0:
            self._save_network_weights()

    def get_eps_threshold(self) -> float:
        """
        Agent return current epsilon value.

        Returns: Float.
        """
        return self.eps_threshold

    def _store_experience(
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

    def _save_network_weights(self) -> None:
        """
        Saves the weights of the target network to the specified file.
        """
        # Create a weights file if it doesn't exist.
        if not os.path.exists(self.weights_file):
            logging.info(
                f"Weight file not found: {self.weights_file}. Creating it now."
            )
            with open(self.weights_file, "w") as _:
                pass
        torch.save(self.target_network.state_dict(), self.weights_file)

    def _load_network_weights(self) -> None:
        """
        Loads the weights of the current/policy and target networks from the
        specified file.
        """
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

    def _update_target_network(self) -> None:
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
            ] * self.config.tau + target_network_state_dict[key] * (1 - self.config.tau)
        self.target_network.load_state_dict(target_network_state_dict)

    def _optimize_model(self) -> None:
        """
        Performs a replay of the experiences stored in the memory buffer.
        """
        if len(self.memory) < self.config.batch_size:
            return

        batch: Experience = self.memory.sample(self.config.batch_size)

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
        next_state_values = torch.zeros(self.config.batch_size).to(self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_network(
                next_state_batch
            ).max(1)[0]

        # Compute the expected Q-values using the Bellman equation.
        expected_q_values = reward_batch + (next_state_values * self.config.gamma)

        # Compute the loss between the predicted and expected Q-values.
        loss = self.loss_fn(current_q_values, expected_q_values.unsqueeze(1))

        # Reset the gradients weights & biases before back propagation.
        self.optimizer.zero_grad()

        # Calculate the gradients of the loss.
        loss.backward()

        # In-place gradient clipping.
        nn.utils.clip_grad_value_(self.network.parameters(), self.config.max_grad_norm)

        # Update the network with the gradients.
        self.optimizer.step()
