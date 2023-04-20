import logging

import matplotlib
import matplotlib.pyplot as plt

import gymnasium as gym

import hydra
from omegaconf import DictConfig

from dqn_modular_code.dqn_agent import DQNAgent


# Set up matplotlib and check if we're in an IPython environment.
is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display
matplotlib.use("TkAgg")
plt.ion()

WEIGHTS_FILE_NAME = "weights/weights.pt"


@hydra.main(version_base="1.2", config_path="../configs", config_name="dqn")
def main(cfg: DictConfig):
    # Setup logging.
    logging.getLogger().setLevel(
        level=logging.getLevelName(str(cfg.training.logging_level))
    )

    # Game: CartPole-v1.
    env = gym.make("CartPole-v1")

    # Get number of actions from gym action space.
    action_size = env.action_space.n

    # Get the number of state observations.
    state_size = env.observation_space.shape[0]

    # Setup DQN Agent with ReplayMemory.
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        capacity=cfg.training.replay_mem_size,
        weights_file=WEIGHTS_FILE_NAME,
        lr=cfg.optimizer.lr,
        gamma=cfg.optimizer.gamma,
        epsilon_start=cfg.optimizer.epsilon_start,
        epsilon_min=cfg.optimizer.epsilon_min,
        epsilon_decay=cfg.optimizer.epsilon_decay,
        tau=cfg.optimizer.tau,
        num_update_target=cfg.training.num_update_target,
        num_save_weights=cfg.training.num_save_weights,
        max_grad_norm=cfg.optimizer.max_grad_norm,
        hidden_1_size=cfg.model.hidden_nodes_1,
        hidden_2_size=cfg.model.hidden_nodes_2,
    )

    # Train.
    if cfg.training.train:
        logging.info(
            "Starting training for {eps} episodes.".format(
                eps=cfg.training.training_episodes
            )
        )
        try:
            agent.train(
                env=env,
                episodes=cfg.training.training_episodes,
                batch_size=cfg.training.batch_size,
                should_plot=cfg.training.plot_training,
                is_ipython=is_ipython,
            )
            logging.info("Finished training!\n")
        except KeyboardInterrupt:
            logging.error("Got interrupted by user, stopping training.")

    # Validate/Test.
    if cfg.training.validate:
        logging.info(
            "Starting validation for {eps} episodes.".format(
                eps=cfg.training.validating_episodes
            )
        )
        try:
            agent.validate(
                env=env,
                episodes=cfg.training.validating_episodes,
                should_plot=cfg.training.plot_validation,
                is_ipython=is_ipython,
            )
            logging.info("Finished validating!\n")
        except KeyboardInterrupt:
            logging.error("Got interrupted by user, stopping validation.")

    # Close our env, since we're done training/validating.
    env.close()


if __name__ == "__main__":
    main()
