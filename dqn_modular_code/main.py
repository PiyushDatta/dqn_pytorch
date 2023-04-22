import logging

import gymnasium as gym

import hydra
from omegaconf import DictConfig

from dqn_modular_code.playground import Playground
from dqn_modular_code.util import AgentConfig, AgentsEnum


def get_config_str(cfg: dict) -> str:
    """
    Get the configurations/hyperparameters
    into a nice string format.
    """
    final_str = ""
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            final_str += "{k}: {v}\n".format(k=k, v=v)

    return final_str


def log_config_info(
    playground: Playground,
    agent_config: AgentConfig,
    log_name: str = "training",
    num_eps: int = 0,
) -> None:
    """
    Before training or validating, pretty print the
    configurations/hyperparameters.
    """
    logging.info(
        "Starting {log_nme} for {eps} episodes.\n\n"
        "Running the playground with:\n{pl_config}\n\n"
        "With below config:\n{config}\n\n".format(
            log_nme=log_name,
            eps=num_eps,
            pl_config=get_config_str(vars(playground)),
            config=get_config_str(vars(agent_config)),
        )
    )


@hydra.main(version_base="1.2", config_path="../configs", config_name="dqn")
def main(cfg: DictConfig):
    # Weights file.
    WEIGHTS_FILE_NAME = "weights/weights.pt"

    # Setup logging.
    logging.getLogger().setLevel(
        level=logging.getLevelName(str(cfg.training.logging_level))
    )

    # Game: CartPole-v1.
    env = gym.make("CartPole-v1")

    # Setup AgentConfig.
    agent_config = AgentConfig(
        hidden_1_size=cfg.model.hidden_nodes_1,
        hidden_2_size=cfg.model.hidden_nodes_2,
        max_grad_norm=cfg.optimizer.max_grad_norm,
        gamma=cfg.optimizer.gamma,
        tau=cfg.optimizer.tau,
        lr=cfg.optimizer.lr,
        epsilon_start=cfg.optimizer.epsilon_start,
        epsilon_min=cfg.optimizer.epsilon_min,
        epsilon_decay=cfg.optimizer.epsilon_decay,
        num_update_target=cfg.training.num_update_target,
        num_save_weights=cfg.training.num_save_weights,
        batch_size=cfg.training.batch_size,
        replay_mem_size=cfg.training.replay_mem_size,
    )

    # Setup a Playground with an Agent.
    playground = Playground(
        gym_env=env,
        agent_type=AgentsEnum.DQNAgent,
        weights_file=WEIGHTS_FILE_NAME,
        config=agent_config,
    )

    # Train.
    if cfg.training.train:
        # Clear metrics.
        playground.clear_metrics()
        # Pretty print config.
        log_config_info(
            playground=playground,
            agent_config=agent_config,
            log_name="training",
            num_eps=cfg.training.training_episodes,
        )
        try:
            playground.train(
                episodes=cfg.training.training_episodes,
                should_plot=cfg.training.plot_training,
            )
            logging.info("Finished training!\n")
        except KeyboardInterrupt:
            logging.error("Got interrupted by user, stopping training.\n")

    # Validate/Test.
    if cfg.training.validate:
        # Clear metrics.
        playground.clear_metrics()
        # Pretty print config.
        log_config_info(
            playground=playground,
            agent_config=agent_config,
            log_name="validation",
            num_eps=cfg.training.validating_episodes,
        )
        try:
            playground.validate(
                episodes=cfg.training.validating_episodes,
                should_plot=cfg.training.plot_training,
            )
            logging.info("Finished validating!\n")
        except KeyboardInterrupt:
            logging.error("Got interrupted by user, stopping validation.\n")

    # Close our env, since we're done training/validating.
    env.close()


if __name__ == "__main__":
    main()
