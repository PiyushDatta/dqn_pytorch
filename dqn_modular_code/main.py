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


def generate_env(env_name: str) -> gym.Env:
    """
    Generates the specified environment
    """
    if env_name == "CartPole-v1":
        return gym.make("CartPole-v1")
    else:
        raise ValueError("Unsupported environment: {env}".format(env=env_name))


@hydra.main(version_base="1.2", config_path="../configs", config_name="dqn")
def main(cfg: DictConfig):
    # Weights file.
    WEIGHTS_FILE_NAME = "weights/weights.pt"

    # Setup logging.
    logging.getLogger().setLevel(level=logging.getLevelName(str(cfg.env.logging_level)))

    # Game: Gym Environment.
    env = generate_env(str(cfg.env.env_name))

    # Setup AgentConfig.
    agent_config = AgentConfig(
        hidden_1_size=cfg.neural_net.hidden_nodes_1,
        hidden_2_size=cfg.neural_net.hidden_nodes_2,
        max_grad_norm=cfg.agent.max_grad_norm,
        gamma=cfg.agent.gamma,
        tau=cfg.agent.tau,
        lr=cfg.agent.lr,
        epsilon_start=cfg.agent.epsilon_start,
        epsilon_min=cfg.agent.epsilon_min,
        epsilon_decay=cfg.agent.epsilon_decay,
        replay_mem_size=cfg.agent.replay_mem_size,
        num_update_target=cfg.env.num_update_target,
        num_save_weights=cfg.env.num_save_weights,
        batch_size=cfg.env.batch_size,
    )

    # Setup a Playground with an Agent.
    playground = Playground(
        gym_env=env,
        agent_type=AgentsEnum.DQNAgent,
        weights_file=WEIGHTS_FILE_NAME,
        config=agent_config,
    )

    # Train.
    if cfg.env.train:
        # Clear metrics.
        playground.clear_metrics()
        # Pretty print config.
        log_config_info(
            playground=playground,
            agent_config=agent_config,
            log_name="training",
            num_eps=cfg.env.training_episodes,
        )
        try:
            playground.train(
                episodes=cfg.env.training_episodes,
                should_plot=cfg.env.plot_training,
            )
            logging.info("Finished training!\n")
        except KeyboardInterrupt:
            logging.error("Got interrupted by user, stopping training.\n")

    # Validate/Test.
    if cfg.env.validate:
        # Clear metrics.
        playground.clear_metrics()
        # Pretty print config.
        log_config_info(
            playground=playground,
            agent_config=agent_config,
            log_name="validation",
            num_eps=cfg.env.validating_episodes,
        )
        try:
            playground.validate(
                episodes=cfg.env.validating_episodes,
                should_plot=cfg.env.plot_training,
            )
            logging.info("Finished validating!\n")
        except KeyboardInterrupt:
            logging.error("Got interrupted by user, stopping validation.\n")

    # Close our env, since we're done training/validating.
    env.close()


if __name__ == "__main__":
    main()
