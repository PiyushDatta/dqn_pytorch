import unittest
from unittest.mock import patch, MagicMock
import gym
from omegaconf import OmegaConf

from dqn_modular_code.main import main as dqn_modular_code_main
from dqn_single_file import main as dqn_single_file_main


# Mock config object.
MOCK_CONFIG = {
    "env": {
        "env_name": "CartPole-v1",
        "train": True,
        "validate": True,
        "training_episodes": 10,
        "validating_episodes": 5,
        "batch_size": 32,
        "plot_training": False,
        "plot_validation": False,
        "logging_level": "INFO",
        "num_update_target": 1,
        "num_save_weights": 1,
    },
    "agent": {
        "agent_type": "DQN_AGENT",
        "replay_mem_size": 1000,
        "lr": 0.1,
        "gamma": 0.55,
        "epsilon_start": 0.01,
        "epsilon_min": 0.01,
        "epsilon_decay": 0.995,
        "tau": 0.01,
        "max_grad_norm": 1.0,
        "weights_file": "weights/testing_dqn_cartpolev1.pt",
    },
    "neural_net": {
        "hidden_nodes_1": 256,
        "hidden_nodes_2": 256,
    },
}


class TestMainFunction(unittest.TestCase):
    @patch("gym.make")
    @patch("dqn_modular_code.dqn.dqn_agent.DQNAgent")
    @patch("logging.getLogger")
    def test_dqn_modular_code_main(self, mock_logger, mock_dqnagent, mock_make):
        print("\nRunning test:", self.test_dqn_modular_code_main.__name__)
        # Mock return values for patched functions.
        cfg = OmegaConf.create(MOCK_CONFIG)
        mock_logger.return_value = MagicMock()
        mock_dqnagent.return_value = MagicMock()
        mock_make.return_value = MagicMock(spec_set=gym.Env)

        # Call main() function.
        dqn_modular_code_main(cfg)

    @patch("gym.make")
    @patch("dqn_single_file.DQNAgent")
    def test_dqn_single_file_main(self, mock_dqnagent, mock_make):
        print("\nRunning test:", self.test_dqn_single_file_main.__name__)
        # Mock return values for patched functions.
        cfg = OmegaConf.create(MOCK_CONFIG)
        mock_dqnagent.return_value = MagicMock()
        mock_make.return_value = MagicMock(spec_set=gym.Env)

        # Call main() function.
        dqn_single_file_main(cfg)

    @patch("gym.make")
    @patch("dqn_modular_code.dqn.dqn_agent.DQNAgent")
    @patch("logging.getLogger")
    def test_dqn_modular_code_main_no_cfg(self, mock_logger, mock_dqnagent, mock_make):
        print("\nRunning test:", self.test_dqn_modular_code_main_no_cfg.__name__)
        # Mock return values for patched functions.
        mock_logger.return_value = MagicMock()
        mock_dqnagent.return_value = MagicMock()
        mock_make.return_value = MagicMock(spec_set=gym.Env)

        # Call main() function.
        dqn_modular_code_main()

    @patch("gym.make")
    @patch("dqn_single_file.DQNAgent")
    def test_dqn_single_file_main_no_cfg(self, mock_dqnagent, mock_make):
        print("\nRunning test:", self.test_dqn_single_file_main_no_cfg.__name__)
        # Mock return values for patched functions.
        mock_dqnagent.return_value = MagicMock()
        mock_make.return_value = MagicMock(spec_set=gym.Env)

        # Call main() function.
        dqn_single_file_main()


if __name__ == "__main__":
    # Create a TestSuite and add the test cases to it.
    suite = unittest.TestSuite()
    suite.addTest(TestMainFunction("test_dqn_modular_code_main"))
    suite.addTest(TestMainFunction("test_dqn_single_file_main"))
    suite.addTest(TestMainFunction("test_dqn_modular_code_main_no_cfg"))
    suite.addTest(TestMainFunction("test_dqn_single_file_main_no_cfg"))

    # Run the tests.
    runner = unittest.TextTestRunner()
    runner.run(suite)
