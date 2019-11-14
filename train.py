"""Main entry point to train Pong agents."""

from training_loop import training_loop
from utils import EasyDict


if __name__ == "__main__":
    # Configure environment and agent.
    num_episodes = 50000000
    update_target_freq = 4
    save_every_n_ep = 100
    player_id = 1
    agent_config = EasyDict(input_shape=(1, 84, 84), num_actions=3,
                            stack_size=4, replay_memory_size=1000000,
                            minibatch_size=128)

    # Call training loop.
    training_loop(num_episodes, player_id, update_target_freq, save_every_n_ep, agent_config)
