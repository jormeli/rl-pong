"""Main entry point to train Pong agents."""

from training_loop import training_loop
from utils import EasyDict

from networks import *

if __name__ == "__main__":
    # Configure environment and agent.
    num_episodes = 50000000
    start_training_at_frame = 5000
    target_epsilon = 0.05
    reach_target_at_frame = 1e6
    update_target_freq = 10000
    save_every_n_ep = 1
    player_id = 1
    log_freq = 50
    agent_config = EasyDict(input_shape=(1, 84, 84),
                            network_fn=CNN,
                            num_actions=3,
                            stack_size=4,
                            replay_memory_size=int(1e6),
                            minibatch_size=128)

    # Call training loop.
    training_loop(num_episodes, target_epsilon, reach_target_at_frame, player_id, start_training_at_frame,
                  update_target_freq, save_every_n_ep, log_freq, agent_config)
