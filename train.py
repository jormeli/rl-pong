"""Main entry point to train Pong agents."""

#TODO: Taalta kutsutaan training loopia erilaisilla agentin configeilla.

from training_loop import training_loop
from utils import EasyDict


if __name__ == "__main__":
    # Configure environment and agent.
    num_episodes = 1000000
    update_target_freq = 1000
    player_id = 1
    history_length = 4
    agent_config = EasyDict(input_shape=[history_length, 200, 200], num_actions=3)

    # Call training loop.
    training_loop(num_episodes, player_id, update_target_freq, agent_config)    
