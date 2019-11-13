"""Main entry point to train Pong agents."""

#TODO: Taalta kutsutaan training loopia erilaisilla agentin configeilla.

from training_loop import training_loop
from utils import EasyDict


if __name__ == "__main__":
    # Configure environment and agent.
    num_episodes = 50000000
    update_target_freq = 1000
    save_every_n_frames = 10000
    player_id = 1
    history_length = 4
    agent_config = EasyDict(input_shape=[history_length, 84, 84], num_actions=3)

    # Call training loop.
    training_loop(num_episodes, player_id, update_target_freq, save_every_n_frames, agent_config)    
