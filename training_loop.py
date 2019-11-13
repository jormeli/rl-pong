"""Training loop."""

#TODO: Tanne varsinainen training loop.

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from agent import Agent
from utils import EasyDict
import wimblepong


def make_pong_environment(fps=30, scale=1):
    """Initialize Pong environment."""
    env = gym.make("WimblepongVisualMultiplayer-v0")  # TODO: Add more options.
    env.unwrapped.fps = fps
    env.unwrapped.scale = scale
    return env


def training_loop(num_episodes, player_id, update_target_freq, save_every_n_frames, agent_config, render=False):
    """Training loop for Pong agents."""
    # Make the environment
    env = make_pong_environment()

    # Set up the agent.
    agent = Agent(**agent_config)

    # Setup the opponent.
    opponent_id = 2
    opponent = wimblepong.SimpleAi(env, opponent_id)

    # Set the names for both SimpleAIs
    env.set_names(agent.get_name(), opponent.get_name())

    # Parameters for GLIE.
    target_eps = 0.1
    reach_target_at_ep = 10000000
    a = np.round(target_eps * reach_target_at_ep / (1 - target_eps))

    # Parameters for training.
    start_training_at_frame = 50000
    train_freq = 4

    # Housekeeping
    states = []
    wins = 0
    frames_seen = 0

    for ep in range(0, num_episodes):
        # Reset the Pong environment
        (agent_state, opp_state), terminal = env.reset(), 0
        done = False
        frame_count = 0

        # Compute new epsilon.
        epsilon = a / (a + ep)

        while not done:
            # Get actions from agent and opponent.
            agent_action = agent.get_action(agent_state, epsilon=epsilon)
            opp_action = opponent.get_action(opp_state)

            # Step the environment and get the rewards and new observations
            (agent_next_state, opp_next_state), (agent_reward, opp_reward), done, info = env.step((agent_action, opp_action))

            # Store transitions.
            agent.store_transition(agent_state, agent_action, agent_next_state, agent_reward, done)

            # See if theres enough frames to start training.            
            if frames_seen > start_training_at_frame:
                if frame_count % train_freq == 0:  # Update agent every 4th frame.
                    agent.td_loss_double_dqn()

                if frame_count % update_target_freq == update_target_freq - 1:  # Update target network.
                    agent.update_target_network()

            # Count the wins
            if agent_reward == 10:
                wins += 1

            if render:
                env.render()

            agent_state = agent_next_state
            opp_state = opp_next_state
            frame_count += 1

        print('episode %i, end at frame %i, tot. frames seen %i, wr %0.2f' % (ep, frame_count, frames_seen, wins / (ep + 1)))
        
        # Reset agent's internal state.
        agent.reset()
        frames_seen += frame_count

        if ep % save_every_n_frames == 0:
            torch.save(agent.policy_net.state_dict(), 'agent.mdl')

        #if ep % 5 == 4:
        #    env.switch_sides()
