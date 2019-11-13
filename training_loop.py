"""Training loop."""

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


def training_loop(num_episodes, player_id, update_target_freq, save_every_n_ep, agent_config, render=False):
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

    # Parameters for epsilon-greedy policy.
    epsilon_start = 1.0
    epsilon_final = 0.1
    epsilon_decay = 500000
    epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * np.exp(-frame_idx / epsilon_decay)

    # Parameters for training.
    start_training_at_frame = 50000
    train_freq = 4

    # Housekeeping
    wins = 0
    frames_seen = 0

    for ep in range(0, num_episodes):
        # Reset the Pong environment
        (agent_state, opp_state) = env.reset()
        done = False
        step = 0
        actions_taken = []

        # Compute new epsilon.
        epsilon = epsilon_by_frame(frames_seen)

        while not done:
            # Get actions from agent and opponent.
            agent_action = agent.get_action(agent_state, epsilon=epsilon)
            opp_action = opponent.get_action(opp_state)

            # Step the environment and get the rewards and new observations
            (agent_next_state, opp_next_state), (agent_reward, _), done, info = env.step((agent_action, opp_action))

            # Store transitions.
            agent.store_transition(agent_state, agent_action, agent_next_state, agent_reward, done)

            # See if theres enough frames to start training.            
            if frames_seen > start_training_at_frame:
                if step % train_freq == 0:  # Update agent every 4th frame.
                    #agent.td_loss_double_dqn()
                    agent.td_loss()

                if frames_seen % update_target_freq == update_target_freq - 1:  # Update target network.
                    agent.update_target_network()

            # Count the wins.
            if agent_reward == 10:
                wins += 1

            if render:
                env.render()

            agent_state = agent_next_state
            opp_state = opp_next_state
            actions_taken.append(agent_action)
            step += 1
            frames_seen += 1

        act_counts, _ = np.histogram(actions_taken, bins=[0, 1, 2, 3])
        actions = 'stay %i, up %i, down %i' % (act_counts[0], act_counts[1], act_counts[2])
        print('episode %i, end frame %i, tot. frames %i, eps %0.2f, %s, wins %i, losses %i' % (ep, step, frames_seen, epsilon, actions, wins, ep + 1 - wins))

        # Reset agent's internal state.
        agent.reset()

        if ep % save_every_n_ep == 0:
            torch.save(agent.policy_net.state_dict(), 'agent.mdl')

        #if ep % 5 == 4:
        #    env.switch_sides()
