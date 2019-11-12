"""Training loop."""

#TODO: Tanne varsinainen training loop.

import gym
import matplotlib.pyplot as plt
import numpy as np

from agent import Agent
from utils import EasyDict
import wimblepong


def make_pong_environment(fps=30, scale=1):
    """Initialize Pong environment."""
    env = gym.make("WimblepongVisualMultiplayer-v0")  # TODO: Add more options.
    env.unwrapped.fps = fps
    env.unwrapped.scale = scale
    return env


def training_loop(num_episodes, player_id, update_target_freq, agent_config, render=False):
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

    # Housekeeping
    states = []
    wins = 0

    for ep in range(0, num_episodes):
        # Reset the Pong environment
        (agent_state, opp_state), terminal = env.reset(), 0
        done = False
        step = 0

        while not done:
            # Get actions from agent and opponent.
            agent_action = agent.get_action(agent_state)
            opp_action = opponent.get_action(opp_state)

            # Step the environment and get the rewards and new observations
            (agent_next_state, opp_next_state), (agent_reward, opp_reward), done, info = env.step((agent_action, opp_action))

            # Store transitions.
            agent.store_transition(agent_state, agent_action, agent_next_state, agent_reward, done)

            # Update agent.
            agent.td_loss_double_dqn()

            # Update target network.
            if step % update_target_freq == update_target_freq - 1:
                agent.update_target_network()

            # Count the wins
            if agent_reward == 10:
                wins += 1

            if render:
                env.render()

            agent_state = agent_next_state
            opp_state = opp_next_state
            step += 1

        print('Episode %i over at step %i. Win record: %0.2f' % (ep, step, wins / (ep + 1)))
        
        # Reset agent's internal state.
        agent.reset()

        #if ep % 5 == 4:
        #    env.switch_sides()
