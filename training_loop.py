"""Training loop."""

import gym
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch.utils.tensorboard import SummaryWriter

from agent import Agent
from utils import EasyDict
import wimblepong


def make_pong_environment(fps=30, scale=1):
    """Initialize Pong environment."""
    env = gym.make("WimblepongVisualMultiplayer-v0")  # TODO: Add more options.
    env.unwrapped.fps = fps
    env.unwrapped.scale = scale
    return env


def epsilon_schedule(episode, target_epsilon, reach_target_at_frame):
    """Schedule for exponentially decaying epsilon."""
    if episode > reach_target_at_frame:
        return target_epsilon

    decay = -np.log(target_epsilon) / reach_target_at_frame 
    return np.exp(-decay * episode)


def training_loop(num_episodes, target_epsilon, reach_target_at_frame, player_id, start_training_at_frame,
                  update_target_freq, save_every_n_ep, log_freq, agent_config, clip_reward=False, run_description='',
                  render=False):
    """Training loop for Pong agents."""
    run_dir = os.path.dirname(os.path.abspath(__file__))

    # Make the environment
    env = make_pong_environment()

    # Set up the agent.
    agent = Agent(**agent_config)

    # Setup the opponent.
    opponent_id = 2
    opponent = wimblepong.SimpleAi(env, opponent_id)

    # Set the names for both SimpleAIs
    env.set_names(agent.get_name(), opponent.get_name())

    # Setup directories for models and logging.
    model_dir = os.path.join(run_dir, 'models')
    log_dir = os.path.join(run_dir, 'logs')

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Initialize summary writer.
    writer = SummaryWriter(log_dir=log_dir, comment=run_description)

    # Housekeeping
    max_reward = 10.0 if not clip_reward else 1.0
    wins = 0
    frames_seen = 0
    game_results = []
    reward_sums = []

    for ep in range(0, num_episodes):
        # Reset the Pong environment
        (agent_state, opp_state) = env.reset()
        done = False
        step = 0
        actions_taken = []
        losses = []
        reward_sum = 0.0

        # Compute new epsilon.
        epsilon = epsilon_schedule(frames_seen, target_epsilon, reach_target_at_frame)

        while not done:
            # Get actions from agent and opponent.
            agent_action = agent.get_action(agent_state, epsilon=epsilon)
            opp_action = opponent.get_action(opp_state)

            # Step the environment and get the rewards and new observations
            (agent_next_state, opp_next_state), (agent_reward, _), done, info = env.step((agent_action, opp_action))

            # Clip reward.
            if clip_reward:
                agent_reward = max(-1., min(1., agent_reward))

            #if agent_reward == 0.0:
            #    agent_reward = 0.2
            #elif agent_reward == 10.0:
            #    agent_reward = 15.0
            #elif agent_reward == -10:
            #    agent_reward = -15.0

            # Store transitions.
            agent.store_transition(agent_state, agent_action, agent_next_state, agent_reward, done)

            # See if theres enough frames to start training.
            if frames_seen > start_training_at_frame:
                loss = agent.compute_loss()

                if frames_seen % update_target_freq == update_target_freq - 1:  # Update target network.
                    agent.update_target_network()

            # Count the wins. Won't work with discounting.
            if agent_reward == max_reward:  # 15.0
                wins += 1
                game_results.append(1)
            else:
                game_results.append(0)

            if render:
                env.render()

            if frames_seen > start_training_at_frame:
                losses.append(loss)
            else:
                losses.append(0)

            agent_state = agent_next_state
            opp_state = opp_next_state
            actions_taken.append(agent_action)
            reward_sum += agent_reward
            step += 1
            frames_seen += 1

        reward_sums.append(reward_sum)
        act_counts, _ = np.histogram(actions_taken, bins=[0, 1, 2, 3])
        actions = 'stay %i, up %i, down %i' % (act_counts[0], act_counts[1], act_counts[2])
        print('buf_count %i, episode %i, end frame %i, tot. frames %i, eps %0.2f, %s, wins %i, losses %i' % (agent.memory.count, ep, step, frames_seen, epsilon, actions, wins, ep + 1 - wins))

        # Log progress.
        if ep % log_freq == 0:
            # Write scalars.
            writer.add_scalar('Episode/Loss', np.mean(losses), ep)
            writer.add_scalar('Episode/Episode-length', step, ep)
            writer.add_scalar('Progress/Epsilon', epsilon, frames_seen)
            writer.add_scalar('Progress/Frames', frames_seen, frames_seen)

            if ep < 100:  # Log results and rewards from last n games.
                last_n_results = game_results
                last_n_reward_sums = reward_sums
            else:
                last_n_results = game_results[-100:]
                last_n_reward_sums = reward_sums[-100:]

            cur_win_rate = np.mean(last_n_results)
            mean_rewards = np.mean(last_n_reward_sums)
            writer.add_scalar('Progress/Cumulative-reward', mean_rewards, ep)
            writer.add_scalar('Progress/Win-rate', cur_win_rate, ep)

            # Show random batch of states.
            (state_batch, _, _, _, _), _  = agent.memory.sample_batch(5)
            n, c, h, w = state_batch.shape
            state_batch = state_batch.reshape(n * c, h, w)[:, None, :, :]
            writer.add_images('ReplayBuffer/Sample states', state_batch, ep)

        # Reset agent's internal state.
        agent.reset()

        if ep % save_every_n_ep == 0:
            torch.save(agent.policy_net.state_dict(), os.path.join(model_dir, 'agent_ep.mdl'))

        #if ep % 5 == 4:
        #    env.switch_sides()
