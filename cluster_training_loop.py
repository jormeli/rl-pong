"""Training loop for cluster runs."""

import gym
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import time
import torch
from torch.utils.tensorboard import SummaryWriter

from agent import Agent
from networks import DuelingDQN, VanillaDQN, CNN
from utils import EasyDict
import wimblepong


def make_pong_environment(fps=30, scale=4):
    """Initialize Pong environment."""
    env = gym.make("WimblepongVisualMultiplayer-v0")
    env.unwrapped.fps = fps
    env.unwrapped.scale = scale
    return env


def epsilon_schedule(frames_seen, target_epsilon, reach_target_at_frame):
    """Schedule for exponentially decaying epsilon."""
    if frames_seen > reach_target_at_frame:
        return target_epsilon

    decay = -np.log(target_epsilon) / reach_target_at_frame 
    return np.exp(-decay * frames_seen)


def beta_schedule(frame, beta_0, reach_target_at_frame):
    """Linearly anneal beta from beta_0 to one."""
    return min(1.0, (1 - beta_0) / reach_target_at_frame * frame + beta_0)


def training_loop(submit_config,
                  num_episodes,
                  target_epsilon,
                  beta_0,
                  reach_target_at_frame,
                  player_id,
                  start_training_at_frame,
                  target_update_freq,
                  model_update_freq,
                  save_every_n_ep,
                  log_freq,
                  agent_config,
                  network_fn_kwargs,
                  clip_reward=False,
                  run_description='',
                  render=False):
    """Training loop for Pong agents."""
    run_dir = submit_config.run_dir

    # Make the environment
    env = make_pong_environment()

    # Set up the agent.
    agent = Agent(**agent_config, network_fn_kwargs=network_fn_kwargs)

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

    # Initialize file for logging KPIs.
    model_perf_file = os.path.join(run_dir, 'model_perf.txt')
    with open(model_perf_file, 'w') as f:
        f.write('ep,mean_rewards,mean_ep_length,mean_wr\n')

    # Housekeeping
    max_reward = 10.0 if not clip_reward else 1.0
    wins = 0
    frames_seen = 0
    game_results = []
    reward_sums = []
    ep_lengths = []

    for ep in range(0, num_episodes):
        # Reset the Pong environment
        (agent_state, opp_state) = env.reset()
        done = False
        step = 0
        losses = []
        reward_sum = 0.0

        # Compute new epsilon and beta.
        epsilon = epsilon_schedule(frames_seen, target_epsilon, reach_target_at_frame)
        beta = beta_schedule(frames_seen, beta_0, reach_target_at_frame)

        start = time.time()
        while not done:
            # Get actions from agent and opponent.
            agent_action = agent.get_action(agent_state, epsilon=epsilon)
            opp_action = opponent.get_action(opp_state)

            # Step the environment and get the rewards and new observations
            (agent_next_state, opp_next_state), (agent_reward, _), done, info = env.step((agent_action, opp_action))

            # Clip reward.
            if clip_reward:
                agent_reward = max(-1., min(1., agent_reward))

            # Store transitions.
            agent.store_transition(agent_state, agent_action, agent_next_state, agent_reward, done)

            # See if theres enough frames to start training.
            if frames_seen > start_training_at_frame:
                if frames_seen % model_update_freq == model_update_freq - 1:
                    # Update policy network.
                    loss = agent.compute_loss(beta=beta)

                    # Update EMA network.
                    agent.update_ema_policy()

                if frames_seen % target_update_freq == target_update_freq - 1:  # Update target network.
                    agent.update_target_network()

            # Count the wins. Won't work with discounting.
            if agent_reward == max_reward:
                wins += 1
                game_results.append(1)

            if agent_reward == -max_reward:
                game_results.append(0)

            if render:
                env.render()

            if frames_seen > start_training_at_frame:
                if frames_seen % model_update_freq == model_update_freq - 1:
                    losses.append(loss)
            else:
                losses.append(0)

            agent_state = agent_next_state
            opp_state = opp_next_state
            reward_sum += agent_reward
            step += 1
            frames_seen += 1

        reward_sums.append(reward_sum)
        ep_lengths.append(step)
        elapsed_time = time.time() - start
        print('buf_count %i, episode %i, end frame %i, tot. frames %i, eps %0.2f, wins %i, losses %i, %gs' % (agent.memory.count, ep, step, frames_seen, epsilon, wins, ep + 1 - wins, elapsed_time))

        # Log progress.
        if ep % log_freq == 0:
            # Write scalars.
            writer.add_scalar('Progress/Epsilon', epsilon, frames_seen)
            writer.add_scalar('Progress/Frames', frames_seen, frames_seen)

            if ep < 100:  # Log results and rewards from last n games.
                last_n_results = game_results
                last_n_reward_sums = reward_sums
                last_n_ep_lengths = ep_lengths
            else:
                last_n_results = game_results[-100:]
                last_n_reward_sums = reward_sums[-100:]
                last_n_ep_lengths = ep_lengths[-100:]

            cur_win_rate = np.mean(last_n_results)
            mean_rewards = np.mean(last_n_reward_sums)
            mean_ep_length = np.mean(last_n_ep_lengths)
            writer.add_scalar('Progress/Cumulative-reward', mean_rewards, ep)
            writer.add_scalar('Progress/Win-rate', cur_win_rate, ep)
            writer.add_scalar('Episode/Average-episode-length', mean_ep_length, ep)
            writer.add_scalar('Episode/Loss', np.mean(losses), ep)

            # Show random batch of states.
            (state_batch, _, _, _, _), _, _  = agent.memory.sample_batch(5)
            n, c, h, w = state_batch.shape
            state_batch = state_batch.reshape(n * c, h, w)[:, None, :, :]
            writer.add_images('ReplayBuffer/Sample states', state_batch, ep)

        # Reset agent's internal state.
        agent.reset()

        if ep % save_every_n_ep == 0:
            torch.save(agent.policy_net.state_dict(), os.path.join(model_dir, 'agent_%s_ep%i.mdl' % (agent_config.network_name, ep)))
            torch.save(agent.policy_net_ema.state_dict(), os.path.join(model_dir, 'ema_agent_%s_ep%i.mdl' % (agent_config.network_name, ep)))

            perf_str = '%i,%g,%g,%g\n' % (ep, mean_rewards, mean_ep_length, cur_win_rate)
            with open(model_perf_file, 'a') as f:
                f.write(perf_str)


def self_play_training_loop(submit_config,
                            player_run_id,
                            player_model_id,
                            opponent_run_ids,
                            opponent_model_ids,
                            total_rounds,
                            num_episodes,
                            start_training_at_frame,
                            target_update_freq,
                            model_update_freq,
                            save_every_n_ep,
                            log_freq,
                            epsilon,
                            learning_rate,
                            run_description='',
                            render=False):
    """Self-play training loop for Pong agents."""
    submission_run_dir = submit_config.run_dir
    run_dir_root = submit_config.run_dir_root

    # Make the environment
    env = make_pong_environment()

    # Locate opponent run dirs.
    run_dirs = []
    for run_id in opponent_run_ids:
        run_dir = [os.path.join(run_dir_root, d) for d in os.listdir(run_dir_root) if str(run_id).zfill(5) in d]
        if run_dir:
            run_dirs.append(run_dir[0])

    # Load agent configs, network configs and network weights.
    opposing_agents = []
    for run_dir, model_id in zip(run_dirs, opponent_model_ids):
        # Load run config.
        with open(os.path.join(run_dir, 'run_func_args.pkl'), 'rb') as f:
            run_config = pickle.load(f)

        # Initialize agent.
        agent_config = run_config.agent_config
        network_fn_kwargs = run_config.network_fn_kwargs
        agent = Agent(network_fn_kwargs=network_fn_kwargs, **agent_config)

        # Load model weights.
        model_dir = os.path.join(run_dir, 'models')
        model_path = os.path.join(model_dir, [f for f in os.listdir(model_dir) if str(model_id) in f][0])
        agent.load_model(model_path)

        # Append to player list.
        opposing_agents.append((agent.get_name() +'_%s' % (agent_config.network_name), agent))

    # Add SimpleAI to opponents.
    opposing_agents.append(('SimpleAI', wimblepong.SimpleAi(env, 2)))
    opponent_run_ids.append(-1)

    # Load agent that is trained.
    target_run_dir = [os.path.join(run_dir_root, d) for d in os.listdir(run_dir_root) if str(player_run_id).zfill(5) in d][0]
    print('Loading traget from: %s' % target_run_dir)

    # Load run config.
    with open(os.path.join(target_run_dir, 'run_func_args.pkl'), 'rb') as f:
        run_config = pickle.load(f)

    # Initialize agent.
    agent_config = run_config.agent_config
    agent_config.learning_rate = learning_rate
    network_fn_kwargs = run_config.network_fn_kwargs
    p1 = Agent(network_fn_kwargs=network_fn_kwargs, **agent_config)

    # Load model weights.
    model_dir = os.path.join(target_run_dir, 'models')
    model_path = os.path.join(model_dir, [f for f in os.listdir(model_dir) if str(player_model_id) in f][0])
    p1.load_model(model_path)
    p1_name  = p1.get_name() +'_%s' % (agent_config.network_name)

    # Setup directories for models and logging.
    model_dir = os.path.join(submission_run_dir, 'models')
    log_dir = os.path.join(submission_run_dir, 'logs')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Initialize summary writer.
    writer = SummaryWriter(log_dir=log_dir, comment=run_description)
    perf_file = os.path.join(submission_run_dir, 'win_rates.txt')

    # Housekeeping.
    num_opponents = len(opposing_agents)
    max_reward = 10.0
    wr_against = {}
    total_frames_seen = 0

    for total_ep in range(total_rounds):
        # Pick two agents uniform random.
        p2_idx = np.random.randint(num_opponents)
        p2_name, p2 = opposing_agents[p2_idx]

        # Setup players and housekeeping.
        env.set_names(p1_name, p2_name)
        frames_seen = 0
        p1_wins = 0
        p1_reward_sums = []
        p1_losses = []
        ep_lengths = []

        print('Training %s vs. %s...' % (p1_name, p2_name))
        for ep in range(num_episodes):
            # Reset the Pong environment.
            (p1_state, p2_state) = env.reset()
            p1_reward_sum = 0
            done = False
            step = 0

            while not done:
                # Get actions from agent and opponent.
                p1_action = p1.get_action(p1_state, epsilon=epsilon)
                p2_action = p2.get_action(p2_state, epsilon=epsilon) if p2_name != 'SimpleAI' else p2.get_action()

                # Step the environment and get the rewards and new observations
                (p1_next_state, p2_next_state), (p1_reward, _), done, info = env.step((p1_action, p2_action))

                # Store transitions.
                p1.store_transition(p1_state, p1_action, p1_next_state, p1_reward, done)

                # See if theres enough frames to start training.
                if frames_seen >= start_training_at_frame:
                    if frames_seen % model_update_freq == model_update_freq - 1:
                        # Update policy networks.
                        loss_p1 = p1.compute_loss()

                        # Update EMA networks.
                        p1.update_ema_policy()
                else:
                    loss_p1 = 0

                if total_frames_seen % target_update_freq == target_update_freq - 1:  # Update target networks.
                    p1.update_target_network()

                # Count the wins. Won't work with discounting.
                if p1_reward == max_reward:
                    p1_wins += 1

                if render:
                    env.render()

                if frames_seen % model_update_freq == model_update_freq - 1:
                    p1_losses.append(loss_p1)

                p1_state = p1_next_state
                p2_state = p2_next_state
                p1_reward_sum += p1_reward
                step += 1
                frames_seen += 1
                total_frames_seen += 1

            p1_reward_sums.append(p1_reward_sum)
            ep_lengths.append(step)
            print('%s vs. %s, episode %i/%i, end frame %i, frames %i, eps %0.2f, wins %i, losses %i' % (p1_name, p2_name, ep, num_episodes, step, frames_seen, epsilon, p1_wins, ep + 1 - p1_wins))

            if ep % save_every_n_ep == 0:
                torch.save(p1.policy_net.state_dict(), os.path.join(model_dir, 'agent_%s.mdl' % (agent_config.network_name)))
                torch.save(p1.policy_net_ema.state_dict(), os.path.join(model_dir, 'ema_agent_%s.mdl' % (agent_config.network_name)))

                # Update WR against current opponent.
                key = 'wr_vs_run_id_%i' % opponent_run_ids[p2_idx]
                wr_against[key] = p1_wins / (ep + 1)

                perf_dict = {**{'round': [total_ep], 'ep': [ep]}, **wr_against}
                df = pd.DataFrame(data=perf_dict)
                df.to_csv(perf_file, header=True, index=False)
                    
        print('WR against %s: %0.2f' % (p2_name, p1_wins / num_episodes))
        with open(os.path.join(submission_run_dir, 'wr.pkl'), 'wb') as f:
            pickle.dump(wr_against, f)
        print()

    # Dump final results.
    with open(os.path.join(submission_run_dir, 'wr.pkl'), 'wb') as f:
        pickle.dump(wr_against, f)
