"""Implementations of loss functions used by the agent."""

import numpy as np
import torch


def double_dqn_loss(policy_net,
                    target_net,
                    states,
                    actions,
                    rewards,
                    next_states,
                    dones,
                    gamma,
                    noisy):
    """TD loss for double DQN, proposed in https://arxiv.org/abs/1509.06461."""

    # Get Q-values from current network and target network.
    q_values = policy_net.forward(states, training=True)
    next_q_values = policy_net.forward(next_states, training=True)
    next_q_state_values = target_net.forward(next_states, training=True)

    q_value = q_values.gather(1, actions).squeeze(1)
    next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
    expected_q_value = rewards + gamma * next_q_value * (1 - dones)
    loss = (q_value - expected_q_value.detach()).pow(2).mean()  # F.smooth_l1_loss(q_value, expected_q_value.detach())
    td_err = np.abs((q_value - expected_q_value).data)

    return loss, td_err


def td_loss(policy_net,
            target_net,
            states,
            actions,
            rewards,
            next_states,
            dones,
            gamma,
            noisy):
    """Compute TD(0) loss."""

    # Get Q-values from current network and target network.
    q_values = policy_net.forward(states, training=True)
    next_q_values = policy_net.forward(next_states, training=True)

    q_value = q_values.gather(1, actions).squeeze(1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = rewards + gamma * next_q_value * (1 - dones)
    loss = (q_value - expected_q_value.detach()).pow(2).mean()  # F.smooth_l1_loss(q_value, expected_q_value.detach())
    td_err = np.abs((q_value - expected_q_value).data)

    return loss, td_err
