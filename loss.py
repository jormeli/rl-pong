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
    priorities = np.abs((q_value - expected_q_value).data)  # Use TD errors as priorities.

    return loss, priorities


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
    priorities = np.abs((q_value - expected_q_value).data)  # Use TD errors as priorities.

    return loss, priorities


def project_bellman_update(target_net,
                           next_states,
                           rewards,
                           dones,
                           V_min,
                           V_max,
                           num_atoms):
    """L2 projection of Bellman update (Tz) onto support of current estimate
       of the distribution of returns (Z_theta)."""

    with torch.no_grad():
        batch_size  = next_states.shape[0]

        delta_z = (V_max - V_min) / (num_atoms - 1)
        support = torch.linspace(V_min, V_max, num_atoms)

        next_dist = target_net(next_states) * support
        next_action = next_dist.sum(2).max(1)[1]
        next_action = next_action.unsqueeze(1).unsqueeze(1).expand(next_dist.size(0), 1, next_dist.size(2))
        next_dist = next_dist.gather(1, next_action).squeeze(1)

        # Reshape rewards, dones and support to contain dimension for z_i's (num_atoms).
        rewards = rewards.unsqueeze(1).expand_as(next_dist)
        dones = dones.unsqueeze(1).expand_as(next_dist)
        support = support.unsqueeze(0).expand_as(next_dist)

        # Calculate Bellman update, i.e., apply Bellman operator T on z_i's.
        Tz = rewards + (1 - dones) * 0.99 * support
        Tz = Tz.clamp(min=V_min, max=V_max)  # Clamp within the support.

        # L2 project Tz to the support of z.
        b  = (Tz - V_min) / delta_z
        l  = b.floor().long()
        u  = b.ceil().long()

        indices = torch.linspace(0, (batch_size - 1) * num_atoms, batch_size).long()
        indices = indices.unsqueeze(1).expand(batch_size, num_atoms)

        # Distribute probability mass of Tz.
        m = torch.zeros(next_dist.size())
        m.view(-1).index_add_(0, (l + indices).view(-1), (next_dist * (u.float() - b)).view(-1))  # m_l = m_l + p_j(s_t, a^*)(u - b_j)
        m.view(-1).index_add_(0, (u + indices).view(-1), (next_dist * (b - l.float())).view(-1))  # m_u = m_u + p_j(s_t, a^*)(b_j - l)

        return m


def categorical_loss(policy_net,
                     target_net,
                     states,
                     actions,
                     rewards,
                     next_states,
                     dones,
                     gamma,
                     noisy,
                     V_min=-10,
                     V_max=10,
                     num_atoms=51):
    """Implements the categorical algorithm (C-51) from https://arxiv.org/abs/1707.06887."""

    batch_size = states.shape[0]

    # Compute Tz projected onto the support of z.
    m = project_bellman_update(target_net,
                               next_states,
                               rewards,
                               dones,
                               V_min,
                               V_max,
                               num_atoms)

    # Get current estimate of the distribution of returns.
    dist = policy_net(states)

    # Expand actions to 3D tensor [N, 1, num_atoms] to
    # lookup corresponding probabilities of p_i(x_t, a_t).
    act = actions.unsqueeze(1).expand(batch_size, 1, num_atoms)
    dist = dist.gather(1, act).squeeze(1)
    dist.data.clamp_(0.001, 0.999)  # Ensure valid probability distribution.

    # Mean cross-entropy loss, minimizes KL-divergence between
    # projected distribution and current distribution of returns.
    loss = -(m * dist.log()).sum(1).mean()
    priorities = (m * dist.log()).detach().sum(1).numpy()

    return loss, priorities
