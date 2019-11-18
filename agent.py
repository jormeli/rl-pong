"""Implementation of Pong agent."""

import numpy as np
import PIL.Image
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from networks import VanillaDQN
from replay_buffer import ReplayBuffer


class Agent():
    def __init__(self,
                 input_shape,
                 num_actions,
                 network_fn=VanillaDQN,
                 minibatch_size=128,
                 replay_memory_size=500000,
                 stack_size=1,
                 gamma=0.98,
                 beta0=0.9,
                 beta1=0.999,
                 learning_rate=1e-4,
                 device='cuda',
                 normalize=False,
                 prioritized=True,
                 **kwargs):
        self.agent_name = 'NBC-pong'
        self.device = torch.device(device)
        self.input_shape = input_shape  # In CHW. (Shape of preprocessed frames)
        self.stack_size = stack_size
        self.stacked_input_shape = (stack_size * input_shape[0],) + input_shape[1:]
        self.num_actions = num_actions
        self.policy_net = network_fn(self.stacked_input_shape, num_actions)
        self.target_net = network_fn(self.stacked_input_shape, num_actions)
        self.optimizer = optim.Adam(self.policy_net.parameters(), betas=(beta0, beta1), lr=learning_rate)
        self.memory = ReplayBuffer(replay_memory_size, input_shape, (1,), prioritized=prioritized, stack_size=stack_size)
        self.minibatch_size = minibatch_size
        self.gamma = gamma
        self.normalize = normalize

        self.state_history = np.zeros(self.stacked_input_shape, dtype=np.uint8)

    def reset(self):
        """Reset agent's state after an episode has finished."""
        self.state_history = np.zeros(self.stacked_input_shape, dtype=np.uint8)

    def get_name(self):
        """Returns the name of the agent."""
        return self.agent_name

    def load_model(self, model_path):
        """Loads model's parameters from path."""
        self.policy_net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.policy_net.eval()

    def _preprocess_state(self, state):
        """Apply preprocessing (grayscale, resize) to a state."""
        # Resize image.
        state = PIL.Image.fromarray(state).resize(self.input_shape[1:], resample=PIL.Image.NEAREST)

        # RGB -> grayscale and convert to numpy array.
        state = np.array(state.convert('L'))

        return state

    def get_action(self, state, epsilon=0.1):
        """Determine action for a given state."""
        # Preprocess state.
        state = self._preprocess_state(state)

        self.state_history = np.roll(self.state_history, -1, axis=0)
        self.state_history[-1] = state
        state = self.state_history[None, :]  # Add batch dimension.

        if random.random() > epsilon:  # Use Q-values.
            with torch.no_grad():
                state = torch.from_numpy(state)
                q_values = self.policy_net(state)
                return torch.argmax(q_values).item()
        else:  # Select action uniform random.
            return random.randrange(self.num_actions)

    def td_loss_double_dqn(self):
        """TD loss for double DQN, proposed in https://arxiv.org/abs/1509.06461."""
        (states, actions, rewards, next_states, dones), idxs  = \
                self.memory.sample_batch(self.minibatch_size)

        # Extract states, next_states, rewards, done signals from transitions
        states = torch.from_numpy(states)
        actions = torch.from_numpy(actions).long()
        rewards = torch.from_numpy(rewards.copy())
        next_states = torch.from_numpy(next_states)
        dones = torch.from_numpy(dones.astype(np.float32))

        # Get Q-values from current network and target network.
        q_values = self.policy_net.forward(states)
        next_q_values = self.policy_net.forward(next_states)
        next_q_state_values = self.target_net.forward(next_states)

        q_value = q_values.gather(1, actions).squeeze(1)
        next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = rewards + self.gamma * next_q_value * (1 - dones)
        loss = (q_value - expected_q_value.detach()).pow(2).mean()  # F.smooth_l1_loss(q_value, expected_q_value.detach())
        td_err = np.abs((q_value- expected_q_value).data)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.memory.update_td_errors(idxs, td_err)

        return loss.item()

    def td_loss(self):
        """Compute TD loss."""
        (states, actions, rewards, next_states, dones), idxs  = \
                self.memory.sample_batch(self.minibatch_size)

        # Extract states, next_states, rewards, done signals from transitions
        states = torch.from_numpy(states)
        actions = torch.from_numpy(actions).long()
        rewards = torch.from_numpy(rewards.copy())
        next_states = torch.from_numpy(next_states)
        dones = torch.from_numpy(dones.astype(np.float32))

        # Get Q-values from current network and target network.
        q_values = self.policy_net.forward(states)
        next_q_values = self.policy_net.forward(next_states)

        q_value = q_values.gather(1, actions).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = rewards + self.gamma * next_q_value * (1 - dones)
        loss = (q_value - expected_q_value.detach()).pow(2).mean()  # F.smooth_l1_loss(q_value, expected_q_value.detach())
        td_err = np.abs((q_value- expected_q_value).data)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.memory.update_td_errors(idxs, td_err)

        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def store_transition(self, state, action, next_state, reward, done):
        # Preprocess next state.
        state = self._preprocess_state(state)
        next_state = self._preprocess_state(next_state)

        self.memory.store_transition(state, action, reward, next_state, done)
