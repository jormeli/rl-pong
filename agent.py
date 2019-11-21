"""Implementation of Pong agent."""

import numpy as np
import PIL.Image
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from loss import double_dqn_loss
from networks import DuelingDQN, VanillaDQN
from noisy_nets import NoisyLinear
from replay_buffer import ReplayBuffer


class Agent():
    def __init__(self,
                 input_shape,
                 num_actions,
                 network_fn=VanillaDQN,
                 network_fn_kwargs=None,
                 loss_fn=double_dqn_loss,
                 minibatch_size=128,
                 replay_memory_size=500000,
                 stack_size=1,
                 gamma=0.98,
                 beta0=0.9,
                 beta1=0.999,
                 learning_rate=1e-4,
                 device='cpu',
                 normalize=False,
                 prioritized=True,
                 **kwargs):

        self.agent_name = 'NBC-pong'
        self.device = torch.device(device)
        self.input_shape = input_shape  # In CHW. (Shape of preprocessed frames)
        self.stack_size = stack_size
        self.stacked_input_shape = (stack_size * input_shape[0],) + input_shape[1:]
        self.num_actions = num_actions
        self.loss_fn = loss_fn

        if network_fn_kwargs is None:
            network_fn_kwargs = {}

        self.policy_net = network_fn(self.stacked_input_shape, num_actions, **network_fn_kwargs)
        self.target_net = network_fn(self.stacked_input_shape, num_actions, **network_fn_kwargs)
        self.optimizer = optim.Adam(self.policy_net.parameters(), betas=(beta0, beta1), lr=learning_rate)
        self.memory = ReplayBuffer(replay_memory_size, input_shape, (1,), prioritized=prioritized, stack_size=stack_size)
        self.minibatch_size = minibatch_size
        self.gamma = gamma
        self.normalize = normalize
        self.noisy = network_fn_kwargs.pop('noisy', False)

        self.state_history = np.zeros(self.stacked_input_shape, dtype=np.uint8)

        # Print policy network architecture.
        print(self.policy_net.eval())

    def reset(self):
        """Reset agent's state after an episode has finished."""
        self.state_history = np.zeros(self.stacked_input_shape, dtype=np.uint8)

    def get_name(self):
        """Returns the name of the agent."""
        return self.agent_name

    def load_model(self, model_path):
        """Loads model's parameters from path."""
        self.policy_net.load_state_dict(torch.load(model_path, map_location=self.device))
        print(self.policy_net.eval())

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

        if np.all(self.state_history == 0):  # Fill state history with observation.
            self.state_history[:, ...] = state
        else:  # Shift state history by one frame and put the newest frame to last channel.
            self.state_history = np.roll(self.state_history, -1, axis=0)
            self.state_history[-1] = state
        state = self.state_history[None, :]  # Add batch dimension.

        if random.random() > epsilon:  # Use Q-values.
            with torch.no_grad():
                state = torch.from_numpy(state)
                q_values = self.policy_net(state, training=False)
                return torch.argmax(q_values).item()
        else:  # Select action uniform random.
            return random.randrange(self.num_actions)

    def compute_loss(self):
        """Compute loss function and update parameters."""
        # Sample a minibatch of observations.
        (states, actions, rewards, next_states, dones), idxs  = \
                self.memory.sample_batch(self.minibatch_size)

        # Extract states, next_states, rewards, done signals from transitions.
        states = torch.from_numpy(states)
        actions = torch.from_numpy(actions).long()
        rewards = torch.from_numpy(rewards.copy())
        next_states = torch.from_numpy(next_states)
        dones = torch.from_numpy(dones.astype(np.float32))

        if self.normalize:  # Normalize rewards within a minibatch.
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # Compute loss function.
        loss, priorities = self.loss_fn(self.policy_net,
                                        self.target_net,
                                        states,
                                        actions,
                                        rewards,
                                        next_states,
                                        dones,
                                        self.gamma,
                                        self.noisy)

        # Minimize loss w.r.t policy network.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update priorities.
        self.memory.update_priorities(idxs, priorities)

        if self.noisy:  # Resample noise epsilons.
            self.policy_net.resample_noise()
            self.target_net.resample_noise()

        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def store_transition(self, state, action, next_state, reward, done):
        # Preprocess next state.
        state = self._preprocess_state(state)
        next_state = self._preprocess_state(next_state)

        self.memory.store_transition(state, action, reward, next_state, done)
