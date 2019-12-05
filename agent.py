"""Implementation of Pong agent."""

import numpy as np
import PIL.Image
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from loss import categorical_loss, double_dqn_loss, td_loss
from networks import DuelingDQN, VanillaDQN, CNN
from replay_buffer import ReplayBuffer


class Agent():
    def __init__(self,
                 input_shape=(1, 84, 84),
                 num_actions=3,
                 network_fn=DuelingDQN,
                 network_fn_kwargs={'noisy': True},
                 loss_fn=td_loss,
                 minibatch_size=32,
                 replay_memory_size=1000000,
                 stack_size=4,
                 gamma=0.99,
                 beta0=0.9,
                 beta1=0.999,
                 learning_rate=2.5e-4,
                 ema_decay=0.99,
                 device='cpu',
                 normalize=False,
                 prioritized=True,
                 **kwargs):

        self.agent_name = 'NBC-labs'
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
        self.policy_net_ema = network_fn(self.stacked_input_shape, num_actions, **network_fn_kwargs)
        self.policy_net_ema.load_state_dict(self.policy_net.state_dict())  # Load initial weights.
        self.optimizer = optim.Adam(self.policy_net.parameters(), betas=(beta0, beta1), lr=learning_rate)
        self.memory = ReplayBuffer(replay_memory_size, input_shape, (1,), prioritized=prioritized, stack_size=stack_size)
        self.minibatch_size = minibatch_size
        self.gamma = gamma
        self.normalize = normalize
        self.ema_decay = ema_decay
        self.noisy = network_fn_kwargs.pop('noisy', False)
        self.categorical = network_fn_kwargs.pop('categorical', False)
        self.V_min = network_fn_kwargs.pop('V_min', -10.0)
        self.V_max = network_fn_kwargs.pop('V_max', 10.0)
        self.num_atoms = network_fn_kwargs.pop('num_atoms', 51)

        self.state_history = np.zeros(self.stacked_input_shape, dtype=np.uint8)

        # Check that categorical loss is used with distributional agent.
        if self.categorical:
            assert self.loss_fn == categorical_loss

        # Print policy network architecture.
        print(self.policy_net.eval())
        print('Loss function: ', self.loss_fn)

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

    def get_action(self, state, epsilon=0.0):
        """Determine action for a given state."""
        # Preprocess state.
        state = self._preprocess_state(state)

        if np.all(self.state_history == 0):  # Fill state history with observation.
            self.state_history[:, ...] = state
        else:  # Shift state history by one frame and put the newest frame to last channel.
            self.state_history = np.roll(self.state_history, -1, axis=0)
            self.state_history[-1] = state
        state = self.state_history[None, :]  # Add batch dimension.

        if random.random() > epsilon:  # Use Q-values / Q-value distribution.
            with torch.no_grad():
                state = torch.from_numpy(state)
                if self.categorical:
                    dist = self.policy_net(state, training=False)
                    dist = dist * torch.linspace(self.V_min, self.V_max, self.num_atoms)
                    action = torch.argmax(dist.sum(2)).item()  # Sum over atoms.
                    return action
                else:
                    q_values = self.policy_net(state, training=False)
                    return torch.argmax(q_values).item()
        else:  # Select action uniform random.
            return random.randrange(self.num_actions)

    def compute_loss(self, beta=None):
        """Compute loss function and update parameters."""
        # Sample a minibatch of observations.
        (states, actions, rewards, next_states, dones), idxs, weights  = \
                self.memory.sample_batch(self.minibatch_size, beta=beta)

        # Extract states, next_states, rewards, done signals from transitions.
        states = torch.from_numpy(states)
        actions = torch.from_numpy(actions).long()
        rewards = torch.from_numpy(rewards.copy())
        next_states = torch.from_numpy(next_states)
        dones = torch.from_numpy(dones.astype(np.float32))
        weights = torch.from_numpy(weights.astype(np.float32))

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
                                        weights,
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
        """Update target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_ema_policy(self):
        """Update EMA weights of policy network."""
        # Apply EMA to current policy network weights.
        new_ema_state_dict = {}
        for ((name, param), (ema_name, ema_param)) in zip(self.policy_net.named_parameters(), self.policy_net_ema.named_parameters()):
            assert name == ema_name
            new_ema_state_dict[ema_name] = (1 - self.ema_decay) * param + self.ema_decay * ema_param

        # Copy registered buffers to new state dict.
        # These are actually newer used but required in order to update EMA model.
        for buf_name, buf in self.policy_net.named_buffers():
            new_ema_state_dict[buf_name] = buf

        # Update EMA policy network from last update with new EMA weights.
        self.policy_net_ema.load_state_dict(new_ema_state_dict)

    def store_transition(self, state, action, next_state, reward, done):
        """Store a transition."""
        # Preprocess next state.
        state = self._preprocess_state(state)
        next_state = self._preprocess_state(next_state)

        self.memory.store_transition(state, action, reward, next_state, done)
