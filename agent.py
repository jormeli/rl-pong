"""Implementation of Pong agent."""

import numpy as np
from skimage import color
import torch
import torch.nn as nn
import torch.optim as optim
import random

from networks import VanillaDQN
from replay_memory import ReplayMemory


#TODO: Tehaan tanne implementaatiot eri agenteista.
#TODO: Voisi tehda jonkinlaisen base agentin, josta muut agentit voisi peria asioita.

class Agent():
    def __init__(self, input_shape, num_actions, minibatch_size=32,
                 replay_memory_size=500000, gamma=0.98, beta0=0.9, beta1=0.999,
                 learning_rate=5e-4, device='cuda', **kwargs):
        self.agent_name = 'NBC-pong'
        self.device = torch.device(device)
        self.input_shape = input_shape  # In CHW.
        self.history_length = input_shape[0]
        self.num_actions = num_actions
        self.policy_net = VanillaDQN(input_shape, num_actions)
        self.target_net = VanillaDQN(input_shape, num_actions)
        self.optimizer = optim.Adam(self.policy_net.parameters(), betas=(beta0, beta1), lr=learning_rate)
        self.memory = ReplayMemory(replay_memory_size)
        self.minibatch_size = minibatch_size
        self.gamma = gamma

        self.state_history = []
        self.next_state_history = []

    def reset(self):
        """Reset agent's state after an episode has finished."""
        self.state_history = []
        self.next_state_history = []

    def get_name(self):
        """Returns the name of the agent."""
        return self.agent_name

    def load_model(self, model_path):
        """Loads model's parameters from path."""
        pass

    def get_action(self, state, epsilon=0.05):
        """Determine action for a given state."""
        # TODO: Add preprocess method, since it's required in two places.
        # Preprocess states. (RGB -> grayscale).
        state = color.rgb2gray(state)[None, :]  # To CHW.

        # Scale from [0, 1] to [-1, 1].
        state = (state - 0.5) / 0.5

        state_history = self.state_history
        if not state_history:
            state_history = self.history_length * [state]
        else:
            state_history.pop(0)
            state_history.append(state)
        state = np.concatenate(state_history)[None, :]  # Add batch dimension.

        if random.random() > epsilon:  # Use Q-values.
            with torch.no_grad():
                state = torch.from_numpy(state).float()
                q_values = self.policy_net(state)
                return torch.argmax(q_values).item()
        else:  # Select action uniform random.
            return random.randrange(self.num_actions)

    def td_loss_double_dqn(self):
        """TD loss for double DQN, proposed in https://arxiv.org/abs/1509.06461."""
        transitions = self.memory.sample(self.minibatch_size)

        # Extract states, next_states, rewards, done signals from transitions
        states = torch.stack([transition.state for transition in transitions], dim=0)
        next_states = torch.stack([transition.next_state for transition in transitions], dim=0)
        actions = torch.stack([transition.action for transition in transitions], dim=0)
        rewards = torch.stack([transition.reward for transition in transitions], dim=0).squeeze(1)
        dones = torch.from_numpy(np.array([int(transition.done) for transition in transitions]))

        # Get Q-values from current network and target network.
        q_values = self.policy_net.forward(states)
        next_q_values = self.policy_net.forward(next_states)
        next_q_state_values = self.target_net.forward(next_states) 

        q_value = q_values.gather(1, actions).squeeze(1)
        next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = rewards + self.gamma * next_q_value * (1 - dones)
        
        loss = (q_value - expected_q_value.data).pow(2).mean()
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def store_transition(self, state, action, next_state, reward, done):
        # Preprocess states. (RGB -> grayscale).
        state = color.rgb2gray(state)[None, :]  # To CHW.
        next_state = color.rgb2gray(next_state)[None, :]

        # Scale from [0, 1] to [-1, 1].
        state = (state - 0.5) / 0.5
        next_state = (next_state - 0.5) / 0.5

        # If there is no previous states/next_states stack the current frame n times.
        if not self.state_history:
            self.state_history = self.history_length * [state]
            self.next_state_history = self.history_length * [next_state]
        else:  # TODO: This is inefficient.
            _ = self.state_history.pop(0)  # Drop first item from history.
            _ = self.next_state_history.pop(0)
            self.state_history.append(state)  # Add new state.
            self.next_state_history.append(next_state)

        action = torch.Tensor([action]).long()
        reward = torch.tensor([reward], dtype=torch.float32)
        next_state = torch.from_numpy(np.concatenate(self.next_state_history)).float()
        state = torch.from_numpy(np.concatenate(self.state_history)).float()
        self.memory.push(state, action, next_state, reward, done)
