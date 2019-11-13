"""Implementation of Pong agent."""

import numpy as np
import PIL.Image
import random
import torch
import torch.nn as nn
import torch.optim as optim

from networks import VanillaDQN
from replay_memory import ReplayMemory


#TODO: Tehaan tanne implementaatiot eri agenteista.
#TODO: Voisi tehda jonkinlaisen base agentin, josta muut agentit voisi peria asioita.

class Agent():
    def __init__(self, input_shape, num_actions, minibatch_size=128,
                 replay_memory_size=100000, gamma=0.98, beta0=0.9, beta1=0.999,
                 learning_rate=1e-4, device='cuda', **kwargs):
        self.agent_name = 'NBC-pong'
        self.device = torch.device(device)
        self.input_shape = input_shape  # In CHW. (Shape of preprocessed frames)
        self.num_actions = num_actions
        self.policy_net = VanillaDQN(input_shape, num_actions)
        self.target_net = VanillaDQN(input_shape, num_actions)
        self.optimizer = optim.Adam(self.policy_net.parameters(), betas=(beta0, beta1), lr=learning_rate)
        self.memory = ReplayMemory(replay_memory_size)
        self.minibatch_size = minibatch_size
        self.gamma = gamma

        self.state_history = np.zeros(input_shape, dtype=np.uint8)
        self.next_state_history = np.zeros(input_shape, dtype=np.uint8)

    def reset(self):
        """Reset agent's state after an episode has finished."""
        self.state_history = np.zeros(self.input_shape, dtype=np.uint8)
        self.next_state_history = np.zeros(self.input_shape, dtype=np.uint8)

    def get_name(self):
        """Returns the name of the agent."""
        return self.agent_name

    def load_model(self, model_path):
        """Loads model's parameters from path."""
        pass

    def _preprocess_state(self, state):
        """Apply preprocessing (grayscale, resize) to a state."""
        # Resize image.
        state = PIL.Image.fromarray(state).resize(self.input_shape[1:], resample=PIL.Image.NEAREST)

        # RGB -> grayscale and convert to numpy array.
        state = np.array(state.convert('L'))

        return state

    def get_action(self, state, epsilon=0.05):
        """Determine action for a given state."""
        # Preprocess state.
        state = self._preprocess_state(state)

        state_history = self.state_history
        if np.all(state_history == 0):
            state_history[:, ...] = state
        else:
            state_history[1:] = state_history[:-1]
            state_history[0] = state
        state = state_history[None, :]  # Add batch dimension.

        if random.random() > epsilon:  # Use Q-values.
            with torch.no_grad():
                state = torch.from_numpy(state)
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
        loss = (q_value - expected_q_value.detach()).pow(2).mean()
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def store_transition(self, state, action, next_state, reward, done):
        # Preprocess next state.
        preprocessed_next_state = self._preprocess_state(next_state)

        # If there is no previous next_states stack the current frame n times.
        # Note state history is saved as np.float32 but replay memory uses np.uint8 to
        # save memory.
        if np.all(self.next_state_history == 0):
            self.next_state_history[:, ...] = preprocessed_next_state
        else:
            self.next_state_history[1:] = self.next_state_history[:-1]
            self.next_state_history[0] = preprocessed_next_state

        # Push frames to replay memory.
        action = torch.Tensor([action]).long()
        reward = torch.tensor([reward], dtype=torch.float32)
        next_state = torch.from_numpy(self.next_state_history)
        state = torch.from_numpy(self.state_history)
        self.memory.push(state, action, next_state, reward, done)
