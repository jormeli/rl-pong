"""Implementations of networks used by the agent."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from noisy_nets import NoisyLinear


class VanillaDQN(nn.Module):
    """Vanilla deep Q network. (https://arxiv.org/abs/1312.5602)"""
    def __init__(self,
                 input_shape,
                 num_actions,
                 conv_fmaps=32,
                 fc_fmaps=512,
                 **kwargs):
        super(VanillaDQN, self).__init__()

        self.input_shape = input_shape  # In format CHW.
        self.num_actions = num_actions
        self.noisy = kwargs.pop('noisy', False)

        # Conv. layers.
        self.conv_features = nn.Sequential(
            nn.Conv2d(input_shape[0], conv_fmaps, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(conv_fmaps, 2 * conv_fmaps, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(2 * conv_fmaps, 2 * conv_fmaps, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Fully-connected layers.
        flat_conv_outputs = self._get_flat_conv_outputs()

        if self.noisy:
            self.fc1_noisy = NoisyLinear(flat_conv_outputs, fc_fmaps)
            self.fc2_noisy = NoisyLinear(fc_fmaps, self.num_actions)
        else:
            self.fc_layers = nn.Sequential(
                nn.Linear(flat_conv_outputs, fc_fmaps),
                nn.ReLU(),
                nn.Linear(fc_fmaps, self.num_actions)
            )

    def _get_flat_conv_outputs(self):
        """Feed zero tensor through conv. features to obtain flattened output size."""
        return self.conv_features(torch.zeros(1, *self.input_shape)).view(1, -1).shape[1]

    def forward(self, x, training=False):
        """Forward pass the network."""
        # Convert to float and scale from [0, 255] to [-1, 1].
        x = x.float()
        x = (x - 127.5) / 127.5

        # Feed x through conv. layers.
        x = self.conv_features(x)
        x = x.view(x.shape[0], -1)  # Flatten (N, C, H, W) -> (N, C * H * W).

        if self.noisy:
            x = F.relu(self.fc1_noisy(x, training=training))
            x = self.fc2_noisy(x, training=training)
        else:
            x = self.fc_layers(x)

        return x

    def resample_noise(self):
        if self.noisy:
            self.fc1_noisy.resample_noise_epsilons()
            self.fc2_noisy.resample_noise_epsilons()


class DuelingDQN(nn.Module):
    """DQN with dueling architecture. (https://arxiv.org/abs/1511.06581)"""
    def __init__(self, input_shape, num_actions, conv_fmaps=32, fc_fmaps=512, **kwargs):
        super(DuelingDQN, self).__init__()

        self.input_shape = input_shape  # In format CHW.
        self.num_actions = num_actions
        self.noisy = kwargs.pop('noisy', False)

        self.conv_features = nn.Sequential(
            nn.Conv2d(input_shape[0], conv_fmaps, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(conv_fmaps, 2 * conv_fmaps, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(2 * conv_fmaps, 2 * conv_fmaps, kernel_size=3, stride=1),
            nn.ReLU()
        )

        flat_conv_outputs = self._get_flat_conv_outputs()

        if self.noisy:
            self.fc1_advantage = NoisyLinear(flat_conv_outputs, fc_fmaps)
            self.fc2_advantage = NoisyLinear(fc_fmaps, num_actions)
            self.fc1_value = NoisyLinear(flat_conv_outputs, fc_fmaps)
            self.fc2_value = NoisyLinear(fc_fmaps, 1)
        else:
            self.advantage_stream = nn.Sequential(
                nn.Linear(flat_conv_outputs, fc_fmaps),
                nn.ReLU(),
                nn.Linear(fc_fmaps, num_actions)
            )
            self.value_stream = nn.Sequential(
                nn.Linear(flat_conv_outputs, fc_fmaps),
                nn.ReLU(),
                nn.Linear(fc_fmaps, 1)
            )

    def _get_flat_conv_outputs(self):
        """Feed zero tensor through conv. features to obtain flattened output size."""
        return self.conv_features(torch.zeros(1, *self.input_shape)).view(1, -1).shape[1]

    def forward(self, x):
        """Forward pass the network."""
        # Convert to float and scale from [0, 255] to [-1, 1].
        x = x.float()
        x = (x - 127.5) / 127.5

        # Feed through conv. layers.
        x = self.conv_features(x)
        x = x.view(x.size(0), -1)  # Flatten (N, C, H, W) -> (N, C * H * W).

        # Separately compute value and advantage streams.
        if self.noisy:
            advantage = F.relu(self.fc1_advantage(x))
            advantage = self.fc2_advantage(advantage)
            value = F.relu(self.fc1_value(x))
            value = self.fc2_value(value)
        else:
            value = self.value_stream(x)
            advantage = self.advantage_stream(x)

        return value + advantage - advantage.mean()

    def resample_noise(self):
        if self.noisy:
            self.fc1_advantage.resample_noise_epsilons()
            self.fc2_advantage.resample_noise_epsilons()
            self.fc1_value.resample_noise_epsilons()
            self.fc2_value.resample_noise_epsilons()


class CNN(nn.Module):
    def __init__(self, input_shape, num_actions, conv_fmaps=32, fc_fmaps=512, **kwargs):
        super(CNN, self).__init__()

        self.input_shape = input_shape  # In format CHW.
        self.num_actions = num_actions
        self.noisy = kwargs.pop('noisy', False)

        # Conv. layers.
        self.conv_features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=7, stride=2, padding=3),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
        )

        # Fully-connected layers.
        flat_conv_outputs = self._get_flat_conv_outputs()
        if self.noisy:
            self.fc1_noisy = NoisyLinear(flat_conv_outputs, 256)
            self.fc2_noisy = NoisyLinear(256, num_actions)
        else:
            self.fc_layers = nn.Sequential(
                nn.Linear(flat_conv_outputs, 256),
                nn.ELU(),
                nn.Linear(256, num_actions)
            )

    def _get_flat_conv_outputs(self):
        """Feed zero tensor through conv. features to obtain flattened output size."""
        return self.conv_features(torch.zeros(1, *self.input_shape)).view(1, -1).shape[1]

    def forward(self, x):
        """Forward pass the network."""
        # Convert to float and scale from [0, 255] to [-1, 1].
        x = x.float()
        x = (x - 127.5) / 127.5

        # Feed x through layers.
        x = self.conv_features(x)
        x = x.view(x.shape[0], -1)  # Flatten (N, C, H, W) -> (N, C * H * W).

        if self.noisy:
            x = F.elu(self.fc1_noisy(x))
            x = self.fc2_noisy(x)
        else:
            x = self.fc_layers(x)

        return x

    def resample_noise(self):
        if self.noisy:
            self.fc1_noisy.resample_noise_epsilons()
            self.fc2_noisy.resample_noise_epsilons()
