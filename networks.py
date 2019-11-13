"""Implementations of networks used by the agent."""

import numpy as np
import torch
import torch.nn as nn


class VanillaDQN(nn.Module):
    """Vanilla deep Q network. (https://arxiv.org/abs/1312.5602)"""
    def __init__(self, input_shape, num_actions, conv_fmaps=32, fc_fmaps=512):
        super(VanillaDQN, self).__init__()

        self.input_shape = input_shape  # In format CHW.
        self.num_actions = num_actions

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
        self.fc_layers = nn.Sequential(
            nn.Linear(flat_conv_outputs, fc_fmaps),
            nn.ReLU(),
            nn.Linear(fc_fmaps, self.num_actions)
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
        x = self.fc_layers(x)
        return x
