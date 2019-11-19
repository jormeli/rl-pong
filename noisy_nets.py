"""Implementation of a noisy linear layer."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """Noisy linear layer."""
    def __init__(self, fmaps_in, fmaps_out, std_init=0.4):
        super(NoisyLinear, self).__init__()
        
        self.fmaps_in = fmaps_in
        self.fmaps_out = fmaps_out
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.FloatTensor(fmaps_out, fmaps_in))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(fmaps_out, fmaps_in))
        self.register_buffer('weight_epsilon', torch.FloatTensor(fmaps_out, fmaps_in))
        
        self.bias_mu = nn.Parameter(torch.FloatTensor(fmaps_out))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(fmaps_out))
        self.register_buffer('bias_epsilon', torch.FloatTensor(fmaps_out))

        # Initialize deterministic and noisy parameters.
        mu_range = 1 / np.sqrt(self.weight_mu.shape[1])
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.weight_sigma.shape[1]))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.bias_sigma.shape[0]))

        # Sample initial noise epsilons.
        self.resample_noise_epsilons()
    
    def forward(self, x, training=True):
        if training:  # Outputs are noisy.
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:  # Outputs are deterministic.
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)

    def resample_noise_epsilons(self):
        epsilon_in  = self._scale_noise(self.fmaps_in)
        epsilon_out = self._scale_noise(self.fmaps_out)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.fmaps_out))
    
    def _scale_noise(self, fmaps):
        x = torch.randn(fmaps)
        x = x.sign() * torch.sqrt(x.abs())
        return x
