"""Implementation of Pong agent."""

import numpy as np
import random
import torch
import torch.nn as nn


#TODO: Tehaan tanne implementaatiot eri agenteista.

class Agent():

    def __init__(self, name, device='cuda'):
        self.agent_name = name
        self.device = torch.devive(device)

    def reset(self):
        """Reset agent's state after an episode has finished."""
        pass

    def get_name(self):
        """Returns the name of the agent."""
        return self.agent_name

    def load_model(self, model_path):
        """Loads model's parameters from path."""
        pass

    def get_action(self, state, epsilon):
        """Determine action for a given state."""
        if random.random() > 1 - epsilon:  # Use Q-values.
            pass
        else:  # Select action uniform random.
            pass
    