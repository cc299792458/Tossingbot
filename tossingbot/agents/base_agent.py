import torch
import numpy as np
import torch.nn as nn

class BaseAgent(nn.Module):
    def __init__(
            self, 
            device: torch.device, 
            perception_module: nn.Module = None, 
            grasping_module: nn.Module = None, 
            throwing_module: nn.Module = None, 
            epsilons: list[float] = [0.5, 0.1],
            total_episodes: int = 10000,
            decay_factor: float = 0.2  # Decay factor for tau, set as a multiple of total_episodes
        ):
        """
        Initialize the BaseAgent for TossingBot.
        
        This agent predicts parameters for grasping and throwing based on visual observations
        and target positions by utilizing separate modules for perception, grasping, and throwing.
        
        Args:
            device (torch.device): The device to run computations on (e.g., 'cpu' or 'cuda').
            perception_module (nn.Module): Neural network module for perception, processes visual input I.
            grasping_module (nn.Module): Neural network module for predicting grasping parameters.
            throwing_module (nn.Module): Neural network module for predicting throwing parameters.
            epsilons (list[float]): Epsilon used for epsilon-greedy.
            total_episodes (int): Total training episodes.
            decay_factor (float): Factor that determines the decay rate of epsilon, as a multiple of total_episodes.
        """
        super(BaseAgent, self).__init__()
        
        # Assign the device
        self.device = device

        # Assign the modules
        self.perception_module = perception_module.to(self.device)
        self.grasping_module = grasping_module.to(self.device)
        self.throwing_module = throwing_module.to(self.device)

        # Assign the initial and final epsilon values
        self.epsilon_start = epsilons[0]
        self.epsilon_end = epsilons[1]
        self.total_episodes = total_episodes
        self.tau = decay_factor * total_episodes  # Set tau as a multiple of total_episodes

        # Initialize the current epsilon to the start value
        self.current_epsilon = self.epsilon_start

    def update_epsilon(self, episode_num):
        """
        Update the epsilon value using exponential decay based on the current episode number.
        The epsilon value decays smoothly from epsilon_start to epsilon_end over total_episodes.
        
        Args:
            episode_num (int): The current episode number.
        """
        # Exponential decay formula
        self.current_epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-episode_num / self.tau)

    def forward(self):
        raise NotImplementedError
    
    def predict(self):
        raise NotImplementedError
