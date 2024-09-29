import torch
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

        # Initialize the current epsilon to the start value
        self.current_epsilon = self.epsilon_start

    def update_epsilon(self, episode_num):
        """
        Update the epsilon value based on the current episode number.
        The epsilon value decays linearly from epsilon_start to epsilon_end over total_episodes.
        
        Args:
            episode_num (int): The current episode number.
        """
        fraction = min(float(episode_num) / self.total_episodes, 1.0)
        self.current_epsilon = self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def forward(self, I):
        raise NotImplementedError
    
    def predict(self, observation, n_rotations=16):
        raise NotImplementedError
