import torch
import torch.nn as nn

class BaseAgent(nn.Module):
    def __init__(self, perception_module, grasping_module, throwing_module):
        """
        Initialize the BaseAgent for TossingBot.
        
        This agent predicts parameters for grasping and throwing based on visual observations
        and target positions by utilizing separate modules for perception, grasping, and throwing.
        
        Args:
            perception_module (nn.Module): Neural network module for perception, processes visual input I.
            grasping_module (nn.Module): Neural network module for predicting grasping parameters.
            throwing_module (nn.Module): Neural network module for predicting throwing parameters.
        """
        super(BaseAgent, self).__init__()
        
        # Assign the modules
        self.perception_module = perception_module
        self.grasping_module = grasping_module
        self.throwing_module = throwing_module

    def forward(self, I, p):
        """
        Forward pass to predict grasping and throwing parameters.

        Args:
            I (torch.Tensor): Visual observation (e.g., image or feature map).
            p (torch.Tensor): 3D target position [x, y, z] of the landing location.

        Returns:
            tuple: Predicted grasping parameters (phi_g) and throwing parameters (phi_t).
        """
        # Step 1: Use perception module to process the visual input and extract spatial features (mu)
        mu = self.perception_module(I)

        # Step 2: Use the perception output (mu) and target position (p) to predict grasping parameters (phi_g)
        phi_g = self.grasping_module(mu, p)

        # Step 3: Use the same perception output (mu) and target position (p) to predict throwing parameters (phi_t)
        phi_t = self.throwing_module(mu, p)

        return phi_g, phi_t