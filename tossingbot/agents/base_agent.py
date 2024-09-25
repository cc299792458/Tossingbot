import torch
import torch.nn as nn

from tossingbot.utils.pytorch_utils import np_image_to_tensor, tensor_to_np_image

class BaseAgent(nn.Module):
    def __init__(self, perception_module, grasping_module, throwing_module, device):
        """
        Initialize the BaseAgent for TossingBot.
        
        This agent predicts parameters for grasping and throwing based on visual observations
        and target positions by utilizing separate modules for perception, grasping, and throwing.
        
        Args:
            perception_module (nn.Module): Neural network module for perception, processes visual input I.
            grasping_module (nn.Module): Neural network module for predicting grasping parameters.
            throwing_module (nn.Module): Neural network module for predicting throwing parameters.
            device (torch.device): The device to run computations on (e.g., 'cpu' or 'cuda').
        """
        super(BaseAgent, self).__init__()
        
        # Assign the device
        self.device = device

        # Assign the modules
        self.perception_module = perception_module.to(self.device)
        self.grasping_module = grasping_module.to(self.device)
        self.throwing_module = throwing_module.to(self.device)

    def forward(self, I):
        """
        Forward pass to predict grasping and throwing parameters.

        Args:
            I (torch.Tensor): Visual observation (e.g., image or feature map).

        Returns:
            tuple: Predicted grasping parameters (phi_g) and throwing parameters (phi_t).
        """
        # Step 1: Use perception module to process the visual input and extract spatial features (mu)
        mu = self.perception_module(I)

        # Step 2: Use the perception output (mu) to predict grasping probability map (q_g)
        q_g = self.grasping_module(mu)

        # Step 3: Use the same perception output (mu) to predict throwing probability map (q_t)
        q_t = self.throwing_module(mu)

        return q_g, q_t
    
    def predict(self, observation):
        I, p = observation

        I_tensor = np_image_to_tensor(I, self.device)

        q_g_tensor, q_t_tensor = self.forward(I_tensor)
        
        

        action = None

        return action