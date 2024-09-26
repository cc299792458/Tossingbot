import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tossingbot.networks import PerceptionModule, GraspingModule, ThrowingModule
from tossingbot.utils.pytorch_utils import np_image_to_tensor, tensor_to_np_image
from tossingbot.envs.pybullet.utils.math_utils import rotate_image_array, rotate_image_tensor

class BaseAgent(nn.Module):
    def __init__(self, perception_module: nn.Module, grasping_module: nn.Module, throwing_module: nn.Module, device: torch.device):
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
    
    def predict(self, observation, n_rotation=16):
        I, p = observation
        grasp_logits = []  # Store network raw outputs (logits) for each rotation

        # Convert the input image to tensor once
        I_tensor = np_image_to_tensor(I, self.device)

        for i in range(n_rotation):
            # Calculate rotation angle for the current rotation step
            theta = 360 / n_rotation * i

            # Rotate the input tensor
            I_rotated = rotate_image_tensor(I_tensor, theta=theta)

            # Forward pass through the network
            q_g_tensor, q_t_tensor = self.forward(I_rotated)

            # Undo the rotation on the grasp logits
            q_g_rotated_back = rotate_image_tensor(q_g_tensor, theta=-theta)

            # Collect the network raw outputs (logits) for this rotation
            grasp_logits.append(q_g_rotated_back)

        # Concatenate all grasp logits along the first dimension (n_rotation)
        concatenated_logits = torch.cat(grasp_logits, dim=0)  # Concatenate along the first dimension

        # Apply softmax to obtain the final grasp affordances (probabilities)
        grasp_affordances = F.softmax(concatenated_logits, dim=1)

        # Detach from the computation graph and move to CPU
        grasp_affordances = grasp_affordances.detach().cpu().numpy()[:, 0, :, :]  # Actual affordances

        # Find the index of the maximum value across the entire (n_rotation, H, W) array
        grasp_pixel_index = np.unravel_index(np.argmax(grasp_affordances, axis=None), grasp_affordances.shape)

        return (grasp_pixel_index, None)

if __name__ == '__main__':
    # Initialize input data
    heightmap = np.zeros([60, 80, 4])  # 60x80 heightmap with 4 channels
    target_position = np.array([2.0, 0.0, 0.1])  # Target position in (x, y, z)

    # Initialize modules and agent
    perception_module = PerceptionModule()
    grasp_module = GraspingModule()
    throw_module = ThrowingModule()
    agent = BaseAgent(
        perception_module=perception_module,
        grasping_module=grasp_module,
        throwing_module=throw_module,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )

    # Make prediction with observation
    observation = (heightmap, target_position)
    agent.predict(observation=observation)