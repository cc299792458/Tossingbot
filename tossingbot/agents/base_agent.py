import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tossingbot.envs.pybullet.utils.camera_utils import plot_heightmaps
from tossingbot.networks import PerceptionModule, GraspingModule, ThrowingModule
from tossingbot.utils.pytorch_utils import np_image_to_tensor, tensor_to_np_image
from tossingbot.envs.pybullet.utils.math_utils import rotate_image_array, rotate_image_tensor

class BaseAgent(nn.Module):
    def __init__(
            self, 
            device: torch.device, 
            perception_module: nn.Module, 
            grasping_module: nn.Module, 
            throwing_module: nn.Module, 
            post_grasp_h: float = 0.3,    # The horizontal distance from robot base to post grasp pose
            post_grasp_z: float = 0.4,    # The height for the post grasp pose
            epsilons: list[float] = [0.5, 0.1]
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
        """
        super(BaseAgent, self).__init__()
        
        # Assign the device
        self.device = device

        # Assign the post_grasp_h and post_grasp_z
        self.post_grasp_h = post_grasp_h
        self.post_grasp_z = post_grasp_z

        # Assign the modules
        self.perception_module = perception_module.to(self.device)
        self.grasping_module = grasping_module.to(self.device)
        self.throwing_module = throwing_module.to(self.device)

        # Assign the epsilons
        self.epsilons = epsilons

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
    
    def predict(self, observation, n_rotations=16):
        I, p = observation
        grasp_logits = []  # Store network raw outputs (logits) for each rotation
        depth_heightmaps = []    # Store the rotated depth heightmap for visualization

        # Convert the input image to tensor once
        I_tensor = np_image_to_tensor(I, self.device)

        for i in range(n_rotations):
            # Calculate rotation angle for the current rotation step
            theta = 360 / n_rotations * i

            # Rotate the input tensor
            I_rotated = rotate_image_tensor(I_tensor, theta=theta)

            # Store the rotated heightmap(only depthmap)
            depth_heightmaps.append(I_rotated[0, -1, :, :].detach().cpu().numpy())

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

        target_x, target_y = p[0], p[1]
        theta = np.arctan2(target_y / target_x)
        post_grasp_pose = ([self.post_grasp_h * np.cos(theta), self.post_grasp_h * np.sin(theta), self.post_grasp_z],
                           [1.0, 0.0, 0.0, 0.0])

        intermidiates = {"depth_heightmaps": depth_heightmaps,
                         "q_t_tensor": q_t_tensor}

        return (grasp_pixel_index, post_grasp_pose, None, None), intermidiates

def draw_arrow_on_last_channel(heightmap, start, end):
    """
    Modify the last channel of the heightmap to include an arrow shape.

    Args:
        heightmap (ndarray): Heightmap with shape (H, W, C) where the last channel is modified.
        start (tuple): Starting point (x, y) of the arrow.
        end (tuple): End point (x, y) of the arrow.
    """
    # Get heightmap dimensions
    H, W, C = heightmap.shape

    # Calculate the vector components for the arrow (direction from start to end)
    arrow_length = max(abs(end[0] - start[0]), abs(end[1] - start[1]))
    dx = (end[0] - start[0]) / arrow_length
    dy = (end[1] - start[1]) / arrow_length

    # Draw the arrow on the last channel (increment values along the arrow's path)
    for i in range(int(arrow_length)):
        x = int(start[0] + i * dx)
        y = int(start[1] + i * dy)
        if 0 <= x < W and 0 <= y < H:
            heightmap[y, x, -1] = 1  # Modify the last channel to mark the arrow path

    # Mark the arrowhead (using a simple cross to denote the arrowhead)
    if 0 <= end[0] < W and 0 <= end[1] < H:
        heightmap[end[1], end[0], -1] = 2  # Arrowhead is a stronger mark
        if end[1] + 1 < H: heightmap[end[1] + 1, end[0], -1] = 2
        if end[1] - 1 >= 0: heightmap[end[1] - 1, end[0], -1] = 2
        if end[0] + 1 < W: heightmap[end[1], end[0] + 1, -1] = 2
        if end[0] - 1 >= 0: heightmap[end[1], end[0] - 1, -1] = 2

    return heightmap

if __name__ == '__main__':
    # Initialize input data
    heightmap = np.zeros([80, 60, 4])  # 80x60 heightmap with 4 channels
    start_point = (10, 20)  # Starting point of the arrow
    end_point = (50, 60)  # End point of the arrow
    heightmap = draw_arrow_on_last_channel(heightmap, start=start_point, end=end_point)
    target_position = np.array([2.0, 0.0, 0.1])  # Target position in (x, y, z)

    # Initialize modules and agent
    perception_module = PerceptionModule()
    grasp_module = GraspingModule()
    throw_module = ThrowingModule()
    agent = BaseAgent(
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        perception_module=perception_module,
        grasping_module=grasp_module,
        throwing_module=throw_module,
    )

    # Make prediction with observation
    observation = (heightmap, target_position)
    action, intermidiates = agent.predict(observation=observation, n_rotations=1)
    plot_heightmaps(intermidiates['depth_heightmaps'])