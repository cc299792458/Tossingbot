import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R
from tossingbot.agents.base_agent import BaseAgent
from tossingbot.utils.pytorch_utils import np_image_to_tensor
from tossingbot.envs.pybullet.utils.math_utils import rotate_image_tensor

class PhysicsController:
    def __init__(self, r_h=0.6, r_z=0.4):
        """
        Initialize the PhysicsController with specified parameters for throw distance and height.

        Args:
            r_h (float): The horizontal distance for the throw position.
            r_z (float): The height for the throw position.
        """
        self.r_h = r_h
        self.r_z = r_z

    def predict(self, target_pos, phi_deg, g=9.81):
        """
        Calculate the throw parameters based on the target position and throw angle.

        Args:
            target_pos (tuple): The target position (p_x, p_y, p_z).
            phi_deg (float): The angle for throwing in the vertical plane (in degrees).
            g (float): Gravitational acceleration, default is 9.81 m/s².

        Returns:
            tuple: The calculated throw position (r_x, r_y, r_z) and velocity components (v_x, v_y, v_z).
        """
        # Target position coordinates
        p_x, p_y, p_z = target_pos

        # Convert the throw angle (phi) from degrees to radians
        phi = np.radians(phi_deg)

        # Calculate the horizontal distance and the angle (theta) in the XY plane
        p_h = np.sqrt(p_x**2 + p_y**2)
        theta = np.arctan2(p_y, p_x)

        # Calculate the throw position (r_x, r_y, r_z)
        r_x = self.r_h * np.cos(theta)
        r_y = self.r_h * np.sin(theta)
        r_z = self.r_z

        # Calculate vertical and horizontal differences
        delta_z = p_z - r_z
        delta_h = p_h - self.r_h

        # Calculate the velocity magnitude using the provided formula
        numerator = g * delta_h**2
        denominator = 2 * np.cos(phi)**2 * (np.tan(phi) * delta_h - delta_z)
        
        if denominator <= 0:
            raise ValueError("Invalid parameters leading to an impossible trajectory.")

        v_magnitude = np.sqrt(numerator / denominator)

        # Compute velocity components
        v_x = v_magnitude * np.cos(phi) * np.cos(theta)
        v_y = v_magnitude * np.cos(phi) * np.sin(theta)
        v_z = v_magnitude * np.sin(phi)

        return (r_x, r_y, r_z), (v_x, v_y, v_z)

class PhysicsAgent(BaseAgent):
    def __init__(
            self, 
            device: torch.device, 
            perception_module: nn.Module = None, 
            grasping_module: nn.Module = None, 
            throwing_module: nn.Module = None, 
            physics_controller: PhysicsController = None,
            epsilons: list[float] = [0.5, 0.1],
            total_episodes: int = 10000,  # Add total episodes for scheduling
        ):
        """
        Initialize the PhysicsAgent for TossingBot, inheriting from BaseAgent.

        This agent will incorporate physics-based decision-making into grasping and throwing actions.
        
        Args:
            device (torch.device): The device to run computations on (e.g., 'cpu' or 'cuda').
            perception_module (nn.Module): Neural network module for perception, processes visual input I.
            grasping_module (nn.Module): Neural network module for predicting grasping parameters.
            throwing_module (nn.Module): Neural network module for predicting throwing parameters.
            physics_controller (PhysicsController): Module to calculate throwing parameters based on physics.
            epsilons (list[float]): Epsilon values for exploration.
        """
        super(PhysicsAgent, self).__init__(device, perception_module, grasping_module, throwing_module, epsilons, total_episodes)
        self.physics_controller = physics_controller

    def forward(self, I, v):
        """
        Forward pass to predict grasping and throwing parameters.

        Args:
            I (torch.Tensor): Visual observation (e.g., image or feature map).
            v (np.array): Predicted velocity for each element in the batch, with length B (same as batch size).
            
        Returns:
            tuple: Predicted grasping parameters (q_g) and throwing parameters (q_t).
        """
        # Step 1: Process the visual input using the perception module to extract visual features (mu)
        mu = self.perception_module(I)

        # Step 2: Concatenate the visual features with the predicted velocity
        v = torch.tensor(v, device=self.device, dtype=torch.float)
        
        B, _, H, W = mu.shape
        assert v.shape[0] == B, f"Length of velocity list/tensor (v) must match batch size (B). Got {v.shape[0]} and {B}."

        # Create a velocity image with the same spatial dimensions as mu and concatenate it
        velocity_images = v.view(B, 1, 1, 1).expand(B, 128, H, W)  # Broadcast v over the spatial dimensions and channels
        fused_features = torch.cat([mu, velocity_images], dim=1)  # Concatenate along the channel dimension

        # Step 3: Predict the grasping probability map (q_g) and throwing probability map (q_t)
        q_g = self.grasping_module(fused_features)
        q_t = self.throwing_module(fused_features)

        return q_g, q_t

    def predict(self, observation, n_rotations=16, phi_deg=45, episode_num=None, use_heuristic=False):
        """
        Predict the best grasping and throwing actions based on the observation.

        Args:
            observation (list of tuples): Each tuple consists of an image (I) and a target position (p).
            n_rotations (int): Number of rotations to consider for grasping.
            phi_deg (float): Angle for throwing in degrees.
            episode_num (int, optional): The current episode number for epsilon decay.
            use_heuristic (bool): If True, use the heuristic method to select the grasping pixel.

        Returns:
            tuple: (grasp_pixel_indices, post_grasp_poses, throw_poses, throw_velocities) and intermediate values.
        """
        # Update epsilon value if episode number is provided
        if episode_num is not None:
            self.update_epsilon(episode_num)

        # Process the observations into batches of images and target positions
        I_batch, p_batch = self._process_observations(observation)

        # Compute throw parameters for each target position
        r_batch, v_batch, v_magnitude_batch = self._compute_throw_parameters(p_batch, phi_deg)

        # Perform forward pass with rotations to get q_g, q_t, and affordances
        grasp_affordances, depth_heightmaps, q_g, q_t = self._compute_affordances(I_batch, v_magnitude_batch, n_rotations)

        if use_heuristic:
            # Use the heuristic method to select grasp pixels
            grasp_pixel_indices = self._grasp_heuristic(depth_heightmaps)
        else:
            # Select grasping actions using epsilon-greedy exploration
            grasp_pixel_indices = self._select_grasp_pixels(grasp_affordances)

        # Compute post-grasp poses, throwing poses, and velocities
        post_grasp_poses, throw_poses, throw_velocities = self._compute_poses(p_batch, r_batch, v_batch, phi_deg)

        # Prepare intermediate results
        intermediates = {
            "depth_heightmaps": depth_heightmaps,
            "grasp_affordances": grasp_affordances,
            "q_g": q_g,
            "q_t": q_t,
        }

        action = self._pack_action(grasp_pixel_indices, post_grasp_poses, throw_poses, throw_velocities)

        return action, intermediates

    def _process_observations(self, observation):
        """Extracts image batches and target positions from the observation."""
        I_batch = np.concatenate([np.expand_dims(obs[0], axis=0) for obs in observation], axis=0)
        p_batch = np.array([obs[1] for obs in observation])
        return I_batch, p_batch

    def _compute_throw_parameters(self, p_batch, phi_deg):
        """Calculates throwing positions, velocities, and magnitudes for each target."""
        r_batch, v_batch, v_magnitude_batch = zip(*[
            (
                (r_x, r_y, r_z),  # Throw position
                (v_x, v_y, v_z),  # Velocity components
                np.sqrt(v_x**2 + v_y**2 + v_z**2)  # Velocity magnitude
            )
            for p in p_batch
            for ((r_x, r_y, r_z), (v_x, v_y, v_z)) in [self.physics_controller.predict(target_pos=p, phi_deg=phi_deg)]
        ])

        return np.array(r_batch), np.array(v_batch), np.array(v_magnitude_batch)

    def _compute_affordances(self, I_batch, v_magnitude_batch, n_rotations):
        """Rotates inputs, performs forward passes, and computes grasp affordances."""
        q_g = []
        q_t = []
        depth_heightmaps = []

        I_tensor = np_image_to_tensor(I_batch, self.device)

        for i in range(n_rotations):
            theta = 360 / n_rotations * i
            I_rotated = rotate_image_tensor(I_tensor, theta=theta)
            depth_heightmaps.append(I_rotated[:, -1, :, :].detach().cpu().unsqueeze(1).numpy()) # [B, R, H, W]

            # Perform forward pass for each rotation
            q_g_tensor, q_t_tensor = self.forward(I=I_rotated, v=v_magnitude_batch)

            # Undo the rotation for grasp logits
            q_g_rotated_back = rotate_image_tensor(q_g_tensor, theta=-theta)
            q_t_rotated_back = rotate_image_tensor(q_t_tensor, theta=-theta)

            # Collect the q_g and q_t for this rotation
            q_g.append(q_g_rotated_back.unsqueeze(1))
            q_t.append(q_t_rotated_back.unsqueeze(1))

        # Concatenate q_g and q_t along the new rotation dimension
        q_g = torch.cat(q_g, dim=1) # [B, R, C, H, W]
        q_t = torch.cat(q_t, dim=1) # [B, R, C, H, W]

        # Concatenate depth_heightmaps along the rotation dimension
        depth_heightmaps = np.concatenate(depth_heightmaps, axis=1)  # [B, R, H, W]

        # Compute grasp affordances
        grasp_affordances = F.softmax(q_g, dim=2).detach().cpu().numpy()[:, :, 0, :, :]

        return grasp_affordances, depth_heightmaps, q_g, q_t
    
    def _grasp_heuristic(self, depth_heightmaps):
        """
        Apply the grasp heuristic to a batch of depth heightmaps and return the best grasp pixel indices.

        Args:
            depth_heightmaps (np.array): A batch of depth heightmaps with shape (B, R, H, W).
            
        Returns:
            best_pix_indices (np.array): The best grasp pixel indices for each image in the batch with shape (B, 3) (r, h, w).
        """
        B, R, H, W = depth_heightmaps.shape  # Batch size, Rotations, Height, Width
        grasp_predictions_batch = []

        # Iterate over the batch
        for b in range(B):
            grasp_predictions = []

            # Iterate over the rotations (R)
            for rotate_idx in range(R):
                rotated_heightmap = depth_heightmaps[b, rotate_idx, :, :]

                # Create valid areas for grasping
                valid_areas = np.zeros(rotated_heightmap.shape)
                valid_areas[np.logical_and(
                    rotated_heightmap - ndimage.interpolation.shift(rotated_heightmap, [0, -8], order=0) > 1.0,
                    rotated_heightmap - ndimage.interpolation.shift(rotated_heightmap, [0, 8], order=0) > 1.0
                )] = 1

                # Apply a blur filter to smooth the valid areas
                blur_kernel = np.ones((8, 8), np.float32) / 9
                valid_areas = cv2.filter2D(valid_areas, -1, blur_kernel)

                # Rotate the valid areas back to the original orientation
                valid_areas_rotated_back = ndimage.rotate(valid_areas, -(rotate_idx * (360.0 / R)), reshape=False, order=0)
                
                # Store valid areas for this rotation after undoing the rotation
                grasp_predictions.append(valid_areas_rotated_back)

            # Stack predictions along the rotation axis
            grasp_predictions = np.stack(grasp_predictions, axis=0)
            grasp_predictions_batch.append(grasp_predictions)

        # Stack all batch predictions [B, R, H, W]
        grasp_predictions_batch = np.stack(grasp_predictions_batch, axis=0)

        # Find the best grasp pixel index for each sample in the batch
        best_pix_indices = np.array([np.unravel_index(np.argmax(grasp_predictions), grasp_predictions.shape) for grasp_predictions in grasp_predictions_batch])

        return best_pix_indices

    def _select_grasp_pixels(self, grasp_affordances):
        """Performs epsilon-greedy exploration to select grasp pixels."""
        B, R, H, W = grasp_affordances.shape
        grasp_pixel_indices = []

        for b in range(B):
            if np.random.rand() < self.current_epsilon:
                grasp_pixel_index = (
                    np.random.randint(R), np.random.randint(H), np.random.randint(W)
                )
            else:
                reshaped_affordances = grasp_affordances[b].reshape(-1)
                max_index = np.argmax(reshaped_affordances)
                r, h, w = np.unravel_index(max_index, (R, H, W))
                grasp_pixel_index = (r, h, w)
            grasp_pixel_indices.append(grasp_pixel_index)

        return np.array(grasp_pixel_indices)

    def _compute_poses(self, p_batch, r_batch, v_batch, phi_deg=45):
        """Computes post-grasp poses, throwing poses, and velocities."""
        post_grasp_poses, throw_poses, throw_velocities = [], [], []

        for b in range(p_batch.shape[0]):
            target_x, target_y, target_z = p_batch[b]
            theta = np.arctan2(target_y, target_x)
            phi = np.radians(phi_deg)

            rotation_z = R.from_euler('z', theta)
            rotation_y = R.from_euler('y', -phi)

            throw_quaternion = (rotation_z * rotation_y).as_quat()
            throw_pose = ([r_batch[b][0], r_batch[b][1], r_batch[b][2]], throw_quaternion)
            throw_poses.append(throw_pose)

            throw_velocity = ([v_batch[b][0], v_batch[b][1], v_batch[b][2]], [0.0, 0.0, 0.0])
            throw_velocities.append(throw_velocity)

            # Determine the post grasp pose based on the throwing pose and throwing velocity
            velocity_magnitude = np.linalg.norm(v_batch[b])
            distance = min(velocity_magnitude * 0.25, 0.5)
            delta_h = distance * np.cos(phi)
            delta_z = distance * np.sin(phi)
            post_grasp_x = r_batch[b][0] - delta_h * np.cos(theta)
            post_grasp_y = r_batch[b][1] - delta_h * np.sin(theta)
            post_grasp_z = r_batch[b][2] - delta_z
            post_grasp_quaternion = rotation_z.as_quat()
            post_grasp_pose = ([post_grasp_x, post_grasp_y, post_grasp_z], post_grasp_quaternion)
            post_grasp_poses.append(post_grasp_pose)

        return post_grasp_poses, throw_poses, throw_velocities

    def _pack_action(self, grasp_pixel_indices, post_grasp_poses, throw_poses, throw_velocities):
        """Packs the action"""
        return [(grasp_pixel_indices[i], post_grasp_poses[i], throw_poses[i], throw_velocities[i]) 
                for i in range(grasp_pixel_indices.shape[0])]

    def extract_logits_for_loss(self, q_g, q_t, grasp_pixel_indices):
        """Extracts logits for loss computation from grasp and throw tensors."""
        q_i_logits = q_g[torch.arange(q_g.shape[0]), grasp_pixel_indices[:, 0], :, grasp_pixel_indices[:, 1], grasp_pixel_indices[:, 2]]
        delta_i = q_t[torch.arange(q_t.shape[0]), grasp_pixel_indices[:, 0], 0, grasp_pixel_indices[:, 1], grasp_pixel_indices[:, 2]]
        return q_i_logits, delta_i

############### Visualization ###############
def plot_trajectory(throw_pos, throw_vel, target_pos, g=9.81, time_steps=100):
    """
    Plot the trajectory based on initial velocity and throw position.

    Args:
        throw_pos (tuple): Initial throw position (x, y, z).
        throw_vel (tuple): Initial velocity components in x, y, z directions.
        target_pos (tuple): Target position (x, y, z).
        g (float): Gravitational acceleration.
        time_steps (int): Number of time steps for simulation.
    """
    # Extract velocity components
    v_x, v_y, v_z = throw_vel

    # Solve the quadratic equation to find total flight time t
    z0 = throw_pos[2]
    z_target = target_pos[2]
    delta_z = z_target - z0

    # Solve quadratic equation for t: (1/2) g t^2 - v_z t + delta_z = 0
    a = 0.5 * g
    b = -v_z
    c = delta_z

    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        raise ValueError("No valid solution for the time, discriminant is negative.")

    # We only take the positive root since time must be positive
    t_total = (-b + np.sqrt(discriminant)) / (2 * a)

    # Time for the trajectory simulation
    times = np.linspace(0, t_total, time_steps)

    # Calculate positions at each time step
    x_positions = throw_pos[0] + v_x * times
    y_positions = throw_pos[1] + v_y * times
    z_positions = throw_pos[2] + v_z * times - 0.5 * g * times**2

    # Plot the 3D trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_positions, y_positions, z_positions, label="Projectile Path", color='b')

    # Mark the throw position
    ax.scatter(throw_pos[0], throw_pos[1], throw_pos[2], color='r', label='Throw Position', marker='o', s=100)

    # Mark the target position
    ax.scatter(target_pos[0], target_pos[1], target_pos[2], color='g', label='Target Position', marker='x', s=100)
    
    # Label the plot
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.legend()

    plt.show()