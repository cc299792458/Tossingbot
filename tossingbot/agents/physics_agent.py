import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tossingbot.agents.base_agent import BaseAgent
from tossingbot.utils.pytorch_utils import np_image_to_tensor
from tossingbot.envs.pybullet.utils.camera_utils import plot_heightmaps
from tossingbot.envs.pybullet.utils.math_utils import rotate_image_tensor
from tossingbot.networks import PerceptionModule, GraspingModule, ThrowingModule

class PhysicsController:
    def __init__(self, r_h=0.4, r_z=0.4):
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
            post_grasp_h: float = 0.3,
            post_grasp_z: float = 0.4,
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
            post_grasp_h (float): Horizontal distance from the robot to the post-grasp pose.
            post_grasp_z (float): Height for the post-grasp pose.
            epsilons (list[float]): Epsilon values for exploration.
        """
        super(PhysicsAgent, self).__init__(device, perception_module, grasping_module, throwing_module, epsilons, total_episodes)
        self.physics_controller = physics_controller
        self.post_grasp_h = post_grasp_h
        self.post_grasp_z = post_grasp_z

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

    def predict(self, observation, n_rotations=16, phi_deg=45, episode_num=None):
        """
        Predict the best grasping and throwing actions based on the observation.

        Args:
            observation (list of tuples): Each tuple consists of an image (I) and a target position (p).
            n_rotations (int): Number of rotations to consider for grasping.
            phi_deg (float): Angle for throwing in degrees.
            episode_num (int, optional): The current episode number for epsilon decay.

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

        # Select grasping actions using epsilon-greedy exploration
        grasp_pixel_indices = self._select_grasp_pixels(grasp_affordances)

        # Compute post-grasp poses, throwing poses, and velocities
        post_grasp_poses, throw_poses, throw_velocities = self._compute_poses(p_batch, r_batch, v_batch, grasp_pixel_indices)

        # Extract logits for loss computation
        q_i_logits, delta_i = self._extract_logits_for_loss(q_g, q_t, grasp_pixel_indices)

        # Prepare intermediate results
        intermediates = {
            "depth_heightmaps": depth_heightmaps,
            "grasp_affordances": grasp_affordances,
            "q_i_logits": q_i_logits,
            "delta_i": delta_i,
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
        q_g = torch.cat(q_g, dim=1)
        q_t = torch.cat(q_t, dim=1)

        # Compute grasp affordances
        grasp_affordances = F.softmax(q_g, dim=2).detach().cpu().numpy()[:, :, 0, :, :]

        return grasp_affordances, depth_heightmaps, q_g, q_t

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

    def _compute_poses(self, p_batch, r_batch, v_batch, grasp_pixel_indices):
        """Computes post-grasp poses, throwing poses, and velocities."""
        post_grasp_poses, throw_poses, throw_velocities = [], [], []

        for b in range(p_batch.shape[0]):
            target_x, target_y, target_z = p_batch[b]
            theta = np.arctan2(target_y, target_x)

            post_grasp_pose = ([self.post_grasp_h * np.cos(theta), self.post_grasp_h * np.sin(theta), self.post_grasp_z],
                            [0.0, 0.0, 0.0, 1.0])
            post_grasp_poses.append(post_grasp_pose)

            throw_pose = ([r_batch[b][0], r_batch[b][1], r_batch[b][2]], [0.0, 0.0, 0.0, 1.0])
            throw_velocity = ([v_batch[b][0], v_batch[b][1], v_batch[b][2]], [0.0, 0.0, 0.0])
            
            throw_poses.append(throw_pose)
            throw_velocities.append(throw_velocity)

        return post_grasp_poses, throw_poses, throw_velocities

    def _extract_logits_for_loss(self, q_g, q_t, grasp_pixel_indices):
        """Extracts logits for loss computation from grasp and throw tensors."""
        q_i_logits = q_g[torch.arange(q_g.shape[0]), grasp_pixel_indices[:, 0], :, grasp_pixel_indices[:, 1], grasp_pixel_indices[:, 2]]
        delta_i = q_t[torch.arange(q_t.shape[0]), grasp_pixel_indices[:, 0], 0, grasp_pixel_indices[:, 1], grasp_pixel_indices[:, 2]]
        return q_i_logits, delta_i
    
    def _pack_action(self, grasp_pixel_indices, post_grasp_poses, throw_poses, throw_velocities):
        """Packs the action"""
        return [(grasp_pixel_indices[i], post_grasp_poses[i], throw_poses[i], throw_velocities[i]) 
                for i in range(grasp_pixel_indices.shape[0])]

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

# Example usage
if __name__ == '__main__':
    r_h = 0.5
    r_z = 0.5
    target_pos = (2, 0, 0)  # Target position
    phi = 0  # Launch angle

    physics_controller = PhysicsController(r_h=r_h, r_z=r_z)
    throw_pos, throw_vel = physics_controller.predict(target_pos, phi)

    print(f"Throw position: {throw_pos}")
    print(f"Throw velocity (linear): {throw_vel}")

    # Plot the trajectory
    plot_trajectory(throw_pos, throw_vel, target_pos)

    # Initialize input data
    heightmap = np.zeros([2, 80, 60, 4])  # 80x60 heightmap with 4 channels
    start_point = (10, 20)  # Starting point of the arrow
    end_point = (50, 60)  # End point of the arrow
    heightmap = draw_arrow_on_last_channel(heightmap, start=start_point, end=end_point)
    target_position = np.array([2.0, 0.0, 0.1])  # Target position in (x, y, z)

    # Initialize modules and agent
    perception_module = PerceptionModule()
    grasp_module = GraspingModule()
    throw_module = ThrowingModule()
    agent = PhysicsAgent(
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        perception_module=perception_module,
        grasping_module=grasp_module,
        throwing_module=throw_module,
        physics_controller=physics_controller
    )

    # Make prediction with observation
    observation = [(heightmap, target_position)]
    action, intermidiates = agent.predict(observation=observation, n_rotations=16)
    plot_heightmaps(intermidiates['depth_heightmaps'])