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
        super(PhysicsAgent, self).__init__(device, perception_module, grasping_module, throwing_module, epsilons)
        self.physics_controller = physics_controller
        self.post_grasp_h = post_grasp_h
        self.post_grasp_z = post_grasp_z

    def forward(self, I, v):
        """
        Forward pass to predict grasping and throwing parameters.

        Args:
            I (torch.Tensor): Visual observation (e.g., image or feature map).
            v (float): Predicted velocity given by physics controller.
            
        Returns:
            tuple: Predicted grasping parameters (q_g) and throwing parameters (q_t).
        """
        # Step 1: Process the visual input using the perception module to extract visual features (mu)
        mu = self.perception_module(I)

        # Step 2: Concatenate the visual features with the predicted velocity
        B, _, H, W = mu.shape
        assert B == 1, "Batch size must be 1"
        
        # Create a velocity image with the same spatial dimensions as mu and concatenate it
        velocity_image = torch.full((B, 128, H, W), v, device=self.device)
        fused_features = torch.cat([mu, velocity_image], dim=1)  # Concatenating along the channel dimension

        # Step 3: Predict the grasping probability map (q_g) and throwing probability map (q_t)
        q_g = self.grasping_module(fused_features)
        q_t = self.throwing_module(fused_features)

        return q_g, q_t
    
    def predict(self, observation, n_rotations=16, phi_deg=45):
        """
        Predict the best grasping and throwing actions based on the observation.

        Args:
            observation (tuple): Observation consisting of an image (I) and the target position (p).
            n_rotations (int): Number of rotations to consider for grasping.
            phi_deg (float): Angle for throwing in degrees.

        Returns:
            tuple: (grasp_pixel_index, post_grasp_pose, throw_pose, throw_velocity) and intermediate values.
        """
        I, p = observation

        # Get throwing parameters from the physics controller
        (r_x, r_y, r_z), (v_x, v_y, v_z) = self.physics_controller.predict(target_pos=p, phi_deg=phi_deg)

        # Calculate the throwing velocity magnitude
        v_magnitude = np.sqrt(v_x**2 + v_y**2 + v_z**2)

        grasp_logits = []  # Store network raw outputs (logits) for each rotation
        depth_heightmaps = []  # Store the rotated depth heightmaps for visualization

        # Convert the input image to tensor once
        I_tensor = np_image_to_tensor(I, self.device)

        for i in range(n_rotations):
            # Calculate rotation angle for the current step
            theta = 360 / n_rotations * i

            # Rotate the input tensor
            I_rotated = rotate_image_tensor(I_tensor, theta=theta)

            # Store the rotated depth heightmap
            depth_heightmaps.append(I_rotated[0, -1, :, :].detach().cpu().numpy())

            # Forward pass through the network
            q_g_tensor, q_t_tensor = self.forward(I=I_rotated, v=v_magnitude)

            # Undo the rotation for grasp logits
            q_g_rotated_back = rotate_image_tensor(q_g_tensor, theta=-theta)

            # Collect the network raw outputs (logits) for this rotation
            grasp_logits.append(q_g_rotated_back)

        # Concatenate all grasp logits along the first dimension (n_rotations)
        concatenated_logits = torch.cat(grasp_logits, dim=0)

        # Apply softmax to obtain the final grasp affordances (probabilities)
        grasp_affordances = F.softmax(concatenated_logits, dim=1)

        # Detach and move to CPU for further processing
        grasp_affordances = grasp_affordances.detach().cpu().numpy()[:, 0, :, :]

        # Find the index of the maximum value across the entire array
        grasp_pixel_index = np.unravel_index(np.argmax(grasp_affordances, axis=None), grasp_affordances.shape)

        # Calculate yaw angle for grasping
        target_x, target_y = p[0], p[1]
        theta = np.arctan2(target_y, target_x)

        # Define the post-grasp pose
        post_grasp_pose = ([self.post_grasp_h * np.cos(theta), self.post_grasp_h * np.sin(theta), self.post_grasp_z],
                           [0.0, 0.0, 0.0, 1.0])

        # Define the throwing pose and velocity
        throw_pose = ([r_x, r_y, r_z], [0.0, 0.0, 0.0, 1.0])
        throw_velocity = ([v_x, v_y, v_z], [0.0, 0.0, 0.0])

        # Return intermediate values and predictions
        intermediates = {
            "depth_heightmaps": depth_heightmaps, 
            "grasp_affordances": grasp_affordances,
            "q_g_tensor": q_g_tensor,
            "q_t_tensor": q_t_tensor,
            "q_i_logits": q_g_tensor[grasp_pixel_index[0], :, grasp_pixel_index[1], grasp_pixel_index[2]],
            "delta_i": q_t_tensor[grasp_pixel_index[0], 0, grasp_pixel_index[1], grasp_pixel_index[2]], 
        }

        return (grasp_pixel_index, post_grasp_pose, throw_pose, throw_velocity), intermediates

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
    heightmap = np.zeros([80, 60, 4])  # 80x60 heightmap with 4 channels
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
    observation = (heightmap, target_position)
    action, intermidiates = agent.predict(observation=observation, n_rotations=16)
    plot_heightmaps(intermidiates['depth_heightmaps'])