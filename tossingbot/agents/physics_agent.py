import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from tossingbot.utils.pytorch_utils import np_image_to_tensor, tensor_to_np_image
from tossingbot.agents.base_agent import BaseAgent

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
            perception_module: nn.Module, 
            grasping_module: nn.Module, 
            throwing_module: nn.Module, 
            post_grasp_h: float,    # The horizontal distance from robot base to post grasp pose
            post_grasp_z: float,    # The height for the post grasp pose
            epsilons: list[float] = [0.5, 0.1],
            physics_controller: PhysicsController = None
        ):
        """
        Initialize the PhysicsAgent for TossingBot, inheriting from BaseAgent.
        
        This agent will incorporate physics-based decision-making into grasping and throwing actions.
        
        Args:
            device (torch.device): The device to run computations on (e.g., 'cpu' or 'cuda').
            perception_module (nn.Module): Neural network module for perception, processes visual input I.
            grasping_module (nn.Module): Neural network module for predicting grasping parameters.
            throwing_module (nn.Module): Neural network module for predicting throwing parameters.
        """
        super(PhysicsAgent, self).__init__(device, perception_module, grasping_module, throwing_module, post_grasp_h, post_grasp_z, epsilons)

        self.physics_controller = physics_controller
    
    def forward(self, I):
        """
        Forward pass to predict grasping and throwing parameters with additional physics considerations.

        Args:
            I (torch.Tensor): Visual observation (e.g., image or feature map).

        Returns:
            tuple: Predicted grasping parameters (phi_g) and throwing parameters (phi_t).
        """
        # Call the base class forward method
        q_g, q_t = super().forward(I)
        
        return q_g, q_t
    
    def predict(self, observation, n_rotations=16, phi_deg=45):
        (grasp_pixel_index, post_grasp_pose, _, _), intermidiates = super().predict(observation=observation, n_rotations=n_rotations)

        I, p = observation

        (r_x, r_y, r_z), (v_x, v_y, v_z) = self.physics_controller.predict(target_pos=p, phi_deg=phi_deg)

        throw_pose = ([r_x, r_y, r_z], [1.0, 0.0, 0.0, 0.0])
        throw_velocity = ([v_x, v_y, v_z], [0.0, 0.0, 0.0])

        return (grasp_pixel_index, post_grasp_pose, throw_pose, throw_velocity), intermidiates

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

if __name__ == '__main__':
    # Example usage
    r_h = 0.5
    r_z = 0.5
    target_pos = (2, 0, 0)  # Target position
    phi = 0  # Launch angle

    controller = PhysicsController(r_h=r_h, r_z=r_z)
    throw_pos, throw_vel = controller.calc_params(target_pos, phi)

    print(f"Throw position: {throw_pos}")
    print(f"Throw velocity (linear): {throw_vel}")

    # Plot the trajectory
    plot_trajectory(throw_pos, throw_vel, target_pos)
