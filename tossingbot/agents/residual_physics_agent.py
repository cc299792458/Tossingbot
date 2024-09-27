import torch
import numpy as np
import torch.nn as nn

from tossingbot.agents.physics_agent import PhysicsAgent, PhysicsController

class ResidualPhysicsAgent(PhysicsAgent):
    """
    ResidualPhysicsAgent combines physics-based control with residuals from a neural network 
    to adjust the throwing velocity.
    """
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
        Initialize the ResidualPhysicsAgent with the same parameters as the PhysicsAgent.

        Args:
            device (torch.device): The device to run computations on (e.g., 'cpu' or 'cuda').
            perception_module (nn.Module): Neural network module for perception, processes visual input.
            grasping_module (nn.Module): Neural network module for predicting grasping parameters.
            throwing_module (nn.Module): Neural network module for predicting throwing parameters.
            physics_controller (PhysicsController): Controller to calculate physics-based parameters.
            post_grasp_h (float): Horizontal distance from the robot to the post-grasp pose.
            post_grasp_z (float): Height for the post-grasp pose.
            epsilons (list[float]): Epsilon values for exploration.
        """
        super(ResidualPhysicsAgent, self).__init__(
            device=device, 
            perception_module=perception_module, 
            grasping_module=grasping_module, 
            throwing_module=throwing_module, 
            physics_controller=physics_controller,
            post_grasp_h=post_grasp_h, 
            post_grasp_z=post_grasp_z, 
            epsilons=epsilons
        )

    def forward(self, I, v):
        """
        Forward pass to predict grasping and throwing parameters.

        Args:
            I (torch.Tensor): Visual observation (e.g., image or feature map).
            v (float): Predicted velocity given by the physics controller.
        
        Returns:
            tuple: Predicted grasping parameters (q_g) and throwing parameters (q_t).
        """
        return super(ResidualPhysicsAgent, self).forward(I, v)

    def predict(self, observation, n_rotations=16, phi_deg=45):
        """
        Predict grasp and throw actions using physics with neural residuals.

        Args:
            observation (tuple): Observation consisting of an image (I) and the target position (p).
            n_rotations (int): Number of rotations to consider for grasping.
            phi_deg (float): Angle for throwing in degrees.

        Returns:
            tuple: (grasp_pixel_index, post_grasp_pose, throw_pose, throw_velocity) and intermediate values.
        """
        (grasp_pixel_index, post_grasp_pose, throw_pose, throw_velocity), intermediates = \
            super(ResidualPhysicsAgent, self).predict(observation, n_rotations, phi_deg)

        # Extract the residual velocity adjustment
        q_t_tensor = intermediates['q_t_tensor']
        residual_velocity = q_t_tensor[grasp_pixel_index[0], 0, grasp_pixel_index[1], grasp_pixel_index[2]].detach().cpu().numpy()

        # Adjust the throw velocity with the residual
        linear_velocity, angular_velocity = throw_velocity
        linear_velocity += residual_velocity * linear_velocity / np.linalg.norm(linear_velocity)

        return (grasp_pixel_index, post_grasp_pose, throw_pose, (linear_velocity, angular_velocity)), intermediates
 