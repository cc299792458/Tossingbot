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
            epsilons: list[float] = [0.5, 0.1],
            total_episodes: int = 10000,
            decay_factor: float = 0.2,
            residual_limits: list[float] = [-0.5, 0.5]
        ):
        """
        Initialize the ResidualPhysicsAgent with the same parameters as the PhysicsAgent.

        Args:
            device (torch.device): The device to run computations on (e.g., 'cpu' or 'cuda').
            perception_module (nn.Module): Neural network module for perception, processes visual input.
            grasping_module (nn.Module): Neural network module for predicting grasping parameters.
            throwing_module (nn.Module): Neural network module for predicting throwing parameters.
            physics_controller (PhysicsController): Controller to calculate physics-based parameters.
            epsilons (list[float]): Epsilon values for exploration.
            total_episodes (int): Total training episodes.
            decay_factor (float): Factor that determines the decay rate of epsilon, as a multiple of total_episodes.
            residual_limits (list[float]): The limits of residual velocity
        """
        super(ResidualPhysicsAgent, self).__init__(
            device=device, 
            perception_module=perception_module, 
            grasping_module=grasping_module, 
            throwing_module=throwing_module, 
            physics_controller=physics_controller,
            epsilons=epsilons,
            total_episodes=total_episodes,
            decay_factor=decay_factor,
        )
        self.residual_limits = residual_limits

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

    def predict(self, observation, n_rotations=16, phi_deg=45, episode_num=None, use_heuristic=False):
        """
        Predict grasp and throw actions using physics with neural residuals.

        Args:
            observation (list of tuples): A batch of observations, each consisting of an image (I) and the target position (p).
            n_rotations (int): Number of rotations to consider for grasping.
            phi_deg (float): Angle for throwing in degrees.
            episode_num (int, optional): Current episode number for epsilon decay.
            use_heuristic (bool): If True, use the heuristic method to select the grasping pixel.

        Returns:
            tuple: (grasp_pixel_indices, post_grasp_poses, throw_poses, throw_velocities) and intermediate values.
        """
        # Step 1: Use the super call to get the packed action from the PhysicsAgent
        packed_action, intermediates = super(ResidualPhysicsAgent, self).predict(observation, n_rotations, phi_deg, episode_num, use_heuristic)

        # Step 2: Unpack the actions (grasp_pixel_indices, post_grasp_poses, throw_poses, throw_velocities)
        grasp_pixel_indices, post_grasp_poses, throw_poses, throw_velocities = zip(*packed_action)

        # Step 3: Extract the residual velocity adjustment for each element in the batch
        residual_velocities = self._compute_residual_velocities(intermediates['q_t'], grasp_pixel_indices)

        # Step 4: Adjust the throw velocities using the residuals
        adjusted_throw_velocities = self._adjust_throw_velocities(throw_velocities, residual_velocities)

        # Step 5: Use _pack_action to repack the adjusted actions
        action = self._pack_action(np.array(grasp_pixel_indices), post_grasp_poses, throw_poses, adjusted_throw_velocities)

        return action, intermediates

    def _compute_residual_velocities(self, q_t, grasp_pixel_indices):
        """
        Compute the residual velocity adjustment for each element in the batch.

        Args:
            q_t (torch.Tensor): Throw probability map tensor [B, R, C, H, W].
            grasp_pixel_indices (list of tuples): Grasp pixel indices [(r, h, w)] for each element in the batch.

        Returns:
            np.array: Residual velocity adjustment for each element in the batch.
        """
        B = len(grasp_pixel_indices)
        residual_velocities = []

        for b in range(B):
            r, h, w = grasp_pixel_indices[b]
            residual_velocity = q_t[b, r, 0, h, w].detach().cpu().numpy()
            # Clip the residual velocity with the limits
            residual_velocity = np.clip(residual_velocity, self.residual_limits[0], self.residual_limits[1])
            residual_velocities.append(residual_velocity)

        return np.array(residual_velocities)

    def _adjust_throw_velocities(self, throw_velocities, residual_velocities):
        """
        Adjust the throw velocities using the residuals.

        Args:
            throw_velocities (list of tuples): A list of throw velocity tuples [(linear_velocity, angular_velocity)].
            residual_velocities (np.array): Residual velocity adjustments for each element in the batch.

        Returns:
            list of tuples: Adjusted throw velocities [(adjusted_linear_velocity, angular_velocity)].
        """
        adjusted_throw_velocities = []

        for b in range(len(throw_velocities)):
            linear_velocity, angular_velocity = throw_velocities[b]

            # Adjust the linear velocity using the residual
            adjusted_linear_velocity = linear_velocity + residual_velocities[b] * np.array(linear_velocity) / np.linalg.norm(linear_velocity)

            # Store the adjusted velocities
            adjusted_throw_velocities.append((adjusted_linear_velocity, angular_velocity))

        return adjusted_throw_velocities