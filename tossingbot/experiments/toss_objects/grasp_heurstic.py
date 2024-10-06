"""
    In this experiment, we test the effectiveness of the grasp heuristic.
"""

import os
import numpy as np
import torch

from tqdm import tqdm
from tossingbot.utils.misc_utils import set_seed
from tossingbot.envs.pybullet.tasks import TossObjects
from tossingbot.agents.physics_agent import PhysicsAgent, PhysicsController
from tossingbot.networks import PerceptionModule, GraspingModule, ThrowingModule

if __name__ == '__main__':
    set_seed()

    # Log directory
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs") 
    
    # Parameters
    use_gui = True
    n_rotations = 1
    phi_deg = 45  # Set phi_deg to 45 degrees
    total_episodes = 100  # Reduced to a smaller number of episodes for heuristic testing

    # Create list to track the history of grasp successes and throw successes
    grasp_success_history = []
    throw_success_history = []

    # Env
    env = TossObjects(
        use_gui=use_gui,
        camera_config={'n_rotations': n_rotations}
    )

    # Networks
    perception_module = PerceptionModule()
    grasping_module = GraspingModule()
    throwing_module = ThrowingModule()

    # Agent
    physics_controller = PhysicsController()
    agent = PhysicsAgent(
        device=torch.device('cpu'),  # CPU is enough for heuristic testing
        perception_module=perception_module, 
        grasping_module=grasping_module, 
        throwing_module=throwing_module,
        physics_controller=physics_controller,
    )

    # Main loop
    obs, info = env.reset()
    progress_bar = tqdm(range(total_episodes), desc="Grasp Heuristic Testing Progress")
    for episode_num in progress_bar:
        # Use the agent to predict the grasp and throw action based on the heuristic
        action, intermediates = agent.predict([obs], n_rotations=n_rotations, phi_deg=phi_deg, episode_num=episode_num, use_heuristic=True)
        next_obs, reward, terminated, truncated, info = env.step(action=action[0])

        # Update current observation
        obs = next_obs

        # Record grasp success and throw success for this episode
        grasp_success_history.append(info['grasp_success'])
        throw_success_history.append(info['throw_success'])

        # Compute average success rate over the completed episodes
        avg_grasp_success = np.mean(grasp_success_history)
        avg_throw_success = np.mean(throw_success_history)

        # Update the tqdm description to display the success rate
        progress_bar.set_postfix({
            "Grasp Success Rate": f"{avg_grasp_success:.3f}",
            "Throw Success Rate": f"{avg_throw_success:.3f}",
        })