"""
    In this experiment, we test the PhysicsAgent by evaluating its grasping and throwing capabilities across multiple episodes.
"""

import os
import torch
import numpy as np

from tqdm import tqdm
from tossingbot.utils.misc_utils import set_seed
from tossingbot.utils.pytorch_utils import load_model
from tossingbot.envs.pybullet.tasks import TossObjects
from tossingbot.agents.physics_agent import PhysicsAgent, PhysicsController
from tossingbot.networks import PerceptionModule, GraspingModule, ThrowingModule

if __name__ == '__main__':
    set_seed()

    # Log directory
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs") 
    
    # Set device (use GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Parameters
    use_gui = False
    box_length = 0.15
    box_n_rows, box_n_cols = 3, 3

    n_rotations = 1
    phi_deg = 45

    total_episodes = 100  # Test for 100 episodes

    # Lists to track the success rates
    grasp_success_history = []
    throw_success_history = []

    # Env
    env = TossObjects(
        use_gui=use_gui,
        scene_config={
            'box_n_rows': box_n_rows,
            'box_n_cols': box_n_cols,
            'box_length': box_length,
        },
        objects_config={"object_types": ['ball', 'cube', 'rod']},
        camera_config={'n_rotations': n_rotations}
    )

    # Networks
    perception_module = PerceptionModule()
    grasping_module = GraspingModule()
    throwing_module = ThrowingModule()

    # Agent
    physics_controller = PhysicsController()
    agent = PhysicsAgent(
        device=device, 
        perception_module=perception_module, 
        grasping_module=grasping_module, 
        throwing_module=throwing_module,
        physics_controller=physics_controller,
        epsilons=[0.0, 0.0]  # Disable epsilon-greedy for this test
    )

    # Load the model
    load_model(agent, None, log_dir, model_name='physics_agent')

    # Main loop for testing
    obs, info = env.reset()
    progress_bar = tqdm(range(total_episodes), desc="Testing Progress")

    for episode_num in progress_bar:
        action, intermediates = agent.predict(
            [obs], 
            n_rotations=n_rotations, 
            phi_deg=phi_deg, 
            episode_num=episode_num, 
            use_heuristic=info['next_grasp_with_heuristic']
        )
        next_obs, reward, terminated, truncated, info = env.step(action=action[0])

        # Update current obs
        obs = next_obs

        # Record grasp success and throw success for this episode
        grasp_success_history.append(info['grasp_success'])

        # Only record throw success if the grasp was successful
        if info['grasp_success']:
            throw_success_history.append(info['throw_success'])

    # Calculate the average success rates
    avg_grasp_success = np.mean(grasp_success_history)
    avg_throw_success = np.mean(throw_success_history) if throw_success_history else 0.0

    # Print final results
    print(f"Average Grasp Success Rate: {avg_grasp_success:.3f}")   # Average Grasp Success Rate: 1.000 / 0.990 / 0.980
    print(f"Average Throw Success Rate (for successful grasps): {avg_throw_success:.3f}")   # Average Throw Success Rate (for successful grasps): 0.990 / 0.556 / 0.806
