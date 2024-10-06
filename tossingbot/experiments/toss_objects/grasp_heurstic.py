"""
    In this experiment, we show the effectiveness of the grasp heuristic.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from tossingbot.utils.misc_utils import set_seed
from tossingbot.envs.pybullet.tasks import TossObjects
from tossingbot.agents.physics_agent import PhysicsAgent, PhysicsController
from tossingbot.networks import PerceptionModule, GraspingModule, ThrowingModule

# NOTE: When we set box_length to 0.25, the throwing success rate is lower than 1.0. 
# The primary reason is that the throwing primitive becomes less stable over longer distances.

def plot_grasp_success(object_poses, grasp_success_history, margin=0.05):
    """
    Visualize the grasp successes and failures.
    An 'O' is placed where the object was successfully grasped, and an 'X' where it failed.
    
    Args:
        object_poses (list): A list of object positions where grasps were attempted.
        grasp_success_history (list): A list of booleans indicating if the grasp was successful or not.
        margin (float): A margin to add around the object positions for better visualization (default: 0.05).
    """
    fig, ax = plt.subplots()

    # Extract all x and y positions
    x_positions = [pose[0][0] for pose in object_poses]
    y_positions = [pose[0][1] for pose in object_poses]

    # Get the min and max values with a margin
    x_min, x_max = min(x_positions) - margin, max(x_positions) + margin
    y_min, y_max = min(y_positions) - margin, max(y_positions) + margin

    for i, pose in enumerate(object_poses):
        x, y = pose[0][0], pose[0][1]
        marker = 'O' if grasp_success_history[i] else 'X'
        color = 'black' if grasp_success_history[i] else 'red'
        ax.text(x, y, marker, color=color, ha='center', va='center', fontsize=12)

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    # Set aspect ratio to be equal for correct scaling
    ax.set_aspect('equal')
    ax.set_title('Grasp Success (O: Success, X: Failure)')
    plt.show()

def plot_throw_success(throw_success_history, target_positions):
    """
    Visualize the throw success rates at each target position.
    
    Args:
        throw_success_history (list): A list of booleans indicating if the throw was successful or not.
        target_positions (list): The target positions (x, y, z) for the throws.
    """
    # Step 1: Convert target positions to a set of unique positions
    target_positions_set = set(map(tuple, target_positions))  # Convert each position to a tuple

    # Step 2: Initialize a dictionary to track total throws and successful throws per target position
    target_stats = {pos: {'success': 0, 'total': 0} for pos in target_positions_set}

    # Step 3: Iterate over the throw success history and target positions, update the stats
    for i, target_pos in enumerate(target_positions):
        target_pos_tuple = tuple(target_pos)  # Convert current position to tuple for dictionary key
        target_stats[target_pos_tuple]['total'] += 1  # Update total attempts
        if throw_success_history[i]:  # If the throw was successful
            target_stats[target_pos_tuple]['success'] += 1  # Update successful throws

    # Step 4: Calculate success rates for each target position
    success_rates = {pos: target_stats[pos]['success'] / target_stats[pos]['total'] 
                     if target_stats[pos]['total'] > 0 else 0 for pos in target_stats}

    # Step 5: Plot the success rates
    fig, ax = plt.subplots()

    # Extract positions and success rates for plotting
    x_vals = [pos[0] for pos in success_rates.keys()]  # Extract X coordinates
    y_vals = [pos[1] for pos in success_rates.keys()]  # Extract Y coordinates
    success_vals = [rate for rate in success_rates.values()]  # Extract success rates

    # Create a scatter plot where size of the marker is proportional to the success rate
    scatter = ax.scatter(x_vals, y_vals, c=success_vals, cmap='Blues', s=[rate * 500 for rate in success_vals], alpha=0.7, vmin=0, vmax=1)
    scatter.set_clim(0, 1)

    # Add a color bar with correct limits from 0 to 1
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Success Rate')

    # Annotate each point with its success rate
    for i, rate in enumerate(success_vals):
        ax.text(x_vals[i], y_vals[i], f'{rate:.2f}', ha='center', va='center', fontsize=9, color='black')

    ax.set_title('Throw Success Rates at Target Positions')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    plt.show()

    return success_rates  # Return the success rates for further analysis if needed

if __name__ == '__main__':
    set_seed()

    # Log directory
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs") 
    
    # Parameters
    use_gui = True
    n_rotations = 1
    phi_deg = 45  # Set phi_deg to 45 degrees
    total_episodes = 100  # Reduced to a smaller number of episodes for heuristic testing

    # Create list to track the history of grasp successes, throw successes, and object poses
    grasp_success_history = []
    throw_success_history = []
    object_poses_history = []  # To track where grasps are attempted
    target_positions = []  # To track target throw positions

    # Env
    env = TossObjects(
        use_gui=use_gui,
        scene_config={'box_length': 0.15},
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

        # Record grasp success, throw success, and object position for this episode
        grasp_success_history.append(info['grasp_success'])
        throw_success_history.append(info['throw_success'])
        object_poses_history.append(info['object_poses'][0])  # Get object position after the grasp
        target_positions.append(obs[1])  # Target throw position

        # Update current observation
        obs = next_obs

        # Compute average success rate over the completed episodes
        avg_grasp_success = np.mean(grasp_success_history)
        avg_throw_success = np.mean(throw_success_history)

        # Update the tqdm description to display the success rate
        progress_bar.set_postfix({
            "Grasp Success Rate": f"{avg_grasp_success:.3f}",
            "Throw Success Rate": f"{avg_throw_success:.3f}",
        })

    # After the loop, plot the grasp and throw success visualizations
    plot_grasp_success(object_poses_history, grasp_success_history)
    throw_success_rates = plot_throw_success(throw_success_history, target_positions)