"""
    In this experiment, we test the stability of the robot grasping at various positions.
    A series of positions are generated on a grid, and a small sphere is created at each position.
    The robot attempts to grasp the sphere, and success is measured by checking the height of the sphere after the grasp.
    Each sphere is removed after the grasp attempt, and the process repeats for the next position.
"""

import os
import time
import numpy as np
import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt

from tqdm import tqdm
from tossingbot.envs.pybullet.robot import Panda
from tossingbot.envs.pybullet.utils.objects_utils import create_sphere, create_plane

def run_grasping_experiment(robot, grid_points, ball_radius=0.02, ball_mass=0.1, success_height_threshold=0.1):
    """
    Runs the grasping experiment by creating a sphere at each position in the grid,
    attempting to grasp it, and checking if the grasp was successful by measuring
    the sphere's height after the grasp. The results are recorded and visualized.

    Args:
        robot: The robot instance (Panda) that will perform the grasp.
        grid_points: List of positions (x, y) where spheres will be created.
        ball_radius: Radius of the sphere to be grasped.
        ball_mass: Mass of the sphere to be grasped.
        success_height_threshold: Height threshold above which a grasp is considered successful.
    
    Returns:
        success_matrix: A matrix of the same size as grid_points showing the success or failure of each grasp.
    """
    
    # Initialize an empty list to store success or failure for each grasp
    results = []
    
    for position in tqdm(grid_points, desc="Running Grasping Experiment"):
        # Create a sphere at the current position
        ball_position = [position[0], position[1], 0.02]  # Z position is fixed for the sphere
        object_id = create_sphere(radius=ball_radius, position=ball_position, mass=ball_mass)

        # Define the grasp and post-grasp poses
        grasp_pose = (ball_position, [0.0, 0.0, 0.0, 1.0])
        post_grasp_pose = ([ball_position[0], ball_position[1], 0.3], [0.0, 0.0, 0.0, 1.0])

        # Attempt to grasp the object
        grasp_completed = False
        while not grasp_completed:
            grasp_completed = robot.grasp(tcp_target_pose=grasp_pose, post_grasp_pose=post_grasp_pose)
            
            # Run simulation for 60 Hz control loop
            for _ in range(int(240 // 60)):
                p.stepSimulation()
                time.sleep(1./240.)
                robot.log_variables()
                if robot.gripper_control_mode == 'torque':
                    robot.keep_gripper_force()
            robot.visualize_tcp_trajectory()

        # Check if the grasp was successful by evaluating the height of the sphere
        ball_position_after_grasp = p.getBasePositionAndOrientation(object_id)[0]
        success = ball_position_after_grasp[2] > success_height_threshold
        results.append(success)  # Store the result (True for success, False for failure)

        # Remove the sphere after the grasp
        p.removeBody(object_id)
    
    # Reshape the results to match the grid layout
    success_matrix = np.array(results).reshape(len(y_points), len(x_points))
    return success_matrix

def plot_grasp_results(success_matrix, x_points, y_points):
    """
    Plots the grasping success matrix. Successful grasps are marked with 'O', failed grasps with 'X'.

    Args:
        success_matrix: A matrix showing the success or failure of each grasp.
        x_points: The x coordinates of the grid.
        y_points: The y coordinates of the grid.

    Returns:
        None
    """
    
    fig, ax = plt.subplots()
    
    # Plot success as 'O' and failure as 'X'
    for i in range(len(y_points)):
        for j in range(len(x_points)):
            marker = 'O' if success_matrix[i, j] else 'X'
            ax.text(x_points[j], y_points[i], marker, ha='center', va='center', fontsize=12)
    
    # Set the limits and labels
    ax.set_xlim(min(x_points) - 0.05, max(x_points) + 0.05)
    ax.set_ylim(min(y_points) - 0.05, max(y_points) + 0.05)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Grasping Results (O: Success, X: Failure)')
    
    # Create directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Save the plot to the logs directory
    plot_filename = os.path.join(log_dir, "grasping_results.png")
    plt.savefig(plot_filename)

    plt.show()

if __name__ == '__main__':
    # Initialize PyBullet simulation
    physics_client_id = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    # Create Panda robot
    gripper_control_mode = 'position'
    use_gripper_gear = True
    visualize_coordinate_frames = True
    robot = Panda(
        1 / 240, 1 / 60, (0, 0.0, 0.0), (0.0, 0.0, 0.0),
        gripper_control_mode=gripper_control_mode, 
        use_gripper_gear=use_gripper_gear,
        visualize_coordinate_frames=visualize_coordinate_frames,
    )

    # Create a plane in the environment
    create_plane()

    # Generate grid points in the workspace
    workspace_xlim = [0.25, 0.55]
    workspace_ylim = [-0.2, 0.2]
    step_size = 0.02  # Define step size for the grid

    # Create grid points within the x and y limits
    x_points = np.arange(workspace_xlim[0], workspace_xlim[1] + step_size, step_size)
    y_points = np.arange(workspace_ylim[0], workspace_ylim[1] + step_size, step_size)
    x_grid, y_grid = np.meshgrid(x_points, y_points)
    grid_points = np.column_stack((x_grid.flatten(), y_grid.flatten()))  # Flatten the grid into a list of points

    # Run the grasping experiment and get the results
    success_matrix = run_grasping_experiment(robot, grid_points)

    # Plot the results
    plot_grasp_results(success_matrix, x_points, y_points)

    # Disconnect from the simulation
    p.disconnect()
