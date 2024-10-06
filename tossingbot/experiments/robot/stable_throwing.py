"""
    In this experiment, we test the stability of the robot throwing at various target positions.
    The PhysicsAgent is used to provide the post_grasp_pose, throw_pose, and throw_velocity.
    The robot first grasps the sphere, then throws it to the target, and success is measured by the final position.
"""

# NOTE: Please also refer to "/Tossingbot/tossingbot/experiments/toss_objects/grasp_heurstic.py"

import os
import time
import numpy as np
import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt

from tqdm import tqdm
from tossingbot.envs.pybullet.robot import Panda
from tossingbot.envs.pybullet.utils.objects_utils import create_plane, create_sphere, create_box
from tossingbot.agents.physics_agent import PhysicsAgent, PhysicsController
from tossingbot.networks.networks import PerceptionModule, GraspingModule, ThrowingModule

def load_boxes_with_dividers(length=0.25, width=0.15, height=0.1, n_rows=3, n_cols=3, position=[0.0, 0.0]):
    """
    Create a grid of hollow boxes separated by dividers using thin box walls.
    This function also returns the center positions of the boxes for use as target positions.

    Args:
        length (float): Length of each individual box (x-dimension).
        width (float): Width of each individual box (y-dimension).
        height (float): Height of the dividers (z-dimension).
        n_rows (int): Number of rows in the grid.
        n_cols (int): Number of columns in the grid.
        position (list): Center position [x, y] of the entire grid.
    
    Returns:
        target_positions: List of the center positions of each box.
    """
    box_ids = []
    divider_thickness = 0.01  # 1 cm thick dividers
    color = [0.545, 0.271, 0.075, 1.0]
    target_positions = []

    # Create outer walls and dividers, also calculate box centers
    for i in range(n_cols + 1):
        y = position[1] - (n_cols / 2 * width) + i * width
        box_ids.append(create_box(half_extents=[length * n_rows / 2, divider_thickness / 2, height / 2],
                                  position=[position[0], y, height / 2], mass=0, color=color))
    for j in range(n_rows + 1):
        x = position[0] - (n_rows / 2 * length) + j * length
        box_ids.append(create_box(half_extents=[divider_thickness / 2, width * n_cols / 2, height / 2],
                                  position=[x, position[1], height / 2], mass=0, color=color))
        
    # Calculate the center positions of each box directly within the loop
    for i in range(n_cols):
        for j in range(n_rows):
            x_central = position[0] + (j - n_rows / 2 + 0.5) * length
            y_central = position[1] + (i - n_cols / 2 + 0.5) * width
            target_positions.append([x_central, y_central, height])

    return box_ids, target_positions

def run_throwing_experiment(
        robot, 
        agent, 
        target_positions, 
        n_rows, n_cols, 
        box_length=0.25, 
        box_width=0.15, 
        box_height=0.1,
        ball_radius=0.02, 
        ball_mass=0.1):
    """
    Runs the throwing experiment where the robot first grasps a sphere and then throws it to different target positions.
    The PhysicsAgent calculates the post-grasp pose, throw pose, and throw velocity.
    Success is determined by checking the final position of the sphere after the throw.

    Args:
        robot: The robot instance (Panda) that will perform the grasp and throw.
        agent: The PhysicsAgent used to calculate the throw parameters.
        target_positions: List of target positions (x, y, z) where the sphere will be thrown.
        n_rows: Number of rows in the box grid.
        n_cols: Number of columns in the box grid.
        box_length, box_width: Dimensions of the target boxes.
        ball_radius: Radius of the sphere to be thrown.
        ball_mass: Mass of the sphere to be thrown.

    Returns:
        success_matrix: A 2D matrix of shape (n_rows, n_cols) indicating success (True) or failure (False) for each throw.
    """
    success_matrix = np.zeros((n_rows, n_cols), dtype=bool)
    debug_lines = []

    for idx, target_pos in enumerate(tqdm(target_positions, desc="Running Throwing Experiment")):
        row = idx // n_cols
        col = idx % n_cols

        # Clear previous debug lines
        for line_id in debug_lines:
            p.removeUserDebugItem(line_id)
        debug_lines.clear()

        # Calculate the limits of the chosen box
        x_min, x_max = target_pos[0] - box_length / 2, target_pos[0] + box_length / 2
        y_min, y_max = target_pos[1] - box_width / 2, target_pos[1] + box_width / 2
        z_max = box_height

        # Create debug lines to highlight the top edges of the target box
        debug_lines.append(p.addUserDebugLine([x_min, y_min, z_max], [x_min, y_max, z_max], [1, 0, 0], 2))
        debug_lines.append(p.addUserDebugLine([x_min, y_max, z_max], [x_max, y_max, z_max], [1, 0, 0], 2))
        debug_lines.append(p.addUserDebugLine([x_max, y_max, z_max], [x_max, y_min, z_max], [1, 0, 0], 2))
        debug_lines.append(p.addUserDebugLine([x_max, y_min, z_max], [x_min, y_min, z_max], [1, 0, 0], 2))

        # Create a sphere for grasping and throwing
        ball_position = [0.3, 0.0, 0.02]  # Sphere starts at a fixed position
        object_id = create_sphere(radius=ball_radius, position=ball_position, mass=ball_mass)

        # Simulate an observation and let the agent predict the action (grasp and throw parameters)
        observation = [(np.zeros([80, 60, 4]), target_pos)]
        action, _ = agent.predict(observation=observation)

        # Extract the grasp pixel index, post-grasp pose, throw pose, and throw velocity from the action
        grasp_pixel_indice, post_grasp_pose, throw_pose, throw_velocity = action[0]

        # Perform the grasp
        grasp_completed = False
        while not grasp_completed:
            grasp_completed = robot.grasp(tcp_target_pose=(ball_position, [0.0, 0.0, 0.0, 1.0]), post_grasp_pose=post_grasp_pose)

            for _ in range(int(240 // 60)):
                p.stepSimulation()
                time.sleep(1./240.)
                robot.log_variables()
                if robot.gripper_control_mode == 'torque':
                    robot.keep_gripper_force()
            robot.visualize_tcp_trajectory()

        # Perform the throw
        throw_completed = False
        while not throw_completed:
            throw_completed = robot.throw(tcp_target_pose=throw_pose, tcp_target_velocity=throw_velocity)
            for _ in range(int(240 // 60)):
                p.stepSimulation()
                time.sleep(1./240.)
                robot.log_variables()
            robot.visualize_tcp_trajectory()

        # Additional 1 second simulation to allow the ball to settle (60 steps of 60 Hz)
        for _ in range(60):
            for _ in range(int(240 // 60)):
                p.stepSimulation()
                time.sleep(1./240.)
                robot.log_variables()

        # Check if the throw was successful by evaluating the final distance to the target
        ball_position_after_throw = p.getBasePositionAndOrientation(object_id)[0]
        success = x_min <= ball_position_after_throw[0] <= x_max and y_min <= ball_position_after_throw[1] <= y_max
        success_matrix[row, col] = success

        # Remove the sphere after the throw
        p.removeBody(object_id)

    return success_matrix

def plot_throw_results(success_matrix, x_points, y_points):
    """
    Plots the throwing success matrix. Successful throws are marked with 'O', failed throws with 'X'.

    Args:
        success_matrix: A 2D matrix showing the success or failure of each throw.
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
    ax.set_title('Throwing Results (O: Success, X: Failure)')

    # Create directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Save the plot to the logs directory
    plot_filename = os.path.join(log_dir, "throwing_results.png")
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
    robot = Panda(
        1 / 240, 1 / 60, (0, 0.0, 0.0), (0.0, 0.0, 0.0),
        gripper_control_mode=gripper_control_mode, 
        use_gripper_gear=use_gripper_gear, 
        visualize_coordinate_frames=True
    )

    # Create a plane in the environment
    create_plane()

    # Create a grid of boxes (targets for throwing)
    boxes_position = [1.0, 0.0]
    _, target_positions = load_boxes_with_dividers(length=0.25, width=0.15, height=0.1, n_rows=3, n_cols=3, position=boxes_position)

    # Create the Physics Agent
    p_module, g_module, t_module = PerceptionModule(), GraspingModule(), ThrowingModule()
    p_controller = PhysicsController()
    agent = PhysicsAgent(device='cpu', perception_module=p_module, grasping_module=g_module, throwing_module=t_module, physics_controller=p_controller)

    # Run the throwing experiment
    success_matrix = run_throwing_experiment(robot, agent, target_positions, n_rows=3, n_cols=3)

    # Plot the results
    x_points = [pos[0] for pos in target_positions[:3]]  # X coordinates for 3 columns
    y_points = [pos[1] for pos in target_positions[::3]]  # Y coordinates for 3 rows
    plot_throw_results(success_matrix, x_points, y_points)

    # Disconnect from the simulation
    p.disconnect()
