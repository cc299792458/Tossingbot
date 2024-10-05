"""
    In this experiment, we demonstrate the visualization of rotated depth heightmaps.
    Depth images are passed through the network, rotated in various orientations,
    and the resulting visual transformations are displayed.
"""

import torch
import numpy as np

from tossingbot.envs.pybullet.utils.camera_utils import plot_heightmaps
from tossingbot.agents.physics_agent import PhysicsAgent, PhysicsController
from tossingbot.networks import PerceptionModule, GraspingModule, ThrowingModule

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

if __name__ == '__main__':
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
    physics_controller = PhysicsController()
    agent = PhysicsAgent(
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        perception_module=perception_module,
        grasping_module=grasp_module,
        throwing_module=throw_module,
        physics_controller=physics_controller
    )

    # Make prediction with observation
    n_rotations = 16
    observation = [(heightmap, target_position), (heightmap, target_position)]
    action, intermidiates = agent.predict(observation=observation, n_rotations=n_rotations)
    depth_heightmaps = [intermidiates['depth_heightmaps'][0, i, :, :] for i in range(n_rotations)]
    plot_heightmaps(depth_heightmaps)