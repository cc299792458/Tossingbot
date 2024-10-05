"""
    In this experiment, we let the robot move its tcp to the post_grasp_pose and throw_pose given by a physics agent.
"""

import time
import numpy as np
import pybullet as p
import pybullet_data

from tossingbot.envs.pybullet.robot import Panda
from tossingbot.envs.pybullet.utils.objects_utils import create_plane
from tossingbot.agents.physics_agent import PhysicsAgent, PhysicsController
from tossingbot.networks.networks import PerceptionModule, GraspingModule, ThrowingModule

if __name__ == '__main__':
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
    create_plane()

    # Create Physics Agent
    p_module, g_module, t_module = PerceptionModule(), GraspingModule(), ThrowingModule()
    p_controller = PhysicsController()
    agent = PhysicsAgent(device='cpu', perception_module=p_module, grasping_module=g_module, throwing_module=t_module, physics_controller=p_controller)

    action, intermidiates = agent.predict(observation=[(np.zeros([80, 60, 4]), [1.25, 0.0, 0.2])])

    grasp_pixel_indice, post_grasp_pose, throw_pose, throw_velocity = action[0]

    move_to_post_grasp_pose_count = 0
    move_to_throw_pose_count = 0

    while True:
        if not move_to_post_grasp_pose_count > 180:
            robot.set_tcp_pose_target(target_tcp_pose=post_grasp_pose)
            move_to_post_grasp_pose_count += 1
        elif not move_to_throw_pose_count > 180:
            robot.set_tcp_pose_target(target_tcp_pose=throw_pose)
            move_to_throw_pose_count += 1
        else:
            continue
        
        # Run simulation for 60 Hz control loop
        for _ in range(int(240 // 60)):
            p.stepSimulation()
            time.sleep(1./240.)
            robot.log_variables()
            if robot.gripper_control_mode == 'torque':
                robot.keep_gripper_force()
        robot.visualize_tcp_trajectory()

    p.disconnect()
