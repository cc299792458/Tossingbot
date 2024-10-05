"""
    In this experiment, we demonstrate the functions to log various variables and visualize them through plots.
"""

import os
import time
import pybullet as p
import pybullet_data

from tossingbot.envs.pybullet.robot import Panda
from tossingbot.envs.pybullet.utils.objects_utils import create_plane, create_sphere

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

    position = [0.3, -0.3, 0.02]
    create_sphere(radius=0.02, position=position)

    post_grasp_pose=([0.3, 0.0, 0.1], [0.0, 0.0, 0.0, 1.0])
    throw_pose = ([0.6, 0.0, 0.4], [0., -0.38268343, 0.0, 0.92387953])
    throw_velocity = ([2.0, 0.0, 2.0], [0.0, 0.0, 0.0])

    grasp_complete = False
    throw_complete = False

    while True:
        if not grasp_complete:  # Grasp the sphere
            grasp_complete = robot.grasp(tcp_target_pose=[position, [0.0, 0.0, 0.0, 1.0]], post_grasp_pose=post_grasp_pose)
        elif not throw_complete:  # Throw the shpere
            throw_complete = robot.throw(tcp_target_pose=throw_pose, tcp_target_velocity=throw_velocity)
        else:
            robot.plot_log_variables(savefig=True, log_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"))
            break
        
        # Run simulation for 60 Hz control loop
        for _ in range(int(240 // 60)):
            p.stepSimulation()
            time.sleep(1./240.)
            robot.log_variables()
            if robot.gripper_control_mode == 'torque':
                robot.keep_gripper_force()
        robot.visualize_tcp_trajectory()

    p.disconnect()
