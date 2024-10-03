"""
    In this experiment, we measure the number of simulation steps required to fully open and close the gripper when there is a box between the jaws.
"""

import time
import numpy as np
import pybullet as p
import pybullet_data

from tossingbot.envs.pybullet.robot import Panda
from tossingbot.envs.pybullet.utils.objects_utils import create_box, create_plane

if __name__ == '__main__':
    physics_client_id = p.connect(p.GUI)  
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    # Create Panda robot
    gripper_control_mode = 'position'
    use_gripper_gear = True
    robot = Panda(
        1 / 240, 1 / 20, (0, 0.0, 0.0), (0.0, 0.0, 0.0),
        gripper_control_mode=gripper_control_mode, 
        use_gripper_gear=use_gripper_gear, 
        visualize_coordinate_frames=True
    )
    create_plane()

    position = np.array([0.3, -0.3, 0.02])
    half_side = 0.02

    # Simulation step counters
    open_count, close_count = 0, 0
    gripper_moved = False
    gripper_opened = False
    gripper_closed = False

    while True:
        if not gripper_moved:   # Move the gripper
            if not np.linalg.norm(robot.get_tcp_pose()[0] - position) < 1e-4:
                robot.set_tcp_pose_target([position, [0.0, 0.0, 0.0, 1.0]])
            else:
                create_box(half_extents=[half_side, half_side, half_side], position=position, mass=0.1)
                gripper_moved = True
        elif not gripper_closed:  # Close the gripper
            if not robot._is_gripper_closed(close_threshold=half_side) or not robot._is_gripper_stopped():
                robot.close_gripper()
                close_count += 1
            else:
                gripper_closed = True
        elif gripper_closed and not gripper_opened:  # Open the gripper
            if not robot._is_gripper_open() or not robot._is_gripper_stopped():
                robot.open_gripper()
                open_count += 1
            else:
                gripper_opened = True
        else:
            break  # Gripper opened and closed, simulation complete

        # Step the simulation
        for _ in range(int(240 // 20)):
            p.stepSimulation()
            time.sleep(1./240.)
            robot.log_variables()
            if gripper_control_mode == 'torque':
                robot.keep_gripper_force()

    p.disconnect()

    # It takes 2 simulation steps to close the gripper, and 6 simulation steps to open the gripper
    print(f"It takes {close_count} simulation steps to close the gripper, and {open_count} simulation steps to open the gripper")
