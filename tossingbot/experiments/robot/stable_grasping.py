"""
    In this experiment, we test the stability of the robot grasping.
"""
import time
import pybullet as p
import pybullet_data

from tossingbot.envs.pybullet.robot import Panda
from tossingbot.envs.pybullet.utils.objects_utils import create_box, create_sphere, create_plane

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
    
    position = [0.3, -0.2, 0.03]
    # object_id = create_box(half_extents=[0.02, 0.02, 0.02], position=position, mass=0.2)
    object_id = create_sphere(radius=0.02, position=position, mass=0.2)
    p.changeDynamics(object_id, -1, lateralFriction=1.0, rollingFriction=0.01)
    grasp_pose = (position, [0.0, 0.0, 0.0, 1.0])
    post_grasp_pose = ([0.3, 0.0, 0.4], [0.0, 0.0, 0.0, 1.0])
    grasp_completed = False
    throw_completed = False

    count = 0
    while True:
        if not grasp_completed:
            grasp_completed = robot.grasp(tcp_target_pose=grasp_pose, post_grasp_pose=post_grasp_pose)
        for _ in range(int(240 // 20)):
            p.stepSimulation()
            time.sleep(1./240.)
            robot.log_variables()
            if gripper_control_mode == 'torque':
                robot.keep_gripper_force()
        robot.visualize_tcp_trajectory()

    p.disconnect()