import math
import time
import numpy as np
import pybullet as p
import pybullet_data

from tossingbot.envs.pybullet.robot.base_robot import BaseRobot
from tossingbot.envs.pybullet.utils.objects_utils import create_box, create_sphere, create_plane

class Panda(BaseRobot):
    def __init__(self, base_position, base_orientation, initial_position=None, visualize_coordinate_frames=False):
        """
        Initialize the Panda robot with default or custom initial joint positions.

        Args:
            base_position (tuple): The position of the robot's base.
            base_orientation (tuple): The orientation of the robot's base.
            initial_position (list or None): A list of joint positions to initialize the robot. 
                                             Defaults to a preset neutral pose if None.
            visualize_coordinate_frames (bool): If True, visualizes the coordinate frames for the robot.
        """
        self.num_arm_dofs = 7
        self.num_gripper_dofs = 2

        # Default joint and gripper positions (in radians)
        self.initial_position = initial_position if initial_position else [
            0.0, -np.pi / 4, 0.0, -np.pi, 0.0, np.pi * 3 / 4, np.pi / 4,  # Joints
            0.04, 0.04  # Gripper (open)
        ]

        self.gripper_range = [0.0, 0.04]    # Gripper fully closed and open limits

        super().__init__(base_position, base_orientation, robot_type='panda')

        if visualize_coordinate_frames:
            self.visualize_coordinate_frames(links_to_visualize=['tcp_link'])

    def _parse_joint_information(self):
        super()._parse_joint_information()
        
        self.gripper_controllable_joints = self.controllable_joints[self.num_arm_dofs:]
        self.gripper_lower_limits = [info.lower_limit for info in self.joints if info.controllable][self.num_arm_dofs:]
        self.gripper_upper_limits = [info.upper_limit for info in self.joints if info.controllable][self.num_arm_dofs:]
        self.gripper_joint_ranges = [info.upper_limit - info.lower_limit for info in self.joints if info.controllable][self.num_arm_dofs:]

    ################## gripper #################
    def set_gripper_position(self, position):
        """
        Set the gripper's position by calculating the corresponding joint angle.
        """
        assert len(position) == self.num_gripper_dofs, "Position length mismatch"
        for joint_id, pos in zip(self.gripper_controllable_joints, position):
            p.resetJointState(self.robot_id, joint_id, pos)

    def set_gripper_position_target(self, target_position):
        """
        Set the target gripper position by calculating the corresponding joint angle.
        """
        assert len(target_position) == self.num_gripper_dofs, "Target position length mismatch"
        for joint_id, target_pos in zip(self.gripper_controllable_joints, target_position):
            p.setJointMotorControl2(
                self.robot_id, joint_id, p.POSITION_CONTROL,
                targetPosition=target_pos,
                force=self.joints[joint_id].max_force,
                maxVelocity=self.joints[joint_id].max_velocity
            )

    def get_gripper_position(self):
        """
        Get the individual positions of the gripper fingers.
        """
        joint_states = p.getJointStates(self.robot_id, self.gripper_controllable_joints)
        finger1_position = joint_states[0][0]
        finger2_position = joint_states[1][0]
        
        # Return both finger positions as a tuple
        return finger1_position, finger2_position
    
    def open_gripper(self):
        """
        Open the gripper.
        """
        self.set_gripper_position_target([self.gripper_range[1], self.gripper_range[1]])

    def close_gripper(self):
        """
        Close the gripper.
        """
        self.set_gripper_position_target([self.gripper_range[0], self.gripper_range[0]])

    def _is_gripper_open(self, tolerance=2e-3, open_threshold=None):
        """
        Check if the gripper is open enough based on a specified threshold.
        
        Args:
            tolerance (float): The tolerance value for checking if the gripper is fully open.
            open_threshold (float, optional): If provided, checks if the gripper has opened by at least 
                                            this threshold value. If None, checks if the gripper is fully open.
                                            
        Returns:
            bool: True if the gripper is open based on the condition, False otherwise.
        """
        gripper_positions = self.get_gripper_position()  # Get current gripper positions
        open_threshold = min(open_threshold, self.gripper_range[1]) if open_threshold is not None else self.gripper_range[1]
        position_condition = max(open_threshold - gripper_positions[0], open_threshold - gripper_positions[1]) < tolerance
        
        return position_condition

    def _is_gripper_closed(self, tolerance=2e-3):
        """
        Check if the gripper is fully closed.
        
        Args:
            tolerance (float): The tolerance value for checking if the gripper is closed.
            
        Returns:
            bool: True if the gripper is closed, False otherwise.
        """
        position_condition = max(self.get_gripper_position()[0] - self.gripper_range[0], self.get_gripper_position()[1] - self.gripper_range[0]) < tolerance
        return position_condition

    def _is_gripper_stopped(self, position_change_tolerance=1e-4, check_steps=10):
        """
        Check if the gripper has stopped moving based on position change within a certain number of steps.
        
        Args:
            position_change_tolerance (float): The tolerance for the gripper position change.
            check_steps (int): Number of steps to check for stopping.
            
        Returns:
            bool: True if the gripper has stopped, False otherwise.
        """
        # Initialize or reset the stopping check
        if not hasattr(self, '_gripper_stop_count'):
            self._gripper_stop_count = 0
            self._gripper_stopped = False
            self._previous_gripper_position = self.get_gripper_position()  # Initialize previous position

        current_position = self.get_gripper_position()
        position_change = max(abs(current_position[0] - self._previous_gripper_position[0]), abs(current_position[1] - self._previous_gripper_position[1]))

        # Update the previous position for the next check
        self._previous_gripper_position = current_position

        # Check if the gripper position change is below the tolerance
        if position_change < position_change_tolerance:
            # If conditions are met, increment the stop count
            self._gripper_stop_count += 1
            # Check if the stop count has reached the required number of steps
            if self._gripper_stop_count >= check_steps:
                self._gripper_stopped = True
                del self._gripper_stop_count
                del self._previous_gripper_position
        else:
            # If conditions are not met, reset the stop count
            self._gripper_stop_count = 0
            self._gripper_stopped = False

        return self._gripper_stopped

if __name__ == '__main__':
    physics_client_id = p.connect(p.GUI)  
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    # Create Panda robot
    robot = Panda((0, 0.0, 0.0), (0.0, 0.0, 0.0), visualize_coordinate_frames=True)
    create_plane()
    position = [0.3, -0.3, 0.03]    
    # object_id = create_box(half_extents=[0.02, 0.02, 0.02], position=position, mass=0.2)
    object_id = create_sphere(radius=0.03, position=position, mass=0.2)
    p.changeDynamics(object_id, -1, lateralFriction=1.0, rollingFriction=0.01)
    grasp_pose = (position, [1.0, 0.0, 0.0, 0.0])
    post_grasp_pose = ([0.3, 0.0, 0.4], [1.0, 0.0, 0.0, 0.0])
    throw_pose = ([0.4, 0.0, 0.4], [1.0, 0.0, 0.0, 0.0])
    throw_vel = ([2.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    grasp_completed = False
    throw_completed = False

    count = 0
    while True:
        if not grasp_completed:
            grasp_completed = robot.grasp(tcp_target_pose=grasp_pose, post_grasp_pose=post_grasp_pose)
        # # Wait for 48 / 240 second
        # elif count < 48:
        #     count += 1
        elif not throw_completed:
            throw_completed = robot.throw(tcp_target_pose=throw_pose, tcp_target_velocity=throw_vel)
        p.stepSimulation()
        robot.visualize_tcp_trajectory()
        time.sleep(1./240.)
