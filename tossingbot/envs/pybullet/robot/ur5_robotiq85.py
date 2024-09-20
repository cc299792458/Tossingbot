import math
import time
import numpy as np
import pybullet as p
import pybullet_data

from tossingbot.envs.pybullet.robot.base_robot import BaseRobot
from tossingbot.envs.pybullet.utils.objects_utils import create_box, create_plane

class UR5Robotiq85(BaseRobot):
    def __init__(self, base_position, base_orientation, initial_position=None, visualize_coordinate_frames=False):
        """
        Initialize the UR5-Robotiq85 robot with default or custom initial joint positions.

        Args:
            base_position (tuple): The position of the robot's base.
            base_orientation (tuple): The orientation of the robot's base.
            initial_position (list or None): A list of joint positions to initialize the robot. 
                                             Defaults to a preset neutral pose if None.
            visualize_coordinate_frames (bool): If True, visualizes the coordinate frames for the robot.
        """
        self.num_arm_dofs = 6  # Already initialized in BaseRobot, but kept here for clarity
        self.initial_position = initial_position if initial_position is not None else [
            0.0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0,  # Joints
            0.085   # Gripper (open)
        ]
        self.gripper_range = [0.0, 0.085]
        super().__init__(base_position, base_orientation, robot_type='ur5_robotiq85')
        if visualize_coordinate_frames:
            self.visualize_coordinate_frames(links_to_visualize=['tcp_link'])

    ############### load robot ###############
    def load_robot(self):
        """
        Load the URDF and set up mimic joints for the gripper.
        """
        super().load_robot()
        self._setup_mimic_joints()

    def _setup_mimic_joints(self):
        """
        Set up mimic joints for the gripper.
        """
        parent_joint_name = 'finger_joint'
        child_joint_multipliers = {
            'right_outer_knuckle_joint': 1,
            'left_inner_knuckle_joint': 1,
            'right_inner_knuckle_joint': 1,
            'left_inner_finger_joint': -1,
            'right_inner_finger_joint': -1
        }
        self.mimic_parent_id = next(
            joint.id for joint in self.joints if joint.name == parent_joint_name
        )
        self.mimic_child_multiplier = {
            joint.id: child_joint_multipliers[joint.name] for joint in self.joints
            if joint.name in child_joint_multipliers
        }

        for joint_id, multiplier in self.mimic_child_multiplier.items():
            constraint = p.createConstraint(
                self.robot_id, self.mimic_parent_id,
                self.robot_id, joint_id,
                jointType=p.JOINT_GEAR,
                jointAxis=[0, 1, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=[0, 0, 0]
            )
            p.changeConstraint(
                constraint, gearRatio=-multiplier, maxForce=100, erp=1
            )

    ############### gripper ###############
    def set_gripper_position(self, open_length):
        """
        Set the gripper's position by calculating the corresponding joint angle.
        """
        open_angle = self._length_to_angle(open_length)
        p.resetJointState(self.robot_id, self.mimic_parent_id, open_angle)

    def set_gripper_position_target(self, open_length):
        """
        Set the target gripper position by calculating the corresponding joint angle.
        """
        open_angle = self._length_to_angle(open_length)
        p.setJointMotorControl2(
            self.robot_id, self.mimic_parent_id, p.POSITION_CONTROL,
            targetPosition=open_angle,
            force=self.joints[self.mimic_parent_id].max_force / 15,     # The default value is too large
            maxVelocity=self.joints[self.mimic_parent_id].max_velocity / 15     # The same
        )

    def get_gripper_position(self):
        """
        Get the current position of the gripper.
        """
        return p.getJointState(self.robot_id, self.mimic_parent_id)[0]
    
    def open_gripper(self):
        """
        Open the gripper.
        """
        self.set_gripper_position_target(self.gripper_range[1])

    def close_gripper(self):
        """
        Close the gripper.
        """
        self.set_gripper_position_target(self.gripper_range[0])

    def _is_gripper_open(self, tolerance=1e-2):
        """
        Check if the gripper is fully open.
        
        Args:
            tolerance (float): The tolerance value for checking if the gripper is open.
            
        Returns:
            bool: True if the gripper is open, False otherwise.
        """
        position_condition = abs(self.get_gripper_position() - self._length_to_angle(self.gripper_range[1])) < tolerance
        return position_condition

    def _is_gripper_closed(self, tolerance=1e-2):
        """
        Check if the gripper is fully closed.
        
        Args:
            tolerance (float): The tolerance value for checking if the gripper is closed.
            
        Returns:
            bool: True if the gripper is closed, False otherwise.
        """
        position_condition = abs(self.get_gripper_position() - self._length_to_angle(self.gripper_range[0])) < tolerance
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
        position_change = abs(current_position - self._previous_gripper_position)

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

    def _length_to_angle(self, open_length):
        """
        Convert gripper open length to joint angle.
        """
        return 0.715 - math.asin((open_length - 0.010) / 0.1143)

    def _angle_to_length(self, joint_angle):
        """
        Convert gripper joint angle to open length.
        """
        return 0.010 + 0.1143 * math.sin(0.715 - joint_angle)

if __name__ == '__main__':
    physics_client_id = p.connect(p.GUI)  
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    # Create UR5 robot
    robot = UR5Robotiq85((0, 0.0, 0.0), (0.0, 0.0, 0.0), visualize_coordinate_frames=True)
    create_plane()
    position = [0.4, -0.4, 0.03]    
    box_id = create_box(half_extents=[0.02, 0.02, 0.02], position=position, mass=0.1)
    p.changeDynamics(box_id, -1, lateralFriction=1.0, rollingFriction=0.01)
    tcp_target_pose = [position, [-0.0006627706705588098, 0.707114179457306, -0.0007339598331235209, 0.7070986913072476]]
    completed = False

    while True:
        if not completed:
            completed = robot.grasp(tcp_target_pose=tcp_target_pose, num_subtargets=10)
        p.stepSimulation()
        time.sleep(1./240.)
