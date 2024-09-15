import math
import time
import numpy as np
import pybullet as p
import pybullet_data

from collections import namedtuple
from tossingbot.scene.objects import create_box

class BaseRobot:
    """
    The base class for robots
    """

    def __init__(self, base_position, base_orientation):
        """
        Initialize the robot with its base position and orientation.
        """
        self.base_position = base_position
        self.base_orientation_quat = p.getQuaternionFromEuler(base_orientation)
        self.links = {}  # Store link information
        self.load_robot()
        self.reset()

    def load_robot(self):
        """
        Load the robot URDF and parse joint information.
        """
        self.robot_id = p.loadURDF(
            './assets/urdf/ur5_robotiq_85.urdf',
            self.base_position,
            self.base_orientation_quat,
            useFixedBase=True,
            flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        )
        self._parse_joint_information()
        self._store_link_information()

    def _parse_joint_information(self):
        """
        Parse the joint information from the loaded URDF.
        """
        num_joints = p.getNumJoints(self.robot_id)
        JointInfo = namedtuple('JointInfo', [
            'id', 'name', 'type', 'damping', 'friction',
            'lower_limit', 'upper_limit', 'max_force',
            'max_velocity', 'controllable'
        ])

        self.joints = []
        self.controllable_joints = []

        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i)
            joint_id = info[0]
            joint_name = info[1].decode("utf-8")
            joint_type = info[2]
            joint_damping = info[6]
            joint_friction = info[7]
            joint_lower_limit = info[8]
            joint_upper_limit = info[9]
            joint_max_force = info[10]
            joint_max_velocity = info[11]
            is_controllable = (joint_type != p.JOINT_FIXED)

            if is_controllable:
                self.controllable_joints.append(joint_id)
                p.setJointMotorControl2(self.robot_id, joint_id, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

            joint_info = JointInfo(
                joint_id, joint_name, joint_type, joint_damping, joint_friction,
                joint_lower_limit, joint_upper_limit, joint_max_force,
                joint_max_velocity, is_controllable
            )
            self.joints.append(joint_info)

        assert len(self.controllable_joints) >= self.num_arm_dofs
        self.arm_controllable_joints = self.controllable_joints[:self.num_arm_dofs]
        self.arm_lower_limits = [info.lower_limit for info in self.joints if info.controllable][:self.num_arm_dofs]
        self.arm_upper_limits = [info.upper_limit for info in self.joints if info.controllable][:self.num_arm_dofs]
        self.arm_joint_ranges = [info.upper_limit - info.lower_limit for info in self.joints if info.controllable][:self.num_arm_dofs]

    def _store_link_information(self):
        """
        Store link information in a dictionary.
        """
        num_joints = p.getNumJoints(self.robot_id)
        self.links[p.getBodyInfo(self.robot_id)[0].decode('utf-8')] = -1
        for joint_id in range(num_joints):
            link_name = p.getJointInfo(self.robot_id, joint_id)[12].decode('utf-8')
            self.links[link_name] = joint_id
        self.tcp_id = self.links.get('tcp_link')

    def reset(self):
        """
        Reset the robot to its initial configuration.
        """
        self.reset_arm()

    def reset_arm(self):
        """
        Reset the arm to its initial position.
        """
        self.set_arm_joint_position(self.initial_position[0:6])
        self.set_arm_joint_position_target(self.initial_position[0:6])

    def set_arm_joint_position(self, position):
        """
        Set the position for the arm joints.
        """
        assert len(position) == self.num_arm_dofs
        for joint_id, pos in zip(self.arm_controllable_joints, position):
            p.resetJointState(self.robot_id, joint_id, pos)

    def set_arm_joint_position_target(self, target_position):
        """
        Set the target position for the arm joints.
        """
        assert len(target_position) == self.num_arm_dofs
        for joint_id, pos in zip(self.arm_controllable_joints, target_position):
            p.setJointMotorControl2(
                self.robot_id, joint_id, p.POSITION_CONTROL, pos,
                force=self.joints[joint_id].max_force,
                maxVelocity=self.joints[joint_id].max_velocity
            )

    def set_tcp_pose(self, tcp_pose):
        """
        Set the TCP pose.
        
        Args:
            tcp_pose (list): A list containing the TCP pose with [position, orientation].
                             - position: [x, y, z]
                             - orientation: [qx, qy, qz, qw]
        """
        joint_position = self.inverse_kinematics(pose=tcp_pose)
        self.set_arm_joint_position(position=joint_position)

    def set_tcp_pose_target(self, target_tcp_pose):
        """
        Set the target TCP pose.
        
        Args:
            target_tcp_pose (list): A list containing the target TCP pose with [position, orientation].
                                    - position: [x, y, z]
                                    - orientation: [qx, qy, qz, qw]
        """
        target_joint_position = self.inverse_kinematics(pose=target_tcp_pose)
        self.set_arm_joint_position_target(target_position=target_joint_position)

    def inverse_kinematics(self, pose):
        """
        Calculate inverse kinematics for the given TCP pose.
        
        Args:
            pose (list): A list containing the pose [position, orientation].
                         - position: [x, y, z]
                         - orientation: [qx, qy, qz, qw]
                         
        Returns:
            list: Joint positions to achieve the desired pose.
        """
        position, orientation = pose
        current_joint_position = self.get_joint_position()
        joint_position = p.calculateInverseKinematics(
            self.robot_id, self.tcp_id, position, orientation,
            self.arm_lower_limits, self.arm_upper_limits,
            self.arm_joint_ranges, current_joint_position,
            maxNumIterations=20
        )
        return joint_position

    def get_joint_position(self):
        """
        Get the current position of the joints.
        
        Returns:
            list: Current positions of all controllable joints.
        """
        return [p.getJointState(self.robot_id, joint_id)[0] for joint_id in self.controllable_joints]

    def get_tcp_pose(self):
        """
        Get the current pose of the TCP.
        
        Returns:
            list: A list containing the TCP pose with [position, orientation].
                  - position: [x, y, z]
                  - orientation: [qx, qy, qz, qw]
        """
        link_state = p.getLinkState(self.robot_id, self.tcp_id)
        position = link_state[0]
        orientation = link_state[1]
        return [position, orientation]

    def _pose_distance(self, pose1, pose2):
        """
        Calculate the distance between two poses.
        
        Args:
            pose1 (list): First pose as [position, orientation].
            pose2 (list): Second pose as [position, orientation].
            
        Returns:
            dict: A dictionary containing 'position_distance' and 'orientation_distance'.
        """
        position1 = np.array(pose1[0])
        position2 = np.array(pose2[0])
        position_distance = np.linalg.norm(position1 - position2)

        quat1 = np.array(pose1[1])
        quat2 = np.array(pose2[1])
        dot_product = np.dot(quat1, quat2)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        orientation_distance = 2 * np.arccos(np.abs(dot_product))

        return {
            'position_distance': position_distance,
            'orientation_distance': orientation_distance
        }

    def is_tcp_reached_target(self, target_pose, position_tolerance=0.01, orientation_tolerance=0.01):
        """
        Check if the TCP has reached the target pose.
        
        Args:
            target_pose (list): Target pose as [position, orientation].
            position_tolerance (float): Tolerance for position difference.
            orientation_tolerance (float): Tolerance for orientation difference in radians.
            
        Returns:
            bool: True if the TCP is within the specified tolerances, False otherwise.
        """
        current_pose = self.get_tcp_pose()
        distances = self._pose_distance(current_pose, target_pose)
        return (
            distances['position_distance'] <= position_tolerance and
            distances['orientation_distance'] <= orientation_tolerance
        )

class UR5Robotiq85(BaseRobot):
    def __init__(self, base_position, base_orientation, initial_position=None, visualize_coordinate_frames=False):
        self.num_arm_dofs = 6
        self.initial_position = initial_position if initial_position is not None else [
            0.0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0, 0.085
        ]
        self.gripper_range = [0, 0.085]
        super().__init__(base_position, base_orientation)
        if visualize_coordinate_frames:
            self.visualize_coordinate_frames()

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

    def reset(self):
        """
        Reset the robot and gripper to the initial state.
        """
        super().reset()
        self.reset_gripper()

    def grasp(self, tcp_target_pose):
        """
        Perform a grasping action at the target TCP pose.
        
        Args:
            tcp_target_pose (list): Target TCP pose as [position, orientation].
        """
        pose_over_target = [tcp_target_pose[0][:2] + [0.365], tcp_target_pose[1]]
        while not self.is_tcp_reached_target(target_pose=pose_over_target):
            self.set_tcp_pose_target(pose_over_target)
        self.open_gripper()
        while not self.is_tcp_reached_target(target_pose=tcp_target_pose):
            self.set_tcp_pose_target(tcp_target_pose)
        self.close_gripper()
        while not self.is_tcp_reached_target(target_pose=pose_over_target):
            self.set_tcp_pose_target(pose_over_target)

    def reset_gripper(self):
        """
        Reset the gripper to its open position.
        """
        self.set_gripper_position(self.initial_position[-1])
        self.set_gripper_position_target(self.initial_position[-1])

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
            force=self.joints[self.mimic_parent_id].max_force,
            maxVelocity=self.joints[self.mimic_parent_id].max_velocity
        )

    def get_gripper_position(self):
        """
        Get the current position of the gripper.
        """
        return p.getJointState(self.robot_id, self.mimic_parent_id)[0]
    
    def get_gripper_velocity(self):
        """
        Get the current velocity of the gripper.
        """
        return p.getJointState(self.robot_id, self.mimic_parent_id)[1]

    def _length_to_angle(self, open_length):
        """
        Convert gripper open length to joint angle.
        """
        return 0.715 - math.asin((open_length - 0.010) / 0.1143)

    def visualize_coordinate_frames(self, axis_length=0.1):
        """
        Draw the coordinate frames for specified links in the URDF.

        Args:
            axis_length (float): The length of the coordinate axes.
        """
        links_to_visualize = ['tcp_link']  # Replace with the link names you want to visualize
        for link_name in links_to_visualize:
            link_index = self.links.get(link_name)
            if link_index is None:
                print(f"Link {link_name} not found.")
                continue
            pos = [0, 0, 0]

            # Draw the X-axis (red)
            p.addUserDebugLine(
                pos, 
                [axis_length, 0, 0], 
                [1, 0, 0],  # Color: Red
                parentObjectUniqueId=self.robot_id, 
                parentLinkIndex=link_index
            )

            # Draw the Y-axis (green)
            p.addUserDebugLine(
                pos, 
                [0, axis_length, 0], 
                [0, 1, 0],  # Color: Green
                parentObjectUniqueId=self.robot_id, 
                parentLinkIndex=link_index
            )

            # Draw the Z-axis (blue)
            p.addUserDebugLine(
                pos, 
                [0, 0, axis_length], 
                [0, 0, 1],  # Color: Blue
                parentObjectUniqueId=self.robot_id, 
                parentLinkIndex=link_index
            )

            # Optionally, add the link name as text
            p.addUserDebugText(
                link_name, pos, textColorRGB=[1, 1, 1],
                parentObjectUniqueId=self.robot_id, parentLinkIndex=link_index
            )

if __name__ == '__main__':
    physics_client_id = p.connect(p.GUI)  
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    initial_position = [-1.569/2, -1.545, 1.344, -1.371, -1.571, 0.001, 0.085]
    robot = UR5Robotiq85((0, 0.0, 0.0), (0.0, 0.0, 0.0), initial_position=initial_position, visualize_coordinate_frames=True)

    tcp_target_pose = [[0.5, 0.0, 0.2], [0.0, 0.0, 0.0, 1.0]]
    robot.grasp(tcp_target_pose=tcp_target_pose)

    while True:
        p.stepSimulation()
        time.sleep(1./240.)
