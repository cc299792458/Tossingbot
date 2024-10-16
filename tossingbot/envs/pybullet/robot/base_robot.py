import os
import math
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from collections import namedtuple
from scipy.spatial.transform import Rotation as R
from tossingbot.envs.pybullet.utils.math_utils import slerp, quaternion_to_euler

class BaseRobot:
    """
    The base class for robots
    """
    def __init__(
            self, 
            timestep, 
            control_timestep, 
            base_position, 
            base_orientation, 
            gripper_control_mode='torque', 
            use_gripper_gear=True,
            robot_type='panda'):
        """
        Initialize the robot with its base position, orientation, and type.
        """
        self.timestep = timestep
        self.control_timestep = control_timestep
        self.sim_step_per_ctrl_step = int(control_timestep // timestep)
        self.base_position = base_position
        self.base_orientation_quat = p.getQuaternionFromEuler(base_orientation)
        assert gripper_control_mode == 'position' or 'torque'
        self.gripper_control_mode = gripper_control_mode
        self.use_gripper_gear = use_gripper_gear
        self.robot_type = robot_type
        assert robot_type in ['ur5_robotiq85', 'panda'], "robot_type must be 'ur5_robotiq85' or 'panda'"
        self.load_robot()
        self.reset()
        self.initialize_targets()
        self.initialize_logs()
    
    ###############  initialization ###############
    def initialize_targets(self):
        """Initialize the targets as the initial configuration"""
        self._joint_target_position = self.get_arm_joint_position()
        self._joint_target_velocity = self.get_arm_joint_velocity()
        self._tcp_target_pose = self.get_tcp_pose()
        self._tcp_target_velocity = self.get_tcp_velocity()
        self._gripper_target_position = self.get_gripper_position()

    def initialize_logs(self):
        # List of log attribute names
        log_attrs = [
            'joint_position_log',
            'joint_velocity_log',
            'target_joint_position_log',
            'target_joint_velocity_log',
            'joint_torque_log',
            'tcp_pose_log',
            'tcp_velocity_log',
            'target_tcp_pose_log',
            'target_tcp_velocity_log',
            'gripper_position_log',
            'target_gripper_position_log',
        ]
        
        # Initialize logs
        for attr in log_attrs:
            setattr(self, attr, [])

    ###############  load robot ###############
    def load_robot(self):
        """
        Load the robot URDF and parse joint information.
        """
        if self.robot_type == 'ur5_robotiq85':
            urdf_path = './assets/urdf/ur5_robotiq_85.urdf'
            use_fixed_base = True
        elif self.robot_type == 'panda':
            urdf_path = './assets/urdf/panda.urdf'
            use_fixed_base = True
        else:
            raise ValueError(f"Unknown robot type: {self.robot_type}")

        self.robot_id = p.loadURDF(
            urdf_path,
            self.base_position,
            self.base_orientation_quat,
            useFixedBase=use_fixed_base,
            flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        )
        self._parse_joint_information()
        self._store_link_information()
        self._set_gripper_information()

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
            joint_max_force = info[10] * 10
            joint_max_velocity = info[11] * 10
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

        assert len(self.controllable_joints) >= self.num_arm_dofs, "Not enough controllable joints"
        self.arm_controllable_joints = self.controllable_joints[:self.num_arm_dofs]
        self.arm_lower_limits = np.array([info.lower_limit for info in self.joints if info.controllable][:self.num_arm_dofs])
        self.arm_upper_limits = np.array([info.upper_limit for info in self.joints if info.controllable][:self.num_arm_dofs])
        self.arm_joint_ranges = np.array([info.upper_limit - info.lower_limit for info in self.joints if info.controllable][:self.num_arm_dofs])

    def _store_link_information(self, change_dynamics=False):
        """
        Store link information in a dictionary and set the TCP link based on the robot type.
        """
        num_joints = p.getNumJoints(self.robot_id)
        self.links = {}
        
        self.links[p.getBodyInfo(self.robot_id)[0].decode('utf-8')] = -1
        for joint_id in range(num_joints):
            link_name = p.getJointInfo(self.robot_id, joint_id)[12].decode('utf-8')
            self.links[link_name] = joint_id
            # Change friction parameter
            if change_dynamics:
                p.changeDynamics(self.robot_id, joint_id, lateralFriction=1.0, rollingFriction=0.01, linearDamping=0, angularDamping=0)
        
        # Set TCP link based on robot type
        if self.robot_type == 'ur5_robotiq85':
            self.tcp_id = self.links.get('tcp_link', -1)
        elif self.robot_type == 'panda':
            self.tcp_id = self.links.get('tcp_link', -1)

        assert self.tcp_id != -1, "TCP link not found for the robot"

    def _set_gripper_information(self):
        raise NotImplementedError

    ############### reset robot ###############
    def reset(self):
        """
        Reset the robot to its initial configuration.
        """
        self._reset_arm()
        self._reset_gripper()

    def _reset_arm(self):
        """
        Reset the arm to its initial position.
        """
        self.set_arm_joint_position(self.initial_position[:self.num_arm_dofs])
        self.set_arm_joint_position_target(self.initial_position[:self.num_arm_dofs])

    def _reset_gripper(self):
        """
        Reset the gripper to its open position.
        """
        self.set_gripper_position(self.initial_position[self.num_arm_dofs:])
        self.set_gripper_position_target(self.initial_position[self.num_arm_dofs:])

    ############### set and get arm position ###############
    def set_arm_joint_position(self, position):
        """
        Set the position for the arm joints.
        """
        assert len(position) == self.num_arm_dofs, "Position length mismatch"
        for joint_id, pos in zip(self.arm_controllable_joints, position):
            p.resetJointState(self.robot_id, joint_id, pos)

    def set_arm_joint_position_target(self, target_position, target_velocity=None):
        """
        Set the target position for the arm joints.
        """
        assert len(target_position) == self.num_arm_dofs, "Target position length mismatch"
        if target_velocity is not None:
            assert len(target_velocity) == self.num_arm_dofs, "Target velocity length mismatch"
        else:
            target_velocity = np.zeros([len(target_position)])
        for joint_id, pos, vel in zip(self.arm_controllable_joints, target_position, target_velocity):
            p.setJointMotorControl2(
            self.robot_id, 
            joint_id, 
            p.POSITION_CONTROL, 
            targetPosition=pos,
            targetVelocity=vel,
            force=self.joints[joint_id].max_force,
            maxVelocity=self.joints[joint_id].max_velocity,
            # positionGain=0.5,
            # velocityGain=0.1,
        )
        self._joint_target_position = target_position
        self._joint_target_velocity = target_velocity

    def set_arm_joint_velocity_target(self, target_velocity):
        """
        Set the target velocity for the arm joints.
        
        Args:
            target_velocity (list): List of target velocities for each arm joint.
        """
        assert len(target_velocity) == self.num_arm_dofs, "Velocity length mismatch"
        for joint_id, vel in zip(self.arm_controllable_joints, target_velocity):
            p.setJointMotorControl2(
                self.robot_id, joint_id, p.VELOCITY_CONTROL, 
                targetVelocity=vel,
                force=self.joints[joint_id].max_force
            )
        self._joint_target_velocity = target_velocity

    def get_arm_joint_position(self):
        """
        Get the current position of the arm joints.
        
        Returns:
            list: Current positions of arm controllable joints.
        """
        return np.array([p.getJointState(self.robot_id, joint_id)[0] for joint_id in self.arm_controllable_joints])

    def get_arm_joint_velocity(self):
        """
        Get the current velocity of the arm joints.
        
        Returns:
            list: Current velocities of arm controllable joints.
        """
        return np.array([p.getJointState(self.robot_id, joint_id)[1] for joint_id in self.arm_controllable_joints])
    
    def get_arm_joint_torque(self):
        """
        Get the current motor torques of the arm joints.

        Returns:
            list: Current motor torques applied by the motors on the arm controllable joints.
        """
        return np.array([p.getJointState(self.robot_id, joint_id)[3] for joint_id in self.arm_controllable_joints])

    ############### set tcp pose, get tcp pose, and inverse kinematics ###############
    def set_tcp_pose(self, tcp_pose):
        """
        Set the TCP pose.
        
        Args:
            tcp_pose (list): A list containing the TCP pose with [position, orientation].
                            - position: [x, y, z]
                            - orientation: [qx, qy, qz, qw]
        """
        joint_position = self.pose_ik(pose=tcp_pose)
        self.set_arm_joint_position(position=joint_position)

    def set_tcp_pose_target(self, target_tcp_pose, target_tcp_velocity=None):
        """
        Set the target TCP pose.
        
        Args:
            target_tcp_pose (tuple): A tuple containing the target TCP pose with [position, orientation].
                                    - position: [x, y, z]
                                    - orientation: [qx, qy, qz, qw]
            target_tcp_velocity (tuple): A tuple containing the target TCP velocity with [linear_velocity, angular_velocity].
                                        - linear_velocity: [vx, vy, vz]
                                        - angular_velocity: [wx, wy, wz]
        """
        target_joint_position = self.pose_ik(pose=target_tcp_pose)
        target_joint_velocity = self.velocity_ik(target_tcp_velocity) if target_tcp_velocity is not None else np.zeros([len(target_joint_position)])
        self.set_arm_joint_position_target(target_position=target_joint_position, target_velocity=target_joint_velocity)

    def set_tcp_velocity_target(self, target_tcp_velocity):
        """
        Set the target velocity for the TCP.
        
        Args:
            target_tcp_velocity (tuple): Desired velocity in Cartesian space [linear_velocity, angular_velocity].
        """
        target_joint_velocity = self.velocity_ik(target_tcp_velocity)
        self.set_arm_joint_velocity_target(target_joint_velocity)

    def get_tcp_pose(self):
        """
        Get the current pose of the TCP.
        
        Returns:
            pose (tuple): A tuple containing the TCP pose with (position, orientation).
                        - position: [x, y, z]
                        - orientation: [qx, qy, qz, qw]
        """
        link_state = p.getLinkState(self.robot_id, self.tcp_id)
        position = np.array(link_state[0])
        orientation = np.array(link_state[1])
        return (position, orientation)
    
    def get_tcp_target_pose(self):
        if hasattr(self, '_tcp_target_pose'):
            return self._tcp_target_pose
        else:
            return None

    def get_tcp_velocity(self):
        """
        Get the current velocity of the TCP.
        
        Returns:
            tuple: A tuple containing the TCP velocity with (linear_velocity, angular_velocity).
                - linear_velocity: [vx, vy, vz]
                - angular_velocity: [wx, wy, wz]
        """
        link_state = p.getLinkState(self.robot_id, self.tcp_id, computeLinkVelocity=True)
        linear_velocity = np.array(link_state[6])
        angular_velocity = np.array(link_state[7])
        return (linear_velocity, angular_velocity)

    ############### Pose and Velocity Inverse Kinematics ###############
    def pose_ik(self, pose, rest_pose=None):
        """
        Calculate inverse kinematics for the given TCP pose.
        
        Args:
            pose (tuple): A tuple containing the pose (position, orientation).
                        - position: [x, y, z]
                        - orientation: [qx, qy, qz, qw]
                        
        Returns:
            list: Joint positions to achieve the desired pose.
        """
        position, orientation = pose    
        # Use rest pose if provided; otherwise, use current joint positions
        if rest_pose is None:
            rest_pose = self.get_arm_joint_position()
        joint_position = p.calculateInverseKinematics(
            bodyUniqueId=self.robot_id,
            endEffectorLinkIndex=self.tcp_id,
            targetPosition=position,
            targetOrientation=orientation,
            lowerLimits=self.arm_lower_limits,
            upperLimits=self.arm_upper_limits,
            jointRanges=self.arm_joint_ranges,
            restPoses=rest_pose,
            maxNumIterations=100
        )
        return np.array(joint_position[:self.num_arm_dofs])

    def velocity_ik(self, velocity):
        """
        Compute joint velocities required to achieve the desired TCP (end-effector) velocity.

        Args:
            velocity (tuple): A tuple containing the velocity (linear_velocity, angular_velocity).
                        - linear_velocity: Desired linear velocity [vx, vy, vz].
                        - angular_velocity: Desired angular velocity [wx, wy, wz].

        Returns:
            np.ndarray: Joint velocities to achieve the desired TCP velocity.
        """
        linear_velocity, angular_velocity = velocity
        joint_positions = self.get_joint_position()
        zero_vec = [0.0] * len(joint_positions)
        jacobian_linear, jacobian_angular = p.calculateJacobian(
            bodyUniqueId=self.robot_id,
            linkIndex=self.tcp_id,
            localPosition=(0, 0, 0),  # TCP origin in local coordinates
            objPositions=joint_positions,
            objVelocities=zero_vec,
            objAccelerations=zero_vec
        )

        jacobian = np.vstack((jacobian_linear, jacobian_angular))  # Combine linear and angular Jacobians
        desired_velocity = np.hstack((linear_velocity, angular_velocity))  # Desired velocity vector
        joint_velocities = np.linalg.pinv(jacobian).dot(desired_velocity)  # Compute joint velocities using Jacobian

        return np.array(joint_velocities[:self.num_arm_dofs])
    
    def get_joint_position(self):
        """
        Get the current position of all the joints.
        
        Returns:
            list: Current positions of controllable joints.
        """
        return [p.getJointState(self.robot_id, joint_id)[0] for joint_id in self.controllable_joints]

    ############### gripper ###############
    def set_gripper_position(self):
        raise NotImplementedError
    
    def set_gripper_position_target(self, target_position):
        self._gripper_target_position = target_position
    
    def get_gripper_position(self):
        raise NotImplementedError
    
    def get_gripper_force(self):
        raise NotImplementedError
    
    def keep_gripper_force(self):
        """
        Keep the gripper force in every PyBullet simulation step.

        This method must be called in each simulation step to apply the correct force 
        based on the control strategy (e.g., position error or impedance control).
        """
        raise NotImplementedError
    
    def open_gripper(self):
        raise NotImplementedError
    
    def close_gripper(self):
        raise NotImplementedError
    
    def _is_gripper_open(self):
        raise NotImplementedError
    
    def _is_gripper_closed(self):
        raise NotImplementedError
    
    def _is_gripper_stopped(self):
        raise NotImplementedError
        
    ############### motion primitives ###############
    def grasp(self, tcp_target_pose, post_grasp_pose, estimate_speed=0.2, gripper_duration=1.0):
        """
        Perform a grasping action at the target TCP pose in a step-by-step manner.
        
        The process involves:
        1. Moving to a position above the target.
        2. Opening the gripper.
        3. Moving down to the target pose.
        4. Closing the gripper to grasp the object.
        5. Lifting back to a safe post-grasp position.
        
        Args:
            tcp_target_pose (list): Target TCP pose as [position, orientation].
            post_grasp_pose (list, optional): The TCP pose to move to after grasping for safety or further actions.
            estimate_speed (float): Speed for moving the TCP.
            gripper_duration (float): Duration for opening/closing the gripper.

        Returns:
            bool: True if the grasping process is completed, False otherwise.
        """
        # Initialize or continue the grasping process
        if not hasattr(self, '_grasp_step'):
            self._initialize_grasp_process(tcp_target_pose, estimate_speed)

        # Step 1: Move to a position above the target
        if self._grasp_step == 0:
            self._move_to_pre_grasp_pose()

        # Step 2: Open the gripper
        if self._grasp_step == 1:
            self._open_gripper(gripper_duration)
        
        # Step 3: Move down to the target position
        if self._grasp_step == 2:
            self._move_to_target_pose()

        # Step 4: Close the gripper to grasp the object
        if self._grasp_step == 3:
            self._close_gripper(gripper_duration, post_grasp_pose, estimate_speed)
        
        # Step 5: Move to the post-grasp pose
        if self._grasp_step == 4:
            return self._move_to_post_grasp_pose()

        return False  # Grasping process not yet complete

    def _initialize_grasp_process(self, tcp_target_pose, estimate_speed):
        """Initializes the grasp process."""
        self._grasp_step = 0
        if not hasattr(self, 'grasp_start_times'):
            self.grasp_start_times = []
        self.grasp_start_times.append(len(self.joint_position_log))

        self.pre_grasp_pose = (np.array(list(tcp_target_pose[0][:2]) + [0.3]), tcp_target_pose[1])  # Pose above the target
        self.tcp_target_pose = tcp_target_pose

        # Generate trajectory to pre-grasp position
        self._tcp_trajectory = self._generate_tcp_trajectory(self.get_tcp_pose(), self.pre_grasp_pose, estimate_speed=estimate_speed)
        self._trajectory_index = 0

    def _move_to_pre_grasp_pose(self):
        """Moves to the pre-grasp pose."""
        if self._trajectory_index < len(self._tcp_trajectory):
            self._tcp_target_pose, self._tcp_target_velocity = self._tcp_trajectory[self._trajectory_index]
            self.set_tcp_pose_target(self._tcp_target_pose, self._tcp_target_velocity)
            self._trajectory_index += 1
        else:
            self._grasp_step = 1

    def _open_gripper(self, gripper_duration):
        """Opens the gripper."""
        if not hasattr(self, '_open_gripper_step'):
            self._open_gripper_step = 0
            self._total_open_gripper_steps = math.ceil(gripper_duration * self.sim_step_per_ctrl_step)
        else:
            self._open_gripper_step += 1

        self.open_gripper()

        if self._open_gripper_step >= self._total_open_gripper_steps:
            self._grasp_step = 2
            self._tcp_trajectory = self._generate_tcp_trajectory(self.pre_grasp_pose, self.tcp_target_pose, estimate_speed=0.2)
            self._trajectory_index = 0
            del self._open_gripper_step, self._total_open_gripper_steps

    def _move_to_target_pose(self):
        """Moves to the target grasp pose."""
        if self._trajectory_index < len(self._tcp_trajectory):
            self._tcp_target_pose, self._tcp_target_velocity = self._tcp_trajectory[self._trajectory_index]
            self.set_tcp_pose_target(self._tcp_target_pose, self._tcp_target_velocity)
            self._trajectory_index += 1
        else:
            self._grasp_step = 3

    def _close_gripper(self, gripper_duration, post_grasp_pose, estimate_speed):
        """Closes the gripper to grasp the object."""
        if not hasattr(self, '_close_gripper_step'):
            self._close_gripper_step = 0
            self._total_close_gripper_steps = math.ceil(gripper_duration * self.sim_step_per_ctrl_step)
        else:
            self._close_gripper_step += 1

        self.close_gripper()

        if self._close_gripper_step >= self._total_close_gripper_steps:
            self._grasp_step = 4
            self._tcp_trajectory = self._generate_tcp_trajectory(self.tcp_target_pose, post_grasp_pose, estimate_speed=estimate_speed)
            self._trajectory_index = 0
            del self._close_gripper_step, self._total_close_gripper_steps

    def _move_to_post_grasp_pose(self):
        """Moves to the post-grasp pose."""
        if self._trajectory_index < len(self._tcp_trajectory):
            self._tcp_target_pose, self._tcp_target_velocity = self._tcp_trajectory[self._trajectory_index]
            self.set_tcp_pose_target(self._tcp_target_pose, self._tcp_target_velocity)
            self._trajectory_index += 1
        else:
            self._finalize_grasp_process()
            return True

        return False

    def _finalize_grasp_process(self):
        """Finalizes the grasp process."""
        del self._grasp_step, self._tcp_trajectory, self._trajectory_index
        if not hasattr(self, 'grasp_end_times'):
            self.grasp_end_times = []
        self.grasp_end_times.append(len(self.joint_position_log))

    def throw(self, tcp_target_pose, tcp_target_velocity, settling_steps=3, open_gripper_steps=1):
        """
        Perform a throwing action with three stages:
        1. Move to the release point with a target velocity.
        2. Release the object by opening the gripper while maintaining the current velocity.
        3. Decelerate the arm after the release.

        Args:
            tcp_target_pose (tuple): Target TCP pose at the release point as (position, orientation).
            tcp_target_velocity (tuple): Target velocity for the TCP at the release point (linear_velocity, angular_velocity).
            settling_steps (int): Steps used to stabilize the TCP velocity before opening the gripper.
            open_gripper_steps (int): Additional steps after stabilizing to release the gripper.

        Returns:
            bool: True if the throwing process is completed, False otherwise.
        """
        # Initialize or continue the throwing process
        if not hasattr(self, '_throw_step'):
            self._initialize_throw_process(tcp_target_pose, tcp_target_velocity, settling_steps)

        # Stage 1: Move to the release point with the target velocity
        if self._throw_step == 0:
            self._move_to_release_point()

        # Stage 2: Release the object by opening the gripper while maintaining the velocity
        if self._throw_step == 1:
            self._release_object(settling_steps, open_gripper_steps, tcp_target_pose)

        # Stage 3: Decelerate the arm by gradually reducing velocity to zero
        if self._throw_step == 2:
            return self._decelerate_arm()

        return False  # Throwing process is not complete

    def _initialize_throw_process(self, tcp_target_pose, tcp_target_velocity, settling_steps):
        """Initialize the throwing process by setting the initial conditions and calculating the pre-throw position."""
        self._throw_step = 0
        if not hasattr(self, 'throw_start_times'):
            self.throw_start_times = []
        self.throw_start_times.append(len(self.joint_position_log))

        self.tcp_target_pose = tcp_target_pose
        self.tcp_target_velocity = tcp_target_velocity

        # Calculate the pre-throw position based on velocity and timestep
        pre_throw_position = np.array([
            tcp_target_pose[0][i] - tcp_target_velocity[0][i] * self.control_timestep * settling_steps
            for i in range(3)
        ])
        self.pre_throw_pose = (pre_throw_position, tcp_target_pose[1])

        # Generate the trajectory towards the release point
        self._tcp_trajectory = self._generate_tcp_trajectory(
            self.get_tcp_pose(),
            self.pre_throw_pose,
            start_tcp_vel=(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])),
            target_tcp_vel=tcp_target_velocity,
        )
        self._trajectory_index = 0

    def _move_to_release_point(self):
        """Move along the trajectory step by step towards the release point."""
        if self._trajectory_index < len(self._tcp_trajectory):
            self._tcp_target_pose, self._tcp_target_velocity = self._tcp_trajectory[self._trajectory_index]
            self.set_tcp_pose_target(self._tcp_target_pose, self._tcp_target_velocity)
            self._trajectory_index += 1
        else:
            self._throw_step = 1

    def _release_object(self, settling_steps, open_gripper_steps, tcp_target_pose):
        """
        Release the object by opening the gripper while maintaining velocity.

        Args:
            settling_steps (int): Steps used to stabilize the TCP velocity.
            open_gripper_steps (int): Steps after stabilizing before opening the gripper.
            tcp_target_pose (list): The target pose of the TCP.
        """
        if not hasattr(self, 'gripper_open_count'):
            self.gripper_open_count = 0

        self.gripper_open_count += 1

        # Apply velocity control
        joint_velocities = self.velocity_ik(self.tcp_target_velocity)
        self.set_arm_joint_velocity_target(joint_velocities)

        # Calculate target pose and velocity for log
        self._update_target(tcp_target_pose, settling_steps)

        # Open gripper when stabilization is reached
        if self.gripper_open_count >= settling_steps:
            self.open_gripper()

        # Wait for object to be released before proceeding
        if self.gripper_open_count >= settling_steps + open_gripper_steps:
            self._throw_step = 2
            del self.gripper_open_count

    def _update_target(self, tcp_target_pose, settling_steps):
        """Update target pose based on current velocity and remaining settling steps."""
        target_pos, target_quat = tcp_target_pose
        remaining_steps = settling_steps - self.gripper_open_count
        current_target_pos = np.array([
            target_pos[i] - self.tcp_target_velocity[0][i] * self.control_timestep * remaining_steps
            for i in range(3)
        ])
        self._tcp_target_pose = (current_target_pos, target_quat)
        self._joint_target_position = self.pose_ik(self._tcp_target_pose)
        self._tcp_target_velocity = self.tcp_target_velocity

    def _decelerate_arm(self):
        """Decelerate the arm by gradually reducing the joint velocities to zero."""
        joint_velocities = self.get_arm_joint_velocity()
        self.set_arm_joint_velocity_target(np.zeros_like(joint_velocities))

        self._tcp_target_velocity = (np.zeros(3), np.zeros(3))

        # Check if all joint velocities are close to zero
        if all(np.abs(vel) < 0.01 for vel in joint_velocities):
            return self._finalize_throw_process()

        return False

    def _finalize_throw_process(self):
        """Finalize the throw process by cleaning up variables and logging the end time."""
        del self._throw_step
        if not hasattr(self, 'throw_end_times'):
            self.throw_end_times = []
        self.throw_end_times.append(len(self.joint_position_log))
        return True  # Throwing process is complete
    
    def _generate_tcp_trajectory(self, start_tcp_pose, target_tcp_pose, start_tcp_vel=None, target_tcp_vel=None, estimate_speed=None):
        """
        Generates a TCP trajectory from start to target pose using quintic polynomial interpolation for both position
        and orientation (converted to Euler angles), considering velocity information.
        
        Args:
            start_tcp_pose (tuple): Starting TCP pose (position, orientation).
            target_tcp_pose (tuple): Target TCP pose (position, orientation).
            start_tcp_vel (tuple): Starting TCP velocity (linear velocity, angular velocity).
            target_tcp_vel (tuple): Target TCP velocity (linear velocity, angular velocity).
            estimate_speed (float): Speed used to estimate the duration of the trajectory.
        
        Returns:
            list: List of subtargets (setpoints) for the TCP to follow, each containing position, orientation, linear velocity, and angular velocity.
        """
        # Extract start and target positions and orientations
        start_pos = np.array(start_tcp_pose[0])
        target_pos = np.array(target_tcp_pose[0])
        start_rot = R.from_quat(start_tcp_pose[1])
        target_rot = R.from_quat(target_tcp_pose[1])
        start_quat = start_rot.as_quat()
        target_quat = target_rot.as_quat()
        
        # Extract start and target translational and rotational velocities
        start_trans_vel = np.array(start_tcp_vel[0]) if start_tcp_vel is not None else np.zeros([3])
        target_trans_vel = np.array(target_tcp_vel[0]) if target_tcp_vel is not None else np.zeros([3])  
        start_rot_vel = np.array(start_tcp_vel[1]) if start_tcp_vel is not None else np.zeros([3])
        target_rot_vel = np.array(target_tcp_vel[1]) if target_tcp_vel is not None else np.zeros([3])
        
        # Estimate the time needed based on the distance and given speed
        if estimate_speed is not None:
            distance = np.linalg.norm(target_pos - start_pos)
            time_estimate = distance / estimate_speed
        else:
            max_vel = np.array(target_trans_vel)
            time_estimates = []
            for dim in range(len(start_pos)):
                if max_vel[dim] > 0:
                    delta_pos = target_pos[dim] - start_pos[dim]
                    time_estimate_vel_dim = (15.0 / 8.0) * (abs(delta_pos) / max_vel[dim])
                    time_estimates.append(time_estimate_vel_dim)
                else:
                    time_estimates.append(np.array(0.0))
            # Take the maximum duration across all dimensions
            time_estimate = max(time_estimates)

        # Compute the number of steps based on control step size (defalult 1/60 seconds)
        num_steps = int(time_estimate / self.control_timestep)
        
        # Generate quintic polynomial trajectory for linear positions and velocities
        pos_trajectory, vel_trajectory = self._generate_quintic_trajectory(
            start_s=start_pos, 
            end_s=target_pos, 
            start_v=start_trans_vel, 
            end_v=target_trans_vel, 
            duration=time_estimate, 
            num_steps=num_steps)

        # Generate quaternion trajectory using slerp
        t_values = np.linspace(0, 1, num_steps)  # Interpolation points for Slerp
        ori_trajectory = slerp(start_quat, target_quat, t_values)

        # Compute rotational velocities using quaternion time derivative
        rot_vel_trajectory = []
        delta_t = time_estimate / num_steps
        for i in range(1, num_steps):
            q0 = R.from_quat(ori_trajectory[i-1])
            q1 = R.from_quat(ori_trajectory[i])
            
            # Compute the relative rotation between two quaternions
            q_diff = q1 * q0.inv()
            
            # Convert quaternion difference to angle-axis (axis * angle), then normalize to get angular velocity
            angle = np.arccos(np.clip(q_diff.as_quat()[-1], -1.0, 1.0)) * 2  # Extract angle (w component in quaternion)
            axis = q_diff.as_rotvec()  # Get axis of rotation
            
            # Compute angular velocity in rad/s
            angular_velocity = axis * (angle / delta_t)
            rot_vel_trajectory.append(angular_velocity)

        # Ensure initial rotational velocity is included
        rot_vel_trajectory.insert(0, start_rot_vel)

        # Generate the final list of subtargets (setpoints) combining positions, orientations, and velocities
        subtargets = []
        for i in range(num_steps):
            # Each subtarget is a list combining position, orientation (quaternion), linear velocity, and rotational velocity
            subtarget = [
                (pos_trajectory[i],    # Position
                ori_trajectory[i]),    # Orientation (quaternion)
                (vel_trajectory[i],    # Linear velocity
                rot_vel_trajectory[i]) # Angular velocity
            ]
            subtargets.append(subtarget)
        
        return subtargets

    def _generate_quintic_trajectory(self, start_s, end_s, start_v=None, end_v=None, start_acc=None, end_acc=None,
                                 duration=1.0, num_steps=100):
        """
        Generates a quintic polynomial trajectory for multiple dimensions (e.g., position components),
        considering maximum velocity and acceleration constraints.

        Args:
            start_s (float or np.ndarray): Initial positions for each dimension (e.g., [x, y, z]).
            end_s (float or np.ndarray): Target positions for each dimension.
            start_v (float or np.ndarray, optional): Initial velocities for each dimension. Defaults to 0 if None.
            end_v (float or np.ndarray, optional): Target velocities for each dimension. Defaults to 0 if None.
            start_acc (float or np.ndarray, optional): Initial accelerations for each dimension. Defaults to 0 if None.
            end_acc (float or np.ndarray, optional): Target accelerations for each dimension. Defaults to 0 if None.
            duration (float): Total duration of the trajectory.
            num_steps (int): Number of points to sample along the trajectory.

        Returns:
            tuple:
                np.ndarray: Array of positions sampled along the quintic polynomial trajectory.
                np.ndarray: Array of velocities sampled along the quintic polynomial trajectory.
        """
        # Convert inputs to numpy arrays
        start_s = np.array(start_s)
        end_s = np.array(end_s)

        # Set velocities and accelerations to zero arrays if they are None
        start_v = np.zeros_like(start_s) if start_v is None else np.array(start_v)
        end_v = np.zeros_like(end_s) if end_v is None else np.array(end_v)
        start_acc = np.zeros_like(start_s) if start_acc is None else np.array(start_acc)
        end_acc = np.zeros_like(end_s) if end_acc is None else np.array(end_acc)

        # Initialize the trajectory for each dimension
        trajectory_s = np.zeros((num_steps, len(start_s)))
        trajectory_v = np.zeros((num_steps, len(start_s)))

        # Time steps
        t_s = np.linspace(0, duration, num_steps)

        # Compute the quintic polynomial for each dimension independently
        for dim in range(len(start_s)):
            delta_s = end_s[dim] - start_s[dim]

            # Calculate quintic polynomial coefficients
            A = (12 * delta_s - 6 * (start_v[dim] + end_v[dim]) * duration -
                (end_acc[dim] - start_acc[dim]) * duration**2) / (2 * duration**5)
            B = (-30 * delta_s + (16 * start_v[dim] + 14 * end_v[dim]) * duration +
                (3 * start_acc[dim] - 2 * end_acc[dim]) * duration**2) / (2 * duration**4)
            C = (20 * delta_s - (12 * start_v[dim] + 8 * end_v[dim]) * duration -
                (3 * start_acc[dim] - end_acc[dim]) * duration**2) / (2 * duration**3)
            D = start_acc[dim] / 2.0
            E = start_v[dim]
            F = start_s[dim]

            # Evaluate the quintic polynomial at each time step for this dimension
            trajectory_s[:, dim] = A * t_s**5 + B * t_s**4 + C * t_s**3 + D * t_s**2 + E * t_s + F
            # Velocity is the derivative of the position polynomial
            trajectory_v[:, dim] = (5 * A * t_s**4 + 4 * B * t_s**3 +
                                    3 * C * t_s**2 + 2 * D * t_s + E)

        return trajectory_s, trajectory_v

    ############### visualization ###############
    def visualize_coordinate_frames(self, axis_length=0.1, links_to_visualize=None):
        """
        Draw the coordinate frames for specified links in the URDF.

        Args:
            axis_length (float): The length of the coordinate axes.
            links_to_visualize (list or None): A list of link names to visualize. 
                                            If None, visualizes all links.
        """
        # If no specific links are provided, visualize all links
        if links_to_visualize is None:
            links_to_visualize = list(self.links.keys())
        
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

    def visualize_tcp_trajectory(self, color_target=[0, 1, 0], color_actual=[1, 0, 0], line_width=3.0, line_duration=5):
        """
        Visualize the TCP target and actual trajectories using debug lines.
        
        Args:
            color_target (list): RGB color for target trajectory lines.
            color_actual (list): RGB color for actual trajectory lines.
            line_width (float): The width of the lines.
            line_duration (float): The duration for which the lines will be visible.
        """

        target_pose = self.get_tcp_target_pose()
        current_pose = self.get_tcp_pose()

        if target_pose is None:
            if not hasattr(self, '_prev_tcp_pose'):
                self._prev_tcp_pose = current_pose
            return
        else:
            if not hasattr(self, '_prev_tcp_target_pose') or not hasattr(self, '_prev_tcp_pose'):
                self._prev_tcp_target_pose = target_pose
                self._prev_tcp_pose = current_pose
                return

        if target_pose is not None:
            # Visualize the target TCP trajectory
            p.addUserDebugLine(self._prev_tcp_target_pose[0], target_pose[0], color_target, lineWidth=line_width, lifeTime=line_duration)
            self._prev_tcp_target_pose = target_pose

        # Visualize the actual TCP trajectory
        p.addUserDebugLine(self._prev_tcp_pose[0], current_pose[0], color_actual, lineWidth=line_width, lifeTime=line_duration)
        self._prev_tcp_pose = current_pose
    
    ############### log ###############
    def log_variables(self):
        # Append current values to logs
        self.joint_position_log.append(self.get_arm_joint_position())
        self.joint_velocity_log.append(self.get_arm_joint_velocity())
        self.target_joint_position_log.append(self._joint_target_position)
        self.target_joint_velocity_log.append(self._joint_target_velocity)
        self.joint_torque_log.append(self.get_arm_joint_torque())
        self.tcp_pose_log.append(self.get_tcp_pose())
        self.tcp_velocity_log.append(self.get_tcp_velocity())
        self.target_tcp_pose_log.append(self._tcp_target_pose)
        self.target_tcp_velocity_log.append(self._tcp_target_velocity)
        self.gripper_position_log.append(self.get_gripper_position())
        self.target_gripper_position_log.append(self._gripper_target_position)

    ############### plot ###############
    def plot_log_variables(self, variables=None, savefig=False, log_dir=None):
        """
        Plot selected variables from the log. If no specific variables are provided, all logs will be plotted.

        Args:
            variables (list): List of variables to plot. Possible values are 'arm_joint_position', 'arm_joint_velocity', 'tcp_position', 'tcp_velocity', 'gripper_position'.
            savefig (bool): If or not to save the figs.
            log_dir (str): The path to save the figs.
        """
        if variables is None:
            variables = [
                'arm_joint_position',
                'arm_joint_velocity',
                'arm_joint_torque',
                'tcp_position',
                'tcp_velocity',
                'gripper_position'
            ]  # Plot all if no specific selection

        if 'arm_joint_position' in variables:
            self.plot_arm_joint_position(savefig=savefig, log_dir=log_dir)
        if 'arm_joint_velocity' in variables:
            self.plot_arm_joint_velocity(savefig=savefig, log_dir=log_dir)
        if 'arm_joint_torque' in variables:
            self.plot_arm_joint_torque(savefig=savefig, log_dir=log_dir)
        if 'tcp_position' in variables:
            self.plot_tcp_position(savefig=savefig, log_dir=log_dir)
        if 'tcp_velocity' in variables:
            self.plot_tcp_velocity(savefig=savefig, log_dir=log_dir)
        if 'gripper_position' in variables:
            self.plot_gripper_position(savefig=savefig, log_dir=log_dir)

        # After all plots are created, show them together
        plt.show()

    def plot_arm_joint_position(self, savefig=False, log_dir=None):
        """
        Plot the arm's joint-related position variables over time, including both actual and target values.
        Joint limits (lower and upper) are also plotted as yellow dotted lines.
        """
        num_joints = self.num_arm_dofs  # Number of joints to plot
        timesteps = np.arange(len(self.joint_position_log)) * self.timestep  # Convert to actual time

        fig, axes = plt.subplots(nrows=num_joints, ncols=1, figsize=(10, 1.5*num_joints))

        # Add a general title for the figure
        fig.suptitle('Arm Joint Positions Over Time', fontsize=16)

        # If there's only one joint, ensure axes is iterable
        if num_joints == 1:
            axes = [axes]

        for i in range(num_joints):
            # Plot actual joint position
            actual_joint_positions = [log[i] for log in self.joint_position_log]
            axes[i].plot(timesteps, actual_joint_positions, label=f'Joint {i} Position', linestyle='-', color='red')

            # Plot target joint position
            target_joint_positions = [log[i] for log in self.target_joint_position_log]
            axes[i].plot(timesteps, target_joint_positions, label=f'Joint {i} Target Position', linestyle='--', color='green')

            # Add joint limits as dotted lines
            lower_limit = self.arm_lower_limits[i]
            upper_limit = self.arm_upper_limits[i]
            axes[i].axhline(lower_limit, color='yellow', linestyle=':', label=f'Joint {i} Lower Limit')
            axes[i].axhline(upper_limit, color='yellow', linestyle=':', label=f'Joint {i} Upper Limit')

            axes[i].set_title(f'Arm Joint {i} Position')
            axes[i].set_xlabel('Time (s)')
            axes[i].set_ylabel('Position')
            axes[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))

            # If there are recorded grasp times, plot vertical lines for each grasp start and end time
            if hasattr(self, 'grasp_start_times') and hasattr(self, 'grasp_end_times'):
                for start_time in self.grasp_start_times:
                    grasp_start_time = start_time * self.timestep
                    axes[i].axvline(x=grasp_start_time, color='blue', linestyle='--', label='Grasp Start')
                for end_time in self.grasp_end_times:
                    grasp_end_time = end_time * self.timestep
                    axes[i].axvline(x=grasp_end_time, color='blue', linestyle='--', label='Grasp End')

            # If there are recorded throw times, plot vertical lines for each grasp start and end time
            if hasattr(self, 'throw_start_times') and hasattr(self, 'throw_end_times'):
                for start_time in self.throw_start_times:
                    throw_start_time = start_time * self.timestep
                    axes[i].axvline(x=throw_start_time, color='purple', linestyle='--', label='Throw Start')
                for end_time in self.throw_end_times:
                    throw_end_time = end_time * self.timestep
                    axes[i].axvline(x=throw_end_time, color='purple', linestyle='--', label='Throw End')

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit suptitle

        # Save the figure if requested
        if savefig and log_dir:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)  # Create the directory if it doesn't exist
            fig_path = os.path.join(log_dir, 'arm_joint_position.png')
            plt.savefig(fig_path)

        plt.pause(0.001)

    def plot_arm_joint_velocity(self, savefig=False, log_dir=None):
        """
        Plot the arm's joint-related velocity variables over time, including both actual and target values.
        Joint velocity limits are also plotted as yellow dotted lines.
        """
        num_joints = self.num_arm_dofs  # Number of joints to plot
        timesteps = np.arange(len(self.joint_velocity_log)) * self.timestep  # Convert to actual time

        fig, axes = plt.subplots(nrows=num_joints, ncols=1, figsize=(10, 1.5*num_joints))

        # Add a general title for the figure
        fig.suptitle('Arm Joint Velocities Over Time', fontsize=16)

        # If there's only one joint, ensure axes is iterable
        if num_joints == 1:
            axes = [axes]

        for i in range(num_joints):
            # Plot actual joint velocity
            actual_joint_velocities = [log[i] for log in self.joint_velocity_log]
            axes[i].plot(timesteps, actual_joint_velocities, label=f'Joint {i} Velocity', linestyle='-', color='red')

            # Plot target joint velocity
            target_joint_velocities = [log[i] for log in self.target_joint_velocity_log]
            axes[i].plot(timesteps, target_joint_velocities, label=f'Joint {i} Target Velocity', linestyle='--', color='green')

            # Add joint velocity limits as dotted lines
            max_velocity = self.joints[i].max_velocity
            axes[i].axhline(-max_velocity, color='yellow', linestyle=':', label=f'Joint {i} Min Velocity')
            axes[i].axhline(max_velocity, color='yellow', linestyle=':', label=f'Joint {i} Max Velocity')

            axes[i].set_title(f'Arm Joint {i} Velocity')
            axes[i].set_xlabel('Time (s)')
            axes[i].set_ylabel('Velocity')
            axes[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))

            # If there are recorded grasp times, plot vertical lines for each grasp start and end time
            if hasattr(self, 'grasp_start_times') and hasattr(self, 'grasp_end_times'):
                for start_time in self.grasp_start_times:
                    grasp_start_time = start_time * self.timestep
                    axes[i].axvline(x=grasp_start_time, color='blue', linestyle='--', label='Grasp Start')
                for end_time in self.grasp_end_times:
                    grasp_end_time = end_time * self.timestep
                    axes[i].axvline(x=grasp_end_time, color='blue', linestyle='--', label='Grasp End')

            # If there are recorded throw times, plot vertical lines for each grasp start and end time
            if hasattr(self, 'throw_start_times') and hasattr(self, 'throw_end_times'):
                for start_time in self.throw_start_times:
                    throw_start_time = start_time * self.timestep
                    axes[i].axvline(x=throw_start_time, color='purple', linestyle='--', label='Throw Start')
                for end_time in self.throw_end_times:
                    throw_end_time = end_time * self.timestep
                    axes[i].axvline(x=throw_end_time, color='purple', linestyle='--', label='Throw End')

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit suptitle

        # Save the figure if requested
        if savefig and log_dir:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)  # Create the directory if it doesn't exist
            fig_path = os.path.join(log_dir, 'arm_joint_velocity.png')
            plt.savefig(fig_path)

        plt.pause(0.001)

    def plot_arm_joint_torque(self, savefig=False, log_dir=None):
        """
        Plot the arm's joint-related motor torque variables over time, including torque limits.
        """
        num_joints = self.num_arm_dofs  # Number of joints to plot
        timesteps = np.arange(len(self.joint_torque_log)) * self.timestep  # Convert to actual time

        fig, axes = plt.subplots(nrows=num_joints, ncols=1, figsize=(10, 1.5*num_joints))

        # Add a general title for the figure
        fig.suptitle('Arm Joint Torques Over Time', fontsize=16)

        # If there's only one joint, ensure axes is iterable
        if num_joints == 1:
            axes = [axes]

        for i in range(num_joints):
            # Plot joint torque
            actual_joint_torques = [log[i] for log in self.joint_torque_log]
            axes[i].plot(timesteps, actual_joint_torques, label=f'Joint {i} Torque', linestyle='-', color='red')

            # Plot max force (torque) limit using self.joints to access max_force
            max_force = self.joints[i].max_force
            axes[i].axhline(-max_force, color='yellow', linestyle=':', label=f'Joint {i} Min Torque')
            axes[i].axhline(max_force, color='yellow', linestyle=':', label=f'Joint {i} Max Torque')
            axes[i].set_title(f'Arm Joint {i} Torque')
            axes[i].set_xlabel('Time (s)')
            axes[i].set_ylabel('Torque (Nm)')
            axes[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))

            # If there are recorded grasp times, plot vertical lines for each grasp start and end time
            if hasattr(self, 'grasp_start_times') and hasattr(self, 'grasp_end_times'):
                for start_time in self.grasp_start_times:
                    grasp_start_time = start_time * self.timestep
                    axes[i].axvline(x=grasp_start_time, color='blue', linestyle='--', label='Grasp Start')
                for end_time in self.grasp_end_times:
                    grasp_end_time = end_time * self.timestep
                    axes[i].axvline(x=grasp_end_time, color='blue', linestyle='--', label='Grasp End')

            # If there are recorded throw times, plot vertical lines for each grasp start and end time
            if hasattr(self, 'throw_start_times') and hasattr(self, 'throw_end_times'):
                for start_time in self.throw_start_times:
                    throw_start_time = start_time * self.timestep
                    axes[i].axvline(x=throw_start_time, color='purple', linestyle='--', label='Throw Start')
                for end_time in self.throw_end_times:
                    throw_end_time = end_time * self.timestep
                    axes[i].axvline(x=throw_end_time, color='purple', linestyle='--', label='Throw End')

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit suptitle

        # Save the figure if requested
        if savefig and log_dir:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)  # Create the directory if it doesn't exist
            fig_path = os.path.join(log_dir, 'arm_joint_torque.png')
            plt.savefig(fig_path)

        plt.pause(0.001)

    def plot_tcp_position(self, savefig=False, log_dir=None):
        """
        Plot the TCP-related position variables over time, including both actual and target values.
        """
        timesteps = np.arange(len(self.tcp_pose_log)) * self.timestep  # Convert to actual time

        # Extract position and orientation (Euler angles) from tcp_pose_log and target_tcp_pose_log
        actual_positions = np.array([log[0] for log in self.tcp_pose_log])
        actual_orientations = np.array([quaternion_to_euler(*log[1]) for log in self.tcp_pose_log])

        target_positions = np.array([log[0] for log in self.target_tcp_pose_log])
        target_orientations = np.array([quaternion_to_euler(*log[1]) for log in self.target_tcp_pose_log])

        # Create subplots for position (x, y, z) and orientation (roll, pitch, yaw)
        fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(10, 9))  # 3 for position and 3 for orientation

        # Add a general title for the figure
        fig.suptitle('TCP Positions and Orientations Over Time', fontsize=16)

        # Plot positions: x, y, z
        position_labels = ['X', 'Y', 'Z']
        for i in range(3):
            axes[i].plot(timesteps, actual_positions[:, i], label=f'Actual {position_labels[i]} Position', linestyle='-', color='red')
            axes[i].plot(timesteps, target_positions[:, i], label=f'Target {position_labels[i]} Position', linestyle='--', color='green')
            axes[i].set_title(f'TCP {position_labels[i]} Position')
            axes[i].set_xlabel('Time (s)')
            axes[i].set_ylabel(f'{position_labels[i]} Position')
            axes[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))

            # If there are recorded grasp times, plot vertical lines for each grasp start and end time
            if hasattr(self, 'grasp_start_times') and hasattr(self, 'grasp_end_times'):
                for start_time in self.grasp_start_times:
                    grasp_start_time = start_time * self.timestep
                    axes[i].axvline(x=grasp_start_time, color='blue', linestyle='--', label='Grasp Start')
                for end_time in self.grasp_end_times:
                    grasp_end_time = end_time * self.timestep
                    axes[i].axvline(x=grasp_end_time, color='blue', linestyle='--', label='Grasp End')

            # If there are recorded throw times, plot vertical lines for each grasp start and end time
            if hasattr(self, 'throw_start_times') and hasattr(self, 'throw_end_times'):
                for start_time in self.throw_start_times:
                    throw_start_time = start_time * self.timestep
                    axes[i].axvline(x=throw_start_time, color='purple', linestyle='--', label='Throw Start')
                for end_time in self.throw_end_times:
                    throw_end_time = end_time * self.timestep
                    axes[i].axvline(x=throw_end_time, color='purple', linestyle='--', label='Throw End')

        # Plot orientations: roll, pitch, yaw
        orientation_labels = ['Roll', 'Pitch', 'Yaw']
        for i in range(3):
            axes[i + 3].plot(timesteps, actual_orientations[:, i], label=f'Actual {orientation_labels[i]}', linestyle='-', color='red')
            axes[i + 3].plot(timesteps, target_orientations[:, i], label=f'Target {orientation_labels[i]}', linestyle='--', color='green')
            axes[i + 3].set_title(f'TCP {orientation_labels[i]}')
            axes[i + 3].set_xlabel('Time (s)')
            axes[i + 3].set_ylabel(f'{orientation_labels[i]} (radians)')
            axes[i + 3].legend(loc='center left', bbox_to_anchor=(1, 0.5))

            # If there are recorded grasp times, plot vertical lines for each grasp start and end time
            if hasattr(self, 'grasp_start_times') and hasattr(self, 'grasp_end_times'):
                for start_time in self.grasp_start_times:
                    grasp_start_time = start_time * self.timestep
                    axes[i + 3].axvline(x=grasp_start_time, color='blue', linestyle='--', label='Grasp Start')
                for end_time in self.grasp_end_times:
                    grasp_end_time = end_time * self.timestep
                    axes[i + 3].axvline(x=grasp_end_time, color='blue', linestyle='--', label='Grasp End')

            # If there are recorded throw times, plot vertical lines for each grasp start and end time
            if hasattr(self, 'throw_start_times') and hasattr(self, 'throw_end_times'):
                for start_time in self.throw_start_times:
                    throw_start_time = start_time * self.timestep
                    axes[i + 3].axvline(x=throw_start_time, color='purple', linestyle='--', label='Throw Start')
                for end_time in self.throw_end_times:
                    throw_end_time = end_time * self.timestep
                    axes[i + 3].axvline(x=throw_end_time, color='purple', linestyle='--', label='Throw End')

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit suptitle

        # Save the figure if requested
        if savefig and log_dir:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)  # Create the directory if it doesn't exist
            fig_path = os.path.join(log_dir, 'tcp_position.png')
            plt.savefig(fig_path)

        plt.pause(0.001)

    def plot_tcp_velocity(self, savefig=False, log_dir=None):
        """
        Plot the TCP-related velocity variables over time, including both actual and target values.
        """
        timesteps = np.arange(len(self.tcp_velocity_log)) * self.timestep  # Convert to actual time

        # Extract linear and angular velocities from tcp_velocity_log and target_tcp_velocity_log
        actual_linear_velocities = np.array([log[0] for log in self.tcp_velocity_log])
        actual_angular_velocities = np.array([log[1] for log in self.tcp_velocity_log])

        target_linear_velocities = np.array([log[0] for log in self.target_tcp_velocity_log])
        target_angular_velocities = np.array([log[1] for log in self.target_tcp_velocity_log])

        # Create subplots for linear (vx, vy, vz) and angular (wx, wy, wz) velocities
        fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(10, 9))  # 3 for linear velocity and 3 for angular velocity

        # Add a general title for the figure
        fig.suptitle('TCP Velocities Over Time', fontsize=16)

        # Plot linear velocities: vx, vy, vz
        linear_velocity_labels = ['Vx', 'Vy', 'Vz']
        for i in range(3):
            axes[i].plot(timesteps, actual_linear_velocities[:, i], label=f'Actual {linear_velocity_labels[i]}', linestyle='-', color='red')
            axes[i].plot(timesteps, target_linear_velocities[:, i], label=f'Target {linear_velocity_labels[i]}', linestyle='--', color='green')
            axes[i].set_title(f'TCP {linear_velocity_labels[i]}')
            axes[i].set_xlabel('Time (s)')
            axes[i].set_ylabel(f'{linear_velocity_labels[i]} (m/s)')
            axes[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))

            # If there are recorded grasp times, plot vertical lines for each grasp start and end time
            if hasattr(self, 'grasp_start_times') and hasattr(self, 'grasp_end_times'):
                for start_time in self.grasp_start_times:
                    grasp_start_time = start_time * self.timestep
                    axes[i].axvline(x=grasp_start_time, color='blue', linestyle='--', label='Grasp Start')
                for end_time in self.grasp_end_times:
                    grasp_end_time = end_time * self.timestep
                    axes[i].axvline(x=grasp_end_time, color='blue', linestyle='--', label='Grasp End')

            # If there are recorded throw times, plot vertical lines for each grasp start and end time
            if hasattr(self, 'throw_start_times') and hasattr(self, 'throw_end_times'):
                for start_time in self.throw_start_times:
                    throw_start_time = start_time * self.timestep
                    axes[i].axvline(x=throw_start_time, color='purple', linestyle='--', label='Throw Start')
                for end_time in self.throw_end_times:
                    throw_end_time = end_time * self.timestep
                    axes[i].axvline(x=throw_end_time, color='purple', linestyle='--', label='Throw End')

        # Plot angular velocities: wx, wy, wz
        angular_velocity_labels = ['Wx', 'Wy', 'Wz']
        for i in range(3):
            axes[i + 3].plot(timesteps, actual_angular_velocities[:, i], label=f'Actual {angular_velocity_labels[i]}', linestyle='-', color='red')
            axes[i + 3].plot(timesteps, target_angular_velocities[:, i], label=f'Target {angular_velocity_labels[i]}', linestyle='--', color='green')
            axes[i + 3].set_title(f'TCP {angular_velocity_labels[i]}')
            axes[i + 3].set_xlabel('Time (s)')
            axes[i + 3].set_ylabel(f'{angular_velocity_labels[i]} (rad/s)')
            axes[i + 3].legend(loc='center left', bbox_to_anchor=(1, 0.5))
            
            # If there are recorded grasp times, plot vertical lines for each grasp start and end time
            if hasattr(self, 'grasp_start_times') and hasattr(self, 'grasp_end_times'):
                for start_time in self.grasp_start_times:
                    grasp_start_time = start_time * self.timestep
                    axes[i + 3].axvline(x=grasp_start_time, color='blue', linestyle='--', label='Grasp Start')
                for end_time in self.grasp_end_times:
                    grasp_end_time = end_time * self.timestep
                    axes[i + 3].axvline(x=grasp_end_time, color='blue', linestyle='--', label='Grasp End')

            # If there are recorded throw times, plot vertical lines for each grasp start and end time
            if hasattr(self, 'throw_start_times') and hasattr(self, 'throw_end_times'):
                for start_time in self.throw_start_times:
                    throw_start_time = start_time * self.timestep
                    axes[i + 3].axvline(x=throw_start_time, color='purple', linestyle='--', label='Throw Start')
                for end_time in self.throw_end_times:
                    throw_end_time = end_time * self.timestep
                    axes[i + 3].axvline(x=throw_end_time, color='purple', linestyle='--', label='Throw End')

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit suptitle

        # Save the figure if requested
        if savefig and log_dir:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)  # Create the directory if it doesn't exist
            fig_path = os.path.join(log_dir, 'tcp_velocity.png')
            plt.savefig(fig_path)

        plt.pause(0.001)

    def plot_gripper_position(self, savefig=False, log_dir=None):
        """
        Plot the gripper-related position variables over time, including both actual and target values.
        This function supports any number of gripper DOFs (radians of freedom).
        """
        num_dofs = self.num_gripper_dofs  # Number of gripper DOFs
        timesteps = np.arange(len(self.gripper_position_log)) * self.timestep  # Convert to actual time

        # Extract gripper positions from logs
        actual_gripper_positions = np.array(self.gripper_position_log)  # Shape: (T, num_dofs)
        target_gripper_positions = np.array(self.target_gripper_position_log)  # Shape: (T, num_dofs)

        # Create subplots for each gripper DOF
        fig, axes = plt.subplots(nrows=num_dofs, ncols=1, figsize=(10, 1.5*num_dofs))

        # Add a general title for the figure
        fig.suptitle('Gripper DOF Positions Over Time', fontsize=16)

        # If there's only one gripper DOF, ensure axes is iterable
        if num_dofs == 1:
            axes = [axes]

        for i in range(num_dofs):
            # Plot actual gripper position for the i-th DOF
            axes[i].plot(timesteps, actual_gripper_positions[:, i], label=f'Gripper DOF {i} Position', linestyle='-', color='red')

            # Plot target gripper position for the i-th DOF
            axes[i].plot(timesteps, target_gripper_positions[:, i], label=f'Gripper DOF {i} Target Position', linestyle='--', color='green')

            axes[i].set_title(f'Gripper DOF {i} Position')
            axes[i].set_xlabel('Time (s)')
            axes[i].set_ylabel('Position')
            axes[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))

            # If there are recorded grasp times, plot vertical lines for each grasp start and end time
            if hasattr(self, 'grasp_start_times') and hasattr(self, 'grasp_end_times'):
                for start_time in self.grasp_start_times:
                    grasp_start_time = start_time * self.timestep
                    axes[i].axvline(x=grasp_start_time, color='blue', linestyle='--', label='Grasp Start')
                for end_time in self.grasp_end_times:
                    grasp_end_time = end_time * self.timestep
                    axes[i].axvline(x=grasp_end_time, color='blue', linestyle='--', label='Grasp End')

            # If there are recorded throw times, plot vertical lines for each grasp start and end time
            if hasattr(self, 'throw_start_times') and hasattr(self, 'throw_end_times'):
                for start_time in self.throw_start_times:
                    throw_start_time = start_time * self.timestep
                    axes[i].axvline(x=throw_start_time, color='purple', linestyle='--', label='Throw Start')
                for end_time in self.throw_end_times:
                    throw_end_time = end_time * self.timestep
                    axes[i].axvline(x=throw_end_time, color='purple', linestyle='--', label='Throw End')

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit suptitle

        # Save the figure if requested
        if savefig and log_dir:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)  # Create the directory if it doesn't exist
            fig_path = os.path.join(log_dir, 'gripper_position.png')
            plt.savefig(fig_path)

        plt.pause(0.001)
