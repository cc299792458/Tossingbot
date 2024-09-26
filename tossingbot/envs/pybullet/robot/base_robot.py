import numpy as np
import pybullet as p

from collections import namedtuple
from scipy.spatial.transform import Rotation as R
from tossingbot.envs.pybullet.utils.math_utils import slerp, pose_distance

class BaseRobot:
    """
    The base class for robots
    """

    def __init__(self, base_position, base_orientation, robot_type='panda'):
        """
        Initialize the robot with its base position, orientation, and type.
        """
        self.base_position = base_position
        self.base_orientation_quat = p.getQuaternionFromEuler(base_orientation)
        self.links = {}  # Store link information
        self.robot_type = robot_type
        assert robot_type in ['ur5_robotiq85', 'panda'], "robot_type must be 'ur5_robotiq85' or 'panda'"
        self.load_robot()
        self.reset()

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

        assert len(self.controllable_joints) >= self.num_arm_dofs, "Not enough controllable joints"
        self.arm_controllable_joints = self.controllable_joints[:self.num_arm_dofs]
        self.arm_lower_limits = [info.lower_limit for info in self.joints if info.controllable][:self.num_arm_dofs]
        self.arm_upper_limits = [info.upper_limit for info in self.joints if info.controllable][:self.num_arm_dofs]
        self.arm_joint_ranges = [info.upper_limit - info.lower_limit for info in self.joints if info.controllable][:self.num_arm_dofs]

    def _store_link_information(self, change_dynamics=False):
        """
        Store link information in a dictionary and set the TCP link based on the robot type.
        """
        num_joints = p.getNumJoints(self.robot_id)
        self.links[p.getBodyInfo(self.robot_id)[0].decode('utf-8')] = -1
        for joint_id in range(num_joints):
            link_name = p.getJointInfo(self.robot_id, joint_id)[12].decode('utf-8')
            self.links[link_name] = joint_id
            # # Change friction parameter
            if change_dynamics:
                p.changeDynamics(self.robot_id, joint_id, lateralFriction=1.0, rollingFriction=0.01, linearDamping=0, angularDamping=0)
        
        # Set TCP link based on robot type
        if self.robot_type == 'ur5_robotiq85':
            self.tcp_id = self.links.get('tcp_link', -1)
        elif self.robot_type == 'panda':
            self.tcp_id = self.links.get('tcp_link', -1)

        assert self.tcp_id != -1, "TCP link not found for the robot"

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
        self.set_arm_joint_position(self.initial_position[0:self.num_arm_dofs])
        self.set_arm_joint_position_target(self.initial_position[0:self.num_arm_dofs])

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
            maxVelocity=self.joints[joint_id].max_velocity
        )

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

    def get_arm_joint_position(self):
        """
        Get the current position of the arm joints.
        
        Returns:
            list: Current positions of arm controllable joints.
        """
        return [p.getJointState(self.robot_id, joint_id)[0] for joint_id in self.arm_controllable_joints]

    def get_arm_joint_velocity(self):
        """
        Get the current velocity of the arm joints.
        
        Returns:
            list: Current velocities of arm controllable joints.
        """
        return [p.getJointState(self.robot_id, joint_id)[1] for joint_id in self.arm_controllable_joints]

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
            target_tcp_pose (list): A list containing the target TCP pose with [position, orientation].
                                    - position: [x, y, z]
                                    - orientation: [qx, qy, qz, qw]
        """
        target_joint_position = self.pose_ik(pose=target_tcp_pose)
        target_joint_velocity = self.velocity_ik(
            linear_velocity=target_tcp_velocity[0],
            angular_velocity=target_tcp_velocity[0]
        ) if target_tcp_velocity is not None else np.zeros([len(target_joint_position)])
        self.set_arm_joint_position_target(target_position=target_joint_position, target_velocity=target_joint_velocity)

    def set_tcp_velocity_target(self, target_tcp_velocity):
        """
        Set the target velocity for the TCP.
        
        Args:
            target_tcp_velocity (list): Desired velocity in Cartesian space [linear, angular].
        """
        target_joint_velocity = self.velocity_ik(
            linear_velocity=target_tcp_velocity[0],
            angular_velocity=target_tcp_velocity[1]
        )
        self.set_arm_joint_velocity_target(target_joint_velocity)

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
    
    def get_tcp_target_pose(self):
        if hasattr(self, '_tcp_target_pose'):
            return self._tcp_target_pose
        else:
            return None

    def get_tcp_velocity(self):
        """
        Get the current velocity of the TCP.
        
        Returns:
            list: A list containing the TCP velocity with [linear_velocity, angular_velocity].
                - linear_velocity: [vx, vy, vz]
                - angular_velocity: [wx, wy, wz]
        """
        link_state = p.getLinkState(self.robot_id, self.tcp_id, computeLinkVelocity=True)
        linear_velocity = link_state[6]
        angular_velocity = link_state[7]
        return [linear_velocity, angular_velocity]

    ############### Pose and Velocity Inverse Kinematics ###############
    def pose_ik(self, pose, rest_pose=None):
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
        return joint_position[:self.num_arm_dofs]

    def velocity_ik(self, linear_velocity, angular_velocity):
        """
        Compute joint velocities required to achieve the desired TCP (end-effector) velocity.

        Args:
            linear_velocity (list or np.ndarray): Desired linear velocity [vx, vy, vz].
            angular_velocity (list or np.ndarray): Desired angular velocity [wx, wy, wz].

        Returns:
            np.ndarray: Joint velocities to achieve the desired TCP velocity.
        """
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

        return joint_velocities[:self.num_arm_dofs]
    
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
    
    def set_gripper_position_target(self):
        raise NotImplementedError
    
    def get_gripper_position(self):
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

    def grasp(self, tcp_target_pose, post_grasp_pose=([0.3, 0.0, 0.3], (1.0, 0.0, 0.0, 0.0))):
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
        
        Returns:
            bool: True if the grasping process is completed, False otherwise.
        """
        # Initialize or continue the grasping process
        if not hasattr(self, '_grasp_step'):
            self._grasp_step = 0  # Initialize the grasp step
            self.pose_over_target = [tcp_target_pose[0][:2] + [0.3], tcp_target_pose[1]]  # Position above the target
            self.tcp_target_pose = tcp_target_pose

            # Generate trajectory for moving to the position above the target
            self._tcp_trajectory = self._generate_tcp_trajectory(self.get_tcp_pose(), self.pose_over_target, estimate_speed=0.5)
            self._trajectory_index = 0  # Initialize trajectory index

        # Step 1: Move to a position above the target
        if self._grasp_step == 0:
            # Move through the trajectory to the position above the target
            if self._trajectory_index < len(self._tcp_trajectory):
                self._tcp_target_pose, self._tcp_target_velocity = self._tcp_trajectory[self._trajectory_index]
                self.set_tcp_pose_target(self._tcp_target_pose, self._tcp_target_velocity)  # Set the current setpoint for the TCP
                self._trajectory_index += 1
            else:
                # Once reached, move to the next step (Step 2)
                self._grasp_step = 1
                self._trajectory_index = 0  # Reset for the next trajectory
                # Generate the next trajectory to move down to the target pose
                self._tcp_trajectory = self._generate_tcp_trajectory(self.pose_over_target, self.tcp_target_pose, estimate_speed=0.5)
            return False  # Grasping process not yet complete

        # Step 2: Open the gripper
        elif self._grasp_step == 1:
            self.open_gripper()  # Open the gripper
            if self._is_gripper_stopped():  # Check if the gripper has finished moving
                self._grasp_step = 2  # Move to the next step
            return False  # Grasping process not yet complete

        # Step 3: Move down to the target position
        elif self._grasp_step == 2:
            # Move through the trajectory to the target position
            if self._trajectory_index < len(self._tcp_trajectory):
                self._tcp_target_pose, self._tcp_target_velocity = self._tcp_trajectory[self._trajectory_index]
                self.set_tcp_pose_target(self._tcp_target_pose, self._tcp_target_velocity)  # Set the current setpoint for the TCP
                self._trajectory_index += 1
            else:
                # Once reached the target, move to the next step (Step 4)
                self._grasp_step = 3
                self._trajectory_index = 0  # Reset for the next trajectory
            return False  # Grasping process not yet complete

        # Step 4: Close the gripper to grasp the object
        elif self._grasp_step == 3:
            self.close_gripper()  # Close the gripper
            if self._is_gripper_stopped():  # Check if the gripper has finished moving
                self._grasp_step = 4  # Move to the next step (post-grasp movement)
                # Generate the trajectory to move to the post-grasp position (lifting up safely)
                self._tcp_trajectory = self._generate_tcp_trajectory(self.tcp_target_pose, post_grasp_pose, estimate_speed=0.5)
            return False  # Grasping process not yet complete

        # Step 5: Move to the post-grasp position
        elif self._grasp_step == 4:
            # Move through the trajectory to the post-grasp position
            if self._trajectory_index < len(self._tcp_trajectory):
                self._tcp_target_pose, self._tcp_target_velocity = self._tcp_trajectory[self._trajectory_index]
                self.set_tcp_pose_target(self._tcp_target_pose, self._tcp_target_velocity)  # Set the current setpoint for the TCP
                self._trajectory_index += 1
            else:
                # Once reached, the grasping process is complete
                del self._grasp_step  # Cleanup the process state
                del self._tcp_trajectory
                del self._trajectory_index
                return True  # Grasping process is complete

        return False  # Grasping process not yet complete

    def throw(self, tcp_target_pose, tcp_target_velocity, correction_gain=0.1):
        """
        Perform a throwing action with three stages: 
        1. Move to the release point with a target velocity while correcting deviations.
        2. Release the object by opening the gripper while maintaining the current velocity.
        3. Decelerate the arm after the release to zero velocity.

        Args:
            tcp_target_pose (list): Target TCP pose at the release point as [position, orientation].
            tcp_target_velocity (list): Target velocity for the TCP at the release point (linear and angular).
            correction_gain (float): Gain for correcting the deviation from the target pose.
            
        Returns:
            bool: True if the throwing process is completed, False otherwise.
        """
        # Initialize or continue the throwing process
        if not hasattr(self, '_throw_step'):
            # Stage 0: Set initial conditions for the throwing process
            self._throw_step = 0
            self.tcp_target_pose = tcp_target_pose
            self.tcp_target_velocity = tcp_target_velocity

        # Stage 1: Move to the release point with velocity, correcting deviations
        if self._throw_step == 0:
            current_pose = self.get_tcp_pose()  # Get the current TCP pose
            position_error = np.array(self.tcp_target_pose[0]) - np.array(current_pose[0])  # Position error

            # Compute orientation error using quaternion difference
            current_orientation = R.from_quat(current_pose[1])
            target_orientation = R.from_quat(self.tcp_target_pose[1])
            orientation_error_axis_angle = (target_orientation * current_orientation.inv()).as_rotvec()

            # Apply corrections to linear and angular velocities
            corrected_linear_velocity = np.array(self.tcp_target_velocity[0]) + correction_gain * position_error
            corrected_angular_velocity = np.array(self.tcp_target_velocity[1]) + correction_gain * orientation_error_axis_angle

            # Use velocity IK to set joint velocities
            joint_velocities = self.velocity_ik(corrected_linear_velocity, corrected_angular_velocity)
            self.set_arm_joint_velocity_target(joint_velocities)

            # Check if position error is small enough to move to the next stage
            if np.linalg.norm(position_error) < 0.01:
                self._throw_step = 1
                self._open_threshold = max(self.get_gripper_position()) + 0.005
                return False  # Throwing not yet complete

        # Stage 2: Release the object by opening the gripper
        elif self._throw_step == 1:
            joint_velocities = self.velocity_ik(self.tcp_target_velocity[0], self.tcp_target_velocity[1])
            self.set_arm_joint_velocity_target(joint_velocities)

            self.open_gripper()
            if self._is_gripper_open(self._open_threshold):
                self._throw_step = 2
                return False  # Throwing not yet complete

        # Stage 3: Decelerate the arm by reducing velocity to zero
        elif self._throw_step == 2:
            self.set_arm_joint_velocity_target(np.zeros([self.num_arm_dofs]))
            joint_velocities = self.get_arm_joint_velocity()
            if all(abs(vel) < 0.01 for vel in joint_velocities):
                del self._throw_step
                del self._open_threshold
                return True  # Throwing process complete

        return False  # Throwing process not yet complete


    # def throw(self, tcp_target_pose, tcp_target_velocity, estimate_speed=0.5):
    #     """
    #     Perform a throwing action with three stages: 
    #     1. Move to the release point with a target velocity.
    #     2. Release the object by opening the gripper while maintaining the current velocity.
    #     3. Decelerate the arm after the release.

    #     Args:
    #         tcp_target_pose (list): Target TCP pose at the release point as [position, orientation].
    #         tcp_target_velocity (list): Target velocity for the TCP at the release point (linear and angular).
    #         deceleration_distance (float): Distance to move along the target velocity direction during the deceleration phase.
    #         estimate_speed (float): Speed used to estimate the time for trajectory generation.

    #     Returns:
    #         bool: True if the throwing process is completed, False otherwise.
    #     """
    #     # Initialize or continue the throwing process
    #     if not hasattr(self, '_throw_step'):
    #         # Stage 0: Set initial conditions for the throwing process
    #         self._throw_step = 0  # Initialize the throw step
    #         self.tcp_target_pose = tcp_target_pose
    #         self.tcp_target_velocity = tcp_target_velocity

    #         # Stage 1: Generate the trajectory towards the release point (target pose with velocity)
    #         self._tcp_trajectory = self._generate_tcp_trajectory(
    #             self.get_tcp_pose(),            # Start at the current TCP pose
    #             self.tcp_target_pose,           # Move to the target pose
    #             start_tcp_vel=([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),  # Starting velocity is zero (from rest)
    #             target_tcp_vel=tcp_target_velocity,  # End with the specified target velocity
    #             estimate_speed=estimate_speed   # Use estimated speed for trajectory timing
    #         )
    #         self._trajectory_index = 0  # Initialize the trajectory index

    #     # Stage 1: Move to the release point with the target velocity
    #     if self._throw_step == 0:
    #         if self._trajectory_index < len(self._tcp_trajectory):
    #             # Move step by step along the trajectory towards the release point
    #             self._tcp_target_pose, self._tcp_target_velocity = self._tcp_trajectory[self._trajectory_index]
    #             self.set_tcp_pose_target(self._tcp_target_pose, self._tcp_target_velocity)  # Set the TCP to the current subtarget
    #             self._trajectory_index += 1
    #         else:
    #             # Once the release point is reached, switch to velocity control for release
    #             self._throw_step = 1
    #             return False  # Throwing not yet complete

    #     # Stage 2: Release the object by opening the gripper while maintaining the velocity
    #     elif self._throw_step == 1:
    #         # Apply velocity control using the velocity IK method
    #         joint_velocities = self.velocity_ik(self.tcp_target_velocity[0], self.tcp_target_velocity[1])
    #         self.set_arm_joint_velocity_target(joint_velocities)

    #         # Open the gripper to release the object
    #         self.open_gripper()
    #         if self._is_gripper_stopped():  # Check if the gripper has fully opened
    #             self._throw_step = 2  # Move to the next step (deceleration phase)
    #             return False  # Throwing not yet complete

    #     # Stage 3: Decelerate the arm by gradually reducing velocity to zero
    #     elif self._throw_step == 2:
    #         # Apply a gradual deceleration to the arm by reducing the joint velocities
    #         joint_velocities = self.velocity_ik(self.tcp_target_velocity[0], self.tcp_target_velocity[1])

    #         # Gradually reduce each joint velocity
    #         deceleration_factor = 0.9  # Factor to gradually reduce velocity each step
    #         joint_velocities = [vel * deceleration_factor for vel in joint_velocities]

    #         # Apply the reduced velocities to the arm joints
    #         self.set_arm_joint_velocity_target(joint_velocities)

    #         # Check if all joint velocities are close to zero
    #         if all(abs(vel) < 0.01 for vel in joint_velocities):
    #             self.set_arm_joint_velocity_target(np.zeros_like(joint_velocities))
    #             del self._throw_step  # Cleanup after completion
    #             return True  # Throwing process is complete

    #         # Update the current velocity for the next iteration
    #         self.tcp_target_velocity = ([v * deceleration_factor for v in self.tcp_target_velocity[0]],
    #                                     [v * deceleration_factor for v in self.tcp_target_velocity[1]])

    #     return False  # Throwing process is not complete
    
    def _generate_tcp_trajectory(self, start_tcp_pose, target_tcp_pose, start_tcp_vel=None, target_tcp_vel=None, estimate_speed=0.5):
        """
        Generates a TCP trajectory from start to target pose using quintic polynomial interpolation for both position
        and orientation (converted to Euler angles), considering velocity information.
        
        Args:
            start_tcp_pose (list): Starting TCP pose [position, orientation].
            target_tcp_pose (list): Target TCP pose [position, orientation].
            start_tcp_vel (list): Starting TCP velocity (linear velocity + angular velocity).
            target_tcp_vel (list): Target TCP velocity (linear velocity + angular velocity).
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
        distance = np.linalg.norm(target_pos - start_pos)
        time_estimate = distance / estimate_speed
        
        # Compute the number of steps based on PyBullet's simulation step size (1/240 seconds)
        num_steps = int(time_estimate / (1/240))
        
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
                (pos_trajectory[i].tolist(),    # Position
                ori_trajectory[i].tolist()),    # Orientation (quaternion)
                (vel_trajectory[i].tolist(),    # Linear velocity
                rot_vel_trajectory[i].tolist()) # Angular velocity
            ]
            subtargets.append(subtarget)
        
        return subtargets

    def _generate_quintic_trajectory(self, start_s, end_s, start_v=None, end_v=None, start_acc=None, end_acc=None, duration=None, num_steps=100):
        """
        Generates a quintic polynomial trajectory for multiple dimensions (e.g., position components).
        
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
        start_v = np.zeros_like(start_s) if start_v is None else start_v
        end_v = np.zeros_like(end_s) if end_v is None else end_v
        start_acc = np.zeros_like(start_s) if start_acc is None else start_acc
        end_acc = np.zeros_like(end_s) if end_acc is None else end_acc

        # Initialize the trajectory for each dimension
        trajectory_s = np.zeros((num_steps, len(start_s)))
        trajectory_v = np.zeros((num_steps, len(start_s)))
        
        # Compute the quintic polynomial for each dimension independently
        for dim in range(len(start_s)):
            delta_s = end_s[dim] - start_s[dim]
            # Corrected quintic polynomial coefficients
            A = (12 * delta_s - 6 * (start_v[dim] + end_v[dim]) * duration - (end_acc[dim] - start_acc[dim]) * duration**2) / (2 * duration**5)
            B = (-30 * delta_s + (16 * start_v[dim] + 14 * end_v[dim]) * duration + (3 * start_acc[dim] - 2 * end_acc[dim]) * duration**2) / (2 * duration**4)
            C = (20 * delta_s - (12 * start_v[dim] + 8 * end_v[dim]) * duration - (3 * start_acc[dim] - end_acc[dim]) * duration**2) / (2 * duration**3)
            D = start_acc[dim] / 2
            E = start_v[dim]
            F = start_s[dim]

            # Time steps
            t_s = np.linspace(0, duration, num_steps)
            # Evaluate the quintic polynomial at each time step for this dimension
            trajectory_s[:, dim] = A * t_s**5 + B * t_s**4 + C * t_s**3 + D * t_s**2 + E * t_s + F
            # Velocity is the derivative of the position polynomial
            trajectory_v[:, dim] = 5 * A * t_s**4 + 4 * B * t_s**3 + 3 * C * t_s**2 + 2 * D * t_s + E

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

    def visualize_tcp_trajectory(self, color_target=[1, 0, 0], color_actual=[0, 1, 0], line_duration=5):
        """
        Visualize the TCP target and actual trajectories using debug lines.
        
        Args:
            color_target (list): RGB color for target trajectory lines.
            color_actual (list): RGB color for actual trajectory lines.
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
            p.addUserDebugLine(self._prev_tcp_target_pose[0], target_pose[0], color_target, line_duration)
            self._prev_tcp_target_pose = target_pose

        # Visualize the actual TCP trajectory
        p.addUserDebugLine(self._prev_tcp_pose[0], current_pose[0], color_actual, line_duration)
        self._prev_tcp_pose = current_pose
    