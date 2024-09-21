import numpy as np
import pybullet as p

from collections import namedtuple
from scipy.spatial.transform import Rotation as R
from tossingbot.envs.pybullet.utils.math_utils import slerp, pose_distance

class BaseRobot:
    """
    The base class for robots
    """

    def __init__(self, base_position, base_orientation, robot_type='ur5_robotiq85'):
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

    def set_arm_joint_position_target(self, target_position):
        """
        Set the target position for the arm joints.
        """
        assert len(target_position) == self.num_arm_dofs, "Target position length mismatch"
        for joint_id, pos in zip(self.arm_controllable_joints, target_position):
            p.setJointMotorControl2(
                self.robot_id, joint_id, p.POSITION_CONTROL, pos,
                force=self.joints[joint_id].max_force,
                maxVelocity=self.joints[joint_id].max_velocity,
            )

    def get_arm_joint_position(self):
        """
        Get the current position of the arm joints.
        
        Returns:
            list: Current positions of arm controllable joints.
        """
        return [p.getJointState(self.robot_id, joint_id)[0] for joint_id in self.arm_controllable_joints]

    ############### set tcp pose, get tcp pose, and inverse kinematics ###############
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

    def inverse_kinematics(self, pose, rest_pose=None):
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

    ############### grasp and throw motion primitives ###############
    # def grasp(self, tcp_target_pose, num_subtargets=10):
    #     """
    #     Perform a grasping action at the target TCP pose in a step-by-step manner.
        
    #     Args:
    #         tcp_target_pose (list): Target TCP pose as [position, orientation].
    #         num_subtargets (int): Number of subtargets for smooth trajectory.
        
    #     Returns:
    #         bool: True if the grasping process is completed, False otherwise.
    #     """
    #     # Initialize or continue the grasping process
    #     if not hasattr(self, '_grasp_step'):
    #         self._grasp_step = 0  # Initialize the grasp step
    #         self.pose_over_target = [tcp_target_pose[0][:2] + [0.365], tcp_target_pose[1]]
    #         self.tcp_target_pose = tcp_target_pose

    #     if self._grasp_step == 0:
    #         # Move to a position over the target with subtargets
    #         if self.set_tcp_trajectory(self.pose_over_target, num_subtargets=num_subtargets, 
    #                                    position_tolerance=0.05, orientation_tolerance=0.05, 
    #                                    final_position_tolerance=0.05, final_orientation_tolerance=0.05):
    #             self._grasp_step = 1  # Move to the next step
    #         return False  # Grasping not yet complete

    #     elif self._grasp_step == 1:
    #         # Open the gripper
    #         self.open_gripper()
    #         # Check if the gripper has stopped moving
    #         if self._is_gripper_stopped():
    #             self._grasp_step = 2  # Move to the next step
    #         return False  # Grasping not yet complete

    #     elif self._grasp_step == 2:
    #         # Move to the target position with subtargets
    #         if self.set_tcp_trajectory(self.tcp_target_pose, num_subtargets=num_subtargets, 
    #                                    position_tolerance=0.05, orientation_tolerance=0.05, 
    #                                    final_position_tolerance=0.01, final_orientation_tolerance=0.01):
    #             self._grasp_step = 3  # Move to the next step
    #         return False  # Grasping not yet complete

    #     elif self._grasp_step == 3:
    #         # Close the gripper
    #         self.close_gripper()
    #         # Check if the gripper has stopped moving
    #         if self._is_gripper_stopped():
    #             self._grasp_step = 4  # Move to the next step
    #         return False  # Grasping not yet complete

    #     elif self._grasp_step == 4:
    #         # Move back to the position over the target with subtargets
    #         if self.set_tcp_trajectory(self.pose_over_target, num_subtargets=num_subtargets, 
    #                                    position_tolerance=0.05, orientation_tolerance=0.05, 
    #                                    final_position_tolerance=0.01, final_orientation_tolerance=0.01):
    #             del self._grasp_step  # Grasping process is complete, cleanup
    #             return True  # Grasping process is complete
    #         return False  # Grasping not yet complete

    #     return False  # Grasping process is not complete

    # def set_tcp_trajectory(self, target_tcp_pose, num_subtargets=10, 
    #                     position_tolerance=0.05, orientation_tolerance=0.05, 
    #                     final_position_tolerance=0.01, final_orientation_tolerance=0.01,
    #                     stop_pose_change_tolerance=1e-6, stop_check_steps=10):
    #     """
    #     Generate subtargets to move the TCP to the target pose smoothly.

    #     Args:
    #         target_tcp_pose (list): A list containing the target TCP pose with [position, orientation].
    #         num_subtargets (int): The number of subtargets to create along the trajectory.
    #         position_tolerance (float): Tolerance for position to switch to the next subtarget.
    #         orientation_tolerance (float): Tolerance for orientation to switch to the next subtarget.
    #         final_position_tolerance (float): Tolerance for final position.
    #         final_orientation_tolerance (float): Tolerance for final orientation.
    #         stop_pose_change_tolerance (float): Pose change threshold to consider the robot stopped.
    #         stop_check_steps (int): Number of steps to confirm the robot has stopped.
            
    #     Returns:
    #         bool: True if the trajectory is completed, False otherwise.
    #     """
    #     # Generate subtargets only if they haven't been created yet
    #     if not hasattr(self, '_subtargets') or not self._subtargets:
    #         self._subtargets = self._generate_smooth_subtargets(self.get_tcp_pose(), target_tcp_pose, num_subtargets)
    #         self._subtarget_index = 0  # Initialize the subtarget index here
    #         self.stopped_counter = 0  # Initialize stopping counter
    #         self.previous_tcp_pose = self.get_tcp_pose()  # Initialize previous TCP pose

    #     # Check if the TCP has stopped
    #     current_tcp_pose = self.get_tcp_pose()
    #     position_change = np.linalg.norm(np.array(current_tcp_pose[0]) - np.array(self.previous_tcp_pose[0]))
    #     orientation_change = np.abs(np.dot(current_tcp_pose[1], self.previous_tcp_pose[1]) - 1)  # Quaternion difference

    #     # Update the previous TCP pose
    #     self.previous_tcp_pose = current_tcp_pose

    #     # Check if the changes are below the threshold
    #     if position_change < stop_pose_change_tolerance and orientation_change < stop_pose_change_tolerance:
    #         self.stopped_counter += 1
    #     else:
    #         self.stopped_counter = 0  # Reset if TCP starts moving again

    #     # If TCP is considered stopped, move to the next subtarget or finish
    #     if self.stopped_counter > stop_check_steps:
    #         self.stopped_counter = 0  # Reset counter for the next check
    #         self._subtarget_index += 1

    #     # Move through subtargets
    #     if self._subtarget_index <= len(self._subtargets):
    #         if self._subtarget_index < len(self._subtargets):
    #             current_subtarget = self._subtargets[self._subtarget_index]
    #             current_position_tolerance = position_tolerance
    #             current_orientation_tolerance = orientation_tolerance
    #         else:
    #             current_subtarget = target_tcp_pose
    #             current_position_tolerance = final_position_tolerance
    #             current_orientation_tolerance = final_orientation_tolerance

    #         # Check if the TCP has reached the current subtarget
    #         if not self.is_tcp_reached_target(target_pose=current_subtarget, 
    #                                           position_tolerance=current_position_tolerance, 
    #                                           orientation_tolerance=current_orientation_tolerance):
    #             # Move to the current subtarget
    #             self.set_tcp_pose_target(current_subtarget)
    #         else:
    #             # Only move to the next subtarget if the current one is reached
    #             self._subtarget_index += 1

    #     # Check if the entire trajectory is completed
    #     if self._subtarget_index > len(self._subtargets):
    #         del self._subtargets  # Clear subtargets
    #         del self._subtarget_index
    #         return True  # Trajectory is completed

    #     return False  # Trajectory is not completed

    # def _generate_smooth_subtargets(self, start_pose, end_pose, num_subtargets):
    #     """
    #     Generate smooth intermediate subtargets using linear interpolation.
        
    #     Args:
    #         start_pose (list): Start pose as [position, orientation].
    #         end_pose (list): End pose as [position, orientation].
    #         num_subtargets (int): Number of subtargets to generate.
            
    #     Returns:
    #         list: List of smooth subtargets [position, orientation].
    #     """
    #     # Linear interpolation for position
    #     start_pos = np.array(start_pose[0])
    #     end_pos = np.array(end_pose[0])
    #     pos_interp = np.linspace(start_pos, end_pos, num_subtargets + 2)[1:-1]

    #     # Slerp for orientation using scipy Rotation
    #     start_rot = R.from_quat(start_pose[1])
    #     end_rot = R.from_quat(end_pose[1])
    #     start_quat = start_rot.as_quat()
    #     end_quat = end_rot.as_quat()
    #     times = np.linspace(0, 1, num_subtargets + 2)[1:-1]
    #     ori_interp = slerp(start_quat, end_quat, times)

    #     subtargets = []
    #     for pos, ori in zip(pos_interp, ori_interp):
    #         subtargets.append([pos.tolist(), ori.tolist()])
    #     return subtargets

    # def is_tcp_reached_target(self, target_pose, position_tolerance=0.01, orientation_tolerance=0.01):
    #     """
    #     Check if the TCP has reached the target pose.
        
    #     Args:
    #         target_pose (list): Target pose as [position, orientation].
    #         position_tolerance (float): Tolerance for position difference.
    #         orientation_tolerance (float): Tolerance for orientation difference in radians.
            
    #     Returns:
    #         bool: True if the TCP is within the specified tolerances, False otherwise.
    #     """
    #     current_pose = self.get_tcp_pose()
    #     distances = pose_distance(current_pose, target_pose)
    #     return (
    #         distances['position_distance'] <= position_tolerance and
    #         distances['orientation_distance'] <= orientation_tolerance
    #     )
    
    def grasp(self, tcp_target_pose):
        """
        Perform a grasping action at the target TCP pose in a step-by-step manner.
        
        Args:
            tcp_target_pose (list): Target TCP pose as [position, orientation].
        
        Returns:
            bool: True if the grasping process is completed, False otherwise.
        """
        # Initialize or continue the grasping process
        if not hasattr(self, '_grasp_step'):
            self._grasp_step = 0  # Initialize the grasp step
            self.pose_over_target = [tcp_target_pose[0][:2] + [0.3], tcp_target_pose[1]]
            self.tcp_target_pose = tcp_target_pose

            # Generate trajectory for moving over the target position
            self._tcp_trajectory = self._generate_tcp_trajectory(self.get_tcp_pose(), self.pose_over_target, 
                                                                start_tcp_vel=[0, 0, 0, 0, 0, 0], 
                                                                target_tcp_vel=[0, 0, 0, 0, 0, 0], 
                                                                estimate_speed=0.5)
            self._trajectory_index = 0

        if self._grasp_step == 0:
            # Move to a position over the target using the generated TCP trajectory
            if self._trajectory_index < len(self._tcp_trajectory):
                current_setpoint = self._tcp_trajectory[self._trajectory_index]
                self.set_tcp_pose_target(current_setpoint)  # Set current setpoint for the TCP
                self._trajectory_index += 1
            else:
                # Once reached the final point, move to the next step
                self._grasp_step = 1
                self._trajectory_index = 0  # Reset for next trajectory
                # Generate trajectory for moving to the target position (step 2)
                self._tcp_trajectory = self._generate_tcp_trajectory(self.pose_over_target, self.tcp_target_pose, 
                                                                    start_tcp_vel=[0, 0, 0, 0, 0, 0], 
                                                                    target_tcp_vel=[0, 0, 0, 0, 0, 0], 
                                                                    estimate_speed=0.5)
            return False  # Grasping not yet complete

        elif self._grasp_step == 1:
            # Open the gripper
            self.open_gripper()
            # Check if the gripper has stopped moving
            if self._is_gripper_stopped():
                self._grasp_step = 2  # Move to the next step
            return False  # Grasping not yet complete

        elif self._grasp_step == 2:
            # Move to the target position using the generated TCP trajectory
            if self._trajectory_index < len(self._tcp_trajectory):
                current_setpoint = self._tcp_trajectory[self._trajectory_index]
                self.set_tcp_pose_target(current_setpoint)  # Set current setpoint for the TCP
                self._trajectory_index += 1
            else:
                # Once reached the final point, move to the next step
                self._grasp_step = 3
                self._trajectory_index = 0  # Reset for next trajectory
            return False  # Grasping not yet complete

        elif self._grasp_step == 3:
            # Close the gripper
            self.close_gripper()
            # Check if the gripper has stopped moving
            if self._is_gripper_stopped():
                self._grasp_step = 4  # Move to the next step
                # Generate trajectory for moving back to the position over the target (step 4)
                self._tcp_trajectory = self._generate_tcp_trajectory(self.tcp_target_pose, self.pose_over_target, 
                                                                    start_tcp_vel=[0, 0, 0, 0, 0, 0], 
                                                                    target_tcp_vel=[0, 0, 0, 0, 0, 0], 
                                                                    estimate_speed=0.5)
            return False  # Grasping not yet complete

        elif self._grasp_step == 4:
            # Move back to the position over the target using the generated TCP trajectory
            if self._trajectory_index < len(self._tcp_trajectory):
                current_setpoint = self._tcp_trajectory[self._trajectory_index]
                self.set_tcp_pose_target(current_setpoint)  # Set current setpoint for the TCP
                self._trajectory_index += 1
            else:
                # Once reached the final point, the grasping process is complete
                del self._grasp_step  # Cleanup
                del self._tcp_trajectory
                del self._trajectory_index
                return True  # Grasping process is complete

        return False  # Grasping process is not complete
    
    def throw(self, tcp_target_pose, tcp_target_velocity, num_subtargets=10):
        """
        Perform a throwing action at the target TCP pose with a specific velocity.

        Args:
            tcp_target_pose (list): Target TCP pose as [position, orientation].
            tcp_target_velocity (list): Target velocity for the TCP.
            num_subtargets (int): Number of subtargets for smooth trajectory.
        
        Returns:
            bool: True if the throwing process is completed, False otherwise.
        """
        # Initialize or continue the throwing process
        if not hasattr(self, '_throw_step'):
            self._throw_step = 0  # Initialize the throw step
            self.tcp_target_pose = tcp_target_pose
            self.tcp_target_velocity = tcp_target_velocity

        if self._throw_step == 0:
            # Move to the target pose with subtargets
            if self.set_tcp_trajectory(self.tcp_target_pose, num_subtargets=num_subtargets, 
                                    position_tolerance=0.05, orientation_tolerance=0.05, 
                                    final_position_tolerance=0.01, final_orientation_tolerance=0.01):
                self._throw_step = 1  # Move to the next step
            return False  # Throwing not yet complete

        elif self._throw_step == 1:
            # Apply velocity to the TCP
            p.setJointMotorControlArray(
                self.robot_id, self.arm_controllable_joints,
                p.VELOCITY_CONTROL,
                targetVelocities=self.tcp_target_velocity,
                forces=[joint.max_force for joint in self.joints if joint.controllable]
            )
            self._throw_step = 2  # Move to the next step
            return False  # Throwing not yet complete

        elif self._throw_step == 2:
            # Open the gripper to release the object
            self.open_gripper()
            if self._is_gripper_stopped():
                del self._throw_step  # Throwing process is complete, cleanup
                return True  # Throwing process is complete
            return False  # Throwing not yet complete

        return False  # Throwing process is not complete
    
    def _generate_tcp_trajectory(self, start_tcp_pose, target_tcp_pose, start_tcp_vel, target_tcp_vel, estimate_speed=0.5):
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
            list: List of subtargets (setpoints) for the TCP to follow, each containing position and orientation.
        """
        # Extract start and target positions and orientations
        start_pos = np.array(start_tcp_pose[0])
        target_pos = np.array(target_tcp_pose[0])
        start_rot = R.from_quat(start_tcp_pose[1])
        target_rot = R.from_quat(target_tcp_pose[1])
        start_quat = start_rot.as_quat()
        target_quat = target_rot.as_quat()
        
        # Extract start and target translational and rotational velocities
        start_trans_vel = np.array(start_tcp_vel[:3])
        target_trans_vel = np.array(target_tcp_vel[:3])
        start_rot_vel = np.array(start_tcp_vel[3:])
        target_rot_vel = np.array(target_tcp_vel[3:])
        
        # Estimate the time needed based on the distance and given speed
        distance = np.linalg.norm(target_pos - start_pos)
        time_estimate = distance / estimate_speed
        
        # Compute the number of steps based on PyBullet's simulation step size (1/240 seconds)
        num_steps = int(time_estimate / (1/240))
        
        # Generate quintic polynomial trajectory for linear positions
        pos_trajectory = self._generate_quintic_trajectory(start_s=start_pos, 
                                                           end_s=target_pos, 
                                                           start_v=start_trans_vel, 
                                                           end_v=target_trans_vel, 
                                                           duration=time_estimate, 
                                                           num_steps=num_steps)

        # Generate quaternion trajectory using slerp
        t_values = np.linspace(0, 1, num_steps)  # Interpolation points for Slerp
        ori_trajectory = slerp(start_quat, target_quat, t_values)
        
        # Generate the final list of subtargets (setpoints) combining positions and orientations
        subtargets = []
        for i in range(num_steps):
            subtarget = (pos_trajectory[i].tolist(), ori_trajectory[i].tolist())
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
            np.ndarray: Array of positions sampled along the quintic polynomial trajectory.
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

        return trajectory_s

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