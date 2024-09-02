import math
import pybullet as p
from collections import namedtuple

class BaseRobot:
    """
    The base class for robots
    """

    def __init__(self, base_position, base_orientation):
        """
        Arguments:
            base_position: [x, y, z]
            base_orientation: [roll, pitch, yaw]

        Attributes:
            robot_id: Int, the ID of the robot
            end_effector_id: Int, the ID of the End-Effector
            num_arm_dofs: Int, the number of DoFs of the arm
            joints: List, a list of joint info
            controllable_joints: List of Ints, IDs for all controllable joints
            arm_controllable_joints: List of Ints, IDs for all controllable joints on the arm
            arm_lower_limits: List, the lower limits for all controllable joints on the arm
            arm_upper_limits: List, the upper limits for all controllable joints on the arm
            arm_joint_ranges: List, the range of motion for each joint
            arm_initial_position: List, the initial position for all controllable joints on the arm
            gripper_range: List[Min, Max]

        Methods:
        
        """
        self.base_position = base_position
        self.base_orientation_quat = p.getQuaternionFromEuler(base_orientation)
        self.load_robot()
        self.reset()

    def load_robot(self):
        self.robot_id = p.loadURDF(
            './assets/urdf/ur5_robotiq_85.urdf',
            self.base_position,
            self.base_orientation_quat,
            useFixedBase=True,
            flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES | p.URDF_USE_SELF_COLLISION
        )
        self._parse_joint_information()

    def _parse_joint_information(self):
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

        self.arm_lower_limits = [
            info.lower_limit for info in self.joints if info.controllable
        ][:self.num_arm_dofs]
        self.arm_upper_limits = [
            info.upper_limit for info in self.joints if info.controllable
        ][:self.num_arm_dofs]
        self.arm_joint_ranges = [
            info.upper_limit - info.lower_limit for info in self.joints if info.controllable
        ][:self.num_arm_dofs]

    def reset(self):
        self.reset_arm()

    def reset_arm(self):
        """
        Reset the arm to its initial position
        """
        self.set_arm_joint_position(self.initial_position[0:6])
        self.set_arm_joint_position_target(self.initial_position[0:6])

    def set_arm_joint_position(self, position):
        """
        Set the position for the arm joints
        """
        assert len(position) == self.num_arm_dofs
        for joint_id, pos in zip(self.arm_controllable_joints, position):
            p.resetJointState(self.robot_id, joint_id, pos)

    def set_arm_joint_position_target(self, target_position):
        """
        Set the target position for the arm joints
        """
        assert len(target_position) == self.num_arm_dofs
        for joint_id, pos in zip(self.arm_controllable_joints, target_position):
            p.setJointMotorControl2(
                self.robot_id, joint_id, p.POSITION_CONTROL, pos,
                force=self.joints[joint_id].max_force,
                maxVelocity=self.joints[joint_id].max_velocity
            )

    def set_arm_joint_velocity_target(self, target_velocity):
        raise NotImplementedError

    def set_end_effector_pose(self, pose):
        joint_position = self.inverse_kinematics(pose)
        self.set_arm_joint_position(joint_position)

    def set_end_effector_pose_target(self, target_pose):
        """
        Set the target pose for the end effector using inverse kinematics
        """
        target_joint_position = self.inverse_kinematics(target_pose)
        self.set_arm_joint_position_target(target_joint_position)

    def set_end_effector_velocity(self, velocity):
        raise NotImplementedError

    def set_end_effector_velocity_target(self, target_velocity):
        raise NotImplementedError

    def inverse_kinematics(self, pose):
        x, y, z, roll, pitch, yaw = pose
        position = (x, y, z)
        orientation = p.getQuaternionFromEuler((roll, pitch, yaw))
        current_joint_position = self.get_joint_position()
        joint_position = p.calculateInverseKinematics(
            self.robot_id, self.end_effector_id, position, orientation,
            self.arm_lower_limits, self.arm_upper_limits,
            self.arm_joint_ranges, current_joint_position,
            maxNumIterations=20
        )

        return joint_position

    def get_joint_position(self):
        """
        Get the current position of the joints
        """
        position = []
        for joint_id in self.controllable_joints:
            pos, _ = p.getJointState(self.robot_id, joint_id)[:2]
            position.append(pos)
        return position

    def get_joint_velocity(self):
        """
        Get the current velocity of the joints
        """
        velocity = []
        for joint_id in self.controllable_joints:
            _, vel = p.getJointState(self.robot_id, joint_id)[:2]
            velocity.append(vel)
        return velocity

    def get_end_effector_position(self):
        """
        Get the current position of the end effector
        """
        end_effector_position = p.getLinkState(self.robot_id, self.end_effector_id)[0]
        return end_effector_position

class UR5Robotiq85(BaseRobot):
    def __init__(self, base_position, base_orientation):
        self.num_arm_dofs = 6
        self.initial_position = [
            -1.569, -1.545, 1.344,
            -1.371, -1.571, 0.001,
            0.085   # Gripper
        ]
        self.end_effector_id = 7
        self.gripper_range = [0, 0.085]

        super().__init__(base_position, base_orientation)

    def load_robot(self):
        super().load_robot()
        self._setup_mimic_joints()

    def _setup_mimic_joints(self):
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
        super().reset()
        self.reset_gripper()

    def reset_gripper(self):
        self.set_gripper_position(self.gripper_range[1])
        self.set_gripper_position_target(self.gripper_range[1])

    def open_gripper(self):
        self.set_gripper_position_target(self.gripper_range[1])

    def close_gripper(self):
        self.set_gripper_position_target(self.gripper_range[0])

    def set_gripper_position(self, open_length):
        open_angle = self._length2angle(open_length=open_length)
        p.resetJointState(self.robot_id, self.mimic_parent_id, open_angle)

    def set_gripper_position_target(self, open_length):
        """
        Set the gripper's position by calculating the corresponding joint angle
        """
        open_angle = self._length2angle(open_length=open_length)
        p.setJointMotorControl2(
            self.robot_id, self.mimic_parent_id, p.POSITION_CONTROL,
            targetPosition=open_angle,
            force=self.joints[self.mimic_parent_id].max_force,
            maxVelocity=self.joints[self.mimic_parent_id].max_velocity
        )

    def _length2angle(self, open_length):
        open_angle = 0.715 - math.asin((open_length - 0.010) / 0.1143)

        return open_angle