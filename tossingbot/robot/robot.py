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
                i.e., the IK for the EE will consider the first `num_arm_dofs` controllable (non-Fixed) joints
            joints: List, a list of joint info
            controllable_joints: List of Ints, IDs for all controllable joints
            arm_controllable_joints: List of Ints, IDs for all controllable joints on the arm (the first `num_arm_dofs` controllable joints)

            ---
            For null-space IK
            ---
            arm_lower_limits: List, the lower limits for all controllable joints on the arm
            arm_upper_limits: List
            arm_joint_ranges: List
            arm_rest_positions: List, the rest position for all controllable joints on the arm

            gripper_range: List[Min, Max]
        """
        self.base_position = base_position
        self.base_orientation_quat = p.getQuaternionFromEuler(base_orientation)

    def load_robot(self):
        self.robot_id = p.loadURDF(
            './assets/urdf/ur5_robotiq_85.urdf', 
            self.base_position, 
            self.base_orientation_quat,
            useFixedBase=True, 
            flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        )
        self._parse_joint_information()
        self._setup_mimic_joints()

    def step_simulation(self):
        raise RuntimeError('`step_simulation` method of BaseRobot Class should be hooked by the environment.')

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

    def _setup_mimic_joints(self):
        raise NotImplementedError

    def reset_robot(self):
        self.reset_arm()
        self.reset_gripper()

    def reset_arm(self):
        """
        Reset to rest positions
        """
        for rest_position, joint_id in zip(self.arm_rest_positions, self.arm_controllable_joints):
            p.resetJointState(self.robot_id, joint_id, rest_position)

        # Wait for a few steps
        for _ in range(10):
            self.step_simulation()

    def reset_gripper(self):
        self.open_gripper()

    def open_gripper(self):
        self.set_gripper_position(self.gripper_range[1])

    def close_gripper(self):
        self.set_gripper_position(self.gripper_range[0])

    def move_end_effector(self, target, control_method):
        assert control_method in ('joint', 'end')

        if control_method == 'end':
            x, y, z, roll, pitch, yaw = target
            position = (x, y, z)
            orientation = p.getQuaternionFromEuler((roll, pitch, yaw))
            joint_positions = p.calculateInverseKinematics(
                self.robot_id, self.end_effector_id, position, orientation,
                self.arm_lower_limits, self.arm_upper_limits, 
                self.arm_joint_ranges, self.arm_rest_positions,
                maxNumIterations=20
            )
        elif control_method == 'joint':
            assert len(target) == self.num_arm_dofs
            joint_positions = target

        # Arm
        for i, joint_id in enumerate(self.arm_controllable_joints):
            p.setJointMotorControl2(
                self.robot_id, joint_id, p.POSITION_CONTROL, joint_positions[i],
                force=self.joints[joint_id].max_force, 
                maxVelocity=self.joints[joint_id].max_velocity
            )

    def set_gripper_position(self, position):
        raise NotImplementedError

    def get_joint_observations(self):
        positions = []
        velocities = []

        for joint_id in self.controllable_joints:
            pos, vel, _, _ = p.getJointState(self.robot_id, joint_id)
            positions.append(pos)
            velocities.append(vel)

        end_effector_position = p.getLinkState(self.robot_id, self.end_effector_id)[0]
        return dict(positions=positions, velocities=velocities, end_effector_position=end_effector_position)


class UR5Robotiq85(BaseRobot):
    def __init__(self, base_position, base_orientation):
        super().__init__(base_position, base_orientation)
        self.end_effector_id = 7
        self.num_arm_dofs = 6
        self.arm_rest_positions = [
            -1.5690622952052096, -1.5446774605904932, 1.343946009733127, 
            -1.3708613585093699, -1.5707970583733368, 0.0009377758247187636
        ]
        self.gripper_range = [0, 0.085]

        self.load_robot()

    def _setup_mimic_joints(self):
        parent_joint_name = 'finger_joint'
        child_joint_multipliers = {
            'right_outer_knuckle_joint': 1,
            'left_inner_knuckle_joint': 1,
            'right_inner_knuckle_joint': 1,
            'left_inner_finger_joint': -1,
            'right_inner_finger_joint': -1
        }
        self.mimic_parent_id = [
            joint.id for joint in self.joints if joint.name == parent_joint_name
        ][0]
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

    def set_gripper_position(self, open_length):
        open_angle = 0.715 - math.asin((open_length - 0.010) / 0.1143)
        p.setJointMotorControl2(
            self.robot_id, self.mimic_parent_id, p.POSITION_CONTROL, 
            targetPosition=open_angle,
            force=self.joints[self.mimic_parent_id].max_force, 
            maxVelocity=self.joints[self.mimic_parent_id].max_velocity
        )
