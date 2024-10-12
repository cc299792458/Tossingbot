import time
import random
import numpy as np
import pybullet as p
import pybullet_data

from scipy.spatial.transform import Rotation as R

class BaseObject:
    def __init__(
            self, 
            position=np.array([0, 0, 1]), 
            orientation=np.array([0, 0, 0]), 
            mass=0.1, 
            color=[0.5, 0.5, 0.5, 1],
            lateral_friction=0.5,
            rolling_friction=0.0,
        ):
        self.object_ids = []
        self.init_pose = [position, orientation]
        self.mass = mass
        self.color = color
        self.lateral_friction = lateral_friction
        self.rolling_friction = rolling_friction

        self._create_object()

    def _create_object(self):
        raise NotImplementedError
    
    def _remove_object(self):
        for object_id in self.object_ids:
            p.removeBody(object_id)

    @property
    def pose(self):
        position, orientation = p.getBasePositionAndOrientation(self.object_ids[0])
        return (np.array(position), np.array(orientation))
    
    @property
    def velocity(self):
        linear_velocity, angular_velocity = p.getBaseVelocity(self.object_ids[0])
        return (np.array(linear_velocity), np.array(angular_velocity))
    
class Ball(BaseObject):
    def __init__(
            self, 
            position=np.array([0, 0, 1]), 
            orientation=np.array([0, 0, 0]), 
            mass=0.1, 
            color=[1.0, 0.0, 0.0, 1], 
            lateral_friction=0.5, 
            rolling_friction=0,
            radius=0.5):
        self.radius = radius
        super().__init__(position, orientation, mass, color, lateral_friction, rolling_friction)
    
    def _create_object(self):
        collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=self.radius)
        position = self.init_pose[0]
        sphere_id = p.createMultiBody(self.mass, collision_shape, -1, position)
        p.changeVisualShape(sphere_id, -1, rgbaColor=self.color)
        p.changeDynamics(sphere_id, -1, lateralFriction=self.lateral_friction, rollingFriction=self.rolling_friction)
        self.object_ids.append(sphere_id)

class Cube(BaseObject):
    def __init__(
            self, 
            position=np.array([0, 0, 1]), 
            orientation=np.array([0, 0, 0]), 
            mass=0.1, 
            color=[0.0, 1.0, 0.0, 1], 
            lateral_friction=0.5, 
            rolling_friction=0,
            half_extents=[0.5, 0.5, 0.5]):
        self.half_extents = half_extents
        super().__init__(position, orientation, mass, color, lateral_friction, rolling_friction)
    
    def _create_object(self):
        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=self.half_extents)
        position, orientation = self.init_pose[0], p.getQuaternionFromEuler(self.init_pose[1])
        cube_id = p.createMultiBody(self.mass, collision_shape, -1, position, orientation)
        p.changeVisualShape(cube_id, -1, rgbaColor=self.color)
        p.changeDynamics(cube_id, -1, lateralFriction=self.lateral_friction, rollingFriction=self.rolling_friction)
        self.object_ids.append(cube_id)

class Rod(BaseObject):
    def __init__(
            self, 
            position=np.array([0, 0, 1]), 
            orientation=np.array([0, 0, 0]), 
            mass=0.1, 
            color=[0.0, 0.0, 1.0, 1], 
            lateral_friction=0.5, 
            rolling_friction=0,
            radius=0.3,
            height=1.0):
        self.radius = radius
        self.height = height
        super().__init__(position, orientation, mass, color, lateral_friction, rolling_friction)
    
    def _create_object(self):
        collision_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=self.radius, height=self.height)
        position, orientation = self.init_pose[0], p.getQuaternionFromEuler(self.init_pose[1])
        rod_id = p.createMultiBody(self.mass, collision_shape, -1, position, orientation)
        p.changeVisualShape(rod_id, -1, rgbaColor=self.color)
        p.changeDynamics(rod_id, -1, lateralFriction=self.lateral_friction, rollingFriction=self.rolling_friction)
        self.object_ids.append(rod_id)

class Hammer(BaseObject):
    def __init__(
            self, 
            position=np.array([0, 0, 1]), 
            orientation=np.array([0, 0, 0]), 
            handle_mass=0.05, 
            head_mass=0.05, 
            color=[0.5, 0.5, 0.5, 1],
            lateral_friction=0.5, 
            rolling_friction=0.0,
            handle_radius=0.1, 
            handle_height=1.0, 
            head_half_extents=[0.3, 0.1, 0.1]):
        self.handle_mass = handle_mass
        self.head_mass = head_mass
        self.handle_radius = handle_radius
        self.handle_height = handle_height
        self.head_half_extents = head_half_extents
        super().__init__(position, orientation, handle_mass + head_mass, color, lateral_friction, rolling_friction)

    def _create_object(self):
        # Compute the center of mass for the combined hammer (handle + head)
        com = self.compute_center_of_mass(cylinder_mass=self.handle_mass, box_mass=self.head_mass, 
                                     cylinder_height=self.handle_height, box_half_extents=self.head_half_extents)

        # Convert Euler angles to quaternion for object orientation
        orientation = p.getQuaternionFromEuler(self.init_pose[1])

        # Create the cylinder (handle) collision shape
        collision_shape_handle = p.createCollisionShape(p.GEOM_CYLINDER, radius=self.handle_radius, height=self.handle_height)
        handle_position_local = np.array([0, 0, self.handle_height / 2]) - com  # Relative to the center of mass
        handle_position_world = np.array(self.init_pose[0]) + R.from_quat(orientation).apply(handle_position_local)
        
        # Create the handle of the hammer in the world
        handle_id = p.createMultiBody(baseMass=self.handle_mass, baseCollisionShapeIndex=collision_shape_handle, 
                                      basePosition=handle_position_world, baseOrientation=orientation)

        # Create the box (head) collision shape
        collision_shape_head = p.createCollisionShape(p.GEOM_BOX, halfExtents=self.head_half_extents)
        head_position_local = np.array([0, 0, self.handle_height + self.head_half_extents[2]]) - com  # Relative to the center of mass
        head_position_world = np.array(self.init_pose[0]) + R.from_quat(orientation).apply(head_position_local)
        
        # Create the head of the hammer in the world
        head_id = p.createMultiBody(baseMass=self.head_mass, baseCollisionShapeIndex=collision_shape_head, 
                                    basePosition=head_position_world, baseOrientation=orientation)

        # Create a fixed joint to attach the handle and the head
        p.createConstraint(parentBodyUniqueId=handle_id, parentLinkIndex=-1,
                           childBodyUniqueId=head_id, childLinkIndex=-1,
                           jointType=p.JOINT_FIXED, 
                           jointAxis=[0, 0, 0], 
                           parentFramePosition=[0, 0, self.handle_height / 2], 
                           childFramePosition=[0, 0, -self.head_half_extents[2]])

        # Set the visual appearance (color) for both handle and head
        p.changeVisualShape(handle_id, -1, rgbaColor=self.color)
        p.changeVisualShape(head_id, -1, rgbaColor=self.color)

        # Set dynamics features for both handle and head
        p.changeDynamics(handle_id, -1, lateralFriction=self.lateral_friction, rollingFriction=self.rolling_friction)
        p.changeDynamics(head_id, -1, lateralFriction=self.lateral_friction, rollingFriction=self.rolling_friction)
        
        # Store both handle and head IDs in the object_ids list
        self.object_ids.extend([handle_id, head_id])

    def compute_center_of_mass(self, cylinder_mass, box_mass, cylinder_height, box_half_extents):
        """Compute the center of mass of the hammer."""
        # Cylinder COM (assuming its center)
        cylinder_com = np.array([0, 0, cylinder_height / 2])
        # Box COM (relative to the cylinder)
        box_com = np.array([0, 0, cylinder_height + box_half_extents[2]])

        # Compute the weighted average to find the COM of the whole hammer
        com = (cylinder_mass * cylinder_com + box_mass * box_com) / self.mass
        return com
    
    @property
    def pose(self):
        """
        The pose (position and orientation) of the hammer based on its center of mass (COM).
        The position is computed as the weighted average of the handle and head positions.
        The orientation can be taken from either part since they are rigidly connected.
        """
        # Get positions of handle and head
        handle_position, handle_orientation = p.getBasePositionAndOrientation(self.object_ids[0])  # Handle's pose
        head_position, head_orientation = p.getBasePositionAndOrientation(self.object_ids[1])  # Head's pose
        
        # Compute center of mass position (weighted average)
        com_position = (self.handle_mass * np.array(handle_position) + self.head_mass * np.array(head_position)) / self.mass
        
        # Orientation is the same for both parts (since they are rigidly connected)
        # We can just use the handle's orientation
        com_orientation = handle_orientation  # Or head_orientation, they are identical
        
        return (com_position, np.array(com_orientation))

    @property
    def velocity(self):
        """
        The velocity (linear and angular) of the hammer based on its center of mass (COM).
        The linear velocity is computed as the weighted average of the handle and head velocities.
        The angular velocity is the same for both parts since they are rigidly connected.
        """
        # Get velocities of handle and head
        handle_linear_velocity, handle_angular_velocity = p.getBaseVelocity(self.object_ids[0])  # Handle's velocity
        head_linear_velocity, head_angular_velocity = p.getBaseVelocity(self.object_ids[1])  # Head's velocity
        
        # Compute center of mass linear velocity (weighted average)
        com_linear_velocity = (self.handle_mass * np.array(handle_linear_velocity) + self.head_mass * np.array(head_linear_velocity)) / self.mass
        
        # Angular velocity is the same for both parts (since they are rigidly connected)
        com_angular_velocity = handle_angular_velocity  # Or head_angular_velocity, they are identical
        
        return (com_linear_velocity, np.array(com_angular_velocity))

# Object Creation Functions
def create_sphere(radius=0.5, position=[0, 0, 1], mass=0.1, color=[1, 0, 0, 1]):
    """Creates a sphere in the PyBullet simulation."""
    collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
    sphere_id = p.createMultiBody(mass, collision_shape, -1, position)
    p.changeVisualShape(sphere_id, -1, rgbaColor=color)
    return sphere_id

def create_box(half_extents=[0.5, 0.5, 0.5], position=[0, 0, 1], orientation=[0, 0, 0], mass=0.1, color=[0, 1, 0, 1]):
    """Creates a box in the PyBullet simulation."""
    collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    box_orientation = p.getQuaternionFromEuler(orientation)
    box_id = p.createMultiBody(mass, collision_shape, -1, position, box_orientation)
    p.changeVisualShape(box_id, -1, rgbaColor=color)
    return box_id

def create_cylinder(radius=0.3, height=1.0, position=[0, 0, 1], orientation=[0, 0, 0], mass=0.1, color=[0, 0, 1, 1]):
    """Creates a cylinder in the PyBullet simulation."""
    collision_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
    cylinder_orientation = p.getQuaternionFromEuler(orientation)
    cylinder_id = p.createMultiBody(mass, collision_shape, -1, position, cylinder_orientation)
    p.changeVisualShape(cylinder_id, -1, rgbaColor=color)
    return cylinder_id

def create_capsule(radius=0.2, height=1.0, position=[0, 0, 1], orientation=[0, 0, 0], mass=0.1, color=[0.5, 0.5, 0.5, 1]):
    """Creates a capsule in the PyBullet simulation."""
    collision_shape = p.createCollisionShape(p.GEOM_CAPSULE, radius=radius, height=height)
    capsule_orientation = p.getQuaternionFromEuler(orientation)
    capsule_id = p.createMultiBody(mass, collision_shape, -1, position, capsule_orientation)
    p.changeVisualShape(capsule_id, -1, rgbaColor=color)
    return capsule_id

def create_mesh(mesh_file, position=[0, 0, 1], orientation=[0, 0, 0], mass=0.1, color=[0.5, 0.5, 0.5, 1]):
    """Creates a mesh object from a given .obj or .stl file."""
    collision_shape = p.createCollisionShape(p.GEOM_MESH, fileName=mesh_file)
    mesh_orientation = p.getQuaternionFromEuler(orientation)
    mesh_id = p.createMultiBody(mass, collision_shape, -1, position, mesh_orientation)
    p.changeVisualShape(mesh_id, -1, rgbaColor=color)
    return mesh_id

def create_plane(position=[0, 0, 0], orientation=[0, 0, 0], color=[1, 1, 1, 1]):
    """Creates a plane in the PyBullet simulation."""
    plane_id = p.loadURDF("plane.urdf", position, p.getQuaternionFromEuler(orientation))
    p.changeVisualShape(plane_id, -1, rgbaColor=color)
    return plane_id

# def create_hammer(position=[0, 0, 1], orientation=[0, 0, 0], 
#                   cylinder_radius=0.1, cylinder_height=1.0, box_half_extents=[0.3, 0.1, 0.1], 
#                   cylinder_mass=0.05, box_mass=0.05, color=[0.5, 0.5, 0.5, 1]):
#     """Create a hammer-like object in the simulation using scipy for quaternion rotations."""
#     # Compute the center of mass of the hammer
#     com = compute_center_of_mass(cylinder_mass=cylinder_mass, box_mass=box_mass, 
#                                  cylinder_height=cylinder_height, box_half_extents=box_half_extents)

#     # Convert Euler orientation to quaternion
#     hammer_orientation = R.from_euler('xyz', orientation).as_quat()

#     # Create cylinder (handle)
#     collision_shape_handle = p.createCollisionShape(p.GEOM_CYLINDER, radius=cylinder_radius, height=cylinder_height)
#     handle_position_local = np.array([0, 0, cylinder_height / 2]) - com  # Relative to COM
#     handle_position_world = np.array(position) + R.from_quat(hammer_orientation).apply(handle_position_local)
    
#     handle_id = p.createMultiBody(baseMass=cylinder_mass, baseCollisionShapeIndex=collision_shape_handle, 
#                                   basePosition=handle_position_world, baseOrientation=hammer_orientation)

#     # Create box (hammer head)
#     collision_shape_head = p.createCollisionShape(p.GEOM_BOX, halfExtents=box_half_extents)
#     head_position_local = np.array([0, 0, cylinder_height + box_half_extents[2]]) - com  # Relative to COM
#     head_position_world = np.array(position) + R.from_quat(hammer_orientation).apply(head_position_local)
    
#     head_id = p.createMultiBody(baseMass=box_mass, baseCollisionShapeIndex=collision_shape_head, 
#                                 basePosition=head_position_world, baseOrientation=hammer_orientation)

#     # Create fixed joint to attach the head to the handle
#     p.createConstraint(parentBodyUniqueId=handle_id, parentLinkIndex=-1,
#                        childBodyUniqueId=head_id, childLinkIndex=-1,
#                        jointType=p.JOINT_FIXED, 
#                        jointAxis=[0, 0, 0], 
#                        parentFramePosition=[0, 0, cylinder_height / 2], 
#                        childFramePosition=[0, 0, -box_half_extents[2]])

#     # Set color for both parts
#     p.changeVisualShape(handle_id, -1, rgbaColor=color)
#     p.changeVisualShape(head_id, -1, rgbaColor=color)
    
    # return handle_id, head_id

# Utility Functions
def random_color():
    """Generates a random RGBA color."""
    return [random.uniform(0, 1) for _ in range(3)] + [1.0]  # RGB + Alpha

# Main Simulation Code
if __name__ == '__main__':
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Set path to URDF files
    p.setGravity(0, 0, -9.8)

    # Create objects in the simulation
    create_plane()
    
    ball = Ball(position=np.array([-0.5, 0.5, 0.2]), mass=0.1, lateral_friction=1.0, rolling_friction=0.5, 
                radius=0.02)
    cube = Cube(position=np.array([-0.5, -0.5, 0.2]), mass=0.1, lateral_friction=1.0, rolling_friction=0.5, 
                half_extents=[0.02, 0.02, 0.02])
    rod = Rod(position=np.array([0.5, -0.5, 0.2]), orientation=np.array([0.0, np.pi / 2, 0.0]), 
              mass=0.1, lateral_friction=1.0, rolling_friction=0.5, radius=0.015, height=0.16)
    hammer = Hammer(position=np.array([0.5, 0.5, 0.2]), orientation=np.array([np.pi / 2, 0.0, np.pi / 2]), handle_mass=0.05, head_mass=0.05,
                    lateral_friction=1.0, rolling_friction=0.5, handle_radius=0.01, handle_height=0.12, head_half_extents=[0.05, 0.02, 0.0125])

    # Example: Load a mesh from an .obj file
    for i in range(8):
        if i != 5:
            mesh_file = f'assets/meshes/objects/blocks/{i}.obj'
            create_mesh(mesh_file=mesh_file, position=[0.0, 0.0, 0.2], color=random_color())

    # Run simulation
    for _ in range(10000):
        p.stepSimulation()
        time.sleep(1. / 240.)

    p.disconnect()
