import time
import random
import numpy as np
import pybullet as p
import pybullet_data

from scipy.spatial.transform import Rotation as R

# Object Creation Functions
def create_box(half_extents=[0.5, 0.5, 0.5], position=[0, 0, 1], orientation=[0, 0, 0], mass=0.1, color=[1, 0, 0, 1]):
    """Creates a box in the PyBullet simulation."""
    collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    box_orientation = p.getQuaternionFromEuler(orientation)
    box_id = p.createMultiBody(mass, collision_shape, -1, position, box_orientation)
    p.changeVisualShape(box_id, -1, rgbaColor=color)
    return box_id

def create_sphere(radius=0.5, position=[0, 0, 1], mass=0.1, color=[0, 1, 0, 1]):
    """Creates a sphere in the PyBullet simulation."""
    collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
    sphere_id = p.createMultiBody(mass, collision_shape, -1, position)
    p.changeVisualShape(sphere_id, -1, rgbaColor=color)
    return sphere_id

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

def create_hammer(position=[0, 0, 1], orientation=[0, 0, 0], 
                  cylinder_radius=0.1, cylinder_height=1.0, box_half_extents=[0.3, 0.1, 0.1], 
                  cylinder_mass=0.05, box_mass=0.05, color=[0.5, 0.5, 0.5, 1]):
    """Create a hammer-like object in the simulation using scipy for quaternion rotations."""
    # Compute the center of mass of the hammer
    com = compute_center_of_mass(cylinder_mass=cylinder_mass, box_mass=box_mass, 
                                 cylinder_height=cylinder_height, box_half_extents=box_half_extents)

    # Convert Euler orientation to quaternion
    hammer_orientation = R.from_euler('xyz', orientation).as_quat()

    # Create cylinder (handle)
    collision_shape_handle = p.createCollisionShape(p.GEOM_CYLINDER, radius=cylinder_radius, height=cylinder_height)
    handle_position_local = np.array([0, 0, cylinder_height / 2]) - com  # Relative to COM
    handle_position_world = np.array(position) + R.from_quat(hammer_orientation).apply(handle_position_local)
    
    handle_id = p.createMultiBody(baseMass=cylinder_mass, baseCollisionShapeIndex=collision_shape_handle, 
                                  basePosition=handle_position_world, baseOrientation=hammer_orientation)

    # Create box (hammer head)
    collision_shape_head = p.createCollisionShape(p.GEOM_BOX, halfExtents=box_half_extents)
    head_position_local = np.array([0, 0, cylinder_height + box_half_extents[2]]) - com  # Relative to COM
    head_position_world = np.array(position) + R.from_quat(hammer_orientation).apply(head_position_local)
    
    head_id = p.createMultiBody(baseMass=box_mass, baseCollisionShapeIndex=collision_shape_head, 
                                basePosition=head_position_world, baseOrientation=hammer_orientation)

    # Create fixed joint to attach the head to the handle
    p.createConstraint(parentBodyUniqueId=handle_id, parentLinkIndex=-1,
                       childBodyUniqueId=head_id, childLinkIndex=-1,
                       jointType=p.JOINT_FIXED, 
                       jointAxis=[0, 0, 0], 
                       parentFramePosition=[0, 0, cylinder_height / 2], 
                       childFramePosition=[0, 0, -box_half_extents[2]])

    # Set color for both parts
    p.changeVisualShape(handle_id, -1, rgbaColor=color)
    p.changeVisualShape(head_id, -1, rgbaColor=color)
    
    return handle_id, head_id

# Utility Functions
def random_color():
    """Generates a random RGBA color."""
    return [random.uniform(0, 1) for _ in range(3)] + [1.0]  # RGB + Alpha

def compute_center_of_mass(cylinder_mass, box_mass, cylinder_height, box_half_extents):
    """Compute the center of mass of the hammer."""
    total_mass = cylinder_mass + box_mass
    # Cylinder COM (assuming its center)
    cylinder_com = np.array([0, 0, cylinder_height / 2])
    # Box COM (relative to the cylinder)
    box_com = np.array([0, 0, cylinder_height + box_half_extents[2]])

    # Compute the weighted average to find the COM of the whole hammer
    com = (cylinder_mass * cylinder_com + box_mass * box_com) / total_mass
    return com

# Main Simulation Code
if __name__ == '__main__':
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Set path to URDF files
    p.setGravity(0, 0, -9.8)

    # Create objects in the simulation
    create_plane()
    
    create_box(position=[0.0, 0.25, 0.2], half_extents=[0.02, 0.02, 0.02], mass=0.1)
    create_sphere(position=[-0.2, 0.25, 0.2], radius=0.02, mass=0.1)
    create_cylinder(position=[0.2, 0.25, 0.2], radius=0.015, height=0.16, mass=0.1)
    create_hammer(position=[0.0, 0.0, 0.2], orientation=[np.pi / 2, 0.0, 0.0], 
                  cylinder_radius=0.01, cylinder_height=0.12, box_half_extents=[0.05, 0.02, 0.0125])

    # Example: Load a mesh from an .obj file
    for i in range(8):
        if i != 5:
            mesh_file = f'assets/meshes/objects/blocks/{i}.obj'
            create_mesh(mesh_file=mesh_file, position=[0, -0.5, 0.2])

    # Run simulation
    for _ in range(10000):
        p.stepSimulation()
        time.sleep(1. / 240.)

    p.disconnect()
