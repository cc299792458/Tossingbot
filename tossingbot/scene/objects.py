import time
import random
import pybullet as p
import pybullet_data

def create_box(half_extents=[0.5, 0.5, 0.5], position=[0, 0, 1], orientation=[0, 0, 0], mass=1, color=[1, 0, 0, 1]):
    """
    Function to create a box in the PyBullet simulation.

    :param color: List of 4 elements specifying RGBA color
    """
    collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    box_orientation = p.getQuaternionFromEuler(orientation)
    box_id = p.createMultiBody(mass, collision_shape, -1, position, box_orientation)
    
    # Set the color of the box
    p.changeVisualShape(box_id, -1, rgbaColor=color)
    
    return box_id

def create_sphere(radius=0.5, position=[0, 0, 1], mass=1, color=[0, 1, 0, 1]):
    """
    Function to create a sphere in the PyBullet simulation.

    :param color: List of 4 elements specifying RGBA color
    """
    collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
    sphere_id = p.createMultiBody(mass, collision_shape, -1, position)
    
    # Set the color of the sphere
    p.changeVisualShape(sphere_id, -1, rgbaColor=color)
    
    return sphere_id

def create_capsule(radius=0.2, height=1.0, position=[0, 0, 1], orientation=[0, 0, 0], mass=1, color=[0, 0, 1, 1]):
    """
    Function to create a capsule in the PyBullet simulation.

    :param color: List of 4 elements specifying RGBA color
    """
    collision_shape = p.createCollisionShape(p.GEOM_CAPSULE, radius=radius, height=height)
    capsule_orientation = p.getQuaternionFromEuler(orientation)
    capsule_id = p.createMultiBody(mass, collision_shape, -1, position, capsule_orientation)
    
    # Set the color of the capsule
    p.changeVisualShape(capsule_id, -1, rgbaColor=color)
    
    return capsule_id

def create_cylinder(radius=0.3, height=1.0, position=[0, 0, 1], orientation=[0, 0, 0], mass=1, color=[1, 1, 0, 1]):
    """
    Function to create a cylinder in the PyBullet simulation.

    :param color: List of 4 elements specifying RGBA color
    """
    collision_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
    cylinder_orientation = p.getQuaternionFromEuler(orientation)
    cylinder_id = p.createMultiBody(mass, collision_shape, -1, position, cylinder_orientation)
    
    # Set the color of the cylinder
    p.changeVisualShape(cylinder_id, -1, rgbaColor=color)
    
    return cylinder_id

def create_mesh(mesh_file, position=[0, 0, 1], orientation=[0, 0, 0], mass=1, color=[1, 0, 1, 1]):
    """
    Create a mesh object from a given .obj or .stl file.

    :param color: List of 4 elements specifying RGBA color
    """
    collision_shape = p.createCollisionShape(p.GEOM_MESH, fileName=mesh_file)
    mesh_orientation = p.getQuaternionFromEuler(orientation)
    mesh_id = p.createMultiBody(mass, collision_shape, -1, position, mesh_orientation)
    
    # Set the color of the mesh
    p.changeVisualShape(mesh_id, -1, rgbaColor=color)
    
    return mesh_id

def create_plane(position=[0, 0, 0], orientation=[0, 0, 0], mass=0, color=[1, 1, 1, 1]):
    """
    Function to create a plane in the PyBullet simulation.

    :param color: List of 4 elements specifying RGBA color
    """
    plane_id = p.loadURDF("plane.urdf", position, p.getQuaternionFromEuler(orientation))
    
    # Set the color of the plane
    p.changeVisualShape(plane_id, -1, rgbaColor=color)
    
    return plane_id

def random_color():
    """
    Generate a random RGBA color.

    :return: A list of 4 elements representing the color in RGBA format.
    """
    r = random.uniform(0, 1)  # Red component
    g = random.uniform(0, 1)  # Green component
    b = random.uniform(0, 1)  # Blue component
    a = 1.0  # Alpha (fully opaque)
    return [r, g, b, a]

# Example usage
if __name__ == '__main__':
    # Initialize PyBullet simulation
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Set path to URDF files
    p.setGravity(0, 0, -9.8)

    # Create different objects, including fixed ones by setting mass=0
    create_plane()  # Fixed plane
    create_box(position=[1, 1, 0.5], half_extents=[0.5, 0.5, 0.5], mass=0)  # Fixed box
    create_sphere(position=[-1, 1, 1], radius=0.5, mass=1)  # Movable sphere
    create_capsule(position=[2, 1, 1], radius=0.2, height=1.0, mass=1)  # Movable capsule
    create_cylinder(position=[-2, 1, 0.5], radius=0.3, height=1.0, mass=0)  # Fixed cylinder

    # Example: Load a mesh from an .obj file
    for i in range(8):
        if i != 5:
            mesh_file = f'assets/meshes/objects/blocks/{i}.obj'
            mesh_id = create_mesh(mesh_file=mesh_file, position=[0, -1, 0.3])  # Replace with path to mesh file

    # Run simulation
    for i in range(10000):
        p.stepSimulation()
        time.sleep(1. / 240.)

    p.disconnect()
