import time
import pybullet as p
import pybullet_data

def create_box(position=[0, 0, 1], orientation=[0, 0, 0], half_extents=[0.5, 0.5, 0.5], mass=1):
    """
    Function to create a box in the PyBullet simulation.

    :param position: List of 3 elements specifying the box position (x, y, z)
    :param orientation: List of 3 elements specifying the box orientation in Euler angles (x, y, z)
    :param half_extents: List of 3 elements specifying the size of the box (half lengths along x, y, z)
    :param mass: Mass of the box. Set to 0 to create a fixed (static) object.
    :return: boxId of the created box
    """
    collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    box_orientation = p.getQuaternionFromEuler(orientation)
    boxId = p.createMultiBody(mass, collision_shape, -1, position, box_orientation)
    return boxId

def create_sphere(radius=0.5, position=[0, 0, 1], mass=1):
    """
    Function to create a sphere in the PyBullet simulation.

    :param radius: Radius of the sphere
    :param position: List of 3 elements specifying the sphere position (x, y, z)
    :param mass: Mass of the sphere. Set to 0 to create a fixed (static) object.
    :return: sphereId of the created sphere
    """
    collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
    sphereId = p.createMultiBody(mass, collision_shape, -1, position)
    return sphereId

def create_capsule(radius=0.2, height=1.0, position=[0, 0, 1], orientation=[0, 0, 0], mass=1):
    """
    Function to create a capsule in the PyBullet simulation.

    :param radius: Radius of the capsule
    :param height: Height of the capsule
    :param position: List of 3 elements specifying the capsule position (x, y, z)
    :param orientation: List of 3 elements specifying the capsule orientation in Euler angles (x, y, z)
    :param mass: Mass of the capsule. Set to 0 to create a fixed (static) object.
    :return: capsuleId of the created capsule
    """
    collision_shape = p.createCollisionShape(p.GEOM_CAPSULE, radius=radius, height=height)
    capsule_orientation = p.getQuaternionFromEuler(orientation)
    capsuleId = p.createMultiBody(mass, collision_shape, -1, position, capsule_orientation)
    return capsuleId

def create_cylinder(radius=0.3, height=1.0, position=[0, 0, 1], orientation=[0, 0, 0], mass=1):
    """
    Function to create a cylinder in the PyBullet simulation.

    :param radius: Radius of the cylinder
    :param height: Height of the cylinder
    :param position: List of 3 elements specifying the cylinder position (x, y, z)
    :param orientation: List of 3 elements specifying the cylinder orientation in Euler angles (x, y, z)
    :param mass: Mass of the cylinder. Set to 0 to create a fixed (static) object.
    :return: cylinderId of the created cylinder
    """
    collision_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
    cylinder_orientation = p.getQuaternionFromEuler(orientation)
    cylinderId = p.createMultiBody(mass, collision_shape, -1, position, cylinder_orientation)
    return cylinderId

def create_plane(position=[0, 0, 0], orientation=[0, 0, 0], mass=0):
    """
    Function to create a plane in the PyBullet simulation.

    :param position: List of 3 elements specifying the plane position (x, y, z)
    :param orientation: List of 3 elements specifying the plane orientation in Euler angles (x, y, z)
    :param mass: Mass of the plane. Typically set to 0 as a plane is generally fixed.
    :return: planeId of the created plane
    """
    planeId = p.loadURDF("plane.urdf", position, p.getQuaternionFromEuler(orientation))
    return planeId

def create_mesh(mesh_file, position=[0, 0, 1], orientation=[0, 0, 0], mass=1):
    """
    Create a mesh object from a given .obj or .stl file.

    :param mesh_file: Path to the mesh file (OBJ or STL format)
    :param position: Position of the mesh object
    :param orientation: Orientation of the mesh in Euler angles
    :param mass: Mass of the object. Set to 0 for a fixed mesh.
    :return: meshId of the created object
    """
    collision_shape = p.createCollisionShape(p.GEOM_MESH, fileName=mesh_file)
    mesh_orientation = p.getQuaternionFromEuler(orientation)
    meshId = p.createMultiBody(mass, collision_shape, -1, position, mesh_orientation)
    return meshId

# Example usage
if __name__ == '__main__':
    # Initialize PyBullet simulation
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Set path to URDF files
    p.setGravity(0, 0, -9.8)

    # Create different objects, including fixed ones by setting mass=0
    create_plane()  # Fixed plane
    create_box(position=[1, 1, 1], half_extents=[0.5, 0.5, 0.5], mass=0)  # Fixed box
    create_sphere(position=[-1, 1, 1], radius=0.5, mass=1)  # Movable sphere
    create_capsule(position=[2, 1, 1], radius=0.2, height=1.0, mass=1)  # Movable capsule
    create_cylinder(position=[-2, 1, 1], radius=0.3, height=1.0, mass=0)  # Fixed cylinder

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
