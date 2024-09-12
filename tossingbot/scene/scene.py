import time
import pybullet as p
import pybullet_data

from tossingbot.robot import UR5Robotiq85
from tossingbot.scene.objects import create_box, create_sphere, create_capsule, create_cylinder, create_mesh

def setup_scene(p):
    """
    Set up the simulation scene in PyBullet.
    
    :param p: PyBullet instance
    :return: None
    """
    # Set search path for URDFs and load plane
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane_id = p.loadURDF("plane.urdf")  # Add a plane

    # Create different objects in the scene
    box_id = create_box(position=[0.4, -0.4, 0.02], half_extents=[0.02, 0.02, 0.02], mass=1)

    return plane_id, [box_id]

if __name__ == '__main__':
    # Connect to physics simulation
    physics_client_id = p.connect(p.GUI)  
    
    # Setup scene with gravity and objects
    p.setGravity(0, 0, -9.81)
    
    # Set up robot
    robot = UR5Robotiq85((0, 0.0, 0.0), (0, 0, 0), visualize_coordinate_frames=True)
    
    # Call the function to set up the scene
    plane_id, object_ids = setup_scene(p)
    
    # Main simulation loop
    while True:
        p.stepSimulation()
        time.sleep(1./240.)

    # Disconnect from the simulation when done
    p.disconnect()
