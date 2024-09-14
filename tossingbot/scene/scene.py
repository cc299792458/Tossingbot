import time
import random
import pybullet as p
import pybullet_data

from tossingbot.robot import UR5Robotiq85
from tossingbot.scene.objects import create_box, create_sphere, create_capsule, create_cylinder, create_mesh

def setup_workspace(length=0.3, width=0.4, position: list = [0.55, 0.0]):
    thickness = 0.01 
    height = 0.1
    box_ids = []
    box_ids.append(create_box(half_extents=[length / 2, width / 2, thickness / 2], position=[position[0], position[1], thickness / 2], mass=0))
    
    box_ids.append(create_box(half_extents=[thickness / 2, width / 2, height / 2], position=[position[0] - length / 2, position[1], height / 2 + thickness], mass=0))
    box_ids.append(create_box(half_extents=[length / 2, thickness / 2, height / 2], position=[position[0], position[1] - width / 2, height / 2 + thickness], mass=0))
    box_ids.append(create_box(half_extents=[thickness / 2, width / 2, height / 2], position=[position[0] + length / 2, position[1], height / 2 + thickness], mass=0))
    box_ids.append(create_box(half_extents=[length / 2, thickness / 2, height / 2], position=[position[0], position[1] + width / 2, height / 2 + thickness], mass=0))
    
    return box_ids

def setup_objects(workspace_length, workspace_width, workspace_position, n_object=1):
    margin = 0.05
    object_ids = []
    for i in range(n_object):
        # TODO: add multiple types of object
        # object_type = random. # 0 box, 1 sphere, ...
        x = random.uniform(workspace_position[0] - workspace_length / 2 + margin, workspace_position[0] + workspace_length / 2 - margin)
        y = random.uniform(workspace_position[1] - workspace_width / 2 + margin, workspace_position[1] + workspace_width / 2 - margin)
    object_ids.append(create_sphere(position=[x, y, 0.02], radius=0.02))

    return object_ids

def setup_boxes_with_dividers(length=0.25, width=0.15, height=0.2, n_rows=4, n_cols=3, position=[0.0, 0.0]):
    """
    Create a grid of hollow boxes separated by dividers using thin box walls.

    :param length: Length of each individual box (x-dimension).
    :param width: Width of each individual box (y-dimension).
    :param height: Height of the dividers (z-dimension).
    :param n_rows: Number of rows in the grid.
    :param n_cols: Number of columns in the grid.
    :param position: Center position [x, y] of the entire grid.
    :return: List of divider box IDs.
    """
    box_ids = []
    
    # Divider thickness (ensure it's less than both length and width of a single box)
    divider_thickness = 0.01  # 1 cm thick dividers

    # Calculate the adjusted total size of the grid
    total_length = n_cols * length - 2 * divider_thickness
    total_width = n_rows * width - 2 * divider_thickness

    # Calculate the center position of each box group
    # This ensures that the entire grid is centered around the specified position
    x_start = position[0] - total_length / 2
    y_start = position[1] - total_width / 2

    # Create outer walls
    # Top wall (along x-axis)
    box_ids.append(create_box(
        half_extents=[total_length / 2 + divider_thickness, divider_thickness / 2, height / 2],
        position=[position[0], y_start + total_width + divider_thickness / 2, height / 2],
        mass=0
    ))
    # Bottom wall (along x-axis)
    box_ids.append(create_box(
        half_extents=[total_length / 2 + divider_thickness, divider_thickness / 2, height / 2],
        position=[position[0], y_start - divider_thickness / 2, height / 2],
        mass=0
    ))
    # Left wall (along y-axis)
    box_ids.append(create_box(
        half_extents=[divider_thickness / 2, total_width / 2 + divider_thickness, height / 2],
        position=[x_start - divider_thickness / 2, position[1], height / 2],
        mass=0
    ))
    # Right wall (along y-axis)
    box_ids.append(create_box(
        half_extents=[divider_thickness / 2, total_width / 2 + divider_thickness, height / 2],
        position=[x_start + total_length + divider_thickness / 2, position[1], height / 2],
        mass=0
    ))

    # Create internal dividers
    # Horizontal dividers (along x-axis)
    for i in range(1, n_rows):
        y = y_start + i * width - divider_thickness
        box_ids.append(create_box(
            half_extents=[total_length / 2, divider_thickness / 2, height / 2],
            position=[position[0], y, height / 2],
            mass=0
        ))

    # Vertical dividers (along y-axis)
    for j in range(1, n_cols):
        x = x_start + j * length - divider_thickness
        box_ids.append(create_box(
            half_extents=[divider_thickness / 2, total_width / 2, height / 2],
            position=[x, position[1], height / 2],
            mass=0
        ))
    
    return box_ids

def setup_scene(workspace_length=0.3, workspace_width=0.4, workspace_position=[0.55, 0],
                box_length=0.25, box_width=0.15, box_height=0.2, box_n_rows=4, box_n_cols=3, box_position=[1.375, 0.0],
                n_object=1):
    """
    Set up the simulation scene in PyBullet.
    """
    p.setGravity(0, 0, -9.81)
    plane_id = p.loadURDF("plane.urdf")
    # Set up workspace
    workspace_ids = setup_workspace(length=workspace_length, 
                                    width=workspace_width, 
                                    position=workspace_position)
    # Set up boxes
    box_ids = setup_boxes_with_dividers(length=box_length, 
                                        width=box_width, 
                                        height=box_height, 
                                        n_rows=box_n_rows, 
                                        n_cols=box_n_cols, 
                                        position=box_position)   # The closes box's side is 1 meter awary from the base of robot.
    # Set up objects
    object_ids = setup_objects(workspace_length, workspace_width, workspace_position, n_object=n_object)
    # Set up camera
    
    # Set up robot
    robot = UR5Robotiq85((0, 0.0, 0.0), (0, 0, 0), visualize_coordinate_frames=True)

    return plane_id, workspace_ids, box_ids, object_ids, robot

if __name__ == '__main__':
    # Connect to physics simulation
    physics_client_id = p.connect(p.GUI)  
    # Set search path for URDFs and load plane
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Call the function to set up the scene
    plane_id, workspace_ids, box_ids, object_ids, robot = setup_scene()
    
    # Main simulation loop
    while True:
        p.stepSimulation()
        time.sleep(1./240.)

    # Disconnect from the simulation when done
    p.disconnect()
