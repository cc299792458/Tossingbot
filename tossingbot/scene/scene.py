import time
import random
import pybullet as p
import pybullet_data

from tossingbot.robot import UR5Robotiq85
from tossingbot.scene.objects import create_box, create_sphere

def setup_workspace(length=0.3, width=0.4, position=[0.55, 0.0]):
    """
    Create a workspace with walls in the simulation.
    
    :param length: Length of the workspace.
    :param width: Width of the workspace.
    :param position: Center position [x, y] of the workspace.
    :return: List of box IDs for the workspace.
    """
    thickness = 0.01
    height = 0.1
    color = [0.8, 0.8, 0.8, 1.0]
    box_ids = []
    
    # Base of the workspace
    box_ids.append(create_box(half_extents=[length / 2, width / 2, thickness / 2], 
                              position=[position[0], position[1], thickness / 2], 
                              mass=0, color=color))
    
    # Walls of the workspace
    box_ids.append(create_box(half_extents=[thickness / 2, width / 2, height / 2], 
                              position=[position[0] - length / 2, position[1], height / 2 + thickness], 
                              mass=0, color=color))
    box_ids.append(create_box(half_extents=[length / 2, thickness / 2, height / 2], 
                              position=[position[0], position[1] - width / 2, height / 2 + thickness], 
                              mass=0, color=color))
    box_ids.append(create_box(half_extents=[thickness / 2, width / 2, height / 2], 
                              position=[position[0] + length / 2, position[1], height / 2 + thickness], 
                              mass=0, color=color))
    box_ids.append(create_box(half_extents=[length / 2, thickness / 2, height / 2], 
                              position=[position[0], position[1] + width / 2, height / 2 + thickness], 
                              mass=0, color=color))
    
    return box_ids

def setup_objects(workspace_length, workspace_width, workspace_position, n_object=1):
    """
    Randomly place objects in the workspace.
    
    :param workspace_length: Length of the workspace.
    :param workspace_width: Width of the workspace.
    :param workspace_position: Center position [x, y] of the workspace.
    :param n_object: Number of objects to place in the workspace.
    :return: List of object IDs.
    """
    margin = 0.1
    object_ids = []
    
    for i in range(n_object):
        x = random.uniform(workspace_position[0] - workspace_length / 2 + margin, 
                           workspace_position[0] + workspace_length / 2 - margin)
        y = random.uniform(workspace_position[1] - workspace_width / 2 + margin, 
                           workspace_position[1] + workspace_width / 2 - margin)
        
        # Create a sphere for now (extendable to other object types)
        object_ids.append(create_sphere(position=[x, y, 0.02], radius=0.02, color=[1, 0, 0, 1]))

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
    divider_thickness = 0.01  # 1 cm thick dividers
    color = [0.545, 0.271, 0.075, 1.0]

    # Calculate the adjusted total size of the grid
    total_length = n_cols * length - 2 * divider_thickness
    total_width = n_rows * width - 2 * divider_thickness

    # Calculate the start position to center the grid around the specified position
    x_start = position[0] - total_length / 2
    y_start = position[1] - total_width / 2

    # Create outer walls
    box_ids.append(create_box(half_extents=[total_length / 2 + divider_thickness, divider_thickness / 2, height / 2],
                              position=[position[0], y_start + total_width + divider_thickness / 2, height / 2],
                              mass=0, color=color))
    box_ids.append(create_box(half_extents=[total_length / 2 + divider_thickness, divider_thickness / 2, height / 2],
                              position=[position[0], y_start - divider_thickness / 2, height / 2],
                              mass=0, color=color))
    box_ids.append(create_box(half_extents=[divider_thickness / 2, total_width / 2 + divider_thickness, height / 2],
                              position=[x_start - divider_thickness / 2, position[1], height / 2],
                              mass=0, color=color))
    box_ids.append(create_box(half_extents=[divider_thickness / 2, total_width / 2 + divider_thickness, height / 2],
                              position=[x_start + total_length + divider_thickness / 2, position[1], height / 2],
                              mass=0, color=color))

    # Create internal dividers
    for i in range(1, n_rows):
        y = y_start + i * width - divider_thickness
        box_ids.append(create_box(half_extents=[total_length / 2, divider_thickness / 2, height / 2],
                                  position=[position[0], y, height / 2],
                                  mass=0, color=color))

    for j in range(1, n_cols):
        x = x_start + j * length - divider_thickness
        box_ids.append(create_box(half_extents=[divider_thickness / 2, total_width / 2, height / 2],
                                  position=[x, position[1], height / 2],
                                  mass=0, color=color))
    
    return box_ids

def setup_scene(workspace_length=0.3, workspace_width=0.4, workspace_position=[0.55, 0],
                box_length=0.25, box_width=0.15, box_height=0.2, box_n_rows=4, box_n_cols=3, box_position=[1.375, 0.0],
                n_object=1):
    """
    Set up the simulation scene in PyBullet.
    
    :param workspace_length: Length of the workspace.
    :param workspace_width: Width of the workspace.
    :param workspace_position: Center position [x, y] of the workspace.
    :param box_length: Length of each box in the grid.
    :param box_width: Width of each box in the grid.
    :param box_height: Height of each divider in the grid.
    :param box_n_rows: Number of rows in the box grid.
    :param box_n_cols: Number of columns in the box grid.
    :param box_position: Center position of the entire grid of boxes.
    :param n_object: Number of objects to place in the workspace.
    :return: IDs of all elements in the simulation (plane, workspace, boxes, objects, robot).
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
                                        position=box_position)
    
    # Set up objects
    object_ids = setup_objects(workspace_length, workspace_width, workspace_position, n_object=n_object)
    
    # Set up robot
    robot = UR5Robotiq85((0, 0.0, 0.0), (0.0, 0.0, 0.0), visualize_coordinate_frames=True)

    return plane_id, workspace_ids, box_ids, object_ids, robot

if __name__ == '__main__':
    # Connect to physics simulation
    physics_client_id = p.connect(p.GUI)
    
    # Set search path for URDFs
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Call the function to set up the scene
    plane_id, workspace_ids, box_ids, object_ids, robot = setup_scene()
    
    # Main simulation loop
    while True:
        p.stepSimulation()
        time.sleep(1./240.)

    # Disconnect from the simulation when done
    p.disconnect()
