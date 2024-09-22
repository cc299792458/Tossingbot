import time
import random
import pybullet as p
import pybullet_data

from tossingbot.envs.pybullet.robot import UR5Robotiq85
from tossingbot.envs.pybullet.utils.objects_utils import create_box, create_sphere

import pybullet as p

class BaseScene:
    def __init__(self, timestep=1/240, gravity=-9.81, use_gui=True):
        """
        Initialize the base simulation scene.

        Args:
            timestep (float): Time step for the simulation.
            gravity (float): Gravity applied to the scene.
            use_gui (bool): Whether to run the simulation in GUI mode or headless mode.
        """
        self.timestep = timestep
        self.gravity = gravity
        self.use_gui = use_gui

        # Initialize the simulation
        self.start_simulation()

        # Load the ground plane and scene-specific elements
        self.load_ground_plane()
        self.load_scene()
        self.load_robot()

        # Reset scene to its initial state
        self.reset()
    
    ############### Initialization ###############
    def start_simulation(self):
        """
        Starts the PyBullet simulation in either GUI or headless mode.
        """
        self.physics_client = p.connect(p.GUI if self.use_gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, self.gravity)
        p.setTimeStep(self.timestep)

    def load_ground_plane(self):
        """
        Loads a flat ground plane into the simulation.
        """
        self.plane_id = p.loadURDF("plane.urdf")

    def load_scene(self):
        """
        Abstract method for loading the scene configuration.
        Should be implemented by subclasses based on scene setup.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def load_robot(self):
        """
        Abstract method for loading the robot. 
        Should be implemented by subclasses based on robot configuration.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    ############### Reset ###############
    def reset(self):
        """
        Resets the robot and objects in the scene. Calls reset methods that should be implemented by subclasses.
        """
        self.reset_robot()
        self.reset_objects()

    def reset_robot(self):
        """
        Abstract method to reset the robot to its initial state.
        Should be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def reset_objects(self):
        """
        Abstract method to reset or reload objects in the scene.
        Should be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    ############### Step ###############
    def step(self, action):
        """
        Takes an action in the environment and progresses the simulation.
        
        Args:
            action (list or array): Action to be performed in the environment.
            
        Returns:
            tuple: Typically returns (next_state, reward, terminated, truncated, info) for reinforcement learning tasks.
        """
        self.step_simulation()

    def step_simulation(self):
        """
        Advances the simulation by one time step.
        """
        p.stepSimulation()

        # Only add a time delay if the GUI is enabled to simulate real-time behavior
        if self.use_gui:
            time.sleep(self.timestep)

    ############### Misc ###############
    def close_simulation(self):
        """
        Disconnects from the PyBullet simulation.
        """
        if self.physics_client is not None:
            p.disconnect(self.physics_client)

# if __name__ == '__main__':
#     # Connect to physics simulation
#     physics_client_id = p.connect(p.GUI)
    
#     # Set search path for URDFs
#     p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
#     # Call the function to set up the scene
#     plane_id, workspace_ids, box_ids, object_ids, robot = setup_scene()
    
#     # Main simulation loop
#     while True:
#         p.stepSimulation()
#         time.sleep(1./240.)

#     # Disconnect from the simulation when done
#     p.disconnect()
