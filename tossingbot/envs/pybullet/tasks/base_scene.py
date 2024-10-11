import time
import pybullet as p
import pybullet_data

import pybullet as p

class BaseScene:
    def __init__(self, timestep=1./240., control_timestep=1./60., gravity=-9.81, use_gui=True):
        """
        Initialize the base simulation scene.

        Args:
            timestep (float): Time step for the simulation.
            control_timestep (float): Control time step for the robot.
            gravity (float): Gravity applied to the scene.
            use_gui (bool): Whether to run the simulation in GUI mode or headless mode.
        """
        self.timestep = timestep
        self.control_timestep = control_timestep
        self.sim_step_per_ctrl_step = int(control_timestep // timestep)
        
        self.gravity = gravity
        self.use_gui = use_gui

        # Initialize the simulation
        self.start_simulation()

        # Load the scene and the robot
        self.load_scene()
        self.load_robot()

        # Reset scene and robot to the initial state
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
        self.reset_robot(init=True)
        self.reset_objects(init=True)
        self.reset_task(init=True)

        obs = self.get_observation()
        info = self.get_info()

        return obs, info

    def reset_robot(self, init=False):
        """
        Abstract method to reset the robot to its initial state.
        Should be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def reset_objects(self, init=False):
        """
        Abstract method to reset or reload objects in the scene.
        Should be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def reset_task(self, init=False):
        """
        Abstract method to reset the task-related parameters in the scene.
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
            tuple: Typically returns (next_obs, reward, terminated, truncated, info) for reinforcement learning tasks.
        """
        completed_and_static = False    # is action completed and are objects static
        self.pre_simulation_process(action)
        while not completed_and_static:
            completed_and_static = self.pre_control_step()
            for _ in range(self.sim_step_per_ctrl_step):
                self.pre_simulation_step()
                self.simulation_step()
                self.post_simulation_step()
            self.post_control_step()
        self.post_simulation_process(completed_and_static)

        next_obs = self.get_observation()
        reward = self.get_reward()
        terminated = self.is_terminated()
        truncated = self.is_truncated()
        info = self.get_info()

        return next_obs, reward, terminated, truncated, info

    def simulation_step(self):
        """
        Advances the simulation by one time step.
        """
        p.stepSimulation()

        # Only add a time delay if the GUI is enabled to simulate real-time behavior
        if self.use_gui:
            time.sleep(self.timestep)

    def pre_simulation_process(self, action):
        pass

    def post_simulation_process(self, completed_and_static):
        pass

    def pre_control_step(self):
        pass

    def post_control_step(self):
        pass

    def pre_simulation_step(self):
        pass

    def post_simulation_step(self):
        pass

    def get_observation(self):
        raise NotImplementedError

    def get_reward(self):
        raise NotImplementedError

    def is_terminated(self):
        raise NotImplementedError

    def is_truncated(self):
        raise NotImplementedError

    def get_info(self):
        return {}

    ############### Visualzation ###############

    ############### Misc ###############
    def close_simulation(self):
        """
        Disconnects from the PyBullet simulation.
        """
        if self.physics_client is not None:
            p.disconnect(self.physics_client)