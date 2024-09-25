import random
import pybullet as p

from tossingbot.utils.misc_utils import set_seed
from tossingbot.envs.pybullet.robot import UR5Robotiq85, Panda
from tossingbot.envs.pybullet.tasks.base_scene import BaseScene
from tossingbot.envs.pybullet.utils.objects_utils import (
    create_sphere,
    create_box, 
    create_cylinder, 
    create_capsule,
    random_color,
)
from tossingbot.envs.pybullet.utils.camera_utils import (
    visualize_camera,
    capture_rgbd_image, 
    depth_to_point_cloud_with_color, 
    point_cloud_to_height_map,
    initialize_visual_plots,
    plot_rgb_pointcloud_heightmap,
    compute_camera_fov_at_height,
)

class TossObjects(BaseScene):
    def __init__(self, 
             timestep=1/240, 
             gravity=-9.81, 
             use_gui=True, 
             visualize_config=None,
             scene_config=None, 
             robot_config=None, 
             objects_config=None, 
             camera_config=None,
        ):
        """
        Initialize the TossObjects scene.

        Args:
            timestep (float): Time step for the simulation.
            gravity (float): Gravity applied to the scene.
            use_gui (bool): Whether to run the simulation in GUI mode or headless mode.
            visualize_config (dict): Configuration for visualization options.
            scene_config (dict): Configuration for the scene setup (e.g., dimensions, objects).
            robot_config (dict): Configuration for the robot setup (e.g., robot type, starting position).
            objects_config (dict): Configuration for the objects in the scene.
            camera_config (dict): Configuration for the camera setup.
        """
        # Default visualize configuration
        default_visualize_config = {
            "visualize_coordinate_frames": True,
            "visualize_camera": True,
            "visualize_visual_plots": True,
            "visualize_target": True,
        }
        if visualize_config is not None:
            default_visualize_config.update(visualize_config)
        self.visualize_config = default_visualize_config
        # Default scene configuration
        default_scene_config = {
            "workspace_length": 0.3,
            "workspace_width": 0.4,
            "workspace_position": [0.3, 0],
            "box_length": 0.25,
            "box_width": 0.15,
            "box_height": 0.1,
            "box_n_rows": 3,
            "box_n_cols": 3,
            "box_position": [1.0, 0.0],
        }
        default_scene_config.update({
            "workspace_xlim": [default_scene_config['workspace_position'][0] - default_scene_config['workspace_length'] / 2, 
                            default_scene_config['workspace_position'][0] + default_scene_config['workspace_length'] / 2],
            "workspace_ylim": [default_scene_config['workspace_position'][1] - default_scene_config['workspace_width'] / 2, 
                            default_scene_config['workspace_position'][1] + default_scene_config['workspace_width'] / 2],
            "workspace_zlim": [0.01, 0.4]
        })
        if scene_config is not None:
            default_scene_config.update(scene_config)
        self.scene_config = default_scene_config

        # Default robot configuration
        default_robot_config = {
            "base_position": [0.0, 0.0, 0.0],
            "base_orientation": [0.0, 0.0, 0.0],
            "robot_type": 'panda',
            "visualize_coordinate_frames": use_gui and self.visualize_config['visualize_coordinate_frames'],
        }
        if robot_config is not None:
            default_robot_config.update(robot_config)
        self.robot_config = default_robot_config

        # Default objects configuration
        default_objects_config = {
            "n_object": 1,
            "object_types": ['sphere', 'box', 'capsule', 'cylindar'],
        }
        if objects_config is not None:
            default_objects_config.update(objects_config)
        self.objects_config = default_objects_config

        # Default camera configuration
        default_camera_config = {
            "cam_target_pos": [*self.scene_config['workspace_position'], 0.0],
            "cam_distance": self.scene_config['workspace_length'] / 2,
            "width": int(self.scene_config["workspace_width"] * 200),
            "height": int(self.scene_config["workspace_length"] * 200),
            "cam_yaw": -90,
            "cam_pitch": -90,
            "cam_roll": 0,
            "fov": 90,
            "aspect": 1.33,
            "near": 0.01,
            "far": 10.0,
        }
        if camera_config is not None:
            default_camera_config.update(camera_config)
        cam_view_xlim, cam_view_ylim = compute_camera_fov_at_height(
            cam_target_pos=default_camera_config['cam_target_pos'],
            cam_distance=default_camera_config['cam_distance'],
            cam_yaw=default_camera_config['cam_yaw'],
            cam_pitch=default_camera_config['cam_pitch'],
            cam_roll=default_camera_config['cam_roll'],
            fov=default_camera_config['fov'],
            aspect=default_camera_config['aspect'],
            target_height=0.0,
            ) 
        default_camera_config.update({
            "cam_view_xlim": cam_view_xlim,
            "cam_view_ylim": cam_view_ylim,
        })
        self.camera_config = default_camera_config

        super().__init__(timestep=timestep, gravity=gravity, use_gui=use_gui)

        if use_gui and self.visualize_config['visualize_camera']:
            visualize_camera(
                cam_target_pos=self.camera_config['cam_target_pos'],
                cam_distance=self.camera_config['cam_distance'],
                cam_yaw=self.camera_config['cam_yaw'],
                cam_pitch=self.camera_config['cam_pitch'],
                cam_roll=self.camera_config['cam_roll'],
                fov=self.camera_config['fov'],
                aspect=self.camera_config['aspect'],
                near=self.camera_config['near'],
                far=self.camera_config['far'],
            )
        if use_gui and self.visualize_config['visualize_visual_plots']:
            # Initialize the plots for real-time rgbd image, pointcloud and height map display
            self.visual_plot_fig, self.visual_plot_axes = initialize_visual_plots()

    ############### Initialization ###############
    def load_scene(self):
        """
        Load the tossing object scene in PyBullet.
        """
        # Load workspace and boxes
        self.load_workspace(
            length=self.scene_config['workspace_length'],
            width=self.scene_config['workspace_width'],
            position=self.scene_config['workspace_position'],
        )
        self.load_boxes_with_dividers(
            length=self.scene_config['box_length'],
            width=self.scene_config['box_width'],
            height=self.scene_config['box_height'],
            n_rows=self.scene_config['box_n_rows'],
            n_cols=self.scene_config['box_n_cols'],
            position=self.scene_config['box_position'],
        )

    def load_workspace(self, length=0.3, width=0.4, position=[0.3, 0.0]):
        """
        Create a workspace with walls in the simulation.

        Args:
            length (float): Length of the workspace.
            width (float): Width of the workspace.
            position (list): Center position [x, y] of the workspace.
        """
        thickness = 0.01
        height = 0.03
        color = [0.8, 0.8, 0.8, 1.0]
        self.workspace_ids = []
        
        # Base of the workspace
        self.workspace_ids.append(create_box(half_extents=[length / 2, width / 2, thickness / 2], 
                                position=[position[0], position[1], thickness / 2], 
                                mass=0, color=color))
        
        # Walls of the workspace
        self.workspace_ids.append(create_box(half_extents=[thickness / 2, width / 2, height / 2], 
                                position=[position[0] - length / 2, position[1], height / 2 + thickness], 
                                mass=0, color=color))
        self.workspace_ids.append(create_box(half_extents=[length / 2, thickness / 2, height / 2], 
                                position=[position[0], position[1] - width / 2, height / 2 + thickness], 
                                mass=0, color=color))
        self.workspace_ids.append(create_box(half_extents=[thickness / 2, width / 2, height / 2], 
                                position=[position[0] + length / 2, position[1], height / 2 + thickness], 
                                mass=0, color=color))
        self.workspace_ids.append(create_box(half_extents=[length / 2, thickness / 2, height / 2], 
                                position=[position[0], position[1] + width / 2, height / 2 + thickness], 
                                mass=0, color=color))

    def load_boxes_with_dividers(self, length=0.25, width=0.15, height=0.2, n_rows=4, n_cols=3, position=[0.0, 0.0]):
        """
        Create a grid of hollow boxes separated by dividers using thin box walls.

        Args:
            length (float): Length of each individual box (x-dimension).
            width (float): Width of each individual box (y-dimension).
            height (float): Height of the dividers (z-dimension).
            n_rows (int): Number of rows in the grid.
            n_cols (int): Number of columns in the grid.
            position (list): Center position [x, y] of the entire grid.
        """
        self.box_ids = []
        divider_thickness = 0.01  # 1 cm thick dividers
        color = [0.545, 0.271, 0.075, 1.0]

        # Calculate the adjusted total size of the grid
        total_length = n_cols * length - 2 * divider_thickness
        total_width = n_rows * width - 2 * divider_thickness

        # Calculate the start position to center the grid around the specified position
        x_start = position[0] - total_length / 2
        y_start = position[1] - total_width / 2

        # Create outer walls
        self.box_ids.append(create_box(half_extents=[total_length / 2 + divider_thickness, divider_thickness / 2, height / 2],
                                position=[position[0], y_start + total_width + divider_thickness / 2, height / 2],
                                mass=0, color=color))
        self.box_ids.append(create_box(half_extents=[total_length / 2 + divider_thickness, divider_thickness / 2, height / 2],
                                position=[position[0], y_start - divider_thickness / 2, height / 2],
                                mass=0, color=color))
        self.box_ids.append(create_box(half_extents=[divider_thickness / 2, total_width / 2 + divider_thickness, height / 2],
                                position=[x_start - divider_thickness / 2, position[1], height / 2],
                                mass=0, color=color))
        self.box_ids.append(create_box(half_extents=[divider_thickness / 2, total_width / 2 + divider_thickness, height / 2],
                                position=[x_start + total_length + divider_thickness / 2, position[1], height / 2],
                                mass=0, color=color))

        # Create internal dividers
        for i in range(1, n_rows):
            y = y_start + i * width - divider_thickness
            self.box_ids.append(create_box(half_extents=[total_length / 2, divider_thickness / 2, height / 2],
                                    position=[position[0], y, height / 2],
                                    mass=0, color=color))

        for j in range(1, n_cols):
            x = x_start + j * length - divider_thickness
            self.box_ids.append(create_box(half_extents=[divider_thickness / 2, total_width / 2, height / 2],
                                    position=[x, position[1], height / 2],
                                    mass=0, color=color))

    def load_robot(self):
        """
        Load the robot based on the robot configuration.
        """
        if self.robot_config['robot_type'] == 'ur5_robotiq':
            self.robot = UR5Robotiq85(
                base_position=self.robot_config['base_position'],
                base_orientation=self.robot_config['base_orientation'],
                visualize_coordinate_frames=self.robot_config['visualize_coordinate_frames'],
            )
        elif self.robot_config['robot_type'] == 'panda':
            self.robot = Panda(
                base_position=self.robot_config['base_position'],
                base_orientation=self.robot_config['base_orientation'],
                visualize_coordinate_frames=self.robot_config['visualize_coordinate_frames'],
            )

    ############### Reset ###############
    def reset_robot(self):
        """
        Reset the robot's position.
        """
        self.robot.reset()

    def reset_objects(self, margin=0.1):
        """
        Randomly place objects in the workspace.
        """
        # Remove existing objects if they exist
        if hasattr(self, 'object_ids'):
            for object_id in self.object_ids:
                p.removeBody(object_id)

        self.object_ids = []

        assert self.objects_config['n_object'] <= 3, "Too many objects"

        for i in range(self.objects_config['n_object']):
            object_type = i
            # object_type = random.randint(0, len(self.objects_config['object_types']) - 1)
            x = random.uniform(
                self.scene_config['workspace_position'][0] - self.scene_config['workspace_length'] / 2 + margin,
                self.scene_config['workspace_position'][0] + self.scene_config['workspace_length'] / 2 - margin,
            )
            y = random.uniform(
                self.scene_config['workspace_position'][1] - self.scene_config['workspace_width'] / 2 + margin,
                self.scene_config['workspace_position'][1] + self.scene_config['workspace_width'] / 2 - margin,
            )
            if self.objects_config['object_types'][object_type] == 'sphere':
                object_id = create_sphere(radius=0.02, position=[x, y, 0.05], color=[1, 0, 0, 1])
            elif self.objects_config['object_types'][object_type] == 'box':
                object_id = create_box(half_extents=[0.02, 0.02, 0.02], position=[x, y, 0.05], color=[0, 1, 0, 1])
            elif self.objects_config['object_types'][object_type] == 'cylindar':
                object_id = create_cylinder(radius=0.02, height=0.02, position=[x, y, 0.05], color=[0, 0, 1, 1])
            elif self.objects_config['object_types'][object_type] == 'capsule':
                object_id = create_capsule(radius=0.02, height=0.02, position=[x, y, 0.05], color=random_color())
            
            self.object_ids.append(object_id)

    ############### Step ###############
    def pre_simulation_step(self):
        self.select_target()
        if self.use_gui and self.visualize_config['visualize_target']:
            self.visualize_target()

    def post_simulation_step(self):
        if self.use_gui and self.visualize_config['visualize_visual_plots']:
            rgb_img, depth_img, point_cloud, colors, \
            color_heightmap, depth_heightmap, \
            color_heightmap_normalized, depth_heightmap_normalized = self.get_visual_observation()
            plot_rgb_pointcloud_heightmap(
                rgb_img=rgb_img, 
                point_cloud=point_cloud, 
                colors=colors,
                depth_heightmap=depth_heightmap,
                fig=self.visual_plot_fig,
                axes=self.visual_plot_axes,
                xlim=self.camera_config['cam_view_xlim'],
                ylim=self.camera_config['cam_view_ylim'],
                zlim=self.scene_config['workspace_zlim'],
            )

    def select_target(self):
        # Get scene configuration
        rows = self.scene_config['box_n_rows']
        cols = self.scene_config['box_n_cols']
        n_boxes = rows * cols

        # Randomly select a target box
        target_box = random.randint(0, n_boxes - 1)
        target_row = target_box // cols
        target_col = target_box % cols

        # Get box dimensions and position
        length = self.scene_config['box_length']
        width = self.scene_config['box_width']
        height = self.scene_config['box_height']
        central_position = self.scene_config['box_position']

        # Calculate the offset for the selected box
        row_offset = target_row - (rows - 1) / 2
        col_offset = target_col - (cols - 1) / 2

        # Set target position
        self.target_position = [
            central_position[0] + row_offset * length,
            central_position[1] + col_offset * width,
            height
        ]

    def get_observation(self):
        rgb_img, depth_img, point_cloud, colors, \
        color_heightmap, depth_heightmap, \
        color_heightmap_normalized, depth_heightmap_normalized = self.get_visual_observation()

    def get_visual_observation(self):
        # Capture rgbd image
        rgb_img, depth_img, view_matrix = capture_rgbd_image(
            cam_target_pos=self.camera_config['cam_target_pos'], 
            cam_distance=self.camera_config['cam_distance'], 
            width=self.camera_config['width'], 
            height=self.camera_config['height'],
            cam_yaw=self.camera_config['cam_yaw'], 
            cam_pitch=self.camera_config['cam_pitch'], 
            cam_roll=self.camera_config['cam_roll'],
            fov=self.camera_config['fov'], 
            aspect=self.camera_config['aspect'], 
            near=self.camera_config['near'], 
            far=self.camera_config['far']
        )

        # Generate point cloud with color
        point_cloud, colors = depth_to_point_cloud_with_color(depth_img, 
                                                              rgb_img, 
                                                              fov=self.camera_config['fov'], 
                                                              aspect=self.camera_config['aspect'], 
                                                              width=self.camera_config['width'], 
                                                              height=self.camera_config['height'], 
                                                              view_matrix=view_matrix, 
                                                              to_world=True)
        
        # Generate height map
        color_heightmap, depth_heightmap, color_heightmap_normalized, depth_heightmap_normalized = point_cloud_to_height_map(point_cloud=point_cloud, 
                                               colors=colors, 
                                               workspace_xlim=self.scene_config['workspace_xlim'], 
                                               workspace_ylim=self.scene_config['workspace_ylim'], 
                                               workspace_zlim=self.scene_config['workspace_zlim'],)
        
        return rgb_img, depth_img, point_cloud, colors, color_heightmap, depth_heightmap, color_heightmap_normalized, depth_heightmap_normalized

    def get_reward(self):
        raise NotImplementedError

    def is_terminated(self):
        raise NotImplementedError

    def is_truncated(self):
        raise NotImplementedError

    def get_info(self):
        return {}
    
    ############### Visulization ###############
    def visualize_target(self):
        if not hasattr(self, 'target_position'):
            raise ValueError("Not set a target yet")
        
        if hasattr(self, 'target_visualization_ids'):
            for vis_id in self.target_visualization_ids:
                p.removeUserDebugItem(vis_id)
        else:
            self.target_visualization_ids = []

        # Get box dimensions and position
        length = self.scene_config['box_length']
        width = self.scene_config['box_width']

        corners = [
            [self.target_position[0] - length / 2, self.target_position[1] - width / 2, self.target_position[2]],
            [self.target_position[0] - length / 2, self.target_position[1] + width / 2, self.target_position[2]],
            [self.target_position[0] + length / 2, self.target_position[1] + width / 2, self.target_position[2]],
            [self.target_position[0] + length / 2, self.target_position[1] - width / 2, self.target_position[2]],
        ]

        # Add debugger lines
        for i in range(len(corners)):
            start_point = corners[i]
            end_point = corners[(i + 1) % len(corners)]
            debug_line_id = p.addUserDebugLine(start_point, end_point, lineColorRGB=[1, 0, 0], lineWidth=4)
            self.target_visualization_ids.append(debug_line_id)

if __name__ == '__main__':
    set_seed()
    env = TossObjects()
    #####----- Main Loop -----#####
    while True:
        env.step(action=None)