import math
import random
import numpy as np
import pybullet as p

from tossingbot.envs.pybullet.robot import UR5Robotiq85, Panda
from tossingbot.envs.pybullet.tasks.base_scene import BaseScene
from tossingbot.envs.pybullet.utils.math_utils import yaw_to_quaternion, pose_distance
from tossingbot.envs.pybullet.utils.objects_utils import (
    create_box, 
    Ball, Cube, Rod, Hammer
)
from tossingbot.envs.pybullet.utils.camera_utils import (
    visualize_camera,
    capture_rgbd_image, 
    depth_to_point_cloud_with_color, 
    point_cloud_to_height_map,
    initialize_visual_plots,
    plot_rgb_pointcloud_heightmap,
    compute_camera_fov_at_height,
    collect_heightmaps_and_stats,
)

class TossObjects(BaseScene):
    def __init__(self, 
             timestep=1/240,
             control_timestep=1/60,
             gravity=-9.81,
             use_gui=True,
             visualize_config=None,
             scene_config=None, 
             robot_config=None,
             objects_config=None,
             task_config=None,
             camera_config=None,
             log_config=None,
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
            task_config (dict): Configuration for the task.
            camera_config (dict): Configuration for the camera setup.
            log_config (dict): Configuration for the log.
        """
        # Default visualize configuration
        default_visualize_config = {
            "visualize_coordinate_frames": True,
            "visualize_camera": True,
            "visualize_visual_plots": False,
            "visualize_target": True,
            "visualize_tcp": False,
        }
        if visualize_config is not None:
            default_visualize_config.update(visualize_config)
        self.visualize_config = default_visualize_config
        
        # Default scene configuration
        default_scene_config = {
            "workspace_length": 0.3,
            "workspace_width": 0.4,
            "workspace_position": [0.4, 0],
            "workspace_thickness": 0.001,
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
            "workspace_zlim": [0.0, 0.4]
        })
        if scene_config is not None:
            default_scene_config.update(scene_config)
        self.scene_config = default_scene_config

        # Default robot configuration
        default_robot_config = {
            "base_position": np.array([0.0, 0.0, 0.0]),
            "base_orientation": np.array([0.0, 0.0, 0.0]),
            "gripper_control_mode": 'position',
            "use_gripper_gear": True,
            "robot_type": 'panda',
            "visualize_coordinate_frames": use_gui and self.visualize_config['visualize_coordinate_frames'],
        }
        if robot_config is not None:
            default_robot_config.update(robot_config)
        self.robot_config = default_robot_config

        # Default objects configuration
        default_objects_config = {
            "n_object": 1,
            "object_types": ['ball', 'cube', 'rod', 'hammer'],
            "lateral_friction": 1.0,
            "rolling_friction": 0.5,
        }
        if objects_config is not None:
            default_objects_config.update(objects_config)
        self.objects_config = default_objects_config

        # Default task configuration
        default_task_config = {
            "use_heuristic": True,
            "consecutive_grasp_failures_threshold": 3,
            "grasp_supervision": 'grasp',
        }
        if task_config is not None:
            default_task_config.update(task_config)
        self.task_config = default_task_config

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
            "heightmap_resolution": 0.005,
            "n_rotations": 16,
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

        # Default log configuration
        default_log_config = {
            "log_robot_variables": False
        }
        if log_config is not None:
            default_log_config.update(log_config)
        self.log_config = default_log_config

        super().__init__(timestep=timestep, control_timestep=control_timestep, gravity=gravity, use_gui=use_gui)

        # Collect heightmap dataset and calculate stats
        rgb_mean, rgb_std, depth_mean, depth_std = collect_heightmaps_and_stats(
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
            far=self.camera_config['far'],
            workspace_xlim=self.camera_config['cam_view_xlim'],
            workspace_ylim=self.camera_config['cam_view_ylim'],
            workspace_zlim=self.scene_config['workspace_zlim'],
            post_func=self.reset_objects
        )
        self.heightmap_stats = {
            "rgb_mean": rgb_mean,
            "rgb_std": rgb_std,
            "depth_mean": depth_mean,
            "depth_std": depth_std, 
        }

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
        # Load plane, workspace and boxes
        self.load_ground_plane()
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

    def load_ground_plane(self):
        """
        Loads a flat ground plane into the simulation.
        """
        self.plane_id = p.loadURDF("plane.urdf")

    def load_workspace(self, length=0.3, width=0.4, height=0.02, position=[0.3, 0.0]):
        """
        Create a workspace with walls in the simulation.

        Args:
            length (float): Length of the workspace.
            width (float): Width of the workspace.
            position (list): Center position [x, y] of the workspace.
        """
        thickness = self.scene_config['workspace_thickness']
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

    def load_boxes_with_dividers(self, length=0.25, width=0.15, height=0.1, n_rows=4, n_cols=3, position=[0.0, 0.0]):
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
        total_length = n_rows * length - 2 * divider_thickness
        total_width = n_cols * width - 2 * divider_thickness

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
        for i in range(1, n_cols):
            y = y_start + i * width - divider_thickness
            self.box_ids.append(create_box(half_extents=[total_length / 2, divider_thickness / 2, height / 2],
                                    position=[position[0], y, height / 2],
                                    mass=0, color=color))

        for j in range(1, n_rows):
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
                timestep=self.timestep,
                control_timestep=self.control_timestep,
                base_position=self.robot_config['base_position'],
                base_orientation=self.robot_config['base_orientation'],
                gripper_control_mode=self.robot_config['gripper_control_mode'],
                use_gripper_gear=self.robot_config['use_gripper_gear'],
                visualize_coordinate_frames=self.robot_config['visualize_coordinate_frames'],
            )

    ############### Reset ###############
    def reset_robot(self, init=False):
        """
        Reset the robot's position.
        """
        self.robot.reset()

        self.grasp_completed = False
        self.throw_completed = False

    def reset_objects(self, init=False, margin=0.1):
        """
        Randomly place objects in the workspace.
        """
        # Remove existing objects if they exist
        if hasattr(self, 'objects'):
            for _object in self.objects:
                _object._remove_object()
                del _object

        self.objects = []
        thickness = self.scene_config['workspace_thickness']

        for i in range(self.objects_config['n_object']):
            object_type = random.randint(0, len(self.objects_config['object_types']) - 1)
            x = random.uniform(
                self.scene_config['workspace_position'][0] - self.scene_config['workspace_length'] / 2 + margin,
                self.scene_config['workspace_position'][0] + self.scene_config['workspace_length'] / 2 - margin,
            )
            y = random.uniform(
                self.scene_config['workspace_position'][1] - self.scene_config['workspace_width'] / 2 + margin,
                self.scene_config['workspace_position'][1] + self.scene_config['workspace_width'] / 2 - margin,
            )
            # yaw = random.uniform(-math.pi, math.pi)
            yaw = random.randint(0, self.camera_config['n_rotations'] - 1) * math.pi / 2

            if self.objects_config['object_types'][object_type] == 'ball':
                _object = Ball(position=np.array([x, y, 0.02 + thickness]), orientation=np.array([0.0, 0.0, yaw]), mass=0.1, 
                               lateral_friction=self.objects_config['lateral_friction'], rolling_friction=self.objects_config['rolling_friction'], 
                               radius=0.02)
            elif self.objects_config['object_types'][object_type] == 'cube':
                _object = Cube(position=np.array([x, y, 0.02 + thickness]), orientation=np.array([0.0, 0.0, yaw]), mass=0.1, 
                               lateral_friction=self.objects_config['lateral_friction'], rolling_friction=self.objects_config['rolling_friction'], 
                               half_extents=[0.02, 0.02, 0.02])
            elif self.objects_config['object_types'][object_type] == 'rod':
                _object = Rod(position=np.array([x, y, 0.015 + thickness]), orientation=np.array([0.0, np.pi / 2, yaw]), mass=0.1, 
                              lateral_friction=self.objects_config['lateral_friction'], rolling_friction=self.objects_config['rolling_friction'], 
                              radius=0.015, height=0.16)
            elif self.objects_config['object_types'][object_type] == 'hammer':
                _object = Hammer(position=np.array([x, y, 0.02 + thickness]), orientation=np.array([np.pi / 2, 0.0, np.pi / 2 + yaw]), 
                                 handle_mass=0.05, head_mass=0.05, lateral_friction=1.0, rolling_friction=0.5, 
                                 handle_radius=0.01, handle_height=0.12, head_half_extents=[0.05, 0.02, 0.0125])
            else:
                raise NotImplementedError
            
            self.objects.append(_object)
        
        # Perform simulation steps for 0.05 second to stabilize the scene
        for _ in range(int(1 / (20 * self.timestep))):
            self.simulation_step()

        return None

    def reset_task(self, init=False):
        self.select_target_box()
        
        self.post_throw_settle_count = 0

        if init:
            self.grasp_success = False
            self.throw_success = False
            self.grasped_object = None
            self.landing_box = (-1, -1)

            if self.task_config['use_heuristic']:
                self.consecutive_grasp_failures = 0

    ############### Step ###############
    def pre_simulation_process(self, action):
        # Unpack the action
        if action is not None:
            grasp_pixel_index, self.post_grasp_pose, self.throw_pose, self.throw_velocity = action
        else:
            return

        # Unpack the grasp pixel index
        yaw_index, pixel_y, pixel_x = grasp_pixel_index

        # Calculate the yaw angle in radians based on the rotation index
        yaw = np.radians(yaw_index * 360 / self.camera_config['n_rotations'])

        # Compute the grasp position in the workspace
        grasp_x = self.scene_config['workspace_xlim'][0] + pixel_x * self.camera_config['heightmap_resolution']
        grasp_y = self.scene_config['workspace_ylim'][1] - pixel_y * self.camera_config['heightmap_resolution']

        # Retrieve the depth value from the visual observation
        # grasp_z = self.visual_observation['depth_heightmap'][pixel_y, pixel_x]
        thickness = self.scene_config['workspace_thickness']
        grasp_z = 0.02 + thickness

        # Define the grasp pose (position and yaw orientation), post grasp pose, throw pose, and throw velocity
        self.grasp_pose = [np.array([grasp_x, grasp_y, grasp_z]), np.array(yaw_to_quaternion(yaw))]  # Grasp pose with quaternion representation

    def post_simulation_process(self, completed_and_static):
        if completed_and_static and self.does_workspace_need_reset():
            self.reset_objects()

        self.reset_robot()
        self.reset_task()

    def does_workspace_need_reset(self):
        for _object in self.objects:
            if self.is_object_in_workspace(_object=_object):
                return False
        return True

    def pre_control_step(self):
        is_action_finished = False
        
        if (not hasattr(self, 'grasp_pose') or not hasattr(self, 'post_grasp_pose') or 
            not hasattr(self, 'throw_pose') or not hasattr(self, 'throw_velocity')):
            return

        if not self.grasp_completed:
            self.grasp_completed = self.robot.grasp(tcp_target_pose=self.grasp_pose, post_grasp_pose=self.post_grasp_pose)
            if self.grasp_completed:
                self.check_grasp_success()
                if self.task_config['use_heuristic']:
                    self.consecutive_grasp_failures = self.consecutive_grasp_failures + 1 if not self.grasp_success else 0
        elif not self.grasp_success:
            is_action_finished = True
        elif not self.throw_completed:
            self.throw_completed = self.robot.throw(tcp_target_pose=self.throw_pose, tcp_target_velocity=self.throw_velocity)
        elif self.post_throw_settle_count < int(1 / self.control_timestep): # Wait 1 second here for settling the objects.
            self.post_throw_settle_count += 1
            if self.post_throw_settle_count == int(1 / self.control_timestep):
                self.check_throw_success()
        else:
            is_action_finished = True
        
        are_objects_static = True
        # for object_id in self.object_ids:
        #     # Set a big angular_threhold here to ignore the angular velocity
        #     if not self.is_object_static(object_id=object_id, linear_threshold=0.05, angular_threshold=10.0):
        #         are_objects_static = False

        return is_action_finished and are_objects_static

    def post_control_step(self):
        if self.use_gui and self.visualize_config['visualize_tcp']:
            self.robot.visualize_tcp_trajectory()

    def pre_simulation_step(self):
        pass

    def post_simulation_step(self):
        if self.robot_config['gripper_control_mode'] == 'torque':
            self.robot.keep_gripper_force()
        if self.log_config['log_robot_variables']:
            self.robot.log_variables()
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

    def select_target_box(self):
        # Get scene configuration
        rows = self.scene_config['box_n_rows']
        cols = self.scene_config['box_n_cols']
        n_boxes = rows * cols

        # Randomly select a target box
        target_box = random.randint(0, n_boxes - 1)
        target_row = target_box // cols
        target_col = target_box % cols

        # # Get box dimensions and position
        height = self.scene_config['box_height']
        target_x, target_y = self.get_box_position(row=target_row, col=target_col)

        # Set target position
        self.target_position = np.array([target_x, target_y, height])

        if self.use_gui and self.visualize_config['visualize_target']:
            self.visualize_target()

    def get_observation(self):
        rgb_img, depth_img, point_cloud, colors, \
        color_heightmap, depth_heightmap, \
        color_heightmap_normalized, depth_heightmap_normalized = self.get_visual_observation()

        I = np.concatenate((color_heightmap_normalized, \
                    depth_heightmap_normalized[..., np.newaxis]), \
                   axis=-1) if color_heightmap_normalized is not None \
                   or depth_heightmap_normalized is not None else None
        
        self.visual_observation = {
            'rgb_img': rgb_img,
            'depth_img': depth_img,
            'point_cloud': point_cloud,
            'colors': colors,
            'color_heightmap': color_heightmap,
            'depth_heightmap': depth_heightmap,
            'color_heightmap_normalized': color_heightmap_normalized,
            'depth_heightmap_normalized': depth_heightmap_normalized,
        }

        return (I, self.target_position)

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
            far=self.camera_config['far'],
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
                                               workspace_zlim=self.scene_config['workspace_zlim'],
                                               heightmap_resolution=self.camera_config['heightmap_resolution'],
                                               rgb_mean=self.heightmap_stats['rgb_mean'] if hasattr(self, 'heightmap_stats') else None,
                                               rgb_std=self.heightmap_stats['rgb_std'] if hasattr(self, 'heightmap_stats') else None,
                                               depth_mean=self.heightmap_stats['depth_mean'] if hasattr(self, 'heightmap_stats') else None,
                                               depth_std=self.heightmap_stats['depth_std'] if hasattr(self, 'heightmap_stats') else None,)
        
        return rgb_img, depth_img, point_cloud, colors, color_heightmap, depth_heightmap, color_heightmap_normalized, depth_heightmap_normalized

    def get_reward(self):
        # NOTE: for self-supervised learning, return label here
        return self.get_label()
    
    def get_label(self):
        is_grasp_supervision = self.task_config['grasp_supervision'] == 'grasp'
        is_throw_supervision = self.task_config['grasp_supervision'] == 'throw'
        if (self.grasp_success and is_grasp_supervision) or (self.throw_success and is_throw_supervision):
            grasp_label = 0
        else:
            grasp_label = 1

        gt_residual_label = self.get_gt_residual_velocity()
            
        return (grasp_label, gt_residual_label)
    
    def get_gt_residual_velocity(self):
        """
            Get the ground truth residual velocity based on the landing position of the object.
        """
        residual_velocity = None

        if self.landing_box != (-1, -1):
            box_x, box_y = self.get_box_position(row=self.landing_box[0], col=self.landing_box[1])
            actual_throw_velocity = self.compute_throw_velocity((box_x, box_y, self.scene_config['box_height']))
            actual_velocity_magnitude = np.linalg.norm(actual_throw_velocity[0])
            velocity_magnitude = np.linalg.norm(self.throw_velocity[0])

            residual_velocity = velocity_magnitude - actual_velocity_magnitude
            
        return residual_velocity

    def compute_throw_velocity(self, landing_position, g=9.81):
        """
        Compute the throw velocity based on the landing position of the object.

        Args:
            landing_position (tuple): Actual landing position as (x, y, z).
            g (float): Gravitational acceleration (default: 9.81 m/s²).

        Returns:
            tuple: The calculated velocity components (v_x, v_y, v_z) and angular velocity (set to zero).
        """
        # Extract throw pose and velocity components
        r_h = np.linalg.norm(self.throw_pose[0][:2])  # Horizontal distance in the throw pose
        r_z = self.throw_pose[0][2]                  # Vertical distance in the throw pose
        v_h = np.linalg.norm(self.throw_velocity[0][:2])  # Horizontal velocity
        v_z = self.throw_velocity[0][2]              # Vertical velocity

        # Extract landing position
        p_h = np.linalg.norm(landing_position[:2])   # Horizontal distance in the landing position
        p_z = landing_position[2]                    # Vertical distance in the landing position

        # Calculate differences in horizontal and vertical positions
        delta_h = p_h - r_h
        delta_z = p_z - r_z

        # Calculate angles
        theta = np.arctan2(self.throw_velocity[0][1], self.throw_velocity[0][0])  # Horizontal angle
        phi = np.arctan2(v_z, v_h)  # Vertical angle

        # Calculate velocity magnitude using physics formula
        numerator = g * delta_h**2
        denominator = 2 * np.cos(phi)**2 * (np.tan(phi) * delta_h - delta_z)

        if denominator <= 0:
            raise ValueError("Invalid parameters leading to an impossible trajectory.")

        v_magnitude = np.sqrt(numerator / denominator)

        # Compute velocity components
        v_x = v_magnitude * np.cos(phi) * np.cos(theta)
        v_y = v_magnitude * np.cos(phi) * np.sin(theta)
        v_z = v_magnitude * np.sin(phi)

        return ([v_x, v_y, v_z], [0.0, 0.0, 0.0])  # Linear velocity and zero angular velocity

    def is_terminated(self):
        return False

    def is_truncated(self):
        return False
    
    def get_info(self):
        return {
            "objects": self.objects,
            "grasp_success": self.grasp_success,
            "grasped_object": self.grasped_object,
            "throw_success": self.throw_success,
            "next_grasp_with_heuristic": (
            self.task_config['use_heuristic'] and
            self.consecutive_grasp_failures >= self.task_config['consecutive_grasp_failures_threshold']
            )
        }
        
    def check_grasp_success(self):
        grasp_success = not self.robot._is_gripper_closed(tolerance=1e-2)
        if grasp_success:
            objects = [
                _object for _object in self.objects 
                if pose_distance(_object.pose, self.robot.get_tcp_pose())[0] < 0.1 
                    and _object.pose[0][2] > 0.02 + self.scene_config['workspace_thickness'] + 0.001
            ]
            assert len(objects) <= 1, "There should be exactly 1 object grasped."
            if len(objects) == 0:
                grasp_success = False   # Double check
                _object = None
            else:
                _object = objects[0]   # Extract the single object ID
        else:
            _object = None

        self.grasp_success = grasp_success
        self.grasped_object = _object

    def check_throw_success(self):
        self.throw_success = self.is_object_in_target_box(_object=self.grasped_object)

        if self.grasp_success:
            for row in range(self.scene_config['box_n_rows']):
                for col in range(self.scene_config['box_n_cols']):
                    if self.is_object_in_box(self.grasped_object, row, col):
                        self.landing_box = (row, col)   
                        return
        self.landing_box = (-1, -1)
    
    def is_object_in_workspace(self, _object, margin=0.1):
        object_pos = _object.pose[0]
        workspace_xlim = self.scene_config['workspace_xlim']
        workspace_ylim = self.scene_config['workspace_ylim']
        
        # Check if object is within the target box
        in_x_range = workspace_xlim[0] + margin < object_pos[0] < workspace_xlim[1] - margin
        in_y_range = workspace_ylim[0] + margin < object_pos[1] < workspace_ylim[1] - margin
        
        return in_x_range and in_y_range
    
    def is_object_in_target_box(self, _object):
        object_pos = _object.pose[0]
        box_length = self.scene_config['box_length']
        box_width = self.scene_config['box_width']
        
        target_x, target_y = self.target_position[0], self.target_position[1]
        
        # Check if object is within the target box
        in_x_range = target_x - box_length / 2 < object_pos[0] < target_x + box_length / 2
        in_y_range = target_y - box_width / 2 < object_pos[1] < target_y + box_width / 2
        
        return in_x_range and in_y_range
    
    def is_object_in_box(self, _object, row, col):
        # Get object position
        object_pos = _object.pose[0]

        # Get box dimensions and position
        box_length = self.scene_config['box_length']
        box_width = self.scene_config['box_width']
        box_x, box_y = self.get_box_position(row=row, col=col)
        
        # Check if object is within the target box
        in_x_range = box_x - box_length / 2 < object_pos[0] < box_x + box_length / 2
        in_y_range = box_y - box_width / 2 < object_pos[1] < box_y + box_width / 2

        return in_x_range and in_y_range
    
    def get_box_position(self, row, col):
        # Get box dimensions and position
        rows = self.scene_config['box_n_rows']
        cols = self.scene_config['box_n_cols']
        box_length = self.scene_config['box_length']
        box_width = self.scene_config['box_width']
        central_position = self.scene_config['box_position']

        # Calculate the offset for the box
        row_offset = row - (rows - 1) / 2
        col_offset = col - (cols - 1) / 2

        # Calculate the box position
        box_x = central_position[0] + row_offset * box_length
        box_y = central_position[1] + col_offset * box_width
        
        return box_x, box_y
        
    def is_object_static(self, _object, linear_threshold=0.01, angular_threshold=0.01):
        object_velocity = _object.velocity
    
        # Ensure object_velocity is not None
        assert object_velocity is not None, f"Object with ID {_object.object_ids} does not exist or has no velocity data."

        linear_velocity, angular_velocity = object_velocity
        return np.all(np.abs(linear_velocity) < linear_threshold) and np.all(np.abs(angular_velocity) < angular_threshold)
    
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
            debug_line_id = p.addUserDebugLine(start_point, end_point, lineColorRGB=[1, 0, 0], lineWidth=3)
            self.target_visualization_ids.append(debug_line_id)

if __name__ == '__main__':
    env = TossObjects()

    # Since no action is specified, the process will continue indefinitely without terminating.
    env.step(action=None)