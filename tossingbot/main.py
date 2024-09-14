import time
import pybullet as p
import pybullet_data

from tossingbot.scene import setup_scene
from tossingbot.utils.misc_utils import set_seed
from tossingbot.scene.camera import capture_rgbd_image, depth_to_point_cloud_with_color, initialize_plots, plot_rgb_pointcloud

if __name__ == '__main__':
    ######----- Set up parameters -----#####
    set_seed()
    
    # Scene parameters
    workspace_length = 0.3
    workspace_width = 0.4
    workspace_position = [0.55, 0]
    box_length = 0.25
    box_width = 0.15
    box_height = 0.2
    box_n_rows = 4
    box_n_cols = 3
    box_position = [1.375, 0.0]
    n_object = 1

    # Camera parameters
    cam_target_pos = [*workspace_position, 0.0]
    cam_distance = workspace_width / 2
    width = 64 * 3  # 192x144 resolution
    height = 48 * 3
    cam_yaw = 90    # TODO: Tune this parameter
    cam_pitch = -90
    cam_roll = 0
    fov = 90
    aspect = 1.33
    near = 0.01
    far = 10.0

    #####----- Set up scence, including objects and robot, etc -----#####
    physics_client_id = p.connect(p.GUI)  
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane_id, workspace_ids, box_ids, object_ids, robot = setup_scene(
        workspace_length=workspace_length,
        workspace_width=workspace_width,
        workspace_position=workspace_position,
        box_length=box_length,
        box_width=box_width,
        box_height=box_height,
        box_n_rows=box_n_rows,
        box_n_cols=box_n_cols,
        box_position=box_position,
        n_object=n_object
    )
    
    #####----- Main Loop -----#####
    # Initialize the plots for real-time display
    fig, axes = initialize_plots()
    while True:
        p.stepSimulation()
        time.sleep(1./240.)

        # Capture image
        rgb_img, depth_img, view_matrix = capture_rgbd_image(
            cam_target_pos=cam_target_pos, cam_distance=cam_distance, 
            width=width, height=height,
            cam_yaw=cam_yaw, cam_pitch=cam_pitch, cam_roll=cam_roll,
            fov=fov, aspect=aspect, near=near, far=far
        )

        # Generate point cloud with color
        point_cloud, colors = depth_to_point_cloud_with_color(depth_img, 
                                                              rgb_img, 
                                                              fov=fov, 
                                                              aspect=aspect, 
                                                              width=width, 
                                                              height=height, 
                                                              view_matrix=view_matrix, 
                                                              to_world=True)

        # Plot rgbd image and pointcloud
        plot_rgb_pointcloud(rgb_img, point_cloud, colors, fig, axes)

    p.disconnect()