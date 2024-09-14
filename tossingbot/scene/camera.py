import time
import numpy as np
import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R
from tossingbot.scene.objects import create_sphere

def capture_rgbd_image(
    cam_target_pos=[0, 0, 0], cam_distance=2, 
    width=640, height=480,
    cam_yaw=45, cam_pitch=-30, cam_roll=0, 
    fov=60, aspect=1.0, near=0.01, far=10.0
):
    """
    Capture an RGB-D image (RGB and Depth) from the scene using the camera.
    """
    # Get view and projection matrices
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=cam_target_pos,
        distance=cam_distance,
        yaw=cam_yaw,
        pitch=cam_pitch,
        roll=cam_roll,
        upAxisIndex=2
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=fov,
        aspect=aspect,
        nearVal=near,
        farVal=far
    )

    # Get camera image (RGB, depth, segmentation masks)
    _, _, rgb_img, depth_buffer, seg_img = p.getCameraImage(
        width=width,        # Image width
        height=height,      # Image height
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_TINY_RENDERER  # Use TINY renderer for fast performance
    )

    # Convert the depth buffer to real depth values
    depth_img = far * near / (far - (far - near) * depth_buffer)  # Depth conversion formula

    # Remove the alpha channel (RGBA -> RGB)
    rgb_img = np.array(rgb_img, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]  # Keep only the RGB channels, discard Alpha

    return rgb_img, depth_img, view_matrix

def depth_to_point_cloud_with_color(
    depth_img, rgb_img, fov, aspect, width, height,
    cam_target_pos, cam_distance, cam_yaw, cam_pitch, cam_roll,
    to_world=True
):
    """
    Convert depth image to a colored point cloud using camera parameters.

    Parameters:
    - depth_img: The depth image (real-world depth values).
    - rgb_img: The RGB image.
    - fov: Field of view of the camera in degrees.
    - aspect: Aspect ratio of the camera.
    - width: Width of the captured image.
    - height: Height of the captured image.
    - cam_target_pos: The target position the camera is looking at.
    - cam_distance: The distance from the camera to the target position.
    - cam_yaw: Yaw angle of the camera in degrees.
    - cam_pitch: Pitch angle of the camera in degrees.
    - cam_roll: Roll angle of the camera in degrees.
    - to_world: If True, transforms the point cloud to world coordinates. Otherwise, stays in camera coordinates.

    Returns:
    - point_cloud: A numpy array of shape (N, 3), where N is the number of points.
                   Each point is represented as (x, y, z).
    - colors: A numpy array of shape (N, 3), where N is the number of points.
              Each color is represented as (r, g, b).
    """

    # Focal length (in pixels) derived from the field of view
    fy = (height / 2.0) / np.tan(np.radians(fov / 2.0))
    fx = fy * aspect

    cx = width / 2.0  # Center of the image width
    cy = height / 2.0  # Center of the image height

    # Create a grid of (u, v) coordinates
    u = np.arange(width)
    v = np.arange(height)
    uu, vv = np.meshgrid(u, v)

    # Flatten the arrays
    uu = uu.flatten()
    vv = vv.flatten()
    depth = depth_img.flatten()
    rgb = rgb_img.reshape(-1, 3)

    # Filter out invalid depth values
    valid = depth > 0
    uu = uu[valid]
    vv = vv[valid]
    depth = depth[valid]
    rgb = rgb[valid]

    # Compute camera coordinates
    x = (uu - cx) / fx * depth
    y = (vv - cy) / fy * depth
    z = depth
    points = np.vstack((x, y, z)).T

    # Normalize RGB colors
    colors = rgb.astype(np.float32) / 255.0

    if to_world:
        R_matrix = R.from_euler('yxz', [cam_yaw, cam_pitch, cam_roll], degrees=True).as_matrix()

        # Compute camera position in world coordinates
        yaw_rad = np.radians(cam_yaw)
        pitch_rad = np.radians(cam_pitch)
        roll_rad = np.radians(cam_roll)

        cam_pos_x = cam_target_pos[0] + cam_distance * np.cos(pitch_rad) * np.sin(yaw_rad)
        cam_pos_y = cam_target_pos[1] + cam_distance * np.cos(pitch_rad) * np.cos(yaw_rad)
        cam_pos_z = cam_target_pos[2] + cam_distance * np.sin(pitch_rad)
        T = np.array([cam_pos_x, cam_pos_y, cam_pos_z])

        # Invert rotation and translation to get world transformation
        R_inv = R_matrix.T
        T_inv = -R_inv @ T

        # Apply rotation and translation to convert to world coordinates
        points = (R_inv @ points.T).T + T_inv

    return points, colors

if __name__ == '__main__':
    # Initialize PyBullet simulation
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    # Create a plane and a box in the simulation
    p.loadURDF("plane.urdf")
    create_sphere()

    # Initialize the plot for real-time display
    plt.ion()  # Turn on interactive mode
    fig = plt.figure(figsize=(12, 6))  # Create a figure

    # Define camera parameters
    cam_target_pos = [0, 0, 0.75]
    cam_distance = 2
    width, height = 64 * 3, 48 * 3
    cam_yaw, cam_pitch, cam_roll = 0, -90, 0
    fov, aspect = 30, 1.33
    near, far = 0.01, 10.0

    # Capture RGB and Depth images
    rgb_img, depth_img, view_matrix = capture_rgbd_image(
        cam_target_pos=cam_target_pos, cam_distance=cam_distance, 
        width=width, height=height,
        cam_yaw=cam_yaw, cam_pitch=cam_pitch, cam_roll=cam_roll,
        fov=fov, aspect=aspect, near=near, far=far
    )
    
    # Generate point cloud with color using camera parameters
    point_cloud, colors = depth_to_point_cloud_with_color(
        depth_img, 
        rgb_img, 
        fov=fov, 
        aspect=aspect, 
        width=width, 
        height=height, 
        cam_target_pos=cam_target_pos, 
        cam_distance=cam_distance, 
        cam_yaw=cam_yaw, 
        cam_pitch=cam_pitch, 
        cam_roll=cam_roll, 
        to_world=True
    )

    # Create a 3D plot for point cloud visualization
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Colored Point Cloud")
    scatter_plot = ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], 
                              c=colors, s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Optionally, enable auto-scaling
    ax.auto_scale_xyz(point_cloud[:,0], point_cloud[:,1], point_cloud[:,2])

    plt.draw()
    plt.pause(0.001)  # Initial pause to display the plot

    # Update the plot in the simulation loop
    for _ in range(1000):
        p.stepSimulation()
        rgb_img, depth_img, view_matrix = capture_rgbd_image(
            cam_target_pos=cam_target_pos, cam_distance=cam_distance, 
            width=width, height=height,
            cam_yaw=cam_yaw, cam_pitch=cam_pitch, cam_roll=cam_roll,
            fov=fov, aspect=aspect, near=near, far=far
        )

        # Update point cloud with color using camera parameters
        point_cloud, colors = depth_to_point_cloud_with_color(
            depth_img, 
            rgb_img, 
            fov=fov, 
            aspect=aspect, 
            width=width, 
            height=height, 
            cam_target_pos=cam_target_pos, 
            cam_distance=cam_distance, 
            cam_yaw=cam_yaw, 
            cam_pitch=cam_pitch, 
            cam_roll=cam_roll, 
            to_world=True
        )

        # Clear and update scatter plot
        ax.cla()  # Clear the current axes
        ax.set_title("Colored Point Cloud")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        scatter_plot = ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], 
                                  c=colors, s=1)
        
        # Dynamically adjust the axes limits based on the point cloud
        ax.auto_scale_xyz(point_cloud[:,0], point_cloud[:,1], point_cloud[:,2])

        plt.draw()
        plt.pause(0.001)  # Pause briefly to allow the plot to update in real-time

    p.disconnect()
