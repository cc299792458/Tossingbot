import time
import numpy as np
import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    rgb_img = rgb_img[:, :, :3]  # Keep only the RGB channels, discard Alpha

    return rgb_img, depth_img, view_matrix

def depth_to_point_cloud_with_color(depth_img, rgb_img, fov, aspect, width, height, view_matrix, to_world=True):
    """
    Convert depth image to a colored point cloud.
    
    Parameters:
    - depth_img: The depth image (real-world depth values).
    - rgb_img: The RGB image.
    - fov: Field of view of the camera in degrees.
    - width: Width of the captured image.
    - height: Height of the captured image.
    - view_matrix: The view matrix for transforming camera to world coordinates.
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

    point_cloud = []
    colors = []

    # Iterate over each pixel in the depth image
    for v in range(height):
        for u in range(width):
            z = depth_img[v, u]
            if z == 0:  # Ignore invalid depth values (e.g., if depth_img contains 0 for no depth)
                continue
            
            # Convert pixel coordinates (u, v) into 3D camera coordinates (x, y, z)
            x = (u - cx) / fx * z
            y = (v - cy) / fy * z
            point_cloud.append([x, y, -z])
            
            # Get the color for the point from the RGB image
            r, g, b = rgb_img[v, u, :]  # Extract RGB values
            colors.append([r / 255.0, g / 255.0, b / 255.0])  # Normalize to [0, 1]

    point_cloud = np.array(point_cloud)
    colors = np.array(colors)

    if to_world:
        # PyBullet's view_matrix is row-major, we need to transpose to get column-major format
        view_matrix_np = np.array(view_matrix).reshape(4, 4).T

        # Extract rotation (3x3) and translation (3x1) from the view matrix
        rotation_matrix = view_matrix_np[:3, :3]
        translation_vector = view_matrix_np[:3, 3]

        # Invert the rotation matrix
        inv_rotation_matrix = np.linalg.inv(rotation_matrix)

        # Invert the translation vector
        inv_translation_vector = -inv_rotation_matrix @ translation_vector

        # Apply rotation and translation to convert to world coordinates
        point_cloud = (inv_rotation_matrix @ point_cloud.T).T + inv_translation_vector

    return point_cloud, colors

if __name__ == '__main__':
    # Initialize PyBullet simulation
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    # Create a plane and a box in the simulation
    p.loadURDF("plane.urdf")
    p.loadURDF("r2d2.urdf", [0, 0, 1])  # Add an example robot

    # Initialize the plot for real-time display
    plt.ion()  # Turn on interactive mode
    fig = plt.figure(figsize=(12, 6))  # Create a figure

    cam_target_pos, cam_distance = [0, 0, 0.75], 2
    width, height = 64 * 3, 48 * 3
    cam_yaw, cam_pitch, cam_roll = 0, -90, 0
    fov, aspect = 45, 1.33
    near, far = 0.01, 10.0

    # Capture RGB and Depth images
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

    # Create a 3D plot for point cloud visualization
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Colored Point Cloud")
    scatter_plot = ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], 
                              c=colors, s=1)

    # Update the plot in the simulation loop
    for _ in range(1000):
        p.stepSimulation()
        rgb_img, depth_img, view_matrix = capture_rgbd_image(
            cam_target_pos=cam_target_pos, cam_distance=cam_distance, 
            width=width, height=height,
            cam_yaw=cam_yaw, cam_pitch=cam_pitch, cam_roll=cam_roll,
            fov=fov, aspect=aspect, near=near, far=far
        )

        # Update point cloud with color
        point_cloud, colors = depth_to_point_cloud_with_color(depth_img, 
                                                              rgb_img, 
                                                              fov=fov, 
                                                              aspect=aspect, 
                                                              width=width, 
                                                              height=height, 
                                                              view_matrix=view_matrix, 
                                                              to_world=True)

        # Clear and update scatter plot
        ax.clear()
        ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c=colors, s=1)
        ax.set_title("Colored Point Cloud")

        plt.pause(0.001)  # Pause briefly to allow the plot to update in real-time

    p.disconnect()
