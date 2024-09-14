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
    # Calculate focal lengths in pixels
    fy = (height / 2.0) / np.tan(np.radians(fov / 2.0))
    fx = fy * aspect
    
    # Image center coordinates
    cx = width / 2.0
    cy = height / 2.0

    # Generate grid of pixel coordinates
    u = np.arange(width)
    v = np.arange(height)
    uu, vv = np.meshgrid(u, v)
    
    # Flatten the arrays for vectorized processing
    uu_flat = uu.flatten()
    vv_flat = vv.flatten()
    depth_flat = depth_img.flatten()
    rgb_flat = rgb_img.reshape(-1, 3)
    
    # Create a mask for valid depth values
    valid_mask = depth_flat > 0
    
    # Filter out invalid points
    u_valid = uu_flat[valid_mask]
    v_valid = vv_flat[valid_mask]
    depth_valid = depth_flat[valid_mask]
    rgb_valid = rgb_flat[valid_mask]
    
    # Convert pixel coordinates to camera coordinates
    x_camera = (u_valid - cx) / fx * depth_valid
    y_camera = (v_valid - cy) / fy * depth_valid
    z_camera = -depth_valid  # Invert z-axis to match PyBullet's coordinate system
    
    # Stack the camera coordinates to form the point cloud
    point_cloud = np.stack((x_camera, y_camera, z_camera), axis=1)
    
    # Normalize RGB colors to [0, 1]
    colors = rgb_valid.astype(np.float32) / 255.0

    if to_world:
        # Convert view_matrix from row-major to column-major
        view_matrix_np = np.array(view_matrix).reshape(4, 4).T
        
        # Extract rotation matrix and translation vector
        rotation_matrix = view_matrix_np[:3, :3]
        translation_vector = view_matrix_np[:3, 3]
        
        # Invert the rotation matrix
        inv_rotation_matrix = np.linalg.inv(rotation_matrix)
        
        # Invert the translation vector
        inv_translation_vector = -inv_rotation_matrix @ translation_vector
        
        # Transform point cloud to world coordinates
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
