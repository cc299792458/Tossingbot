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
    fov=60, aspect=1.0, near=0.1, far=100.0
):
    """
    Capture an RGB-D image (RGB and Depth) from the scene using the camera.

    Parameters:
    - cam_target_pos: List specifying the camera target position [x, y, z].
    - cam_distance: Distance from the camera to the target position.
    - width: Width of the captured image (default is 640).
    - height: Height of the captured image (default is 480).
    - cam_yaw: Yaw angle (horizontal rotation) of the camera.
    - cam_pitch: Pitch angle (vertical rotation) of the camera.
    - cam_roll: Roll angle of the camera.
    - fov: Field of view of the camera in degrees.
    - aspect: Aspect ratio of the camera (width/height).
    - near: Near clipping plane distance.
    - far: Far clipping plane distance.

    Returns:
    - rgb_img: The captured RGB image.
    - real_depth: The corresponding depth image with real-world depth values.
    - view_matrix: The view matrix for camera-world transformation.
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
    _, _, rgb_img, depth_img, seg_img = p.getCameraImage(
        width=width,        # Image width
        height=height,      # Image height
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_TINY_RENDERER  # Use TINY renderer for fast performance
    )

    # Convert the depth buffer to real depth values
    depth_img = far * near / (far - (far - near) * depth_img)  # Depth conversion formula

    # Remove the alpha channel (RGBA -> RGB)
    rgb_img = rgb_img[:, :, :3]  # Keep only the RGB channels, discard Alpha

    return rgb_img, depth_img, view_matrix

def depth_to_point_cloud(depth_img, fov, width, height, near, far, view_matrix, to_world=False):
    """
    Convert depth image to a point cloud, either in camera or world coordinates.
    
    Parameters:
    - depth_img: The depth image (real-world depth values).
    - fov: Field of view of the camera in degrees.
    - width: Width of the captured image.
    - height: Height of the captured image.
    - near: Near clipping plane distance.
    - far: Far clipping plane distance.
    - view_matrix: The view matrix for transforming camera to world coordinates.
    - to_world: If True, transforms the point cloud to world coordinates. Otherwise, stays in camera coordinates.

    Returns:
    - point_cloud: A numpy array of shape (N, 3), where N is the number of points.
                   Each point is represented as (x, y, z).
    """
    # Calculate the camera intrinsic parameters
    cx = width / 2.0  # Center of the image width
    cy = height / 2.0  # Center of the image height
    
    # Focal length (in pixels) derived from the field of view
    f = width / (2 * np.tan(np.radians(fov / 2)))
    
    point_cloud = []

    # Iterate over each pixel in the depth image
    for v in range(height):
        for u in range(width):
            z = depth_img[v, u]
            if z == 0:  # Ignore invalid depth values (e.g., if depth_img contains 0 for no depth)
                continue
            
            # Convert pixel coordinates (u, v) into 3D camera coordinates (x, y, z)
            x = (u - cx) * z / f
            y = (v - cy) * z / f
            point_cloud.append([x, y, z])
    
    point_cloud = np.array(point_cloud)

    if to_world:
        # PyBullet's view_matrix is row-major, we need to transpose to get column-major format
        view_matrix_np = np.array(view_matrix).reshape(4, 4).T

        # Extract rotation (3x3) and translation (3x1) from the view matrix
        rotation_matrix = view_matrix_np[:3, :3]
        translation_vector = view_matrix_np[:3, 3]

        # Invert the rotation matrix
        inv_rotation_matrix = np.linalg.inv(rotation_matrix)

        # Apply rotation and translation to convert to world coordinates
        point_cloud = (inv_rotation_matrix @ point_cloud.T).T + translation_vector

    return point_cloud

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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # Create a figure with two subplots

    # Subplot 1: Real-time RGB image
    rgb_img, depth_img, view_matrix = capture_rgbd_image(
        cam_target_pos=[0, 0, 0.75], cam_distance=2, 
        width=640, height=480,
        cam_yaw=0, cam_pitch=-45, cam_roll=0,
        fov=45, aspect=1.33, near=0.2, far=5.0
    )
    
    img_plot = ax1.imshow(rgb_img)
    ax1.set_title("Real-Time RGB Image")
    ax1.axis('off')

    # Subplot 2: 3D point cloud in camera or world coordinates
    ax2 = fig.add_subplot(122, projection='3d')
    to_world = True  # Set to True for world coordinates, False for camera coordinates
    point_cloud = depth_to_point_cloud(depth_img, fov=45, width=640, height=480, near=0.2, far=5.0, view_matrix=view_matrix, to_world=to_world)
    scatter_plot = ax2.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c=point_cloud[:, 2], cmap='jet', s=1)
    ax2.set_title("Real-Time Point Cloud")

    # Update the plots in the simulation loop
    for _ in range(1000):
        p.stepSimulation()
        rgb_img, depth_img, view_matrix = capture_rgbd_image(
            cam_target_pos=[0, 0, 0.75], cam_distance=2, 
            width=640, height=480,
            cam_yaw=0, cam_pitch=-45, cam_roll=0,
            fov=45, aspect=1.33, near=0.2, far=5.0
        )  # Capture the RGB-D images with custom parameters

        # Update RGB image plot
        img_plot.set_data(rgb_img)

        # Update point cloud plot (in world or camera coordinates)
        point_cloud = depth_to_point_cloud(depth_img, fov=45, width=640, height=480, near=0.2, far=5.0, view_matrix=view_matrix, to_world=to_world)
        ax2.clear()  # Clear the previous scatter plot
        ax2.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c=point_cloud[:, 2], cmap='jet', s=1)
        ax2.set_title("Real-Time Point Cloud")

        plt.pause(0.001)  # Pause briefly to allow the plot to update in real-time

        time.sleep(1. / 240.)

    p.disconnect()
