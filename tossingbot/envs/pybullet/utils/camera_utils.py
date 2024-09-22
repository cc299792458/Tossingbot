import time
import math
import numpy as np
import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt

from math_utils import rotation_matrix_to_quaternion

# TODO: debug when pitch is not 90

def capture_rgbd_image(
    cam_target_pos=[0, 0, 0], cam_distance=2, 
    width=640, height=480,
    cam_yaw=45, cam_pitch=-30, cam_roll=0, 
    fov=60, aspect=1.0, near=0.01, far=10.0
):
    """
    Capture an RGB-D image (RGB and Depth) from the scene using the camera.
    
    Parameters:
    - cam_target_pos: Target position the camera is looking at.
    - cam_distance: Distance of the camera from the target position.
    - width: Width of the captured image in pixels.
    - height: Height of the captured image in pixels.
    - cam_yaw: Yaw angle of the camera in degrees.
    - cam_pitch: Pitch angle of the camera in degrees.
    - cam_roll: Roll angle of the camera in degrees.
    - fov: Field of view of the camera in degrees.
    - aspect: Aspect ratio of the camera.
    - near: Near clipping plane distance.
    - far: Far clipping plane distance.
    
    Returns:
    - rgb_img: The captured RGB image as a (H, W, 3) numpy array.
    - depth_img: The captured Depth image as a (H, W) numpy array.
    - view_matrix: The view matrix used for capturing the image.
    """
    # Get the view and projection matrices
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

    # Convert to numpy array
    rgb_img = np.array(rgb_img)
    depth_img = np.array(depth_img)

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
    y_camera = -(v_valid - cy) / fy * depth_valid   # Invert y-axis to match PyBullet's coordinate system   # NOTE: this may be a wrong way.
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

def initialize_plots(figsize=(12, 6)):
    """
    Initialize the matplotlib figure and subplots for real-time display.

    Parameters:
    - figsize: Tuple specifying the size of the figure.

    Returns:
    - fig: The matplotlib figure object.
    - axes: A tuple containing the axes objects (ax_rgb, ax_pointcloud).
    """
    plt.ion()  # Turn on interactive mode
    fig = plt.figure(figsize=figsize)  # Create a figure

    # Create two subplots: RGB and Point Cloud
    ax_rgb = fig.add_subplot(1, 2, 1)
    ax_pointcloud = fig.add_subplot(1, 2, 2, projection='3d')
    
    # Set initial titles and labels
    ax_rgb.set_title("RGB Image")
    ax_rgb.axis('off')

    ax_pointcloud.set_title("Colored Point Cloud")
    ax_pointcloud.set_xlabel("X")
    ax_pointcloud.set_ylabel("Y")
    ax_pointcloud.set_zlabel("Z")

    # Pack the axes into a tuple for easy passing
    axes = (ax_rgb, ax_pointcloud)

    return fig, axes

def plot_rgb_pointcloud(rgb_img, point_cloud, colors, fig, axes, xlim=[-1.0, 1.0], ylim=[-1.0, 1.0], zlim=[0.0, 2.0]):
    """
    Plot the RGB image and Point Cloud in a single figure with two subplots.

    Parameters:
    - rgb_img: The RGB image as a (H, W, 3) numpy array.
    - point_cloud: The point cloud as a (N, 3) numpy array.
    - colors: The colors for each point in the point cloud as a (N, 3) numpy array.
    - fig: The matplotlib figure object.
    - axes: A tuple of matplotlib axes objects (ax_rgb, ax_pointcloud).

    Returns:
    - None
    """
    ax_rgb, ax_pointcloud = axes

    # Update RGB Image
    ax_rgb.clear()
    ax_rgb.imshow(rgb_img)
    ax_rgb.set_title("RGB Image")
    ax_rgb.axis('off')

    # Update Point Cloud
    ax_pointcloud.clear()
    ax_pointcloud.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], 
                          c=colors, s=1)
    ax_pointcloud.set_title("Colored Point Cloud")
    ax_pointcloud.set_xlabel("X")
    ax_pointcloud.set_ylabel("Y")
    ax_pointcloud.set_zlabel("Z")

    # Set axis limits if provided
    if xlim is not None:
        ax_pointcloud.set_xlim(xlim)
    if ylim is not None:
        ax_pointcloud.set_ylim(ylim)
    if zlim is not None:
        ax_pointcloud.set_zlim(zlim)

    # Redraw the figure
    plt.draw()
    plt.pause(0.001)  # Pause briefly to allow the plot to update in real-time

def draw_camera_frustum(camera_pos, camera_orientation, fov, aspect_ratio, near, far):
    """
    Visualize the camera frustum in the PyBullet environment using debug lines.
    
    Args:
        camera_pos (list): Camera position [x, y, z].
        camera_orientation (list): Camera orientation as quaternion.
        fov (float): Field of view in degrees.
        aspect_ratio (float): Aspect ratio (width/height).
        near (float): Near clipping plane distance.
        far (float): Far clipping plane distance.
    """
    # Convert FOV from degrees to radians
    fov_rad = fov * (math.pi / 180.0)

    # Calculate frustum dimensions at near and far planes
    near_height = 2 * near * math.tan(fov_rad / 2)
    near_width = near_height * aspect_ratio
    far_height = 2 * far * math.tan(fov_rad / 2)
    far_width = far_height * aspect_ratio

    # Define frustum corners in the camera's local space
    near_plane = np.array([
        [near_width / 2, near_height / 2, -near],
        [-near_width / 2, near_height / 2, -near],
        [-near_width / 2, -near_height / 2, -near],
        [near_width / 2, -near_height / 2, -near]
    ])

    far_plane = np.array([
        [far_width / 2, far_height / 2, -far],
        [-far_width / 2, far_height / 2, -far],
        [-far_width / 2, -far_height / 2, -far],
        [far_width / 2, -far_height / 2, -far]
    ])

    # Convert the orientation quaternion to a rotation matrix
    rot_matrix = np.array(p.getMatrixFromQuaternion(camera_orientation)).reshape(3, 3)

    # Rotate and translate the frustum corners from local to world coordinates
    near_plane_world = [np.dot(rot_matrix, corner) + camera_pos for corner in near_plane]
    far_plane_world = [np.dot(rot_matrix, corner) + camera_pos for corner in far_plane]

    # Draw frustum lines using PyBullet debug lines
    for i in range(4):
        # Near plane lines
        p.addUserDebugLine(near_plane_world[i], near_plane_world[(i + 1) % 4], [0, 1, 0])
        # Far plane lines
        p.addUserDebugLine(far_plane_world[i], far_plane_world[(i + 1) % 4], [0, 1, 0])
        # Lines connecting near and far planes
        p.addUserDebugLine(near_plane_world[i], far_plane_world[i], [0, 1, 0])

def draw_camera_axes(camera_pos, camera_orientation, axis_length=0.2):
    """
    Draw the camera's local coordinate frame (X, Y, Z axes) in the PyBullet environment.
    
    Args:
        camera_pos (list): Camera position [x, y, z].
        camera_orientation (list): Camera orientation as quaternion.
        axis_length (float): Length of the axes lines to be drawn.
    """
    # Convert the camera orientation quaternion to a rotation matrix
    rot_matrix = np.array(p.getMatrixFromQuaternion(camera_orientation)).reshape(3, 3)

    # Define unit vectors for the camera's local coordinate frame
    x_axis = np.array([1, 0, 0])  # Local X-axis (red)
    y_axis = np.array([0, 1, 0])  # Local Y-axis (green)
    z_axis = np.array([0, 0, 1])  # Local Z-axis (blue)

    # Transform the unit vectors to world coordinates using the rotation matrix
    x_axis_world = np.dot(rot_matrix, x_axis) * axis_length
    y_axis_world = np.dot(rot_matrix, y_axis) * axis_length
    z_axis_world = np.dot(rot_matrix, z_axis) * axis_length

    # Draw the axes in PyBullet
    p.addUserDebugLine(camera_pos, camera_pos + x_axis_world, [1, 0, 0])  # X-axis (red)
    p.addUserDebugLine(camera_pos, camera_pos + y_axis_world, [0, 1, 0])  # Y-axis (green)
    p.addUserDebugLine(camera_pos, camera_pos + z_axis_world, [0, 0, 1])  # Z-axis (blue)

def extract_camera_position_from_view_matrix(view_matrix):
    """
    Extract the camera position from the view matrix considering OpenGL conventions.
    
    Args:
        view_matrix (list): The view matrix as a flat list of 16 elements.
    
    Returns:
        list: The camera position [x, y, z].
    """
    # Convert the view matrix to a numpy array and reshape into a 4x4 matrix
    view_mat = np.array(view_matrix).reshape(4, 4, order='F')  # Column-major order

    # Extract the rotation matrix (upper-left 3x3)
    rotation_matrix = view_mat[:3, :3]

    # Extract the translation vector (first 3 elements of the fourth column)
    translation_vector = view_mat[:3, 3]

    # The camera position is given by:
    # camera_pos = -rotation_matrix.T @ translation_vector
    camera_pos = -np.dot(rotation_matrix.T, translation_vector)

    return camera_pos.tolist()

def extract_camera_position_from_view_matrix(view_matrix):
    """
    Extract the camera position from the view matrix considering OpenGL conventions.
    
    Args:
        view_matrix (list): The view matrix as a flat list of 16 elements.
    
    Returns:
        list: The camera position [x, y, z].
    """
    # Convert the view matrix to a numpy array and reshape into a 4x4 matrix
    view_mat = np.array(view_matrix).reshape(4, 4, order='F')  # Column-major order

    # Extract the rotation matrix (upper-left 3x3)
    rotation_matrix = view_mat[:3, :3]

    # Extract the translation vector (first 3 elements of the fourth column)
    translation_vector = view_mat[:3, 3]

    # The camera position is given by:
    # camera_pos = -rotation_matrix.T @ translation_vector
    camera_pos = -np.dot(rotation_matrix.T, translation_vector)

    return camera_pos.tolist()

def extract_camera_orientation_from_view_matrix(view_matrix):
    """
    Extract the camera orientation quaternion from the view matrix.
    
    Args:
        view_matrix (list): The view matrix as a flat list of 16 elements.
    
    Returns:
        list: The camera orientation as a quaternion [x, y, z, w].
    """
    # Convert the view matrix to a numpy array and reshape into a 4x4 matrix
    view_mat = np.array(view_matrix).reshape(4, 4, order='F')  # Column-major order

    # Extract the rotation matrix (upper-left 3x3)
    rotation_matrix = view_mat[:3, :3]

    # Transpose the rotation matrix to get the correct orientation
    rotation_matrix_T = rotation_matrix.T

    # Convert rotation matrix to quaternion
    quaternion = rotation_matrix_to_quaternion(rotation_matrix_T)

    return quaternion

if __name__ == '__main__':
    # Initialize PyBullet simulation
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    # Load a plane and a robot into the simulation
    p.loadURDF("plane.urdf")
    p.loadURDF("r2d2.urdf", [0, 0, 1])  # Add an example robot
    # p.loadURDF("r2d2.urdf", [0, 0, 1], [0, 0, 0.707, 0.707])  # Add an example robot

    # Initialize the plots for real-time display
    fig, axes = initialize_plots()

    # Camera parameters
    cam_target_pos, cam_distance = [0, 0, 0.75], 2
    width, height = 64 * 4, 64 * 3  # resolution
    cam_yaw, cam_pitch, cam_roll = 90, -45, 0
    fov, aspect = 45, 1.33
    near, far = 0.01, 10.0

    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=cam_target_pos,
        distance=cam_distance,
        yaw=cam_yaw,
        pitch=cam_pitch,
        roll=cam_roll,
        upAxisIndex=2
    )

    camera_pos = extract_camera_position_from_view_matrix(view_matrix)
    camera_orientation = extract_camera_orientation_from_view_matrix(view_matrix)

    draw_camera_frustum(
        camera_pos=camera_pos,
        camera_orientation=camera_orientation,
        fov=fov,
        aspect_ratio=aspect,
        near=near,
        far=far
    )

    draw_camera_axes(camera_pos=camera_pos, camera_orientation=camera_orientation)

    # Simulation loop
    for _ in range(1000):
        p.stepSimulation()
        time.sleep(1./240.)  # Adjust sleep time as needed for real-time simulation

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

        # Update the plots
        plot_rgb_pointcloud(rgb_img, point_cloud, colors, fig, axes)

    # Disconnect the PyBullet simulation
    p.disconnect()
