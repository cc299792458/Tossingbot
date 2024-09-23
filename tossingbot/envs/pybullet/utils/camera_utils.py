import time
import math
import numpy as np
import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt

from tossingbot.envs.pybullet.utils.math_utils import rotation_matrix_to_quaternion

############### Observation for TossObjects env ###############
def capture_rgbd_image(cam_target_pos=[0, 0, 0], cam_distance=2, width=640, height=480,
                       cam_yaw=90, cam_pitch=-90, cam_roll=0, fov=60, aspect=1.0, near=0.01, far=10.0):
    """
    Capture an RGB-D image from the scene using the camera.
    """
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=cam_target_pos, distance=cam_distance,
        yaw=cam_yaw, pitch=cam_pitch, roll=cam_roll, upAxisIndex=2
    )
    proj_matrix = p.computeProjectionMatrixFOV(fov=fov, aspect=aspect, nearVal=near, farVal=far)
    
    _, _, rgb_img, depth_buffer, _ = p.getCameraImage(width=width, height=height,
                                                      viewMatrix=view_matrix, projectionMatrix=proj_matrix,
                                                      renderer=p.ER_TINY_RENDERER)

    depth_img = far * near / (far - (far - near) * depth_buffer)
    rgb_img = np.array(rgb_img[:, :, :3])  # Discard alpha channel

    return rgb_img, np.array(depth_img), view_matrix

def depth_to_point_cloud_with_color(depth_img, rgb_img, fov, aspect, width, height, view_matrix, to_world=True):
    """
    Convert depth image to a colored point cloud.
    """
    fy = (height / 2.0) / np.tan(np.radians(fov / 2.0))
    fx = fy * aspect
    cx, cy = width / 2.0, height / 2.0
    
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    uu_flat, vv_flat = u.flatten(), v.flatten()
    depth_flat, rgb_flat = depth_img.flatten(), rgb_img.reshape(-1, 3)

    valid_mask = depth_flat > 0
    u_valid, v_valid = uu_flat[valid_mask], vv_flat[valid_mask]
    depth_valid, rgb_valid = depth_flat[valid_mask], rgb_flat[valid_mask]

    x_camera = (u_valid - cx) / fx * depth_valid
    y_camera = -(v_valid - cy) / fy * depth_valid   # # Invert y-axis to match PyBullet's upward direction
    z_camera = -depth_valid # Invert z-axis to match PyBullet's coordinate system
    
    point_cloud = np.stack((x_camera, y_camera, z_camera), axis=1)
    colors = rgb_valid.astype(np.float32) / 255.0

    if to_world:
        view_matrix_np = np.array(view_matrix).reshape(4, 4).T
        rotation_matrix, translation_vector = view_matrix_np[:3, :3], view_matrix_np[:3, 3]
        inv_rotation_matrix = np.linalg.inv(rotation_matrix)
        inv_translation_vector = -inv_rotation_matrix @ translation_vector
        point_cloud = (inv_rotation_matrix @ point_cloud.T).T + inv_translation_vector

    return point_cloud, colors

def point_cloud_to_height_map(point_cloud, colors, workspace_xlim, workspace_ylim, workspace_zlim, heightmap_resolution=0.005):
    # Compute heightmap size based on workspace limits and resolution
    heightmap_size = np.round((
        (workspace_xlim[1] - workspace_xlim[0]) / heightmap_resolution,
        (workspace_ylim[1] - workspace_ylim[0]) / heightmap_resolution
    )).astype(int)

    # Sort point cloud by z (height)
    sort_z_ind = np.argsort(point_cloud[:, 2])
    point_cloud = point_cloud[sort_z_ind]
    colors = colors[sort_z_ind]

    # Filter points within workspace boundaries
    valid_mask = (
        (point_cloud[:, 0] >= workspace_xlim[0]) & (point_cloud[:, 0] < workspace_xlim[1]) &
        (point_cloud[:, 1] >= workspace_ylim[0]) & (point_cloud[:, 1] < workspace_ylim[1]) &
        (point_cloud[:, 2] >= workspace_zlim[0]) & (point_cloud[:, 2] < workspace_zlim[1])
    )
    point_cloud, colors = point_cloud[valid_mask], colors[valid_mask]

    # Project points to heightmap pixel coordinates
    heightmap_pix_x = np.floor((point_cloud[:, 0] - workspace_xlim[0]) / heightmap_resolution).astype(int)
    heightmap_pix_y = np.floor((point_cloud[:, 1] - workspace_ylim[0]) / heightmap_resolution).astype(int)

    # Initialize heightmaps for RGB channels and depth
    color_heightmap = np.zeros((heightmap_size[1], heightmap_size[0], 3), dtype=np.uint8)
    depth_heightmap = np.zeros((heightmap_size[1], heightmap_size[0]))

    # Assign RGB values to heightmap
    color_heightmap[heightmap_pix_y, heightmap_pix_x] = colors[:, :3]

    # Assign depth values to heightmap (height from the bottom of the workspace)
    z_bottom = workspace_zlim[0]
    depth_heightmap[heightmap_pix_y, heightmap_pix_x] = point_cloud[:, 2] - z_bottom

    # Remove invalid depth values (set as 0 or NaN)
    depth_heightmap[depth_heightmap < 0] = 0

    return color_heightmap, depth_heightmap

############ Visualization ############
def initialize_visual_plots(figsize=(18, 6)):
    """
    Initialize the matplotlib figure for real-time display with RGB image, point cloud, and heightmap.
    """
    plt.ion()
    fig = plt.figure(figsize=figsize)
    ax_rgb = fig.add_subplot(1, 3, 1)
    ax_pointcloud = fig.add_subplot(1, 3, 2, projection='3d')
    ax_heightmap = fig.add_subplot(1, 3, 3)
    
    ax_rgb.set_title("RGB Image")
    ax_rgb.axis('off')

    ax_pointcloud.set_title("Colored Point Cloud")
    ax_pointcloud.set_xlabel("X")
    ax_pointcloud.set_ylabel("Y")
    ax_pointcloud.set_zlabel("Z")

    ax_heightmap.set_title("Depth Heightmap")
    ax_heightmap.axis('off')

    return fig, (ax_rgb, ax_pointcloud, ax_heightmap)

def plot_rgb_pointcloud_heightmap(rgb_img, point_cloud, colors, depth_heightmap, fig, axes, xlim=None, ylim=None, zlim=None):
    """
    Plot the RGB image, Point Cloud, and Heightmap (Depth Map) in subplots.
    
    Args:
    - rgb_img: RGB image (H, W, 3) numpy array.
    - point_cloud: Point cloud (N, 3) numpy array.
    - colors: Colors corresponding to the point cloud (N, 3) numpy array.
    - depth_heightmap: Depth heightmap (H, W) numpy array.
    - fig: Matplotlib figure object.
    - axes: A tuple containing the RGB image axis, point cloud axis, and heightmap axis.
    - xlim: Limits for the x-axis in the point cloud plot.
    - ylim: Limits for the y-axis in the point cloud plot.
    - zlim: Limits for the z-axis in the point cloud plot.
    """
    ax_rgb, ax_pointcloud, ax_heightmap = axes

    # Update RGB Image
    ax_rgb.clear()
    ax_rgb.imshow(rgb_img)
    ax_rgb.set_title("RGB Image")
    ax_rgb.axis('off')

    # Update Point Cloud
    ax_pointcloud.clear()
    ax_pointcloud.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c=colors, s=1)
    ax_pointcloud.set_title("Colored Point Cloud")
    ax_pointcloud.set_xlabel("X")
    ax_pointcloud.set_ylabel("Y")
    ax_pointcloud.set_zlabel("Z")
    ax_pointcloud.set_xlim(xlim)
    ax_pointcloud.set_ylim(ylim)
    ax_pointcloud.set_zlim(zlim)

    # Plot the depth heightmap using a colormap to represent the depth values
    ax_heightmap.clear()
    depth_img = ax_heightmap.imshow(depth_heightmap, cmap='viridis')
    fig.colorbar(depth_img, ax=ax_heightmap)  # Add colorbar to represent depth values
    ax_heightmap.set_title("Depth Heightmap")
    ax_heightmap.axis('off')

    # Redraw the figure
    plt.draw()
    plt.pause(0.001)

def draw_camera_frustum(camera_pos, camera_orientation, fov, aspect, near, far):
    """
    Visualize the camera frustum in PyBullet.
    """
    fov_rad = math.radians(fov)
    near_height, far_height = 2 * near * math.tan(fov_rad / 2), 2 * far * math.tan(fov_rad / 2)
    near_width, far_width = near_height * aspect, far_height * aspect
    
    near_plane = np.array([[near_width / 2, near_height / 2, -near],
                           [-near_width / 2, near_height / 2, -near],
                           [-near_width / 2, -near_height / 2, -near],
                           [near_width / 2, -near_height / 2, -near]])
    
    far_plane = np.array([[far_width / 2, far_height / 2, -far],
                          [-far_width / 2, far_height / 2, -far],
                          [-far_width / 2, -far_height / 2, -far],
                          [far_width / 2, -far_height / 2, -far]])

    rot_matrix = np.array(p.getMatrixFromQuaternion(camera_orientation)).reshape(3, 3)
    near_plane_world = [np.dot(rot_matrix, corner) + camera_pos for corner in near_plane]
    far_plane_world = [np.dot(rot_matrix, corner) + camera_pos for corner in far_plane]

    for i in range(4):
        p.addUserDebugLine(near_plane_world[i], near_plane_world[(i + 1) % 4], [0, 1, 0])
        p.addUserDebugLine(far_plane_world[i], far_plane_world[(i + 1) % 4], [0, 1, 0])
        p.addUserDebugLine(near_plane_world[i], far_plane_world[i], [0, 1, 0])

def draw_camera_axes(camera_pos, camera_orientation, axis_length=0.1):
    """
    Draw the camera's local coordinate frame (X, Y, Z axes) in the PyBullet environment.
    """
    rot_matrix = np.array(p.getMatrixFromQuaternion(camera_orientation)).reshape(3, 3)
    axes = {'X': [1, 0, 0], 'Y': [0, 1, 0], 'Z': [0, 0, 1]}
    colors = {'X': [1, 0, 0], 'Y': [0, 1, 0], 'Z': [0, 0, 1]}

    for axis in axes:
        world_axis = np.dot(rot_matrix, np.array(axes[axis])) * axis_length
        p.addUserDebugLine(camera_pos, camera_pos + world_axis, colors[axis])

def visualize_camera(cam_target_pos, cam_distance, cam_yaw, cam_pitch, cam_roll, fov, aspect, near, far):
    view_matrix = p.computeViewMatrixFromYawPitchRoll(cam_target_pos, cam_distance, cam_yaw, cam_pitch, cam_roll, upAxisIndex=2)
    camera_pos = extract_camera_position_from_view_matrix(view_matrix)
    camera_orientation = extract_camera_orientation_from_view_matrix(view_matrix)

    draw_camera_frustum(camera_pos, camera_orientation, fov, aspect, near, far)
    draw_camera_axes(camera_pos, camera_orientation)

############### Transformation ############### 
def extract_camera_position_from_view_matrix(view_matrix):
    """
    Extract the camera position from the view matrix.
    """
    view_mat = np.array(view_matrix).reshape(4, 4, order='F')
    rotation_matrix = view_mat[:3, :3]
    translation_vector = view_mat[:3, 3]
    return (-np.dot(rotation_matrix.T, translation_vector)).tolist()

def extract_camera_orientation_from_view_matrix(view_matrix):
    """
    Extract the camera orientation quaternion from the view matrix.
    """
    view_mat = np.array(view_matrix).reshape(4, 4, order='F')
    rotation_matrix_T = view_mat[:3, :3].T
    return rotation_matrix_to_quaternion(rotation_matrix_T)

def compute_camera_fov_at_height(view_matrix, fov, aspect, target_height):
    """
    Calculate the x and y limits (field of view) at a given height based on camera parameters.

    Args:
        view_matrix (list or np.array): The view matrix of the camera.
        fov (float): Vertical field of view in degrees.
        aspect (float): Aspect ratio (width / height).
        target_height (float): The height (z) at which to calculate the field of view.

    Returns:
        tuple: (xlim, ylim) representing the field of view at the given height in world coordinates.
    """
    # Extract camera position and orientation
    camera_pos = np.array(extract_camera_position_from_view_matrix(view_matrix))
    camera_orientation = extract_camera_orientation_from_view_matrix(view_matrix)

    # Convert quaternion to rotation matrix
    rot_matrix = np.array(p.getMatrixFromQuaternion(camera_orientation)).reshape(3, 3)

    # Compute the vertical and horizontal field of view in radians
    fov_y = np.radians(fov)
    fov_x = 2 * np.arctan(np.tan(fov_y / 2) * aspect)

    # Define image plane corners in normalized device coordinates (NDC)
    ndc_corners = [(-1, -1), (-1, 1), (1, -1), (1, 1)]  # (x_ndc, y_ndc)

    # Compute the directions of the corner rays in camera space
    dirs_camera = []
    for x_ndc, y_ndc in ndc_corners:
        x = x_ndc * np.tan(fov_x / 2)
        y = y_ndc * np.tan(fov_y / 2)
        z = -1  # Camera looks along negative Z-axis
        dir_camera = np.array([x, y, z])
        dir_camera /= np.linalg.norm(dir_camera)
        dirs_camera.append(dir_camera)

    # Transform direction vectors to world space
    dirs_world = [rot_matrix @ dir_camera for dir_camera in dirs_camera]

    # Compute intersection points with the plane z = target_height
    x_coords = []
    y_coords = []

    for dir_world in dirs_world:
        denom = dir_world[2]
        if np.isclose(denom, 0):
            # The ray is parallel to the plane; skip or handle appropriately
            continue

        t = (target_height - camera_pos[2]) / denom
        # Intersection point
        intersection_point = camera_pos + dir_world * t

        # Collect x and y coordinates
        x_coords.append(intersection_point[0])
        y_coords.append(intersection_point[1])

    # Determine xlim and ylim
    xlim = (min(x_coords), max(x_coords))
    ylim = (min(y_coords), max(y_coords))

    return xlim, ylim


if __name__ == '__main__':
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    p.loadURDF("plane.urdf")
    p.loadURDF("r2d2.urdf", [0, 0, 1])

    fig, axes = initialize_visual_plots()

    cam_target_pos, cam_distance = [0, 0, 0.0], 2.0
    width, height = 64 * 4, 64 * 3
    cam_yaw, cam_pitch, cam_roll = 90, -90, 0
    fov, aspect = 45, 1.33
    near, far = 0.01, 10.0
    visualize_camera(cam_target_pos, cam_distance, cam_yaw, cam_pitch, cam_roll, fov, aspect, near, far)
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=cam_target_pos, distance=cam_distance,
        yaw=cam_yaw, pitch=cam_pitch, roll=cam_roll, upAxisIndex=2
    )
    workspace_xlim, workspace_ylim = compute_camera_fov_at_height(
        view_matrix=view_matrix,
        fov=fov, aspect=aspect, target_height=0.0
    )
    workspace_zlim = [0.0, 1.0]

    # Simulation loop
    for _ in range(1000):
        p.stepSimulation()
        time.sleep(1./240.)

        # Capture RGB and Depth images
        rgb_img, depth_img, view_matrix = capture_rgbd_image(
            cam_target_pos=cam_target_pos, cam_distance=cam_distance,
            width=width, height=height,
            cam_yaw=cam_yaw, cam_pitch=cam_pitch, cam_roll=cam_roll,
            fov=fov, aspect=aspect, near=near, far=far
        )

        # Generate point cloud with color
        point_cloud, colors = depth_to_point_cloud_with_color(
            depth_img, rgb_img, fov=fov, aspect=aspect,
            width=width, height=height, view_matrix=view_matrix, to_world=True
        )

        color_heightmap, depth_heightmap = point_cloud_to_height_map(
            point_cloud=point_cloud, colors=colors,
            workspace_xlim=workspace_xlim, workspace_ylim=workspace_ylim, workspace_zlim=workspace_zlim)

        # Update the plots
        plot_rgb_pointcloud_heightmap(rgb_img, point_cloud, colors, depth_heightmap, fig, axes)

    # Disconnect the PyBullet simulation
    p.disconnect()
