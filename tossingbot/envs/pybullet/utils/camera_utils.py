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

def point_cloud_to_height_map(point_cloud, colors, workspace_width, workspace_length, workspace_position):
    height_map = None

    return height_map

############ Visualization ############
def initialize_visual_plots(figsize=(12, 6)):
    """
    Initialize the matplotlib figure for real-time display.
    """
    plt.ion()
    fig = plt.figure(figsize=figsize)
    ax_rgb = fig.add_subplot(1, 2, 1)
    ax_pointcloud = fig.add_subplot(1, 2, 2, projection='3d')
    ax_rgb.set_title("RGB Image")
    ax_rgb.axis('off')
    ax_pointcloud.set_title("Colored Point Cloud")
    ax_pointcloud.set_xlabel("X")
    ax_pointcloud.set_ylabel("Y")
    ax_pointcloud.set_zlabel("Z")
    return fig, (ax_rgb, ax_pointcloud)

def plot_rgb_pointcloud(rgb_img, point_cloud, colors, fig, axes, xlim=None, ylim=None, zlim=None):
    """
    Plot the RGB image and Point Cloud in two subplots.
    """
    ax_rgb, ax_pointcloud = axes
    ax_rgb.clear()
    ax_rgb.imshow(rgb_img)
    ax_rgb.set_title("RGB Image")
    ax_rgb.axis('off')

    ax_pointcloud.clear()
    ax_pointcloud.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c=colors, s=1)
    ax_pointcloud.set_title("Colored Point Cloud")
    ax_pointcloud.set_xlabel("X")
    ax_pointcloud.set_ylabel("Y")
    ax_pointcloud.set_zlabel("Z")
    ax_pointcloud.set_xlim(xlim)
    ax_pointcloud.set_ylim(ylim)
    ax_pointcloud.set_zlim(zlim)

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

if __name__ == '__main__':
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    p.loadURDF("plane.urdf")
    p.loadURDF("r2d2.urdf", [0, 0, 1])

    fig, axes = initialize_visual_plots()

    cam_target_pos, cam_distance = [0, 0, 0.75], 2
    width, height = 64 * 4, 64 * 3
    cam_yaw, cam_pitch, cam_roll = 90, -90, 0
    fov, aspect = 45, 1.33
    near, far = 0.01, 10.0

    visualize_camera(cam_target_pos, cam_distance, cam_yaw, cam_pitch, cam_roll, fov, aspect, near, far)

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

        # Update the plots
        plot_rgb_pointcloud(rgb_img, point_cloud, colors, fig, axes)

    # Disconnect the PyBullet simulation
    p.disconnect()
