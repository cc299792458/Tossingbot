import time
import numpy as np
import pybullet as p
import pybullet_data

def capture_rgbd_image(
    cam_target_pos=[0, 0, 0], cam_distance=2, cam_yaw=45, cam_pitch=-30, cam_roll=0, 
    fov=60, aspect=1.0, near=0.1, far=100.0
):
    """
    Capture an RGB-D image (RGB and Depth) from the scene using the camera.

    Parameters:
    - cam_target_pos: List specifying the camera target position [x, y, z].
    - cam_distance: Distance from the camera to the target position.
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
    width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
        width=640,
        height=480,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_TINY_RENDERER  # Use TINY renderer for fast performance
    )

    # Convert the depth buffer to real depth values
    depth_buffer = np.array(depth_img).reshape(height, width)
    real_depth = far * near / (far - (far - near) * depth_buffer)  # Depth conversion formula

    return rgb_img, real_depth

if __name__ == '__main__':
    # Initialize PyBullet simulation
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    # Create a plane and a box in the simulation
    p.loadURDF("plane.urdf")
    p.loadURDF("r2d2.urdf", [0, 0, 1])  # Add an example robot

    # Capture RGB and Depth images in a loop
    for _ in range(1000):
        p.stepSimulation()
        rgb_img, depth_img = capture_rgbd_image(
            cam_target_pos=[0, 0, 1], cam_distance=2, cam_yaw=30, cam_pitch=-20, cam_roll=0,
            fov=45, aspect=1.33, near=0.2, far=5.0
        )  # Capture the RGB-D images with custom parameters
        time.sleep(1. / 240.)

    p.disconnect()
