import numpy as np

def slerp(q0, q1, t_array):
    # Ensure inputs are unit quaternions
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot_product = np.dot(q0, q1)

    # Clamp the dot product to avoid numerical errors
    dot_product = np.clip(dot_product, -1.0, 1.0)
    theta_0 = np.arccos(dot_product)  # Angle between q0 and q1
    sin_theta_0 = np.sin(theta_0)

    # If the angle is small, use linear interpolation to avoid division by zero
    if sin_theta_0 < 1e-6:
        return np.array([(1 - t) * q0 + t * q1 for t in t_array])

    # Perform the spherical linear interpolation (slerp)
    return np.array([
        (np.sin((1 - t) * theta_0) / sin_theta_0) * q0 +
        (np.sin(t * theta_0) / sin_theta_0) * q1
        for t in t_array
    ])


def pose_distance(pose1, pose2):
    """
    Calculate the distance between two poses.
    
    Args:
        pose1 (list): First pose as [position, orientation].
        pose2 (list): Second pose as [position, orientation].
        
    Returns:
        dict: A dictionary containing 'position_distance' and 'orientation_distance'.
    """
    position1 = np.array(pose1[0])
    position2 = np.array(pose2[0])
    position_distance = np.linalg.norm(position1 - position2)

    quat1 = np.array(pose1[1])
    quat2 = np.array(pose2[1])
    dot_product = np.dot(quat1, quat2)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    orientation_distance = 2 * np.arccos(np.abs(dot_product))

    return {
        'position_distance': position_distance,
        'orientation_distance': orientation_distance
    }

def rotation_matrix_to_quaternion(rotation_matrix):
    """
    Converts a 3x3 rotation matrix to a quaternion.
    
    Args:
        rotation_matrix (np.ndarray): 3x3 rotation matrix.
        
    Returns:
        list: Quaternion [x, y, z, w].
    """
    # Ensure the matrix is 3x3
    assert rotation_matrix.shape == (3, 3), "Rotation matrix must be 3x3"

    # Extract elements from the rotation matrix
    R = rotation_matrix
    trace = np.trace(R)

    if trace > 0:
        s = 2.0 * np.sqrt(1.0 + trace)
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    return [qx, qy, qz, qw]

def rotate_image_array(image_array, theta):
    """
    Rotate a 2D array (such as an image or heightmap) by a specified angle.

    Parameters:
        image_array (numpy.ndarray): The input 2D array of shape (H, W) or (H, W, C).
        theta (float): The angle in degrees to rotate the image.

    Returns:
        rotated_image: The rotated 2D array.
    """
    # Get the original height and width
    H, W = image_array.shape[:2]

    # Calculate diagonal length
    diag_length = np.sqrt(H**2 + W**2)

    # Determine padding size to make it a multiple of 32
    padding_height = int((diag_length - H) / 2)
    padding_width = int((diag_length - W) / 2)

    new_height, new_width = H + padding_height * 2, W + padding_width * 2

    # Apply padding
    if image_array.ndim == 3:
        padded_image = np.pad(image_array, 
                              ((padding_height, padding_height), 
                               (padding_width, padding_width), 
                               (0, 0)), 
                              mode='constant', constant_values=0)
    else:
        padded_image = np.pad(image_array, 
                              ((padding_height, padding_height), 
                               (padding_width, padding_width)), 
                              mode='constant', constant_values=0)

    # Generate the rotation matrix
    rotate_theta = np.radians(theta)
    rotation_matrix = np.array([[np.cos(-rotate_theta), np.sin(-rotate_theta)],
                                 [-np.sin(-rotate_theta), np.cos(-rotate_theta)]])

    # Generate rotated coordinates
    grid_y, grid_x = np.indices(padded_image.shape[:2])
    coords = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(float)

    # Adjust for padding
    coords -= np.array([(W / 2) + padding_width - 0.5, (H / 2) + padding_height - 0.5])

    # Apply rotation
    rotated_coords = (rotation_matrix @ coords.T).T

    # Convert coordinates to image indices
    x_rotated = np.clip(rotated_coords[:, 0] + (W / 2 + padding_width - 0.5), 0, padded_image.shape[1] - 1)
    y_rotated = np.clip(rotated_coords[:, 1] + (H / 2 + padding_height - 0.5), 0, padded_image.shape[0] - 1)

    # Use nearest neighbor interpolation to fill the rotated image
    rotated_image = padded_image[y_rotated.astype(int), x_rotated.astype(int)]

    rotated_image = rotated_image.reshape(new_height, new_width, -1 if image_array.ndim == 3 else 1)

    # Remove padding from the result
    return rotated_image[padding_height:-padding_height, padding_width:-padding_width]