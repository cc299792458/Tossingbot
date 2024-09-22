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