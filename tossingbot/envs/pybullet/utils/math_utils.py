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