import numpy as np

def slerp(q0, q1, t_array):
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot_product = np.dot(q0, q1)

    # Clamp to avoid numerical errors
    dot_product = np.clip(dot_product, -1.0, 1.0)
    theta_0 = np.arccos(dot_product)  # Angle between q0 and q1
    sin_theta_0 = np.sin(theta_0)

    # If the angle is small, use linear interpolation to avoid division by zero
    if sin_theta_0 < 1e-6:
        return np.array([(1 - t) * q0 + t * q1 for t in t_array])

    return np.array([
        (np.sin((1 - t) * theta_0) / sin_theta_0) * q0 +
        (np.sin(t * theta_0) / sin_theta_0) * q1
        for t in t_array
    ])