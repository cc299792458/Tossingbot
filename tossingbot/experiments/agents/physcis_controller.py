"""
    In this experiment, we test the PhysicsController by predicting the 
    release point and velocity for an object that we want to toss to a 
    target position with a given launch angle.
"""

import numpy as np

from tossingbot.agents.physics_agent import PhysicsController

if __name__ == '__main__':
    controller = PhysicsController(r_h=0.5, r_z=0.4)

    phi_degs = [0, 30, 45, 60]

    for phi_deg in phi_degs:
        # Predict the release point and velocity based on the target position and launch angle
        (r_x, r_y, r_z), (v_x, v_y, v_z) = controller.predict(target_pos=[1.0, 0.0, 0.2], phi_deg=phi_deg)
        v_magnitude = np.linalg.norm([v_x, v_y, v_z])
        print(f"Predicted velocity magnitude when phi is {phi_deg} degrees: {v_magnitude}")