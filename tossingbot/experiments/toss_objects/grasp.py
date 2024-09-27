import torch

from tossingbot.utils.misc_utils import set_seed
from tossingbot.envs.pybullet.tasks import TossObjects
from tossingbot.envs.pybullet.utils.camera_utils import plot_heightmaps
from tossingbot.agents.physics_agent import PhysicsAgent, PhysicsController
from tossingbot.networks import PerceptionModule, GraspingModule, ThrowingModule

if __name__ == '__main__':
    set_seed()

    # Set device (use GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Parameters
    box_n_rows = 1
    box_n_cols = 1

    r_h = 0.5
    post_grasp_h = 0.3
    n_rotations = 1
    phi_deg = 0

    # Env
    env = TossObjects(
        scene_config={
            'box_n_rows': box_n_rows,
            'box_n_cols': box_n_cols,
        },
        camera_config={'n_rotations': n_rotations})

    # Networks
    perception_module = PerceptionModule()
    grasping_module = GraspingModule()
    throwing_module = ThrowingModule()

    # Agent
    physics_controller = PhysicsController(r_h=r_h)
    agent = PhysicsAgent(
        device=device, 
        perception_module=perception_module, 
        grasping_module=grasping_module, 
        throwing_module=throwing_module,
        physics_controller=physics_controller,
        post_grasp_h=post_grasp_h,
    )

    # Main loop
    obs, info = env.reset()
    while True:
        action, intermidiates = agent.predict(obs, n_rotations=n_rotations, phi_deg=phi_deg)
        # plot_heightmaps(intermidiates['depth_heightmaps'])
        obs, reward, terminated, truncated, info = env.step(action=action)