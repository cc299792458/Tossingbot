import torch

from tossingbot.utils.misc_utils import set_seed
from tossingbot.agents.base_agent import BaseAgent
from tossingbot.envs.pybullet.tasks import TossObjects
from tossingbot.envs.pybullet.utils.camera_utils import plot_heightmaps
from tossingbot.networks import PerceptionModule, GraspingModule, ThrowingModule

if __name__ == '__main__':
    set_seed()

    # Set device (use GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Parameters
    n_rotations = 1

    # Env
    env = TossObjects(camera_config={'n_rotations': n_rotations})

    # Networks
    perception_module = PerceptionModule()
    grasping_module = GraspingModule()
    throwing_module = ThrowingModule()

    # Agent
    agent = BaseAgent(device=device, perception_module=perception_module, grasping_module=grasping_module, throwing_module=throwing_module)

    # Main loop
    obs, info = env.reset()
    while True:
        action, intermidiates = agent.predict(obs, n_rotations=n_rotations)
        # plot_heightmaps(intermidiates['depth_heightmaps'])
        obs, reward, terminated, truncated, info = env.step(action=action)