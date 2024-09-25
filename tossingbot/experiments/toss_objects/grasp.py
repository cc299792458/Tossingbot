from tossingbot.utils.misc_utils import set_seed
from tossingbot.agents.base_agent import BaseAgent
from tossingbot.envs.pybullet.tasks import TossObjects
from tossingbot.networks import PerceptionModule, GraspingModule, ThrowingModule

if __name__ == '__main__':
    set_seed()
    # Env
    env = TossObjects()

    # Networks
    perception_module = PerceptionModule()
    grasping_module = GraspingModule()
    throwing_module = ThrowingModule()

    # Agent
    agent = BaseAgent(perception_module=perception_module, grasping_module=grasping_module, throwing_module=throwing_module)

    # Main loop
    obs, info = env.reset()
    while True:
        action = agent.predict(obs)
        env.step(action=action)