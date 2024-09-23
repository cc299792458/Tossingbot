from tossingbot.utils.misc_utils import set_seed
from tossingbot.envs.pybullet.tasks import TossObjects

if __name__ == '__main__':
    set_seed()

    env = TossObjects()
    while True:
        env.step()