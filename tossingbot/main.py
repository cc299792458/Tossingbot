import time
import pybullet as p
import pybullet_data

from tossingbot.robot import UR5Robotiq85

if __name__ == '__main__':
    physics_client_id = p.connect(p.GUI)  
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,-9.81)
    planeId = p.loadURDF("plane.urdf") 
    
    robot = UR5Robotiq85((0, 0.0, 0.0), (0, 0, 0))
    
    for i in range(10000):
        p.stepSimulation()
        time.sleep(1./240.)

    p.disconnect()