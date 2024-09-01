import time
import pybullet as p
import pybullet_data

from tossingbot.robot import UR5Robotiq85

if __name__ == '__main__':
    physicsClient = p.connect(p.GUI)  
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,-9.81)
    planeId = p.loadURDF("plane.urdf") 
    
    startPos = [0,0,1]
    startOrientation = p.getQuaternionFromEuler([0,0,0])
    # boxId = p.loadURDF("r2d2.urdf",startPos, startOrientation)
    robot = UR5Robotiq85((0, 0.0, 0.0), (0, 0, 0))
    
    for i in range(10000):
        p.stepSimulation() 
        time.sleep(1./240.)

    # cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
    # print(cubePos,cubeOrn) 
    p.disconnect()