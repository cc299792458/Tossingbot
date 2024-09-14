import time
import pybullet as p
import pybullet_data

from tossingbot.scene import setup_scene
from tossingbot.utils.misc_utils import set_seed
from tossingbot.scene.camera import capture_rgbd_image, depth_to_point_cloud_with_color

if __name__ == '__main__':
    ###### Set up XXXX #####
    set_seed()
    physics_client_id = p.connect(p.GUI)  
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    ##### Set up scence, including objects and robot #####
    plane_id, workspace_ids, box_ids, object_ids, robot = setup_scene()
    
    ##### Main Loop #####
    while True:
        p.stepSimulation()
        time.sleep(1./240.)

    p.disconnect()