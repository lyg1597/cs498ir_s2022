from sympy import C
from simulated_robot import createRobotController
from klampt.control.interop import RobotInterfacetoVis
from klampt.control import StepContext
from klampt.math import so3,se3,vectorops
from klampt.model import sensing
from klampt.io import resource
from klampt import *
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import scipy.cluster.hierarchy as hcluster
import time

class TwoLayerNet(torch.nn.Module):
    def __init__(self):
        super(TwoLayerNet, self).__init__()
        self.control1 = torch.nn.Linear(2,32)
        self.control2 = torch.nn.Linear(32,32)
        self.control3 = torch.nn.Linear(32,2)

    def forward(self,x):
        h2 = torch.relu(self.control1(x))
        h3 = torch.relu(self.control2(h2))
        u = self.control3(h3)
        return u


TARGET_LOCATION = [0,0.2,0]

if __name__ == '__main__':
    controller = createRobotController()
    rgbd_sensor = controller.robotModel().sensor('rgbd_camera')
    Tcamera_world = sensing.get_sensor_xform(rgbd_sensor)  #transform of the camera in the world frame (which is also the robot base frame)
    q_out_of_the_way = resource.get('out_of_the_way.config')

    controllerVis = RobotInterfacetoVis(controller.arm)
    plotShown = False
    im = None

    block_location = [0,0,0]
    state = "OperateMode.IDLE"
    block_idx = 0

    model = TwoLayerNet()
    model.load_state_dict(torch.load("perception_model"))
    
    def initVis():
        vis.add("world",controller.world)
        vis.addAction(controller.toggleVacuum,'Toggle vacuum','v')
        
    def loopVis():
        global plotShown,im
        global block_location, state, block_idx
        with StepContext(controller):

            #print the flow sensor if the vacuum is on
            # if controller.getVacuumCommand() > 0:
            #     print("Flow:",controller.getVacuumFlow())

            #update the Matplotlib window if the sensor is working
            rgb,depth = controller.rgbdImages()
            if rgb is not None:
                #funky stuff to make sure that the image window updates quickly
                if not plotShown:
                    im = plt.imshow(rgb)
                    plt.show(block=False)
                    plotShown = True
                else:
                    im.set_array(rgb)
                    plt.pause(0.01)

            #TODO: fill me out to perform image-based pick and place planning to create as much of a stack as you can at TARGET_LOCATION
            #
            #You are NOT allowed to cheat and access controller.sim or controller.world.
            #
            #You will want to implement a state machine...
            #
            if state == 'OperateMode.IDLE' and controller.arm.destinationTime()-controller.arm.clock() < 0.05:
                print('OperateMode.IDLE')
                controller.setArmPosition(q_out_of_the_way)
                state = 'move_out_of_way_wait'
            elif state == 'move_out_of_way_wait':
                if controller.arm.destinationTime()-controller.arm.clock() < 0.05:
                    state = 'OperateMode.APPROACH'
            elif state == "OperateMode.APPROACH" and controller.arm.destinationTime()-controller.arm.clock() < 0.05:
                print("APPROACH")
                # Compute location of block
                true_array = depth<0.38
                idx_array = np.argwhere(true_array)
                if idx_array.size == 0:
                    state = "OperateMode.STOP"
                    return
                clusters = hcluster.fclusterdata(idx_array, 5, criterion="distance")
                idx = np.random.randint(clusters.min(), clusters.max()+1)
                idx_array = idx_array[np.argwhere(clusters==idx).flatten(),:]
                pixel = np.mean(idx_array,axis=0)
                pixel_x = pixel[1]
                pixel_y = pixel[0]
                pos = model(torch.FloatTensor([pixel_x, pixel_y])).detach().numpy()
                pos_x = pos[0]
                pos_y = pos[1]
                block_location[0] = pos_x 
                block_location[1] = pos_y
                block_location[2] = 0

                # for i in range(clusters.min(), clusters.max()+1):
                #     tmp = np.argwhere(clusters==i)

                controller.arm.moveToCartesianPosition((so3.identity(),vectorops.add(block_location,[0,0,0.1])))
                state = "OperateMode.PICK_DOWN"
            elif state == "OperateMode.PICK_DOWN" and controller.arm.destinationTime()-controller.arm.clock() < 0.05:
                print("PICK_DOWN")
                controller.arm.moveToCartesianPosition((so3.identity(),vectorops.add(block_location,[0,0,0.025])))
                state = "OperateMode.PICK"
            elif state == "OperateMode.PICK":
                print("PICK")
                controller.setVacuumOn()
                state = "OperateMode.PICK_UP"
            elif state == "OperateMode.PICK_UP" and controller.arm.destinationTime()-controller.arm.clock() < 0.05:
                print("PICK_UP")
                controller.arm.moveToCartesianPosition((so3.identity(),vectorops.add(block_location,[0,0,0.2])))
                state = "OperateMode.TRANSIT"
            elif state == "OperateMode.TRANSIT" and controller.arm.destinationTime()-controller.arm.clock() < 0.05:
                print("TRANSIT")
                controller.arm.moveToCartesianPosition((so3.identity(),vectorops.add(TARGET_LOCATION,[0,0,0.2])))
                state = "OperateMode.RELEASE_DOWN"
            elif state == "OperateMode.RELEASE_DOWN" and controller.arm.destinationTime()-controller.arm.clock() < 0.05:
                print("RELEASE_DOWN")
                controller.arm.moveToCartesianPosition((so3.identity(),vectorops.add(TARGET_LOCATION,[0,0,0.01+block_idx*0.025+0.025])))
                block_idx += 1
                state = "OperateMode.RELEASE"
            elif state == "OperateMode.RELEASE" and controller.arm.destinationTime()-controller.arm.clock() < 0.05:
                print("RELEASE")
                controller.setVacuumOff()
                state = "OperateMode.RELEASE_UP"
            elif state == "OperateMode.RELEASE_UP":
                print("RELEASE_UP")
                controller.arm.moveToCartesianPosition((so3.identity(),vectorops.add(TARGET_LOCATION,[0,0,0.2])))
                state = "OperateMode.IDLE"
            elif state == "OperateMode.STOP":
                pass
            controllerVis.update()

            
    def closeVis():
        controller.close()

    #maximum compability with Mac
    vis.loop(initVis,loopVis,closeVis)
