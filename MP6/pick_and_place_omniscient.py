from ast import operator
from webbrowser import Opera
from rosdep2 import DownloadFailure
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
from enum import Enum,auto
import time

TARGET_LOCATION = [0,0.2,0]

class OperateMode(Enum):
    APPROACH = auto()
    PICK_DOWN = auto()
    PICK_UP = auto()
    TRANSIT = auto()
    RELEASE_DOWN = auto() 
    RELEASE_UP = auto()
    IDLE = auto()
    STOP = auto()
    PICK = auto()
    RELEASE = auto()

object_idx = 0    
state = OperateMode.IDLE
tmp_time = time.time()

if __name__ == '__main__':
    controller = createRobotController()
    rgbd_sensor = controller.robotModel().sensor('rgbd_camera')
    Tcamera_world = sensing.get_sensor_xform(rgbd_sensor)  #transform of the camera in the world frame (which is also the robot base frame)

    controllerVis = RobotInterfacetoVis(controller.arm)

    def initVis():
        vis.add("world",controller.world)
        vis.addAction(controller.toggleVacuum,'Toggle vacuum','v')
        
    def loopVis():
        global object_idx, state, tmp_time
        with StepContext(controller):

            #Fake sensor that produces ground truth object positions and orientations
            block_transforms = []
            for i in range(controller.world.numRigidObjects()):
                body = controller.sim.body(controller.world.rigidObject(i))
                block_transforms.append(body.getTransform())

            #TODO: fill me out to perform pick and place planning to create as much of a stack as you can at TARGET_LOCATION
            #
            #You will want to implement a state machine...
            #
            # state = OperateMode.IDLE
            # for i in range(len(block_transforms)):
            #     # controller.beginStep()
            #     controller.arm.moveToCartesianPosition((block_transforms[i][0],vectorops.add(block_transforms[i][1],[0,0,0.01])))
            #     # controller.endStep()
            #     # time.sleep(10)
            if state == OperateMode.IDLE:
                tmp_time = time.time()
                state = OperateMode.APPROACH
            elif state == OperateMode.APPROACH and controller.arm.destinationTime()-controller.arm.clock() < 0.05:
                tmp_time = time.time()
                state = OperateMode.PICK_DOWN
            elif state == OperateMode.PICK_DOWN and controller.arm.destinationTime()-controller.arm.clock() < 0.05:
                tmp_time = time.time()
                state = OperateMode.PICK
            elif state == OperateMode.PICK:
                tmp_time = time.time()
                state = OperateMode.PICK_UP
            elif state == OperateMode.PICK_UP and controller.arm.destinationTime()-controller.arm.clock() < 0.05:
                tmp_time = time.time()
                state = OperateMode.TRANSIT
            elif state == OperateMode.TRANSIT and controller.arm.destinationTime()-controller.arm.clock() < 0.05:
                tmp_time = time.time()
                state = OperateMode.RELEASE_DOWN
            elif state == OperateMode.RELEASE_DOWN and controller.arm.destinationTime()-controller.arm.clock() < 0.05:
                tmp_time = time.time()
                state = OperateMode.RELEASE
            elif state == OperateMode.RELEASE:
                tmp_time = time.time()
                state = OperateMode.RELEASE_UP
            elif state == OperateMode.RELEASE_UP and controller.arm.destinationTime()-controller.arm.clock() < 0.05:
                tmp_time = time.time()
                if object_idx == len(block_transforms)-1:
                    state = OperateMode.STOP
                else:
                    state = OperateMode.IDLE
            
            if state == OperateMode.IDLE:
                object_idx += 1
            elif state == OperateMode.APPROACH:
                controller.arm.moveToCartesianPosition((block_transforms[object_idx][0],vectorops.add(block_transforms[object_idx][1],[0,0,0.1])))
            elif state == OperateMode.PICK_DOWN:
                controller.arm.moveToCartesianPosition((block_transforms[object_idx][0],vectorops.add(block_transforms[object_idx][1],[0,0,0.01])))
            elif state == OperateMode.PICK:
                controller.setVacuumOn()
            elif state == OperateMode.PICK_UP:
                controller.arm.moveToCartesianPosition((block_transforms[object_idx][0],vectorops.add(block_transforms[object_idx][1],[0,0,0.1])))
            elif state == OperateMode.TRANSIT:
                controller.arm.moveToCartesianPosition((block_transforms[object_idx][0],vectorops.add(TARGET_LOCATION,[0,0,0.1])))
            elif state == OperateMode.RELEASE_DOWN:
                controller.arm.moveToCartesianPosition((block_transforms[object_idx][0],vectorops.add(TARGET_LOCATION,[0,0,0.01+object_idx*0.025])))
            elif state == OperateMode.RELEASE:
                controller.setVacuumOff()
            elif state == OperateMode.RELEASE_UP:
                controller.arm.moveToCartesianPosition((block_transforms[object_idx][0],vectorops.add(TARGET_LOCATION,[0,0,0.1])))
            elif state == OperateMode.STOP:
                return

            controllerVis.update()

            
    def closeVis():
        controller.close()

    #maximum compability with Mac
    vis.loop(initVis,loopVis,closeVis)
    