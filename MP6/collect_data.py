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
import time
import pickle

DATASET_FOLDER = 'generated_data'

def randomize_world(world : WorldModel, sim : Simulator):
    """Helper function to help reset the world state. """
    for i in range(world.numRigidObjects()):
        posx = np.random.uniform(0.1,0.3)
        posy = np.random.uniform(-0.2,0.2)
        obj = world.rigidObject(i)
        #TODO: sample object positions
        #Bad things will happen to the sim if the objects are colliding!
        T = obj.getTransform()
        obj.setTransform(T[0],vectorops.add([posx,posy,0],[0,0,0.015]))

    #reset the sim bodies -- this code doesn't need to be changed
    for i in range(world.numRigidObjects()):
        model = world.rigidObject(i)
        body = sim.body(model)
        body.setVelocity([0,0,0],[0,0,0])
        body.setObjectTransform(*model.getTransform())


if __name__ == '__main__':
    controller = createRobotController()
    rgbd_sensor = controller.robotModel().sensor('rgbd_camera')
    Tcamera_world = sensing.get_sensor_xform(rgbd_sensor)  #transform of the camera in the world frame (which is also the robot base frame)
    q_out_of_the_way = resource.get('out_of_the_way.config')

    controllerVis = RobotInterfacetoVis(controller.arm)
    plotShown = False
    im = None
    numExamples = 0
    grasp_data = None 
    grasp_image = None 
    state = 'move_out_of_way'
    data_input = []
    data_output = []

    def initVis():
        vis.add("world",controller.world)
        vis.addAction(controller.toggleVacuum,'Toggle vacuum','v')
        vis.addAction(lambda : randomize_world(controller.world,controller.sim),"Randomize world",'r')
        
    def loopVis():
        global plotShown,im,numExamples
        global state
        global data_input 
        global data_output
        with StepContext(controller):
            #TODO: fill me out to perform self-supervised data generation -- will want to generate a
            #target, try grasping, and try lifting. Then use your sensors to determine whether you
            #have grasped the object, and then save the image and grasp location.
            #
            #You will want to implement a state machine...
            #   

            rgb, depth = None, None
            if state == 'move_out_of_way':
                print('move_out_of_way')
                controller.setArmPosition(q_out_of_the_way)
                state = 'move_out_of_way_wait'
            elif state == 'move_out_of_way_wait':
                if controller.arm.destinationTime()-controller.arm.clock() < 0.05:
                    state = "collect"
            elif state == "collect":
                print(f"collect{numExamples}")
                randomize_world(controller.world,controller.sim)
                controllerVis.update()

                time.sleep(0.01)
                # Get ground truth             
                posx = 0
                posy = 0
                for i in range(controller.world.numRigidObjects()):
                    obj = controller.world.rigidObject(i)
                    #TODO: sample object positions
                    #Bad things will happen to the sim if the objects are colliding!
                    T = obj.getTransform()
                    posx = T[1][0]
                    posy = T[1][1]
                data_output.append([posx,posy])
                rgb, depth = controller.rgbdImages()
                rgb, depth = controller.rgbdImages()

                true_array = depth<0.38
                idx_array = np.argwhere(true_array)
                mean_pos = np.mean(idx_array, axis = 0)
                data_input.append(mean_pos)
                numExamples += 1

            #update the Matplotlib window if the sensor is working
            if rgb is not None:
                #funky stuff to make sure that the image window updates quickly
                if not plotShown:
                    im = plt.imshow(rgb)
                    plt.show(block=False)
                    plotShown = True
                else:
                    im.set_array(rgb)
                    plt.gcf().canvas.draw()
            if numExamples > 5000:
                vis.kill()
            controllerVis.update()

    def closeVis():
        controller.close()

    #maximum compability with Mac
    vis.loop(initVis,loopVis,closeVis)
    # print(data_input, data_output)
    data_input = data_input[1:]
    data_output = data_output[:-1]
    assert len(data_input) == len(data_output)
    with open('train_data.pickle','wb+') as f:
        pickle.dump((data_input, data_output),f)
