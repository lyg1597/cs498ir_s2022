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

DATASET_FOLDER = 'generated_data'

def randomize_world(world : WorldModel, sim : Simulator):
    """Helper function to help reset the world state. """
    region_code_choice = np.random.choice([0,1,2,3,4,5], size = world.numRigidObjects, replace = False)
    for i in range(world.numRigidObjects()):
        obj = world.rigidObject(i)
        #TODO: sample object positions
        #Bad things will happen to the sim if the objects are colliding!
        T = obj.getTransform()
        region_code = region_code_choice[i]
        tmp = int(region_code)/int(3)
        x_low, x_high = (0.1+tmp*0.1, 0.1+(tmp+1)*0.1)
        tmp = int(region_code)%3
        y_low, y_high = (-0.2 + tmp*0.4/3, -0.2+(tmp+1)*0.4/3)
        x = np.random.uniform(x_low, x_high)
        y = np.random.uniform(y_low, y_high)
        obj.setTransform(T[0],vectorops.add([x,y,T[1][2]],[0.01,0,0]))

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

    def initVis():
        vis.add("world",controller.world)
        vis.addAction(controller.toggleVacuum,'Toggle vacuum','v')
        vis.addAction(lambda : randomize_world(controller.world,controller.sim),"Randomize world",'r')
        
    def loopVis():
        global plotShown,im,numExamples
        global state
        global grasp_data,grasp_image
        with StepContext(controller):
            #TODO: fill me out to perform self-supervised data generation -- will want to generate a
            #target, try grasping, and try lifting. Then use your sensors to determine whether you
            #have grasped the object, and then save the image and grasp location.
            #
            #You will want to implement a state machine...
            #                
            
            rgb, depth = controller.rgbdImages()
            if state == 'move_out_of_way':
                print('move_out_of_way')
                controller.setArmPosition(q_out_of_the_way)
                state = 'move_out_of_way_wait'
            elif state == 'move_out_of_way_wait':
                if controller.arm.destinationTime()-controller.arm.clock() < 0.05:
                    state = 'move_target'
            elif state == 'move_target':
                state = 'move_target_wait'
                # Red rgbd image
                # Determine a position
                true_array = depth<0.38
                idx_array = np.argwhere(true_array)
                y,x = idx_array[np.random.randint(0,idx_array.shape[0]-1),:]
                # y = y-controller.image_h/2
                # x = x-controller.image_w/2
                # y = y/(controller.image_h/2*7)
                # x = x/(controller.image_w/2*5)
                A = np.array([[-0.0006,    0.0010],[-0.0006,    0.0012]])
                x,y = A@np.array([x,y])
                x,y,_ = se3.apply(Tcamera_world, [x,y,0])
                print(f'move_target {x},{y}')
                # grasp_image = rgb 
                grasp_data = [x,y]

                controller.arm.moveToCartesianPosition((so3.identity(),[grasp_data[0], grasp_data[1], 0.05]))
            elif state == 'move_target_wait':
                if controller.arm.destinationTime()-controller.arm.clock() < 0.05:
                    state = 'grasp'
            elif state == 'grasp':
                grasp_image = rgb 
                print('grasp')
                controller.setVacuumOn()
                state = 'move_up'
            elif state == 'move_up':
                print('move_up')
                flow = controller.getVacuumFlow()
                controller.arm.moveToCartesianPosition((so3.identity(),[grasp_data[0], grasp_data[1],0.1]))
                state = 'move_up_wait'
            elif state == 'move_up_wait':
                if controller.arm.destinationTime()-controller.arm.clock() < 0.05:
                    state = 'store'
            elif state == 'store':
                print('store')
                state = 'move_out_of_way'
                controller.setVacuumOff()
                tmp = Image.fromarray(grasp_image)
                tmp.save(f'{DATASET_FOLDER}/{numExamples}.png')
                numExamples += 1

            #print the flow sensor if the vacuum is on
            if controller.getVacuumCommand() > 0:
                print("Flow:",controller.getVacuumFlow())

            rgb,depth = controller.rgbdImages()
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

            controllerVis.update()

    def closeVis():
        controller.close()

    #maximum compability with Mac
    vis.loop(initVis,loopVis,closeVis)
    