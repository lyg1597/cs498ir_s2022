#Imports

#If you have wurlitzer installed, this will help you catch printouts from Klamp't
#Note: doesn't work on Windows
#%load_ext wurlitzer

import time
from klampt import *
from klampt import vis
from klampt.math import vectorops,so3,se3
from klampt.model.trajectory import Trajectory
from klampt.io import numpy_convert
from klampt.model.contact import ContactPoint
import numpy as np
import math
import random
import os
import sys
sys.path.append('../common')
import known_grippers
vis.init('IPython')

closeup_viewport = {'up': {'z': 0, 'y': 1, 'x': 0}, 'target': {'z': 0, 'y': 0, 'x': 0}, 'near': 0.1, 'position': {'z': 1.0, 'y': 0.5, 'x': 0.0}, 'far': 1000}

finger_radius = 0.01

class AntipodalGrasp:
    """A structure containing information about antipodal grasps.
    
    Attributes:
        center (3-vector): the center of the fingers (object coordinates).
        axis (3-vector): the direction of the line through the
            fingers (object coordinates).
        approach (3-vector, optional): the direction that the fingers
            should move forward to acquire the grasp.
        finger_width (float, optional): the width that the gripper should
            open between the fingers.
        contact1 (ContactPoint, optional): a point of contact on the
            object.
        contact2 (ContactPoint, optional): another point of contact on the
            object.
    """
    def __init__(self,center,axis):
        self.center = center
        self.axis = axis
        self.approach = None
        self.finger_width = None
        self.contact1 = None
        self.contact2 = None

    def add_to_vis(self,name,color=(1,0,0,1)):
        if self.finger_width == None:
            w = 0.05
        else:
            w = self.finger_width*0.5+finger_radius
        a = vectorops.madd(self.center,self.axis,w)
        b = vectorops.madd(self.center,self.axis,-w)
        vis.add(name,[a,b],color=color)
        if self.approach is not None:
            vis.add(name+"_approach",[self.center,vectorops.madd(self.center,self.approach,0.05)],color=(1,0.5,0,1))

#define some quantities of the gripper
gripper = known_grippers.robotiq_85
finger_tip = vectorops.madd(gripper.center,gripper.primaryAxis,gripper.fingerLength-0.005)
finger_closure_axis = gripper.secondaryAxis

temp_world = WorldModel()
res = temp_world.readFile(os.path.join('../data/gripperinfo',gripper.klamptModel))
if res == False:
    raise IOError("Unable to load file",gripper.klamptModel)
#merge the gripper parts into a static geometry
gripper_geom = gripper.getGeometry(temp_world.robot(0))

world2 = WorldModel()
obj2 = world2.makeRigidObject("object1")
obj2.geometry().loadFile("../data/objects/ycb-select/048_hammer/nontextured.ply")

#this will perform a reasonable center of mass / inertia estimate
m = obj2.getMass()
m.estimate(obj2.geometry(),mass=0.908,surfaceFraction=0.0)
obj2.setMass(m)

#make the object transparent yellow
obj2.appearance().setColor(0.8,0.8,0.2,0.5)
world2.readFile("../data/terrains/plane.env")
world2.terrain(0).geometry().scale(0.1)
world2.terrain(0).appearance().setColor(0,0,0.5,0.5)

########################## Problem 1.A code goes here ##############################

def match_grasp(finger_tip,finger_closure_axis,grasp):
    """
    Args:
        finger_tip (3-vector): local coordinates of the center-point between the gripper's fingers.
        finger_closure_axis (3-vector): local coordinates of the axis connecting the gripper's fingers.
        grasp (AntipodalGrasp): the desired grasp
        
    Returns:
        (R,t): a Klampt se3 element describing the maching gripper transform
    """
    return se3.identity()

####################################################################################

#Problem 1.A. Find a rotation to match the gripper to the antipodal grasp

grasp1 = AntipodalGrasp([0.025,-0.15,0.015],[math.cos(math.radians(20)),math.sin(math.radians(20)),0])
grasp1.finger_width = 0.05
gripper_geom.setCurrentTransform(*match_grasp(finger_tip,finger_closure_axis,grasp1))

vis.createWindow()
vis.setViewport(closeup_viewport)
vis.add("world",world2)
vis.add("gripper",gripper_geom)

grasps = [grasp1]
for i,g in enumerate(grasps):
    name = "grasp{}".format(i)
    g.add_to_vis(name,(1,0,0,1)) 
vis.run()