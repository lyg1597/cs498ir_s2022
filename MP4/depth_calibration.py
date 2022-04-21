import numpy as np
from klampt.math import vectorops,se3
from klampt.model.geometry import fit_plane
from scipy.optimize import minimize
import random
import sys
sys.path.append("../common")
from rgbd import *
from rgbd_realsense import load_rgbd_dataset,sr300_factory_calib
import json

#set which problem you are working on. You can also set it on the command line
PROBLEM = '3a'
#PROBLEM = '3b'
#PROBLEM = '3c'
CALIBRATION = 'spec'
#CALIBRATION = 'camera'
SUBSET_PLANES = 1000

def mutual_orthogonality(scans,planesets,fx,fy,cx,cy):
    """Returns an objective function that returns the sum of orthogonality
    errors across the planes.
    
    Args:
        scans: list of RGBDScan objects (see common.rgbd)
        planesets: a list of lists of point indices forming the estimated
            planes.  These are read from the output of problem 2.
        fx,fy,cx,cy: the intrinsic parameters being calibrated.
    """
    cost = 0
    for scanno,(scan,planeset) in enumerate(zip(scans,planesets)):
        if len(planeset)<=1: continue
        # print("TODO: problem 3.A")
        #TODO: set up the point cloud that would have been obtained using the given intrinsic parameters
        scan.camera.depth_intrinsics['fx'] = fx 
        scan.camera.depth_intrinsics['fy'] = fy
        scan.camera.depth_intrinsics['cx'] = cx
        scan.camera.depth_intrinsics['cy'] = cy
        pc = scan.get_point_cloud(colors=False,normals=False,structured=True)
        # w,h = 640,480
        # # X = X*Z*(1.0/fx)
        # pc[:,0] = pc[:,0]/(pc[:,2]*(1.0/fx))
        # # Y = Y*Z*(1.0/fy)
        # pc[:,1] = pc[:,1]/(pc[:,2]*(1.0/fy))

        # # X, Y = np.meshgrid(np.array(range(w))-cx,np.array(range(h))-cy)
        # pc[:,0] = pc[:,0] + cx 
        # pc[:,1] = pc[:,1] + cy

        plane_normals = []
        for plane in planeset:
            plane_eqn = fit_plane(pc[plane])
            plane_normals.append(np.array(plane_eqn[:3]))
        #do something with the cost
        # print(len(plane_normals))
        for i in range(len(plane_normals)):
            for j in range(i+1, len(plane_normals)):
                dot_product = abs(np.dot(plane_normals[i], plane_normals[j]))
                cross_product = np.linalg.norm(np.cross(plane_normals[i], plane_normals[j]))
                if cross_product > 0.001:
                    cost += dot_product
    return cost

def fxfy_objective(x, cx, cy, scans, planesets):
    fx, fy = x 
    return mutual_orthogonality(scans, planesets, fx, fy, cx, cy)

def calibrate_intrinics_fxfy(scans,planesets):
    cam = scans[0].camera
    fx = cam.depth_intrinsics['fx']
    fy = cam.depth_intrinsics['fy']
    cx = cam.depth_intrinsics['cx']
    cy = cam.depth_intrinsics['cy']
    # print("TODO... problem 3.B")
    print(fx, fy)
    initial_guess = np.array([fx,fy])
    print(fxfy_objective(initial_guess, cx, cy, scans, planesets))
    res = minimize(fxfy_objective, initial_guess, args = (cx, cy, scans, planesets,), method = "Nelder-Mead", options={"maxiter":50})
    fx, fy = res.x
    print(fx, fy) 
    print(fxfy_objective(np.array([fx, fy]), cx, cy, scans, planesets))
    return fx, fy 

def all_objective(x, scans, planesets):
    fx, fy, cx, cy = x 
    return mutual_orthogonality(scans, planesets, fx, fy, cx, cy)

def calibrate_intrinics_all(scans,planesets):
    cam = scans[0].camera
    fx = cam.depth_intrinsics['fx']
    fy = cam.depth_intrinsics['fy']
    cx = cam.depth_intrinsics['cx']
    cy = cam.depth_intrinsics['cy']
    print("TODO... problem 3.C")
    print(fx, fy, cx, cy)
    initial_guess = np.array([fx,fy, cx, cy])
    print(all_objective(initial_guess, scans, planesets))
    res = minimize(all_objective, initial_guess, args = (scans, planesets,), method = "Nelder-Mead", options={"maxiter":50})
    fx, fy, cx, cy = res.x
    print(fx, fy, cx, cy) 
    print(all_objective(np.array([fx, fy, cx, cy]), scans, planesets))
    return fx, fy, cx, cy 

if __name__ == '__main__':
    #read problem from command line, if provided
    if len(sys.argv) > 1:
        PROBLEM = sys.argv[1]
        
    with open("planesets.json","r") as f:
        planesets = json.load(f)
    if SUBSET_PLANES is not None:
        #select a smaller subset of the planesets
        for i,planeset in enumerate(planesets):
            for j,plane in enumerate(planeset):
                if len(plane) > SUBSET_PLANES:
                    planeset[j] = list(random.sample(plane,SUBSET_PLANES))
    scans = load_rgbd_dataset('calibration')
    # print(len(planesets), len(scans))
    assert len(planesets) == len(scans)
    #reset to factory calibration
    if CALIBRATION=='spec':
        cam = sr300_factory_calib
        for s in scans:
            s.camera = cam
    else:
        #use calibration from camera
        cam = scans[0].camera
    if PROBLEM == '3a':
        fx = cam.depth_intrinsics['fx']
        fy = cam.depth_intrinsics['fy']
        cx = cam.depth_intrinsics['cx']
        cy = cam.depth_intrinsics['cy']
        print("Cost():",mutual_orthogonality(scans,planesets,fx,fy,cx,cy))
        print()
        print("Some random testing...")
        print("Cost(fx,fy*1.01):",mutual_orthogonality(scans,planesets,fx*1.01,fy*1.01,cx,cy))
        print("Cost(fx,fy*1.1):",mutual_orthogonality(scans,planesets,fx*1.1,fy*1.1,cx,cy))
        print("Cost(fx,fy*1.2):",mutual_orthogonality(scans,planesets,fx*1.2,fy*1.2,cx,cy))
        print("Cost(fx,fy*.99):",mutual_orthogonality(scans,planesets,fx*0.99,fy*0.99,cx,cy))
        print("Cost(fx,fy*.95):",mutual_orthogonality(scans,planesets,fx*0.95,fy*0.95,cx,cy))
        print("Cost(fx,fy*.9):",mutual_orthogonality(scans,planesets,fx*0.9,fy*0.9,cx,cy))
        print("Cost(cx offset):",mutual_orthogonality(scans,planesets,fx,fy,cx+10,cy+30))
    elif PROBLEM == '3b':
        res = calibrate_intrinics_fxfy(scans,planesets)
    else:
        res = calibrate_intrinics_all(scans,planesets)
