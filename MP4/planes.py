import numpy as np
from klampt import vis
from klampt.model.geometry import fit_plane,fit_plane3
from klampt.model.trajectory import Trajectory
from klampt.math import vectorops
from klampt.io import numpy_convert
from klampt import PointCloud
import random
import sys
sys.path.append("../common")
from rgbd import *
from rgbd_realsense import load_rgbd_dataset
import json

#set the problem you are working on. you can also set this from the command line
# PROBLEM = '2a'
# PROBLEM = '2b'
PROBLEM = '2c'
DONT_EXTRACT = False #if you just want to see the point clouds, turn this to true

def extract_planes_ransac_a(pc,N=100,m=3,inlier_threshold=0.01,inlier_count=20000):
    """Uses RANSAC to determine which planes make up the scene

    Args:
        pc: an Nx3 numpy array of points
        N: the number of iterations used to sample planes
        m: the number of points to fit on each iteration
        inlier_threshold: the distance between plane / point to consider
            it an inlier
        inlier_count: consider a plane to be an inlier (and output it!) if this
            many points are inliers
    
    Returns:
        list of lists of int: a list of lists of point indices that belong to
        planes. If `plane_indices` is the result, each entry represents a plane,
        and the plane equation can be obtained using `fit_plane(pc[plane_indices[i]])`.
    """
    #to fit a plane through 3 points:
    #(a,b,c,d) = fit_plane3(p1,p2,p3)

    #to fit a plane through N>=3 points:
    #(a,b,c,d) = fit_plane([p1,p2,p3,p4])
    planes = []
    # planes.append([0,1,2,3,4])
    # planes.append([5,6,7,8])

    # Sampling for N iterations
    tmp_pc = pc.tolist()
    for i in range(N):
        sampled_points = random.sample(tmp_pc, m)
        try:
            a,b,c,d = fit_plane(sampled_points)
        except:
            continue
        inlier = []
        res = np.abs(pc@np.array([[a,b,c]]).T+d)
        indices_array = np.where(res < inlier_threshold)[0].tolist()
        inlier = indices_array
        if len(inlier) > inlier_count:
            planes.append(inlier) 
    return planes
    
def extract_planes_ransac_b(pc,N=100,m=3,inlier_threshold=0.01,inlier_count=20000):
    """Uses RANSAC to determine which planes make up the scene

    Args:
        pc: an Nx3 numpy array of points
        N: the number of iterations used to sample planes
        m: the number of points to fit on each iteration
        inlier_threshold: the distance between plane / point to consider
            it an inlier
        inlier_count: consider a plane to be an inlier (and output it!) if this
            many points are inliers
    
    Returns:
        list of lists of int: a list of lists of point indices that belong to
        planes. If `plane_indices` is the result, each entry represents a plane,
        and the plane equation can be obtained using `fit_plane(pc[plane_indices[i]])`.
    """
    #to fit a plane through 3 points:
    #(a,b,c,d) = fit_plane3(p1,p2,p3)

    #to fit a plane through N>=3 points:
    #(a,b,c,d) = fit_plane([p1,p2,p3,p4])
    planes = []
    tmp_planes = []
    # planes.append([0,1,2,3,4])
    # planes.append([5,6,7,8])
    # tmp_pc = pc.tolist()
    # Sampling for N iterations
    points_in_plane = []
    print("Perform Main RANSAC Iteration")
    point_dict = {}
    for i in range(N):
        # Make a copy of the original pc
        sample_pc_list = np.copy(pc)
        # Remove all elements that's already in plane
        np.delete(sample_pc_list, points_in_plane, axis=0)
        sample_pc_list = sample_pc_list.tolist()

        sampled_points = random.sample(sample_pc_list, m)
        try:
            a,b,c,d = fit_plane(sampled_points)
        except:
            continue
        inlier = []
        res = np.abs(pc@np.array([[a,b,c]]).T+d)
        indices_array = np.where(res < inlier_threshold)[0].tolist()
        inlier = indices_array

        if len(inlier) > inlier_count:
            tmp_planes.append(inlier) 
            points_in_plane += inlier
    
    print("Assign Points to largest plane")
    # Sort the planes according to size
    # tmp = [len(plane) for plane in tmp_planes]
    # indices = np.argsort(-np.array(tmp)).flatten().tolist()
    # planes = []
    # for val in indices:
    #     planes.append(tmp_planes[val])
    planes = tmp_planes
    for i in range(len(planes)):
        plane1 = planes[i]
        if plane1 == []:
            continue
        for j in range(i+1, len(planes)):
            plane2 = planes[j]
            if len(plane1) > len(plane2):
                large_plane = plane1 
                small_plane = plane2 
                tmp = np.setdiff1d(small_plane, np.intersect1d(large_plane, small_plane)).tolist()
                planes[j] = tmp
            else:
                large_plane = plane2 
                small_plane = plane1 
                tmp = np.setdiff1d(small_plane, np.intersect1d(large_plane, small_plane)).tolist()
                planes[i] = tmp
                plane1 = tmp
            if plane1 == []:
                break
    
    print("Remove small planes")
    idx = 0
    while idx < len(planes):
        if len(planes[idx]) < inlier_count:
            planes.pop(idx)
        else:
            idx = idx + 1
    return planes

def extract_planes_ransac_c(pc,N=100,m=3,inlier_threshold=0.015,inlier_count=50000):
    """Uses RANSAC to determine which planes make up the scene

    Args:
        pc: an Nx3 numpy array of points
        N: the number of iterations used to sample planes
        m: the number of points to fit on each iteration
        inlier_threshold: the distance between plane / point to consider
            it an inlier
        inlier_count: consider a plane to be an inlier (and output it!) if this
            many points are inliers
    
    Returns:
        list of lists of int: a list of lists of point indices that belong to
        planes. If `plane_indices` is the result, each entry represents a plane,
        and the plane equation can be obtained using `fit_plane(pc[plane_indices[i]])`.
    """
    #to fit a plane through 3 points:
    #(a,b,c,d) = fit_plane3(p1,p2,p3)

    #to fit a plane through N>=3 points:
    #(a,b,c,d) = fit_plane([p1,p2,p3,p4])
    #to fit a plane through 3 points:
    #(a,b,c,d) = fit_plane3(p1,p2,p3)

    #to fit a plane through N>=3 points:
    #(a,b,c,d) = fit_plane([p1,p2,p3,p4])
    planes = []
    tmp_planes = []
    # planes.append([0,1,2,3,4])
    # planes.append([5,6,7,8])
    # tmp_pc = pc.tolist()
    # Sampling for N iterations
    points_in_plane = []
    print("Perform Main RANSAC Iteration")
    point_dict = {}
    for i in range(N):
        # Make a copy of the original pc
        sample_pc_list = np.copy(pc)
        # Remove all elements that's already in plane
        np.delete(sample_pc_list, points_in_plane, axis=0)
        sample_pc_list = sample_pc_list.tolist()

        sampled_points = random.sample(sample_pc_list, m)
        try:
            a,b,c,d = fit_plane(sampled_points)
        except:
            continue
        inlier = []
        res = np.abs(pc@np.array([[a,b,c]]).T+d)
        indices_array = np.where(res < inlier_threshold)[0].tolist()
        inlier = indices_array

        if len(inlier) > inlier_count:
            tmp_planes.append(inlier) 
            points_in_plane += inlier
    
    print("Assign Points to largest plane")
    # Sort the planes according to size
    # tmp = [len(plane) for plane in tmp_planes]
    # indices = np.argsort(-np.array(tmp)).flatten().tolist()
    # planes = []
    # for val in indices:
    #     planes.append(tmp_planes[val])
    planes = tmp_planes
    for i in range(len(planes)):
        plane1 = planes[i]
        if plane1 == []:
            continue
        for j in range(i+1, len(planes)):
            plane2 = planes[j]
            if len(plane1) > len(plane2):
                large_plane = plane1 
                small_plane = plane2 
                tmp = np.setdiff1d(small_plane, np.intersect1d(large_plane, small_plane)).tolist()
                planes[j] = tmp
            else:
                large_plane = plane2 
                small_plane = plane1 
                tmp = np.setdiff1d(small_plane, np.intersect1d(large_plane, small_plane)).tolist()
                planes[i] = tmp
                plane1 = tmp
            if plane1 == []:
                break
    
    print("Remove small planes")
    idx = 0
    while idx < len(planes):
        if len(planes[idx]) < inlier_count:
            planes.pop(idx)
        else:
            idx = idx + 1
    
    # Recomput inlier based on the computed planes
    print("Recomput inlier based on the computed planes")
    tmp_planes = []
    for plane in planes:
        point_list = pc[plane,:].tolist()
        a,b,c,d = fit_plane(point_list)
        res = np.abs(pc@np.array([[a,b,c]]).T+d)
        indices_array = np.where(res < inlier_threshold)[0].tolist()
        inlier = indices_array
        tmp_planes.append(inlier)

    # Get points that fall inside multiple planes and remove those points from tmp_planes
    print("Get points that fall inside multiple planes and remove those points from tmp_planes")
    multiple_plane = []
    for i in range(len(tmp_planes)):
        for j in range(i+1, len(tmp_planes)):
            tmp = np.intersect1d(tmp_planes[i], tmp_planes[j]).tolist()
            tmp_planes[i] = np.setdiff1d(tmp_planes[i], tmp).tolist()
            tmp_planes[j] = np.setdiff1d(tmp_planes[j], tmp).tolist()
            multiple_plane += tmp

    # Fit planes again using the remaining points
    print("Fit planes again using the remaining points")
    plane_abc = []
    plane_d = []
    for plane in tmp_planes:
        point_list = pc[plane,:].tolist()
        a,b,c,d = fit_plane(point_list)
        plane_abc.append([a,b,c])
        plane_d.append(d)

    # Compute the distance between points and each plane
    print("Compute the distance between points and each plane")
    multiple_plane_points = pc[multiple_plane,:]
    dist = np.abs(multiple_plane_points@np.array(plane_abc).T+np.array(plane_d))

    print("Determine min distance")
    point_idx = np.argmin(dist,axis=1)

    print("Append points to min distance plane")
    for i in range(point_idx.shape[0]):
        tmp_planes[point_idx[i]].append(multiple_plane[i])
       
    planes = tmp_planes
    return planes

if __name__ == '__main__':
    #read problem from command line, if provided
    if len(sys.argv) > 1:
        PROBLEM = sys.argv[1]
        
    scans = load_rgbd_dataset('calibration')
    planesets = []
    for scanno,s in enumerate(scans):
        if scanno!=0 :
            continue
        pc = s.get_point_cloud(colors=True,normals=True,structured=True,format='PointCloud')
        vis.clear()
        vis.setWindowTitle("Scan "+str(scanno))
        vis.add("PC",pc)
        if not DONT_EXTRACT:
            pc2 = s.get_point_cloud(colors=False,normals=False,structured=True)
            if PROBLEM=='2a':
                planes = extract_planes_ransac_a(pc2)
            elif PROBLEM=='2b':
                planes = extract_planes_ransac_b(pc2)
            else:
                planes = extract_planes_ransac_c(pc2)
            planesets.append(planes)
            for j,plane in enumerate(planes):
                color = (random.random(),random.random(),random.random())
                for i in plane:
                    pc.setProperty(i,0,color[0])
                    pc.setProperty(i,1,color[1])
                    pc.setProperty(i,2,color[2])
                plane_eqn = fit_plane(pc2[plane])
                centroid = np.average(pc2[plane],axis=0).tolist()
                assert len(centroid)==3
                vis.add("Plane "+str(j),Trajectory(milestones=[centroid,vectorops.madd(centroid,plane_eqn[:3],0.1)]),color=(1,1,0,1))
        vis.dialog()
    if not DONT_EXTRACT:
        print("Dumping plane identities to planesets.json")
        with open("planesets.json","w") as f:
            json.dump(planesets,f)

