import cv2
import sys
import argparse
import configparser
import matplotlib.pyplot as plt
import rawpy
import numpy as np
import imageio
import os
import glob
import json
import pyvista as pv

from tools import apply_h_transform
from tools import inverse_H
from tools import H_to_euler
from tools import euler_to_H

def parse():
    parser = argparse.ArgumentParser('Plot Image')
    parser.add_argument('camP', help='File containing camera parameters')
    parser.add_argument('image', help='image file')
    parser.add_argument('imNum', type=int, help='which pose number is the image')
    args = parser.parse_args(None if (sys.argv[1:]) else ['-h'])
    return args

def read_calib_io_fisheye_model(json_path):
    # 1. Load the raw JSON file
    with open(json_path, 'r') as f:
        calib_data = json.load(f)
    
    camera_params = calib_data["Calibration"]["cameras"][0]["model"]["ptr_wrapper"]["data"]["parameters"]
    f = camera_params["f"]["val"]
    cx = camera_params["cx"]["val"]
    cy = camera_params["cy"]["val"]
    k1 = camera_params["k1"]["val"]
    k2 = camera_params["k2"]["val"]
    k3 = camera_params["k3"]["val"]
    k4 = camera_params["k4"]["val"]

    print(" f: %.3f\n cx: %.3f\n cy: %.3f\n k1: %.6f\n k2: %.6f\n k3: %.6f\n k4: %.6f\n" 
            % (f, cx, cy, k1, k2, k3, k4))

    return f, cx, cy, k1, k2, k3, k4

def read_calib_io_pose(json_path, pNum):
    # 1. Load the raw JSON file
    with open(json_path, 'r') as f:
        calib_data = json.load(f)

    camera_pose = calib_data["Calibration"]["poses"][pNum]["transform"]
    rx = camera_pose["rotation"]["rx"]
    ry = camera_pose["rotation"]["ry"]
    rz = camera_pose["rotation"]["rz"]
    tx = camera_pose["translation"]["x"]
    ty = camera_pose["translation"]["y"]
    tz = camera_pose["translation"]["z"]

    print(" pose#: %d\n rx: %.3f\n ry: %.3f\n rz: %.3f\n tx: %.3f\n ty: %.3f\n tz: %.3f\n" 
            % (pNum, rx, ry, rz, tx, ty, tz))

    return rx, ry, rz, tx, ty, tz

def read_calib_io_targets(json_path):
    
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Navigate to the first target's object points
    obj_points_list = data["Calibration"]["targets"][0]["objectPoints"]

    # Extract x, y, z into a structured list
    parsed_points = [[pt["x"], pt["y"], pt["z"]] for pt in obj_points_list]

    # Convert to standard OpenCV-ready float32 array
    # Shape will be (N, 3), e.g., (56, 3)
    object_points_3d = np.array(parsed_points, dtype=np.float32)
    #print(object_points_3d)

    return object_points_3d

def xyz_to_fisheye(x, y, z, coeff):

    f = coeff[0]
    cx = coeff[1]
    cy = coeff[2]
    k1 = coeff[3]
    k2 = coeff[4]
    k3 = coeff[5]
    k4 = coeff[6]

    theta = np.arctan2(np.sqrt(x**2 + y**2), z)
    phi = np.arctan2(y, x)
    
    # Radius in fisheye depends on FOV; assume radius is half of image width
    dTheta = theta + k1*theta**3 + k2*theta**5 + k3*theta**7 + k4*theta**9
    
    # Fisheye 2D coordinates
    u = cx + dTheta * f * np.cos(phi)
    v = cy + dTheta * f * np.sin(phi)

    return u, v

def main():
    #code
    file = args.camP
    poseNum = args.imNum
    image = plt.imread(args.image)

    f, cx, cy, k1, k2, k3, k4 = read_calib_io_fisheye_model(file)
    rx, ry, rz, tx, ty, tz = read_calib_io_pose(file, poseNum)
    objectPts = read_calib_io_targets(file)

    coeff = np.array([f, cx, cy, k1, k2, k3, k4])

    r_vector = np.array([rx, ry, rz], dtype=np.float64)
    R, _ = cv2.Rodrigues(r_vector)

    H = np.zeros((4, 4))
    H[:-1, -1] = np.array([tx, ty, tz])
    H[:3, :3] = R
    H[3, 3] = 1

    print(H)

    camPts = apply_h_transform(objectPts.T, H).T

    u, v = xyz_to_fisheye(camPts[:, 0], camPts[:, 1], camPts[:, 2], coeff)

    plt.figure(1)
    plt.imshow(image, cmap='gray')
    plt.scatter(u, v, marker='x', s=20, color='red')
    plt.show()

    xyz = pv.PolyData(apply_h_transform(objectPts.T, H).T)
    #xyz = pv.PolyData(objectPts)

    pl = pv.Plotter()
    pl.add_mesh(xyz, color='green', render_points_as_spheres=True, point_size=10)
    #pl.add_mesh(xyzR, color='blue', render_points_as_spheres=True, point_size=10)

    scale = 0.1
    ax_start = np.zeros((3, 3))
    ax_end = np.zeros((3, 3))
    ax_end[0, 0] = scale
    ax_end[1, 1] = scale
    ax_end[2, 2] = scale
    
    xAx = pv.Line((ax_start[0, 0], ax_start[1, 0], ax_start[2, 0]), (ax_end[0, 0], ax_end[1, 0], ax_end[2, 0]))
    yAx = pv.Line((ax_start[0, 1], ax_start[1, 1], ax_start[2, 1]), (ax_end[0, 1], ax_end[1, 1], ax_end[2, 1]))
    zAx = pv.Line((ax_start[0, 2], ax_start[1, 2], ax_start[2, 2]), (ax_end[0, 2], ax_end[1, 2], ax_end[2, 2]))

    pl.add_mesh(xAx, color='red', render_lines_as_tubes=True, line_width=2)
    pl.add_mesh(yAx, color='green', render_lines_as_tubes=True, line_width=2)
    pl.add_mesh(zAx, color='blue', render_lines_as_tubes=True, line_width=2)

    pl.background_color='gray'
    pl.show()



if __name__=="__main__":
    args = parse()
    main()