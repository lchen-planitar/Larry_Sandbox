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

def read_calib_io_DS_model(json_path):
    # 1. Load the raw JSON file
    with open(json_path, 'r') as f:
        calib_data = json.load(f)
    
    camera_params = calib_data["Calibration"]["cameras"][0]["model"]["ptr_wrapper"]["data"]["parameters"]
    f = camera_params["f"]["val"]
    cx = camera_params["cx"]["val"]
    cy = camera_params["cy"]["val"]
    xi = camera_params["xi"]["val"]
    alpha = camera_params["alpha"]["val"]

    print(" f: %.3f\n cx: %.3f\n cy: %.3f\n xi: %.6f\n alpha: %.6f\n" 
            % (f, cx, cy, xi, alpha))

    return f, cx, cy, xi, alpha

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

class DoubleSphereCamera:
    def __init__(self, fx, fy, cx, cy, xi, alpha):
        """
        Initializes the Double Sphere Camera Model.
        Parameters:
            fx, fy : Focal lengths
            cx, cy : Principal point (optical center)
            xi     : First sphere shift parameter
            alpha  : Second sphere distortion balance parameter [0, 1]
        """
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.xi = xi
        self.alpha = alpha

    def project(self, points_3d):
        """
        Projects 3D points (Nx3 array) in the camera frame to 2D pixel coordinates (Nx2 array).
        """
        points_3d = np.atleast_2d(points_3d)
        x = points_3d[:, 0]
        y = points_3d[:, 1]
        z = points_3d[:, 2]

        # Step 1: Project onto the first sphere
        d1 = np.sqrt(x**2 + y**2 + z**2)
        
        # Step 2: Shift and project onto the second sphere
        z_shifted = self.xi * d1 + z
        d2 = np.sqrt(x**2 + y**2 + z_shifted**2)

        # Step 3: Compute denominator and project to pixel plane
        denom = self.alpha * d2 + (1.0 - self.alpha) * z_shifted
        
        u = self.fx * (x / denom) + self.cx
        v = self.fy * (y / denom) + self.cy

        return np.stack((u, v), axis=-1)

    def unproject(self, pixels):
        """
        Unprojects 2D pixel coordinates (Nx2 array) back into 3D rays (Nx3 array).
        Uses the closed-form inverse derivation.
        """
        pixels = np.atleast_2d(pixels)
        u = pixels[:, 0]
        v = pixels[:, 1]

        # Step 1: Normalize pixel coordinates using pinhole intrinsics
        mx = (u - self.cx) / self.fx
        my = (v - self.cy) / self.fy
        r2 = mx**2 + my**2

        # Step 2: Closed-form inverse calculations
        # Prevent division by zero errors for edge conditions
        alpha_factor = 1.0 - 2.0 * self.alpha + 0.0
        
        # Compute intermediate scaling factor 'mz'
        if self.alpha != 0.5:
            mz = (1.0 - self.alpha**2 * r2) / (self.alpha * np.sqrt(1.0 + (1.0 - 2.0 * self.alpha) * r2) + (1.0 - self.alpha))
        else:
            mz = (1.0 - 0.25 * r2) / 2.0

        # Step 3: Back-project through the two spheres
        mz2 = mz**2
        denom_sphere = self.xi * np.sqrt(r2 + mz2) + mz
        
        # Scale factor to project back to the unit sphere
        scale = (self.xi * mz + np.sqrt(mz2 + (1.0 - self.xi**2) * r2)) / (r2 + mz2)
        
        # Calculate final 3D vector coordinates
        x_3d = scale * mx
        y_3d = scale * my
        z_3d = scale * mz - self.xi

        return np.stack((x_3d, y_3d, z_3d), axis=-1)

def main():
    #code
    file = args.camP
    poseNum = args.imNum
    image = plt.imread(args.image)

    f, cx, cy, xi, alpha = read_calib_io_DS_model(file)
    rx, ry, rz, tx, ty, tz = read_calib_io_pose(file, poseNum)
    objectPts = read_calib_io_targets(file)

    cam = DoubleSphereCamera(fx=f, fy=f, cx=cx, cy=cy, xi=xi, alpha=alpha)
    #coeff = np.array([f, cx, cy, xi, alpha])

    r_vector = np.array([rx, ry, rz], dtype=np.float64)
    R, _ = cv2.Rodrigues(r_vector)

    H = np.zeros((4, 4))
    H[:-1, -1] = np.array([tx, ty, tz])
    H[:3, :3] = R
    H[3, 3] = 1

    print(H)

    camPts = apply_h_transform(objectPts.T, H).T

    pixels = cam.project(camPts)
    u = pixels[:, 0]
    v = pixels[:, 1]
    #u, v = xyz_to_fisheye(camPts[:, 0], camPts[:, 1], camPts[:, 2], coeff)
    print(pixels)

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