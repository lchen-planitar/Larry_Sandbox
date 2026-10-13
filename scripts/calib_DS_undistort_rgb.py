import cv2
import sys
import argparse
import configparser
import matplotlib.pyplot as plt
import rawpy
import numpy as np
import imageio
import os
from skimage import exposure
import pandas as pd
import pyvista as pv
import json
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from tools import apply_h_transform
from tools import inverse_H
from tools import H_to_euler
from tools import euler_to_H

def parse():
    parser = argparse.ArgumentParser('Plot Image')
    parser.add_argument('image', help='image file')
    parser.add_argument('camP', help='file containing left and right camera intrinsics')
    #parser.add_argument('camP_R', help='file containing right camera intrinsics')
    #parser.add_argument('camExt', help='file containing LR camera extrinsics')
    args = parser.parse_args(None if (sys.argv[1:]) else ['-h'])
    return args

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

def read_calib_io_DS_intrinsic(json_path, camNum):
    # 1. Load the raw JSON file
    with open(json_path, 'r') as f:
        calib_data = json.load(f)
    
    camera_params = calib_data["Calibration"]["cameras"][camNum]["model"]["ptr_wrapper"]["data"]["parameters"]
    f = camera_params["f"]["val"]
    cx = camera_params["cx"]["val"]
    cy = camera_params["cy"]["val"]
    xi = camera_params["xi"]["val"]
    alpha = camera_params["alpha"]["val"]

    print(" f: %.3f\n cx: %.3f\n cy: %.3f\n xi: %.6f\n alpha: %.6f\n" 
            % (f, cx, cy, xi, alpha))

    return f, cx, cy, xi, alpha

def read_calib_io_DS_extrinsic(json_path, camNum):
    # 1. Load the raw JSON file
    with open(json_path, 'r') as f:
        calib_data = json.load(f)

    camera_pose = calib_data["Calibration"]["cameras"][camNum]["transform"]
    rx = camera_pose["rotation"]["rx"]
    ry = camera_pose["rotation"]["ry"]
    rz = camera_pose["rotation"]["rz"]
    tx = camera_pose["translation"]["x"]*1000
    ty = camera_pose["translation"]["y"]*1000
    tz = camera_pose["translation"]["z"]*1000

    print(" pose#: %d\n rx: %.3f\n ry: %.3f\n rz: %.3f\n tx: %.3f\n ty: %.3f\n tz: %.3f\n" 
            % (camNum, rx, ry, rz, tx, ty, tz))

    return rx, ry, rz, tx, ty, tz

def pv_cloud_plot(xyz):
    print("pyvista plotting\n")
    mesh = pv.PolyData(xyz[::100, :])

    pl = pv.Plotter()
    pl.add_mesh(mesh, color='green', render_points_as_spheres=True, point_size=5, opacity=0.5)

    scale = 500
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
    pl.show()

    return 0

def DS_reproject(img, outShape, fov, cam, H, reproj_R, right=False):
    h_src, w_src = img.shape[:2]
    # Define output equirectangular dimensions (2:1 aspect ratio)
    w_dest = outShape[1]
    h_dest = outShape[0]

    print("Output Shape: ", outShape)
    print("FOV: %.1fdeg" % fov)
    #print("Camera Parameters (k1, k2, k3, k4, k5, cx, cy): ", coeff)
    print("Reprojection Transform (Rz, Ry, Rx, Tx, Ty, Tz, deg/mm): ", H_to_euler(H))
    print("Reprojection Depth Assumption: %.2fmm" % reproj_R)

    # Create meshgrid for target equirectangular image
    x, y = np.meshgrid(np.arange(w_dest), np.arange(h_dest))

    az_lim = fov * np.pi / 180
    el_lim = np.pi
    
    # Normalize coordinates to +/-0.5FOV (longitude) and +/-0.5FOV (latitude)
    theta = (az_lim * x / w_dest) - (az_lim / 2)
    phi = (el_lim * y / h_dest) - (el_lim / 2)

    if right:
        theta2 = theta.copy()
        theta2[theta > 0] = theta[theta > 0] - np.pi
        theta2[theta <= 0] = theta[theta <= 0] + np.pi
        theta = theta2
    
    # Spherical to Cartesian
    psph_x = reproj_R * np.cos(phi) * np.sin(theta)
    psph_z = reproj_R * np.cos(phi) * np.cos(theta)
    psph_y = reproj_R * np.sin(phi)

    # Reproject to different view point (i.e. center between 2 lenses)

    x = psph_x.flatten()
    y = psph_y.flatten()
    z = psph_z.flatten()

    xyz = np.stack((x, y, z), axis=0)
    trans_xyz = apply_h_transform(xyz, inverse_H(H))

    #pv_cloud_plot(trans_xyz.T)

    pixels = cam.project(trans_xyz.T)
    u = pixels[:, 0]
    v = pixels[:, 1]

    map_x = u.reshape(psph_x.shape)
    map_y = v.reshape(psph_y.shape)

    #print(np.max(map_y))
    
    # Remap image using OpenCV's optimized function
    return cv2.remap(img, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR)

def get_poses_in_middle_frame(T1, T2):
    """
    Finds the middle transform between two 4x4 matrices and returns T1 and T2 
    expressed in the coordinate frame of that middle transform.
    
    Args:
        T1 (np.ndarray): 4x4 homogeneous transformation matrix 1
        T2 (np.ndarray): 4x4 homogeneous transformation matrix 2
        
    Returns:
        tuple: (T1_in_mid, T2_in_mid, T_mid)
    """
    # 1. Extract Translations
    t1, t2 = T1[:3, 3], T2[:3, 3]
    
    # 2. Extract and SLERP Rotations
    r1 = R.from_matrix(T1[:3, :3])
    r2 = R.from_matrix(T2[:3, :3])
    rotations = R.concatenate([r1, r2])

    slerp_instance = Slerp([0.0, 1.0], rotations)
    slerp_rot = slerp_instance(0)
    
    # 3. Calculate Middle Transformation
    t_mid = (t1 + t2) / 2.0
    R_mid = slerp_rot.as_matrix()
    
    T_mid = np.eye(4)
    T_mid[:3, :3] = R_mid
    T_mid[:3, 3] = t_mid
    
    # 4. Compute the Inverse of the Middle Transform
    T_mid_inv = np.linalg.inv(T_mid)
    
    # 5. Transform original poses into the middle frame (A_in_B = B_inv * A)
    T1_in_mid = T_mid_inv @ T1
    T2_in_mid = T_mid_inv @ T2
    
    return T1_in_mid, T2_in_mid, T_mid

def apply_gamma(img, gamma=2.2, linear_slope=4.5):
    # 1. Load image and convert to YCbCr (standard YUV for OpenCV)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, u, v = cv2.split(img_yuv)

    # 2. Normalize Luma channel to [0, 1]
    y_norm = y / 255.0

    # 3. Piecewise Gamma Correction (Linear ends + Gamma center)
    # Similar to sRGB/BT.709 transfer function
    y_corr = np.where(y_norm <= 0.018, 
                      y_norm * linear_slope, 
                      (1.0 + 0.055) * np.power(y_norm, 1/gamma) - 0.055)

    # 4. Scale back to [0, 255] and replace Y channel
    y_new = np.clip(y_corr * 255.0, 0, 255).astype(np.uint8)
    
    # 5. Merge and convert back to BGR
    img_yuv_new = cv2.merge((y_new, u, v))
    result = cv2.cvtColor(img_yuv_new, cv2.COLOR_YCrCb2BGR)
    return result

def main():
    #code

    #fileL = args.camP_L
    #fileR = args.camP_R
    fileLR = args.camP
    image = plt.imread(args.image)

    height = image.shape[0]
    width = image.shape[1]

    f, cx, cy, xi, alpha = read_calib_io_DS_intrinsic(fileLR, 0)
    camL = DoubleSphereCamera(fx=f, fy=f, cx=cx, cy=cy, xi=xi, alpha=alpha)
    rx, ry, rz, tx, ty, tz = read_calib_io_DS_extrinsic(fileLR, 0)
    r_vector = np.array([rx, ry, rz], dtype=np.float64)
    R, _ = cv2.Rodrigues(r_vector)
    Hl = np.zeros((4, 4))
    Hl[:-1, -1] = np.array([tx, ty, tz])
    Hl[:3, :3] = R
    Hl[3, 3] = 1
    #transform_L = [0, 0, 0, 0, 0, 0] #Rz, Ry, Rx, Tx, Ty, Tz

    f, cx, cy, xi, alpha = read_calib_io_DS_intrinsic(fileLR, 1)
    camR = DoubleSphereCamera(fx=f, fy=f, cx=cx, cy=cy, xi=xi, alpha=alpha)
    rx, ry, rz, tx, ty, tz = read_calib_io_DS_extrinsic(fileLR, 1)
    r_vector = np.array([rx, ry, rz], dtype=np.float64)
    R, _ = cv2.Rodrigues(r_vector)
    Hr = np.zeros((4, 4))
    Hr[:-1, -1] = np.array([tx, ty, tz])
    Hr[:3, :3] = R
    Hr[3, 3] = 1
    fudge_factor = [0, 0, 0, 0, -12, 0] #Rz, Ry, Rx, Tx, Ty, Tz
    Hf = euler_to_H(fudge_factor)
    Hr = Hr @ Hf

    Hl_mid, Hr_mid, _ = get_poses_in_middle_frame(Hl, Hr)
    #transform_R = [0, 180, 0, -1.019, 1.554, -28.764] #Rz, Ry, Rx, Tx, Ty, Tz
    print(H_to_euler(Hl_mid))
    print(H_to_euler(Hr_mid))

    imgL = image[:, :int(image.shape[1]/2), :].astype(np.float32)
    imgR = image[:, int(image.shape[1]/2):, :].astype(np.float32)
    #imgL = rgb[:, :int(rgb.shape[1]/2), :]
    #imgR = rgb[:, int(rgb.shape[1]/2):, :]
    outShape = [5952, 5952]
    reproj_rad = 6000
    fov = 180
    outShape2 = [5952, int(5952*fov/180)]
    
    print("\nUndistorting Left...\n")
    equi_imgL = DS_reproject(imgL, outShape2, fov, camL, Hl_mid, reproj_rad, right=False)
    
    print("\nUndistorting Right...\n")
    equi_imgR = DS_reproject(imgR, outShape2, fov, camR, Hr_mid, reproj_rad, right=True)


    equi_img = np.zeros((5952, 2*int(5952*fov/180), 3))
    #equi_img[:, :int(ldr_8bit.shape[1]/2), :] = equi_imgL
    #equi_img[:, int(ldr_8bit.shape[1]/2):, :] = equi_imgR

    #equi_img[:, :int(ldr_8bit.shape[1]/4), :] = equi_imgL[:, int(ldr_8bit.shape[1]/4):, :]
    #equi_img[:, int(ldr_8bit.shape[1]/4):int(ldr_8bit.shape[1]*3/4), :] = equi_imgR
    #equi_img[:, int(ldr_8bit.shape[1]*3/4):, :] = equi_imgL[:, :int(ldr_8bit.shape[1]/4), :]
    equi_img[:, :int(equi_img.shape[1]/4), :] = equi_imgL[:, -int(equi_img.shape[1]/4):, :]
    equi_img[:, int(equi_img.shape[1]/4):int(equi_img.shape[1]*3/4), :] = equi_imgR
    equi_img[:, -int(equi_img.shape[1]/4):, :] = equi_imgL[:, :int(equi_img.shape[1]/4), :]

    equi_img = np.clip((equi_img*255.0), 0, 255).astype('uint8')


    # Display using Matplotlib

    print("\nPlotting...\n")
    plt.figure(1)
    #fig, ax = plt.subplots()
    #plt.imshow(image, cmap='gray')
    #plt.imshow(equi_imgL.astype("uint8"), cmap='gray')
    #plt.imshow(equi_imgR.astype("uint8"))
    plt.imshow(equi_img)

    #plt.figure(2)
    #plt.imshow((equi_img[:, :, 1]/255).astype(np.float32), cmap="gray")
    #plt.imshow(ldr_8bit[:, :int(ldr_8bit.shape[1]/2), :])

    #plt.figure(3)
    #plt.hist(equi_imgL.ravel(), 256, [0, 256], alpha=0.5)
    #plt.hist(equi_imgR.ravel(), 256, [0, 256], alpha=0.5)
    #plt.colorbar()
    plt.show()

    print("\nSaving Image Output...\n")
    input_filename = os.path.basename(args.image)
    input_filename = os.path.splitext(input_filename)[0]
    output_filename = f"output_undistort_{input_filename}.jpg"
    print("Output Filename: ", output_filename)
    #imageio.imsave(output_filename, equi_img.astype("uint8"), quality=100)
    imageio.imsave(output_filename, np.clip((equi_imgR*255.0), 0, 255).astype('uint8'), quality=100)


if __name__=="__main__":
    args = parse()
    main()

