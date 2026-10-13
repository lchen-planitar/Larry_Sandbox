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

from tools import apply_h_transform
from tools import inverse_H
from tools import H_to_euler
from tools import euler_to_H

def parse():
    parser = argparse.ArgumentParser('Plot Image')
    parser.add_argument('image', help='image file')
    parser.add_argument('camP_L', help='file containing left camera intrinsics')
    parser.add_argument('camP_R', help='file containing right camera intrinsics')
    #parser.add_argument('camExt', help='file containing LR camera extrinsics')
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

def fisheye_reproject(img, outShape, fov, coeff, H, reproj_R, right=False):
    h_src, w_src = img.shape[:2]
    # Define output equirectangular dimensions (2:1 aspect ratio)
    w_dest = outShape[1]
    h_dest = outShape[0]
    
    f = coeff[0]
    cx = coeff[1]
    cy = coeff[2]
    k1 = coeff[3]
    k2 = coeff[4]
    k3 = coeff[5]
    k4 = coeff[6]

    print("Output Shape: ", outShape)
    print("FOV: %.1fdeg" % fov)
    print("Camera Parameters (k1, k2, k3, k4, k5, cx, cy): ", coeff)
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
    #trans_xyz = apply_h_transform(xyz, H)

    x2 = trans_xyz[0, :]
    y2 = trans_xyz[1, :]
    z2 = trans_xyz[2, :]

    #pv_cloud_plot(trans_xyz.T)

    psph_x2 = x2.reshape(psph_x.shape)
    psph_y2 = y2.reshape(psph_y.shape)
    psph_z2 = z2.reshape(psph_z.shape)
    
    # Cartesian to Fisheye polar coordinates
    #fov_rad = np.radians(fov)
    angle = np.arctan2(np.sqrt(psph_x2**2 + psph_y2**2), psph_z2)
    beta = np.arctan2(psph_y2, psph_x2)
    
    # Radius in fisheye depends on FOV; assume radius is half of image width
    dTheta = angle + k1*angle**3 + k2*angle**5 + k3*angle**7 + k4*angle**9
    
    # Fisheye 2D coordinates
    map_x = cx + dTheta * f * np.cos(beta)
    map_y = cy + dTheta * f * np.sin(beta)
    
    # Remap image using OpenCV's optimized function
    return cv2.remap(img, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR)


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

    fileL = args.camP_L
    fileR = args.camP_R
    image = plt.imread(args.image)

    height = image.shape[0]
    width = image.shape[1]

    f, cx, cy, k1, k2, k3, k4 = read_calib_io_fisheye_model(fileL)
    sym_coeff_L = [f, cx, cy, k1, k2, k3, k4]
    transform_L = [0, 0, 0, 0, 0, 15] #Rz, Ry, Rx, Tx, Ty, Tz

    f, cx, cy, k1, k2, k3, k4 = read_calib_io_fisheye_model(fileR)
    sym_coeff_R = [f, cx, cy, k1, k2, k3, k4]
    transform_R = [0, 180, 0, 0, 0, -15] #Rz, Ry, Rx, Tx, Ty, Tz

    imgL = image[:, :int(image.shape[1]/2)].astype(np.float32)
    imgR = image[:, int(image.shape[1]/2):].astype(np.float32)
    #imgL = rgb[:, :int(rgb.shape[1]/2), :]
    #imgR = rgb[:, int(rgb.shape[1]/2):, :]
    outShape = [5952, 5952]
    reproj_rad = 900
    fov = 200
    outShape2 = [5952, int(5952*fov/180)]
    
    print("\nUndistorting Left...\n")
    equi_imgL = fisheye_reproject(imgL, outShape2, fov, sym_coeff_L, euler_to_H(transform_L), reproj_rad, right=False)
    
    print("\nUndistorting Right...\n")
    equi_imgR = fisheye_reproject(imgR, outShape2, fov, sym_coeff_R, euler_to_H(transform_R), reproj_rad, right=True)


    equi_img = np.zeros((5952, 2*int(5952*fov/180)))
    #equi_img[:, :int(ldr_8bit.shape[1]/2), :] = equi_imgL
    #equi_img[:, int(ldr_8bit.shape[1]/2):, :] = equi_imgR

    #equi_img[:, :int(ldr_8bit.shape[1]/4), :] = equi_imgL[:, int(ldr_8bit.shape[1]/4):, :]
    #equi_img[:, int(ldr_8bit.shape[1]/4):int(ldr_8bit.shape[1]*3/4), :] = equi_imgR
    #equi_img[:, int(ldr_8bit.shape[1]*3/4):, :] = equi_imgL[:, :int(ldr_8bit.shape[1]/4), :]
    equi_img[:, :int(equi_img.shape[1]/4)] = equi_imgL[:, -int(equi_img.shape[1]/4):]
    equi_img[:, int(equi_img.shape[1]/4):int(equi_img.shape[1]*3/4)] = equi_imgR
    equi_img[:, -int(equi_img.shape[1]/4):] = equi_imgL[:, :int(equi_img.shape[1]/4)]

    equi_img = np.clip((equi_img*255.0), 0, 255).astype('uint8')


    # Display using Matplotlib

    print("\nPlotting...\n")
    plt.figure(1)
    #fig, ax = plt.subplots()
    #plt.imshow(image, cmap='gray')
    #plt.imshow(equi_imgL.astype("uint8"), cmap='gray')
    #plt.imshow(equi_imgR.astype("uint8"))
    plt.imshow(equi_img, cmap='gray')

    #plt.figure(2)
    #plt.imshow((equi_img[:, :, 1]/255).astype(np.float32), cmap="gray")
    #plt.imshow(ldr_8bit[:, :int(ldr_8bit.shape[1]/2), :])

    #plt.figure(3)
    #plt.hist(equi_imgL.ravel(), 256, [0, 256], alpha=0.5)
    #plt.hist(equi_imgR.ravel(), 256, [0, 256], alpha=0.5)
    plt.colorbar()
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

