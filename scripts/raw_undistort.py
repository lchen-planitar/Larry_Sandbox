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

from tools import apply_h_transform
from tools import inverse_H
from tools import H_to_euler
from tools import euler_to_H

def parse():
    parser = argparse.ArgumentParser('Plot Image')
    parser.add_argument('input', help='image file')
    parser.add_argument('camP', help='file containing camera parameters')
    args = parser.parse_args(None if (sys.argv[1:]) else ['-h'])
    return args

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

def fisheye_undistort(img, outShape, fov, coeff):
    h_src, w_src = img.shape[:2]
    # Define output equirectangular dimensions (2:1 aspect ratio)
    w_dest = outShape[1]
    h_dest = outShape[0]
    
    k1 = coeff[0]
    k2 = coeff[1]
    k3 = coeff[2]
    k4 = coeff[3]
    k5 = coeff[4]
    cx = coeff[5]
    cy = coeff[6]

    # Create meshgrid for target equirectangular image
    x, y = np.meshgrid(np.arange(w_dest), np.arange(h_dest))

    ang_lim = fov * np.pi / 180
    
    # Normalize coordinates to +/-0.5FOV (longitude) and +/-0.5FOV (latitude)
    theta = (ang_lim * x / w_dest) - (ang_lim / 2)
    phi = (ang_lim * y / h_dest) - (ang_lim / 2)
    
    # Spherical to Cartesian
    psph_x = np.cos(phi) * np.sin(theta)
    psph_y = np.cos(phi) * np.cos(theta)
    psph_z = np.sin(phi)
    
    # Cartesian to Fisheye polar coordinates
    #fov_rad = np.radians(fov)
    angle = np.arctan2(np.sqrt(psph_x**2 + psph_z**2), psph_y)
    beta = np.arctan2(psph_z, psph_x)
    
    # Radius in fisheye depends on FOV; assume radius is half of image width
    r = k1*angle + k2*angle**3 + k3*angle**5 + k4*angle**7 + k5*angle**9
    
    # Fisheye 2D coordinates
    map_x = cx + r * np.cos(beta)
    map_y = cy + r * np.sin(beta)
    
    # Remap image using OpenCV's optimized function
    return cv2.remap(img, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR)

def fisheye_reproject(img, outShape, fov, coeff, H, reproj_R, right=False):
    h_src, w_src = img.shape[:2]
    # Define output equirectangular dimensions (2:1 aspect ratio)
    w_dest = outShape[1]
    h_dest = outShape[0]
    
    k1 = coeff[0]
    k2 = coeff[1]
    k3 = coeff[2]
    k4 = coeff[3]
    k5 = coeff[4]
    cx = coeff[5]
    cy = coeff[6]

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
    r = k1*angle + k2*angle**3 + k3*angle**5 + k4*angle**7 + k5*angle**9
    
    # Fisheye 2D coordinates
    map_x = cx + r * np.cos(beta)
    map_y = cy + r * np.sin(beta)
    
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
    file = args.input
    df_cam = pd.read_csv(args.camP, index_col=0)

    print(df_cam)

    with rawpy.imread(file) as raw:
        # Post-process to get an RGB array
        print("\nReading Raw File...\n")
        print(f'Raw type: {raw.raw_type}')
        print(f'Number of colors: {raw.num_colors}')
        print(f'Color description: {raw.color_desc}')
        print(f'Color Matrix: {raw.color_matrix}')
        print(f'Black levels: {raw.black_level_per_channel}')
        print(f'White level: {raw.white_level}')
        print(f'Camera white balance: {raw.camera_whitebalance}')
        black = raw.black_level_per_channel
        raw_im = raw.raw_image_visible.astype(np.uint16)
        #raw_left = raw_im[:, :int(raw_im.shape[1]/2)]
        #corr_left = LCA_correction(raw_left, 0.9998, 0.9998)
        #corr_left = LCA_correction(raw_left, 1, 1)
        #corr_im = raw_im.copy()
        #corr_im = np.clip(raw_im, black[0], 65535).astype(np.uint16)
        #raw.raw_image_visible[:] = corr_im
        raw2 = raw.raw_image_visible.astype(np.uint16)
        print("\nDemosaicing...\n")
        rgb = raw.postprocess(gamma=(1, 1), no_auto_bright=True, use_camera_wb=True, auto_bright_thr=0.03, output_bps=16, 
                            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD, output_color=rawpy.ColorSpace.sRGB)
        

    rgb_float = (rgb / (2**16)).astype(np.float32)
    print("Image Shape: ", rgb_float.shape)

    raw_max = np.percentile(raw2, 99)
    #print("raw image scale top: ", raw_max)

    print("\nTone Mapping...\n")
    gamma = 1
    light_scale = 2
    print("Gamma: %.2f, Light Scale: %.2f" % (gamma, light_scale))
    tonemapReinhard = cv2.createTonemapReinhard(gamma, light_scale, 0, 0.4)
    ldr = tonemapReinhard.process(rgb_float)
    ldr_8bit = np.clip(ldr * 255, 0, 255).astype('uint8')
    ldr_8bit = apply_gamma(ldr_8bit, 1.5, 4.5)
    #ldr_8bit = np.clip(rgb_float * 255, 0, 255).astype('uint8')

    
    imgL = ldr_8bit[:, :int(ldr_8bit.shape[1]/2), :]
    imgR = ldr_8bit[:, int(ldr_8bit.shape[1]/2):, :]
    #imgL = rgb[:, :int(rgb.shape[1]/2), :]
    #imgR = rgb[:, int(rgb.shape[1]/2):, :]
    outShape = [5952, 5952]
    reproj_rad = 3000

    R = 2400
    k1 = R/(np.pi/2)
    k2 = 0
    k3 = 0
    k4 = 0
    k5 = 0
    cx = 5952/2
    cy = 5952/2
    sym_coeff_Lg = [k1, k2, k3, k4, k5, cx, cy]
    transform_Lg = [0, 0, 0, 0, 0, 8] #Rz, Ry, Rx, Tx, Ty, Tz
    sym_coeff_Rg = [k1, k2, k3, k4, k5, cx, cy]
    transform_Rg = [0, 180, 0, 0, 0, -8] #Rz, Ry, Rx, Tx, Ty, Tz

    #R = 2400 #px
    k1 = df_cam.at['k1', 'L Param']
    k2 = df_cam.at['k2', 'L Param']
    k3 = df_cam.at['k3', 'L Param']
    k4 = df_cam.at['k4', 'L Param']
    k5 = df_cam.at['k5', 'L Param']
    cx = df_cam.at['cx', 'L Param']
    cy = df_cam.at['cy', 'L Param']
    sym_coeff_L = [k1, k2, k3, k4, k5, cx, cy]
    transform_L = df_cam.iloc[-6:, 0].to_numpy().copy() #Rz, Ry, Rx, Tx, Ty, Tz
    #transform_L[5] = 40
    #transform_L = [0,0,0,0,0,8] #Rz, Ry, Rx, Tx, Ty, Tz
    #transform_L = [0,0,0,0,0,0] #Rz, Ry, Rx, Tx, Ty, Tz

    #print(sym_coeff_L)
    #print(transform_L)

    k1 = df_cam.at['k1', 'R Param']
    k2 = df_cam.at['k2', 'R Param']
    k3 = df_cam.at['k3', 'R Param']
    k4 = df_cam.at['k4', 'R Param']
    k5 = df_cam.at['k5', 'R Param']
    cx = df_cam.at['cx', 'R Param']
    cy = df_cam.at['cy', 'R Param']
    sym_coeff_R = [k1, k2, k3, k4, k5, cx, cy]
    transform_R = df_cam.iloc[-6:, 1].to_numpy().copy() #Rz, Ry, Rx, Tx, Ty, Tz
    #transform_R = [0, 180, 0, 0, 0, -40]
    #transform_R = transform_R.copy()
    #transform_R[4] += 3.0
    #transform_R[2] += 1.0
    #transform_R = [0,180,0,0,0,-8] #Rz, Ry, Rx, Tx, Ty, Tz
    #transform_R = [0,180,0,0,0,0] #Rz, Ry, Rx, Tx, Ty, Tz

    fov = 180
    outShape2 = [5952, int(5952*fov/180)]
    
    print("\nUndistorting Left...\n")
    equi_imgL = fisheye_reproject(imgL, outShape2, fov, sym_coeff_L, euler_to_H(transform_L), reproj_rad, right=False)
    
    print("\nUndistorting Right...\n")
    equi_imgR = fisheye_reproject(imgR, outShape2, fov, sym_coeff_R, euler_to_H(transform_R), reproj_rad, right=True)

    #equi_imgR = cv2.rotate(equi_imgR, cv2.ROTATE_180)

    equi_img = ldr.copy()
    equi_img = np.zeros((5952, 2*int(5952*fov/180), 3))
    #equi_img[:, :int(ldr_8bit.shape[1]/2), :] = equi_imgL
    #equi_img[:, int(ldr_8bit.shape[1]/2):, :] = equi_imgR

    #print(equi_imgL[:, 5951, :3])
    #lr_mean = (np.mean(equi_imgL[:, -1, 0])) # + np.mean(equi_imgL[:, -20:, :1]))/2
    #lg_mean = (np.mean(equi_imgL[:, -1, 1]))
    #lb_mean = (np.mean(equi_imgL[:, -1, 2]))
    #rr_mean = (np.mean(equi_imgR[:, 0, 0])) # + np.mean(equi_imgL[:, -20:, :1]))/2
    #rg_mean = (np.mean(equi_imgR[:, 0, 1]))
    #rb_mean = (np.mean(equi_imgR[:, 0, 2]))

    #right_mean = (np.mean(equi_imgR[:, :20, :1]) + np.mean(equi_imgR[:, -20:, :1]))/2
    #Rscale = rr_mean / lr_mean
    #Gscale = rg_mean / lg_mean
    #Bscale = rb_mean / lb_mean

    #print(Rscale, Gscale, Bscale)
    #equi_imgR[:, :, 0] = equi_imgR[:, :, 0]/Rscale
    #equi_imgR[:, :, 1] = equi_imgR[:, :, 1]/Gscale
    #equi_imgR[:, :, 2] = equi_imgR[:, :, 2]/Bscale
    #equi_imgR = np.clip(equi_imgR, 0, 255).astype(np.uint8)

    #scale_factor = right_mean / left_mean
    #delta = right_mean - left_mean

    #equi_imgL = exposure.match_histograms(equi_imgL, equi_imgR, channel_axis=-1)
    #equi_imgL = cv2.convertScaleAbs(equi_imgL, alpha=scale_factor, beta=0)
    #equi_imgL = equi_imgL * scale_factor
    #equi_imgL = equi_imgL + delta
    #equi_imgL = np.clip(equi_imgL, 0, 255).astype(np.uint8)

    #equi_img[:, :int(ldr_8bit.shape[1]/4), :] = equi_imgL[:, int(ldr_8bit.shape[1]/4):, :]
    #equi_img[:, int(ldr_8bit.shape[1]/4):int(ldr_8bit.shape[1]*3/4), :] = equi_imgR
    #equi_img[:, int(ldr_8bit.shape[1]*3/4):, :] = equi_imgL[:, :int(ldr_8bit.shape[1]/4), :]
    equi_img[:, :int(equi_img.shape[1]/4), :] = equi_imgL[:, -int(equi_img.shape[1]/4):, :]
    equi_img[:, int(equi_img.shape[1]/4):int(equi_img.shape[1]*3/4), :] = equi_imgR
    equi_img[:, -int(equi_img.shape[1]/4):, :] = equi_imgL[:, :int(equi_img.shape[1]/4), :]

    equi_img[:, :int(equi_img.shape[1]/2), :] = equi_imgL
    equi_img[:, int(equi_img.shape[1]/2):, :] = equi_imgR


    # Display using Matplotlib

    print("\nPlotting...\n")
    plt.figure(1)
    #fig, ax = plt.subplots()
    #plt.imshow(raw2, cmap='gray', vmin=black[0], vmax=raw_max)
    plt.imshow(equi_img.astype("uint8"))
    #plt.imshow(equi_imgR.astype("uint8"))
    #plt.imshow(equi_img/2**16)

    #plt.figure(2)
    #plt.imshow((equi_img[:, :, 1]/255).astype(np.float32), cmap="gray")
    #plt.imshow(ldr_8bit[:, :int(ldr_8bit.shape[1]/2), :])

    #plt.figure(3)
    #plt.hist(equi_imgL.ravel(), 256, [0, 256], alpha=0.5)
    #plt.hist(equi_imgR.ravel(), 256, [0, 256], alpha=0.5)
    plt.show()

    print("\nSaving Image Output...\n")
    input_filename = os.path.basename(file)
    input_filename = os.path.splitext(input_filename)[0]
    output_filename = f"output_undistort_{input_filename}.jpg"
    print("Output Filename: ", output_filename)
    imageio.imsave(output_filename, equi_img.astype("uint8"), quality=100)


if __name__=="__main__":
    args = parse()
    main()

