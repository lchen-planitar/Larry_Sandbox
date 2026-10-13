import cv2
import sys
import argparse
import configparser
import matplotlib.pyplot as plt
import rawpy
import numpy as np
import imageio
from omnicv import fisheyeImgConv



def parse():
    parser = argparse.ArgumentParser('Plot Image')
    parser.add_argument('input', help='File containing camera poses')
    args = parser.parse_args(None if (sys.argv[1:]) else ['-h'])
    return args


def LCA_correction(image, r_scale, b_scale):
    r_image = image[1::2, 0::2]
    b_image = image[0::2, 1::2]

    (h, w) = r_image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 0, r_scale)
    r_image_scaled = cv2.warpAffine(r_image, M, (w, h))

    (h, w) = b_image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 0, b_scale)
    b_image_scaled = cv2.warpAffine(b_image, M, (w, h))

    out_image = image.copy()
    out_image[1::2, 0::2] = r_image_scaled
    out_image[0::2, 1::2] = b_image_scaled

    return out_image

def fisheye_to_equirectangular(img, outShape, fov, radius, dx, dy):
    h_src, w_src = img.shape[:2]
    # Define output equirectangular dimensions (2:1 aspect ratio)
    w_dest = outShape[1]
    h_dest = outShape[0]
    
    # Create meshgrid for target equirectangular image
    x, y = np.meshgrid(np.arange(w_dest), np.arange(h_dest))
    
    # Normalize coordinates to -pi to pi (longitude) and -pi/2 to pi/2 (latitude)
    theta = (np.pi * x / w_dest) - (np.pi / 2)
    phi = (np.pi * y / h_dest) - (np.pi / 2)
    
    # Spherical to Cartesian (assuming Y-up for the camera axis)
    psph_x = np.cos(phi) * np.sin(theta)
    psph_y = np.cos(phi) * np.cos(theta)
    psph_z = np.sin(phi)
    
    # Cartesian to Fisheye polar coordinates
    fov_rad = np.radians(fov)
    angle = np.arctan2(np.sqrt(psph_x**2 + psph_z**2), psph_y)
    beta = np.arctan2(psph_z, psph_x)
    
    # Radius in fisheye depends on FOV; assume radius is half of image width
    r = (radius) * (angle / (fov_rad / 2))
    
    # Fisheye 2D coordinates
    map_x = ((w_src / 2)+dx) + r * np.cos(beta)
    map_y = ((h_src / 2)+dy) + r * np.sin(beta)
    
    # Remap image using OpenCV's optimized function
    return cv2.remap(img, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR)



    return im_out

def main():
    #code
    file = args.input

    with rawpy.imread(file) as raw:
        # Post-process to get an RGB array
        print(f'Raw type: {raw.raw_type}')
        print(f'Number of colors: {raw.num_colors}')
        print(f'Color description: {raw.color_desc}')
        print(f'Color Matrix: {raw.color_matrix}')
        print(f'Black levels: {raw.black_level_per_channel}')
        print(f'White level: {raw.white_level}')
        print(f'Camera white balance: {raw.camera_whitebalance}')
        raw_im = raw.raw_image_visible.astype(np.uint16)
        raw_left = raw_im[:, :int(raw_im.shape[1]/2)]
        corr_left = LCA_correction(raw_left, 0.9998, 0.9998)
        corr_left = LCA_correction(raw_left, 1, 1)
        corr_im = raw_im.copy()
        corr_im[:, :int(raw_im.shape[1]/2)] = corr_left
        raw.raw_image_visible[:] = corr_im
        raw2 = raw.raw_image_visible.astype(np.uint16)
        rgb = raw.postprocess(gamma=(1, 1), no_auto_bright=True, use_camera_wb=True, auto_bright_thr=0.03, output_bps=16, 
                            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD, output_color=rawpy.ColorSpace.sRGB)
        black = raw.black_level_per_channel

    rgb_float = (rgb / (2**16)).astype(np.float32)
    print(rgb_float.shape)

    raw_max = np.percentile(raw2, 99)
    print("raw image scale top: ", raw_max)

    tonemapReinhard = cv2.createTonemapReinhard(1.5, 0,0,0.4)
    ldr = tonemapReinhard.process(rgb_float)
    ldr_8bit = np.clip(ldr * 255, 0, 255).astype('uint8')

    #circle1 = plt.Circle((5952/2, 5952/2), 2700, color='red', fill=False)
    #circle2 = plt.Circle((5970+5952/2, 5952/2), 2700, color='red', fill=False)

    imgL = ldr_8bit[:, :int(ldr_8bit.shape[1]/2), :]
    imgR = ldr_8bit[:, int(ldr_8bit.shape[1]/2):, :]
    imgL = rgb[:, :int(rgb.shape[1]/2), :]
    imgR = rgb[:, int(rgb.shape[1]/2):, :]
    outShape = [5952, 5952]
    radL = 2376
    radR = 2376
    dx_L = 0
    dy_L = 0
    dx_R = 0
    dy_R = 0

    equi_imgL = fisheye_to_equirectangular(imgL, outShape, 180, radL, dx_L, dy_L)
    equi_imgR = fisheye_to_equirectangular(imgR, outShape, 180, radR, dx_R, dy_R)
    equi_img = ldr.copy()
    equi_img[:, :int(ldr_8bit.shape[1]/2), :] = equi_imgL
    equi_img[:, int(ldr_8bit.shape[1]/2):, :] = equi_imgR
    # Display using Matplotlib
    plt.figure(1)
    #fig, ax = plt.subplots()
    #plt.imshow(raw2, cmap='gray', vmin=black[0], vmax=raw_max)
    plt.imshow(equi_img.astype("uint8"))
    #plt.imshow(equi_img/(2**16))
    #ax.add_patch(circle1)
    #ax.add_patch(circle2)
    plt.show()
    imageio.imsave('output.jpg', ldr_8bit, quality=100)


if __name__=="__main__":
    args = parse()
    main()

