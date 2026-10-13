import cv2
import sys
import argparse
import configparser
import matplotlib.pyplot as plt
import rawpy
import numpy as np


def parse():
    parser = argparse.ArgumentParser('Plot Image')
    parser.add_argument('input', help='File containing camera poses')
    args = parser.parse_args(None if (sys.argv[1:]) else ['-h'])
    return args

def gray_world_wb(img):
    # Convert to float to prevent clipping during calculation
    img_float = img.astype(float)
    # Calculate the mean for each channel (B, G, R)
    mean_b = np.mean(img_float[:, :, 0])
    mean_g = np.mean(img_float[:, :, 1])
    mean_r = np.mean(img_float[:, :, 2])
    
    # Calculate the overall gray value (average of means)
    gray = (mean_b + mean_g + mean_r) / 3
    
    # Scale each channel
    img_float[:, :, 0] *= (gray / mean_b)
    img_float[:, :, 1] *= (gray / mean_g)
    img_float[:, :, 2] *= (gray / mean_r)
    
    # Clip values to [0, 255] and convert back to uint8
    return np.clip(img_float, 0, 255).astype(np.uint8)

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

    #with rawpy.imread(file) as raw:
    #    # Post-process to get an RGB array
    #    rgb = raw.postprocess()

    ## Display using Matplotlib
    #plt.figure(1)
    #plt.imshow(rgb)
    #plt.show()

    image = cv2.imread(file)
    print(image[0,0,:])
    #image = gray_world_wb(image)
    #image = apply_gamma(image)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(1)
    plt.imshow(img_rgb)
    plt.show()


if __name__=="__main__":
    args = parse()
    main()