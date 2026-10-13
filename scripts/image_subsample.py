import cv2
import sys
import argparse
import configparser
import matplotlib.pyplot as plt
import rawpy
import numpy as np
import os
from PIL import Image


def parse():
    parser = argparse.ArgumentParser('Plot Image')
    parser.add_argument('input', help='File containing camera poses')
    parser.add_argument('sub', type=int, help='subsample factor')
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

    image = plt.imread(file)
    image_sub = image[::args.sub, ::args.sub, :]

    plt.figure(1)
    plt.imshow(image_sub)
    plt.show()

    input_basename = os.path.basename(file)
    input_filename = os.path.splitext(input_basename)[0]
    output_filename = f"{input_filename}_subsampled.jpg"
    outputIm = Image.fromarray(image_sub, "RGB")
    outputIm.save(output_filename)


if __name__=="__main__":
    args = parse()
    main()