import cv2
import sys
import argparse
import configparser
import matplotlib.pyplot as plt
import glob
import numpy as np


def parse():
    parser = argparse.ArgumentParser('Plot Image')
    parser.add_argument('inputs', help='File containing camera poses')
    args = parser.parse_args(None if (sys.argv[1:]) else ['-h'])
    return args

def main():
    #code
    files = glob.glob(args.inputs)
    count = 0
    for file in files:
        if count == 0:
            max_image = plt.imread(file).astype(float)
            min_image = plt.imread(file).astype(float)
        else:
            image = plt.imread(file).astype(float)
            max_image = np.maximum(max_image, image)
            min_image = np.minimum(min_image, image)
        print(file)
        count += 1
    
    range_image = max_image - min_image
    noise = range_image[:,:,2]
    #noise = np.mean(range_image, axis=2)
    
    print(noise.shape)
    print("RMS: %.3f" % (np.mean(noise**2)**0.5) )
    plt.figure(1)
    plt.imshow(noise, vmin=0, vmax=20, cmap='jet')
    plt.colorbar()
    plt.show()


if __name__=="__main__":
    args = parse()
    main()