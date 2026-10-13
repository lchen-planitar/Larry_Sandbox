import cv2
import sys
import argparse
import configparser
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import math


def parse():
    parser = argparse.ArgumentParser('Plot Image')
    parser.add_argument('input', help='File containing camera poses')
    args = parser.parse_args(None if (sys.argv[1:]) else ['-h'])
    return args

def main():
    #code
    file = args.input
    image = plt.imread(file)
    height = image.shape[0]
    width = image.shape[1]
    print(width, height)

    front_img = np.zeros((height, int(width/2), 3), dtype=np.uint8)
    back_img = np.zeros((height, int(width/2), 3), dtype=np.uint8)
    front_img[:,:,:] = image[:, int(width/4):int(width*3/4), :]
    back_img[:,int(width/4):,:] = image[:, :int(width/4), :]
    back_img[:,:int(width/4),:] = image[:, int(width*3/4):, :]
    
    #print(front_img.shape)
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow(front_img)
    ax[1].imshow(back_img)
    plt.show()

    front_pts = np.zeros((int(front_img.shape[0]*front_img.shape[1]), 3))
    front_color = np.zeros((int(front_img.shape[0]*front_img.shape[1]), 3), dtype=np.uint8)

    back_pts = np.zeros((int(back_img.shape[0]*back_img.shape[1]), 3))
    back_color = np.zeros((int(back_img.shape[0]*back_img.shape[1]), 3), dtype=np.uint8)

    count = 0
    radius = 4000
    for i in range(0, front_img.shape[1]):
        asmuth = (i - front_img.shape[1]/2)/(front_img.shape[1]) * 180
        #asmuth2 = (i - front_img.shape[1]/2)/(front_img.shape[1]) * -180
        print(i, asmuth)
        for j in range(0, front_img.shape[0]):
            elevation = (j - front_img.shape[0]/2)/(front_img.shape[0]) * 180
            #elevation2 = (j - front_img.shape[0]/2)/(front_img.shape[0]) * -180
            #print(elevation)
            front_pts[count, 0] = radius*np.cos(elevation*math.pi/180)*np.sin(asmuth*math.pi/180)
            front_pts[count, 1] = -radius*np.cos(elevation*math.pi/180)*np.cos(asmuth*math.pi/180)
            front_pts[count, 2] = radius*np.sin(elevation*math.pi/180)
            front_color[count, 0] = front_img[j, i, 0]
            front_color[count, 1] = front_img[j, i, 1]
            front_color[count, 2] = front_img[j, i, 2]

            back_pts[count, 0] = -radius*np.cos(elevation*math.pi/180)*np.sin(asmuth*math.pi/180)
            back_pts[count, 1] = radius*np.cos(elevation*math.pi/180)*np.cos(asmuth*math.pi/180)
            back_pts[count, 2] = radius*np.sin(elevation*math.pi/180)
            back_color[count, 0] = back_img[j, i, 0]
            back_color[count, 1] = back_img[j, i, 1]
            back_color[count, 2] = back_img[j, i, 2]
            count += 1

    front_cloud = pv.PolyData(front_pts[::2, :])
    front_cloud["colors"] = front_color[::2, :]
    back_cloud = pv.PolyData(back_pts[::2, :])
    back_cloud["colors"] = back_color[::2, :]

    pl = pv.Plotter()
    pl.camera_position = [(2000, 2000, 2000), (0, 0, 0), (0, 0, 1)]
    pl.add_mesh(front_cloud, rgb=True, render_points_as_spheres=True, point_size=5, lighting=False)
    pl.add_mesh(back_cloud, rgb=True, render_points_as_spheres=True, point_size=5, lighting=False)
    #pl.add_points(front_pts[::10, :], scalars=front_color[::10, :], rgb=True, render_points_as_spheres=True, point_size=25)
    pl.add_points(np.array([0,0,0], np.float32), color="blue", render_points_as_spheres=True, point_size=20)
    pl.background_color='gray'
    pl.show_axes()
    #pl.disable_shadows()
    pl.show()


if __name__=="__main__":
    args = parse()
    main()