import cv2
import sys
import argparse
import configparser
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import math
from matplotlib.widgets import Button, Slider

from tools import euler_to_H
from tools import H_to_euler
from tools import apply_h_transform
from tools import inverse_H

sensor_width = 1280
sensor_height = 1280
focal_length = 1000
cam_az = 0
cam_el = 0



def unwrap_360(image, points, sensor_height, sensor_width, cam_az, cam_el):
    #image -> raw image (height, width, 3, dtype=np.uint8)
    #points -> normalized position of cropped sensor pixels in xyz (sensor_height*sensor_width, 3)
    #sensor_height -> height of cropped region in pixels
    #sensor_width -> width of cropped region in pixels
    #cam_az -> asmuth of center of cropped region in degrees
    #cam_el -> elevation of center of cropped region in degrees
    #returns cropped region as unwrapped image

    cam_Euler = np.array([0, cam_az, cam_el, 0, 0, 0])
    H_cam = euler_to_H(cam_Euler)
    height = image.shape[0]
    width = image.shape[1]

    trans_pts = apply_h_transform(points.T, H_cam)
    trans_pts = trans_pts.T
    elevation = np.arcsin(trans_pts[:, 1]) * 180 / math.pi

    trans_xz = np.zeros((sensor_height*sensor_width, 2))
    trans_xz[:, 0] = trans_pts[:, 0]
    trans_xz[:, 1] = trans_pts[:, 2]
    trans_xz = trans_xz.T / np.linalg.norm(trans_xz.T, axis=0)
    trans_xz = trans_xz.T
    asmuth = np.arccos(trans_xz[:, 1]) *180 / math.pi
    filter = (trans_xz < 0)
    asmuth[filter[:, 0]] = -1*asmuth[filter[:, 0]]

    asmuth_px = (asmuth * width / 360) + (width / 2)
    elevation_px = (elevation * height / 180) + (height / 2)
    
    d1_index = elevation_px.astype(int)
    d2_index = asmuth_px.astype(int)
    d1_index[d1_index >= height] = height - 1
    d2_index[d2_index >= width] = width - 1
    unwrapped = image[d1_index, d2_index, :].copy()
    unwrapped = unwrapped.reshape((sensor_height, sensor_width, 3))

    return unwrapped, trans_pts

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

    #front_img = np.zeros((height, int(width/2), 3), dtype=np.uint8)
    #back_img = np.zeros((height, int(width/2), 3), dtype=np.uint8)
    #front_img[:,:,:] = image[:, int(width/4):int(width*3/4), :]
    #back_img[:,int(width/4):,:] = image[:, :int(width/4), :]
    #back_img[:,:int(width/4),:] = image[:, int(width*3/4):, :]

    sphere = pv.Sphere(radius=1, center=(0,0,0))
    x_ax = pv.Line(pointa=(0,0,0), pointb=(1,0,0))
    y_ax = pv.Line(pointa=(0,0,0), pointb=(0,1,0))
    z_ax = pv.Line(pointa=(0,0,0), pointb=(0,0,1))

    cam_Euler = np.array([0, cam_az, cam_el, 0, 0, 0])
    H_cam = euler_to_H(cam_Euler)

    prime_ax = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]], dtype=np.float32)
    
    trans_ax = apply_h_transform(prime_ax, H_cam)

    x_ax2 = pv.Line(pointa=(0,0,0), pointb=trans_ax[:, 0])
    y_ax2 = pv.Line(pointa=(0,0,0), pointb=trans_ax[:, 1])
    z_ax2 = pv.Line(pointa=(0,0,0), pointb=trans_ax[:, 2])

    window_out = np.zeros((sensor_height, sensor_width, 3))
    for i in range (0, sensor_width):
        window_out[:, i, 0] = i
    for j in range (0, sensor_height):
        window_out[j, :, 1] = j
    window_out[:, :, 2] = focal_length

    points = window_out.copy().reshape((sensor_height*sensor_width, 3))
    points[:, 0] = points[:, 0] - 0.5*sensor_width
    points[:, 1] = points[:, 1] - 0.5*sensor_height
    points = points.T / np.linalg.norm(points.T, axis=0)
    points = points.T

    print("start")
    unwrapped, trans_pts = unwrap_360(image, points, sensor_height, sensor_width, cam_az, cam_el)
    print("end")

    #fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    #ax1.imshow(image)
    ax2.imshow(unwrapped)
    fig2.subplots_adjust(left=0.25, bottom=0.25)

    def update(val):
        new_unwrap,_ = unwrap_360(image, points, sensor_height, sensor_width, az_slider.val, el_slider.val)
        ax2.imshow(new_unwrap)

    asmuth_ax = fig2.add_axes([0.25, 0.1, 0.65, 0.03])
    az_slider = Slider(
        ax=asmuth_ax,
        label='Asmuth [Deg]',
        valmin=-180,
        valmax=180,
        valstep=20,
        valinit=0,
        )

    elevation_ax = fig2.add_axes([0.1, 0.25, 0.0225, 0.63])
    el_slider = Slider(
        ax=elevation_ax,
        label='Elevation [Deg]',
        valmin=-90,
        valmax=90,
        valstep=10,
        valinit=0,
        orientation="vertical"
        )

    az_slider.on_changed(update)
    el_slider.on_changed(update)
    plt.show()

    pl = pv.Plotter()
    #pl.add_points(points, color="blue")
    #pl.add_points(trans_pts, color="red")
    pl.add_mesh(sphere, color="cyan", opacity=0.2)
    pl.add_mesh(x_ax, color="red", line_width=8, render_lines_as_tubes=True)
    pl.add_mesh(y_ax, color="green", line_width=8, render_lines_as_tubes=True)
    pl.add_mesh(z_ax, color="blue", line_width=8, render_lines_as_tubes=True)

    pl.add_mesh(x_ax2, color="red", line_width=2, render_lines_as_tubes=True)
    pl.add_mesh(y_ax2, color="green", line_width=2, render_lines_as_tubes=True)
    pl.add_mesh(z_ax2, color="blue", line_width=2, render_lines_as_tubes=True)
    pl.background_color='gray'
    pl.show_axes()
    #pl.show()


if __name__=="__main__":
    args = parse()
    main()