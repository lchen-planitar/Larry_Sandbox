import cv2
import sys
import argparse
import configparser
import matplotlib.pyplot as plt
import pyvista as pv
import numpy as np

from tools import apply_h_transform
from tools import inverse_H
from tools import H_to_euler
from tools import euler_to_H

def cone_volume(t_ang, b_ang, r):
    # Define parameters
    t_ang = t_ang * np.pi / 180
    b_ang = b_ang * np.pi / 180

    # Generate angles from 0 to 2*pi
    az = np.linspace(0, 2 * np.pi, 360)
    el = np.linspace(t_ang, b_ang, 180)

    theta, phi = np.meshgrid(az, el)
    noise = np.random.normal(0, 1e-2, size=(theta.shape[0], theta.shape[1]))
    theta = theta + noise
    phi = phi + noise

    # Parametric equations for the ellipse
    x = r * np.cos(phi) * np.sin(theta)
    z = r * np.cos(phi) * np.cos(theta)
    y = r * np.sin(phi)

    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    return x, y, z

def parse():
    parser = argparse.ArgumentParser('Plot Image')
    parser.add_argument('input', help='ply mesh file')
    args = parser.parse_args(None if (sys.argv[1:]) else ['-h'])
    return args

def main():
    #code

    offset_r = 50

    x, y, z = cone_volume(-7, 52, 1000)
    points = np.stack((x, y, z), axis=0).T
    transform = [0, 0, 50, 0, 0, 0] #Rz, Ry, Rx, Tx, Ty, Tz
    transform = [180, 0, 0, 0, 0, 0] #Rz, Ry, Rx, Tx, Ty, Tz
    H = euler_to_H(transform)
    points_T = apply_h_transform(points.T, H).T
    pcd1 = pv.PolyData(points_T)

    x, y, z = cone_volume(-7, 52, 1000)
    points = np.stack((x, y, z), axis=0).T
    transform = [120, 0, 50, 0, 0, 0] #Rz, Ry, Rx, Tx, Ty, Tz
    transform = [-30, 0, 0, 0, 0, 0] #Rz, Ry, Rx, Tx, Ty, Tz
    H = euler_to_H(transform)
    points_T = apply_h_transform(points.T, H).T
    pcd2 = pv.PolyData(points_T)

    x, y, z = cone_volume(-7, 52, 1000)
    points = np.stack((x, y, z), axis=0).T
    transform = [240, 0, 50, 0, 0, 0] #Rz, Ry, Rx, Tx, Ty, Tz
    transform = [30, 0, 0, 0, 0, 0] #Rz, Ry, Rx, Tx, Ty, Tz
    H = euler_to_H(transform)
    points_T = apply_h_transform(points.T, H).T
    pcd3 = pv.PolyData(points_T)

    x, y, z = cone_volume(-7, 52, 1000)
    points = np.stack((x, y, z), axis=0).T
    transform = [0, 0, -90, 0, 0, 0] #Rz, Ry, Rx, Tx, Ty, Tz
    #transform = [270, 0, 10, 0, 0, 0] #Rz, Ry, Rx, Tx, Ty, Tz
    H = euler_to_H(transform)
    points_T = apply_h_transform(points.T, H).T
    pcd4 = pv.PolyData(points_T)

    scale = 500
    xAx = pv.Line((0, 0, 0), (scale, 0, 0))
    yAx = pv.Line((0, 0, 0), (0, scale, 0))
    zAx = pv.Line((0, 0, 0), (0, 0, scale))

    pl = pv.Plotter()
    pl.add_mesh(pcd1, rgb=False, color='r', render_points_as_spheres=True, point_size=3, lighting=True, opacity=1)
    pl.add_mesh(pcd2, rgb=False, color='g', render_points_as_spheres=True, point_size=3, lighting=True, opacity=1)
    pl.add_mesh(pcd3, rgb=False, color='b', render_points_as_spheres=True, point_size=3, lighting=True, opacity=1)
    #pl.add_mesh(pcd4, rgb=False, color='yellow', render_points_as_spheres=True, point_size=3, lighting=True, opacity=1)



    pl.add_mesh(xAx, color='red', render_lines_as_tubes=True, line_width=2)
    pl.add_mesh(yAx, color='green', render_lines_as_tubes=True, line_width=2)
    pl.add_mesh(zAx, color='blue', render_lines_as_tubes=True, line_width=2)
    pl.background_color='white'
    pl.show_axes()
    pl.show()

    


if __name__=="__main__":
    #args = parse()
    main()