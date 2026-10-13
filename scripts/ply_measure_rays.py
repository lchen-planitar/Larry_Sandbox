import cv2
import sys
import argparse
import configparser
import matplotlib.pyplot as plt
import pyvista as pv
import math
import numpy as np

from tools import apply_h_transform
from tools import inverse_H
from tools import H_to_euler
from tools import euler_to_H

def parse():
    parser = argparse.ArgumentParser('Plot Image')
    parser.add_argument('input', help='ply mesh file')
    args = parser.parse_args(None if (sys.argv[1:]) else ['-h'])
    return args

def main():
    #code
    file = args.input
    mesh = pv.read(file)

    scale = 0.5
    xAx = pv.Line((0, 0, 0), (scale, 0, 0))
    yAx = pv.Line((0, 0, 0), (0, scale, 0))
    zAx = pv.Line((0, 0, 0), (0, 0, scale))

    point_count = 1000
    thetas = np.linspace(0,2*math.pi,point_count)
    r = 10000
    origins = np.zeros((point_count, 3))
    directions = np.zeros((point_count, 3))
    directions[:, 0] = np.sin(thetas)
    directions[:, 1] = np.cos(thetas)

    transform = [0, 25, 0, 0, 0, 0] #Rz, Ry, Rx, Tx, Ty, Tz
    H = euler_to_H(transform)

    directions = apply_h_transform(directions.T, H).T

    intersects,_,_ = mesh.multi_ray_trace(origins=origins, directions=directions, first_point=True)

    H_inv = inverse_H(H)
    intersects_trans = apply_h_transform(intersects.T, H_inv).T

    plt.figure(1)
    plt.scatter(intersects_trans[:, 0], intersects_trans[:, 1], color='red', s=3)
    plt.axis('equal')
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.show()

    pl = pv.Plotter()
    pl.add_mesh(mesh, rgb=True, lighting=False, opacity=0.7)
    #pl.add_points(np.array([0,0,0], np.float32), color="blue", render_points_as_spheres=True, point_size=20)
    pl.add_points(intersects, color="red", render_points_as_spheres=True, point_size=5)
    pl.add_mesh(xAx, color='red', render_lines_as_tubes=True, line_width=2)
    pl.add_mesh(yAx, color='green', render_lines_as_tubes=True, line_width=2)
    pl.add_mesh(zAx, color='blue', render_lines_as_tubes=True, line_width=2)
    pl.background_color='white'
    pl.show_axes()
    pl.show()


if __name__=="__main__":
    args = parse()
    main()