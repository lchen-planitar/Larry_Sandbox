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

def parse():
    parser = argparse.ArgumentParser('Plot Image')
    parser.add_argument('input', help='ply mesh file')
    args = parser.parse_args(None if (sys.argv[1:]) else ['-h'])
    return args

def main():
    #code
    file = args.input
    mesh = pv.read(file)

    pl = pv.Plotter()
    

    transform = [0, 0, 50, 0, 75, 0] #Rz, Ry, Rx, Tx, Ty, Tz
    H = euler_to_H(transform)
    pl.add_mesh(mesh, user_matrix=H)

    transform = [120, 0, 50, -65, -37.5, 0] #Rz, Ry, Rx, Tx, Ty, Tz
    H = euler_to_H(transform)
    pl.add_mesh(mesh, user_matrix=H)

    transform = [240, 0, 50, 65, -37.5, 0] #Rz, Ry, Rx, Tx, Ty, Tz
    H = euler_to_H(transform)
    pl.add_mesh(mesh, user_matrix=H)

    transform = [0, 0, -90, 0, 0, -100] #Rz, Ry, Rx, Tx, Ty, Tz
    H = euler_to_H(transform)
    #pl.add_mesh(mesh, user_matrix=H)

    
    """
    x, y, z = cone_volume(-7, 60, 1000)
    points = np.stack((x, y, z), axis=0).T
    transform = [240, 0, 53, 0, 0, 0] #Rz, Ry, Rx, Tx, Ty, Tz
    #transform = [180, 0, 0, 0, 0, 0] #Rz, Ry, Rx, Tx, Ty, Tz
    H = euler_to_H(transform)
    points_T = apply_h_transform(points.T, H).T
    pcd3 = pv.PolyData(points_T)

    x, y, z = cone_volume(-7, 60, 1000)
    points = np.stack((x, y, z), axis=0).T
    transform = [0, 0, -90, 0, 0, 0] #Rz, Ry, Rx, Tx, Ty, Tz
    #transform = [270, 0, 0, 0, 0, 0] #Rz, Ry, Rx, Tx, Ty, Tz
    H = euler_to_H(transform)
    points_T = apply_h_transform(points.T, H).T
    pcd4 = pv.PolyData(points_T)
    """
    scale = 100
    xAx = pv.Line((0, 0, 0), (scale, 0, 0))
    yAx = pv.Line((0, 0, 0), (0, scale, 0))
    zAx = pv.Line((0, 0, 0), (0, 0, scale))

    
    #pl.add_points(np.array([0,0,0], np.float32), color="blue", render_points_as_spheres=True, point_size=20)
    #pl.add_mesh(xAx, color='red', render_lines_as_tubes=True, line_width=2)
    #pl.add_mesh(yAx, color='green', render_lines_as_tubes=True, line_width=2)
    #pl.add_mesh(zAx, color='blue', render_lines_as_tubes=True, line_width=2)
    pl.background_color='white'
    pl.show_axes()
    pl.show()


if __name__=="__main__":
    args = parse()
    main()