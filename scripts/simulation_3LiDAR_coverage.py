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
    az = np.linspace(0, 2 * np.pi, 1000)
    el = np.linspace(t_ang, b_ang, 500)

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
    file = args.input
    mesh = pv.read(file)

    offset_r = 60
    rise_ang = 45
    low_ang = -10
    high_ang = 60

    pl = pv.Plotter()
    sphere = pv.Sphere(radius=1000).triangulate()
    cube = pv.Cube(
                    center=(0, 0, 0),  # Position of the cube's center
                    x_length=2500.0,  # Width along X-axis
                    y_length=2500.0,  # Height along Y-axis
                    z_length=2400.0,  # Depth along Z-axis
                ).triangulate()

    roll_ang = 0
    x, y, z = cone_volume(low_ang, high_ang, 1)
    vectors = np.stack((x, y, z), axis=0).T
    transform = [roll_ang, 0, rise_ang, 0, 0, 0] #Rz, Ry, Rx, Tx, Ty, Tz
    H = euler_to_H(transform)
    vectors_T = apply_h_transform(vectors.T, H).T
    transform = [roll_ang, 0, rise_ang, -np.sin(roll_ang*np.pi/180)*offset_r, np.cos(roll_ang*np.pi/180)*offset_r, 0] #Rz, Ry, Rx, Tx, Ty, Tz
    H = euler_to_H(transform)
    origins = np.zeros((vectors_T.shape[0], 3))
    origins_T = apply_h_transform(origins.T, H).T
    print(origins_T.shape)
    #print(origins_T[0, :])
    pcd1, _, _ = cube.multi_ray_trace(origins_T, vectors_T)
    pl.add_mesh(mesh, color='r', user_matrix=H)

    roll_ang = 120
    x, y, z = cone_volume(low_ang, high_ang, 1)
    vectors = np.stack((x, y, z), axis=0).T
    transform = [roll_ang, 0, rise_ang, 0, 0, 0] #Rz, Ry, Rx, Tx, Ty, Tz
    #transform = [0, 0, 0, 0, 0, 0] #Rz, Ry, Rx, Tx, Ty, Tz
    H = euler_to_H(transform)
    vectors_T = apply_h_transform(vectors.T, H).T
    transform = [roll_ang, 0, rise_ang, -np.sin(roll_ang*np.pi/180)*offset_r, np.cos(roll_ang*np.pi/180)*offset_r, 0] #Rz, Ry, Rx, Tx, Ty, Tz
    H = euler_to_H(transform)
    origins = np.zeros((vectors_T.shape[0], 3))
    origins_T = apply_h_transform(origins.T, H).T
    print(origins_T.shape)
    pcd2, _, _ = cube.multi_ray_trace(origins_T, vectors_T)
    pl.add_mesh(mesh, color='g', user_matrix=H)

    roll_ang = 240
    x, y, z = cone_volume(low_ang, high_ang, 1)
    vectors = np.stack((x, y, z), axis=0).T
    transform = [roll_ang, 0, rise_ang, 0, 0, 0] #Rz, Ry, Rx, Tx, Ty, Tz
    #transform = [0, 0, 0, 0, 0, 0] #Rz, Ry, Rx, Tx, Ty, Tz
    H = euler_to_H(transform)
    vectors_T = apply_h_transform(vectors.T, H).T
    transform = [roll_ang, 0, rise_ang, -np.sin(roll_ang*np.pi/180)*offset_r, np.cos(roll_ang*np.pi/180)*offset_r, 0] #Rz, Ry, Rx, Tx, Ty, Tz
    H = euler_to_H(transform)
    origins = np.zeros((vectors_T.shape[0], 3))
    origins_T = apply_h_transform(origins.T, H).T
    print(origins_T.shape)
    pcd3, _, _ = cube.multi_ray_trace(origins_T, vectors_T)
    pl.add_mesh(mesh, color='b', user_matrix=H)

    roll_ang = 270
    x, y, z = cone_volume(low_ang, high_ang, 1)
    vectors = np.stack((x, y, z), axis=0).T
    transform = [roll_ang, 0, rise_ang, 0, 0, 0] #Rz, Ry, Rx, Tx, Ty, Tz
    #transform = [0, 0, 0, 0, 0, 0] #Rz, Ry, Rx, Tx, Ty, Tz
    H = euler_to_H(transform)
    vectors_T = apply_h_transform(vectors.T, H).T
    transform = [roll_ang, 0, rise_ang, -np.sin(roll_ang*np.pi/180)*offset_r, np.cos(roll_ang*np.pi/180)*offset_r, 0] #Rz, Ry, Rx, Tx, Ty, Tz
    H = euler_to_H(transform)
    origins = np.zeros((vectors_T.shape[0], 3))
    origins_T = apply_h_transform(origins.T, H).T
    print(origins_T.shape)
    pcd4, _, _ = cube.multi_ray_trace(origins_T, vectors_T)
    #pl.add_mesh(mesh, color='y', user_matrix=H)


    scale = 500
    xAx = pv.Line((0, 0, 0), (scale, 0, 0))
    yAx = pv.Line((0, 0, 0), (0, scale, 0))
    zAx = pv.Line((0, 0, 0), (0, 0, scale))

    #pl.add_mesh(cube, color='gray', opacity=0.1)
    pl.add_mesh(pcd1, rgb=False, color='r', render_points_as_spheres=True, point_size=1, lighting=True, opacity=1)
    pl.add_mesh(pcd2, rgb=False, color='g', render_points_as_spheres=True, point_size=1, lighting=True, opacity=1)
    pl.add_mesh(pcd3, rgb=False, color='b', render_points_as_spheres=True, point_size=1, lighting=True, opacity=1)
    #pl.add_mesh(pcd4, rgb=False, color='y', render_points_as_spheres=True, point_size=1, lighting=True, opacity=1)
    #pl.add_mesh(pcd4, rgb=False, color='yellow', render_points_as_spheres=True, point_size=3, lighting=True, opacity=1)



    pl.add_mesh(xAx, color='red', render_lines_as_tubes=True, line_width=2)
    pl.add_mesh(yAx, color='green', render_lines_as_tubes=True, line_width=2)
    pl.add_mesh(zAx, color='blue', render_lines_as_tubes=True, line_width=2)
    pl.background_color='white'
    pl.show_axes()
    pl.show()

    


if __name__=="__main__":
    args = parse()
    main()