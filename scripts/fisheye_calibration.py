import pandas as pd
import numpy as np
import sys
import argparse
import configparser
import matplotlib.pyplot as plt
import math
from scipy.optimize import least_squares
from matplotlib.colors import Normalize
import pyvista as pv

from tools import apply_h_transform
from tools import inverse_H
from tools import H_to_euler
from tools import euler_to_H

def fisheye_to_xyz(u, v, cx, cy, f, scale=500):
    """
    Projects 2D fisheye pixel coordinates to 3D Cartesian coordinates.
    
    Parameters:
    u, v      : Pixel coordinates (can be arrays)
    cx, cy    : Optical center of the image (pixels)
    f         : Focal length (pixels)
    lens_type : 'equidistant', 'equisolid', 'orthographic', 'stereographic'
    """
    # 1. Shift origin to the image center
    x_centered = u - cx
    y_centered = v - cy
    
    # 2. Calculate pixel radius (r) and azimuth angle (theta)
    r = np.sqrt(x_centered**2 + y_centered**2)
    theta = np.arctan2(y_centered, x_centered)
    
    # Prevent division by zero at the exact center
    r = np.maximum(r, 1e-10)
    phi = r / f
        
    # 4. Convert Spherical (phi, theta) to Cartesian (XYZ) on a unit sphere
    X = np.sin(phi) * np.cos(theta)
    Y = np.sin(phi) * np.sin(theta)
    Z = np.cos(phi)
    
    return X*scale, Y*scale, Z*scale

def xyz_to_fisheye(x, y, z, coeff):

    k1 = coeff[0]
    k2 = coeff[1]
    k3 = coeff[2]
    k4 = coeff[3]
    k5 = 0#coeff[4]
    cx = coeff[5]
    cy = coeff[6]

    theta = np.arctan2(np.sqrt(x**2 + y**2), z)
    phi = np.arctan2(y, x)
    
    # Radius in fisheye depends on FOV; assume radius is half of image width
    r = k1*theta + k2*theta**3 + k3*theta**5 + k4*theta**7 + k5*theta**9
    
    # Fisheye 2D coordinates
    u = cx + r * np.cos(phi)
    v = cy + r * np.sin(phi)

    return u, v

def calibrate_fisheye_lens(coeff_guess, transform_guess, meas_u, meas_v, d3):

    cam_guess = np.concatenate((coeff_guess, transform_guess))
    coeff = cam_guess[:7]
    transform = cam_guess[7:]

    def minFunc(cam):
        coeff = cam[:7]
        transform = cam[7:]
        H = euler_to_H(transform)
        d3_trans = apply_h_transform(d3, H)
        reproj_u, reproj_v = xyz_to_fisheye(d3_trans[0, :], d3_trans[1, :], d3_trans[2, :], coeff)
        cost = np.concatenate(( np.abs(meas_u-reproj_u), np.abs(meas_v-reproj_v) ))
        return cost

    result = least_squares(minFunc, cam_guess, loss='linear', jac='3-point', method='trf', verbose=2,
                               gtol=1e-12, ftol=1e-12, xtol=1e-15, max_nfev=200)

    camParam = result.x
    coeff = camParam[:7]
    transform = camParam[7:]

    return coeff, transform

def ring_gen(az=500, el=20, fov_az=205, fov_el=180):

    x, y = np.meshgrid(np.arange(az), np.arange(el))

    az_lim = fov_az * np.pi / 180
    el_lim = fov_el * np.pi / 180
    
    # Normalize coordinates to +/-0.5FOV (longitude) and +/-0.5FOV (latitude)
    theta = (az_lim * x / az) - (az_lim / 2)
    phi = (el_lim * y / el) - (el_lim / 2)

    #theta = np.linspace(-np.pi/2, np.pi/2, 360)
    #phi = np.linspace(-np.pi/2, np.pi/2, 11)
    
    # Spherical to Cartesian
    x = np.cos(phi) * np.sin(theta)
    z = np.cos(phi) * np.cos(theta)
    y = np.sin(phi)

    xyz = np.stack((x, y, z), axis=0)

    return xyz

def parse():
    parser = argparse.ArgumentParser('Plot Image')
    parser.add_argument('coordL', help='input left coordinate file')
    parser.add_argument('coordR', help='input right coordinate file')
    parser.add_argument('image', help='input image file')
    parser.add_argument('out_file', help='output file')
    args = parser.parse_args(None if (sys.argv[1:]) else ['-h'])
    return args

def main():
    
    ########  Loading Inputs #######

    image = plt.imread(args.image)
    height = image.shape[0]
    width = image.shape[1]
    df_L = pd.read_csv(args.coordL)
    df_R = pd.read_csv(args.coordR)

    d3L = np.zeros((3, df_L.shape[0]))
    d3R = np.zeros((3, df_R.shape[0]))

    d3L[0, :] = df_L.iloc[:, 3]
    d3L[1, :] = df_L.iloc[:, 4]
    d3L[2, :] = df_L.iloc[:, 5]

    d3R[0, :] = df_R.iloc[:, 3]
    d3R[1, :] = df_R.iloc[:, 4]
    d3R[2, :] = df_R.iloc[:, 5]

    meas_uL = df_L.iloc[:, 1]
    meas_vL = df_L.iloc[:, 2]

    meas_uR = df_R.iloc[:, 1]
    meas_vR = df_R.iloc[:, 2]

    ########  Initial Guess #######

    R = 2500
    k1 = R/(math.pi/2)
    k2 = 0
    k3 = 0
    k4 = 0
    k5 = 0
    cx = 5952/2
    cy = 5952/2
    coeff_guess_L = [k1, k2, k3, k4, k5, cx, cy]
    transform_guess_L = [0, 180, 0, 0, 0, 0] #Rz, Ry, Rx, Tx, Ty, Tz
    coeff_guess_R = [k1, k2, k3, k4, k5, cx, cy]
    transform_guess_R = [0, 0, 0, 0, 0, 0] #Rz, Ry, Rx, Tx, Ty, Tz

    #H = euler_to_H(transform_guess)

    #d3_trans = apply_h_transform(d3.copy(), H)

    #x, y, z = fisheye_to_xyz(meas_uL, meas_vL, cx, cy, k1, scale=500)
    #reproj_u, reproj_v = xyz_to_fisheye(d3_trans[0, :], d3_trans[1, :], d3_trans[2, :], coeff_guess)
    #reproj_u, reproj_v = xyz_to_fisheye(x, y, z, coeff)

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax1.imshow(image, cmap="gray")
    ax1.scatter(meas_uL, meas_vL, s=4)
    ax1.scatter(meas_uR+int(width/2), meas_vR, s=4)

    #for i in range(0, df.shape[0]):
    #    ax1.annotate(
    #        f"({int(df.iloc[i, 0])})", 
    #        (df.iloc[i, 1], df.iloc[i, 2]),              # Point to label
    #        textcoords="offset points",   # Specify positioning type
    #        xytext=(0, 10),               # Distance from text to point (x,y)
    #        ha='center'                   # Horizontal alignment
    #    )

    #ax1.scatter(reproj_u, reproj_v, s=4)

    #for i in range(0, df.shape[0]):
    #    ax1.annotate(
    #        f"({int(df.iloc[i, 0])})", 
    #        (reproj_u[i], reproj_v[i]),              # Point to label
    #        textcoords="offset points",   # Specify positioning type
    #        xytext=(0, 10),               # Distance from text to point (x,y)
    #        ha='center'                   # Horizontal alignment
    #    )

    ax2.scatter(d3L[0, :], d3L[1, :], d3L[2, :])
    ax2.scatter(d3R[0, :], d3R[1, :], d3R[2, :])
    #ax2.scatter(x, y, z)
    ax1.set_aspect("equal")
    ax2.set_aspect("equal")
    plt.show()

    ########  Calibrate Camera Parameters #######

    coeff_L, transform_L = calibrate_fisheye_lens(coeff_guess_L, transform_guess_L, meas_uL, meas_vL, d3L)
    coeff_R, transform_R = calibrate_fisheye_lens(coeff_guess_R, transform_guess_R, meas_uR, meas_vR, d3R)

    ########  Evaluate Calibration Quality #######

    Hl = euler_to_H(transform_L)
    d3L_trans = apply_h_transform(d3L, Hl)
    ringsL = ring_gen()

    Hr = euler_to_H(transform_R)
    d3R_trans = apply_h_transform(d3R, Hr)
    ringsR = ring_gen()

    reproj_uL, reproj_vL = xyz_to_fisheye(d3L_trans[0, :], d3L_trans[1, :], d3L_trans[2, :], coeff_L)
    reproj_uR, reproj_vR = xyz_to_fisheye(d3R_trans[0, :], d3R_trans[1, :], d3R_trans[2, :], coeff_R)
    rings_uL, rings_vL = xyz_to_fisheye(ringsL[0, :], ringsL[1, :], ringsL[2, :], coeff_R)
    rings_uR, rings_vR = xyz_to_fisheye(ringsR[0, :], ringsR[1, :], ringsR[2, :], coeff_guess_R)

    UL_vec = reproj_uL - meas_uL
    VL_vec = reproj_vL - meas_vL

    error_L = (UL_vec**2 + VL_vec**2)**0.5
    errorL_rms = np.mean(error_L**2)**0.5
    errorL_max = np.max(error_L)

    print("\nRMS Error: %.3f pixels" % errorL_rms)
    print("Max Error: %.3f pixels" % errorL_max)

    UR_vec = reproj_uR - meas_uR
    VR_vec = reproj_vR - meas_vR

    error_R = (UR_vec**2 + VR_vec**2)**0.5
    errorR_rms = np.mean(error_R**2)**0.5
    errorR_max = np.max(error_R)

    print("\nRMS Error: %.3f pixels" % errorR_rms)
    print("Max Error: %.3f pixels" % errorR_max)


    Hlr = Hr @ inverse_H(Hl)
    transform_RL = H_to_euler(Hlr)
    print(transform_RL)
    transform_RC = transform_RL.copy()
    transform_RC[3:] = transform_RC[3:]/2
    
    Hcr = euler_to_H(transform_RC)
    Hcl = Hcr @ inverse_H(Hlr)
    transform_CR = H_to_euler(Hcr)
    transform_CL = H_to_euler(Hcl)

    Hmc = inverse_H(inverse_H(Hr) @ Hcr) #moument to center

    ########  Results Compile #######

    ParamOutput = np.array([['err_RMS', 'k1', 'k2', 'k3', 'k4', 'k5', 'cx', 'cy', 'Rz', 'Ry', 'Rx', 'Tx', 'Ty', 'Tz'], 
                            [(np.mean(error_L**2)**0.5), coeff_L[0], coeff_L[1], coeff_L[2], coeff_L[3], coeff_L[4], coeff_L[5], coeff_L[6], 
                            transform_CL[0], transform_CL[1], transform_CL[2], transform_CL[3], transform_CL[4], transform_CL[5]], 
                            [(np.mean(error_R**2)**0.5), coeff_R[0], coeff_R[1], coeff_R[2], coeff_R[3], coeff_R[4], coeff_R[5], coeff_R[6], 
                            transform_CR[0], transform_CR[1], transform_CR[2], transform_CR[3], transform_CR[4], transform_CR[5]] ])

    output_df = pd.DataFrame(data=ParamOutput[1:, :].T, index=ParamOutput[0, :], columns=["L Param", "R Param"])
    output_df.to_csv(args.out_file, index=True)
    print("\n", output_df)


    ########  Results Plotting #######

    fig = plt.figure()
    ax1 = fig.add_subplot()

    norm_range = Normalize(vmin=0, vmax=2.0)

    ax1.imshow(image, cmap="gray")
    ax1.scatter(meas_uL, meas_vL, s=4)
    ax1.scatter(reproj_uL, reproj_vL, s=4)
    ax1.scatter(meas_uR+int(width/2), meas_vR, s=4)
    ax1.scatter(reproj_uR+int(width/2), reproj_vR, s=4)
    ax1.scatter(rings_uL, rings_vL, s=0.2, c="red", alpha=0.2)
    ax1.scatter(rings_uR+int(width/2), rings_vR, s=0.2, c="green", alpha=0.2)
    quiver = ax1.quiver(meas_uL, meas_vL, UL_vec, VL_vec, error_L, scale=0.009, scale_units='xy', width=0.003, cmap="jet", norm=norm_range, angles="xy")
    quiver = ax1.quiver(meas_uR+int(width/2), meas_vR, UR_vec, VR_vec, error_R, scale=0.009, scale_units='xy', width=0.003, cmap="jet", norm=norm_range, angles="xy")
    ax1.set_aspect("equal")
    ax1.set_title("RMS error L: %.3f pixels\nRMS error R: %.3f pixels" % (errorL_rms, errorR_rms))
    fig.colorbar(quiver, ax=ax1)

    plt.show()

    xyzL = pv.PolyData(apply_h_transform(d3L, Hmc).T)
    xyzR = pv.PolyData(apply_h_transform(d3R, Hmc).T)

    pl = pv.Plotter()
    pl.add_mesh(xyzL, color='green', render_points_as_spheres=True, point_size=15, opacity=0.5)
    pl.add_mesh(xyzR, color='blue', render_points_as_spheres=True, point_size=10)

    scale = 100
    ax_start = np.zeros((3, 3))
    ax_end = np.zeros((3, 3))
    ax_end[0, 0] = scale
    ax_end[1, 1] = scale
    ax_end[2, 2] = scale
    
    xAx = pv.Line((ax_start[0, 0], ax_start[1, 0], ax_start[2, 0]), (ax_end[0, 0], ax_end[1, 0], ax_end[2, 0]))
    yAx = pv.Line((ax_start[0, 1], ax_start[1, 1], ax_start[2, 1]), (ax_end[0, 1], ax_end[1, 1], ax_end[2, 1]))
    zAx = pv.Line((ax_start[0, 2], ax_start[1, 2], ax_start[2, 2]), (ax_end[0, 2], ax_end[1, 2], ax_end[2, 2]))

    pl.add_mesh(xAx, color='red', render_lines_as_tubes=True, line_width=2)
    pl.add_mesh(yAx, color='green', render_lines_as_tubes=True, line_width=2)
    pl.add_mesh(zAx, color='blue', render_lines_as_tubes=True, line_width=2)

    H = Hcr
    new_start = apply_h_transform(ax_start, H)
    new_end = apply_h_transform(ax_end, H)
    xAx = pv.Line((new_start[0, 0], new_start[1, 0], new_start[2, 0]), (new_end[0, 0], new_end[1, 0], new_end[2, 0]))
    yAx = pv.Line((new_start[0, 1], new_start[1, 1], new_start[2, 1]), (new_end[0, 1], new_end[1, 1], new_end[2, 1]))
    zAx = pv.Line((new_start[0, 2], new_start[1, 2], new_start[2, 2]), (new_end[0, 2], new_end[1, 2], new_end[2, 2]))
    pl.add_mesh(xAx, color='red', render_lines_as_tubes=True, line_width=5)
    pl.add_mesh(yAx, color='green', render_lines_as_tubes=True, line_width=5)
    pl.add_mesh(zAx, color='blue', render_lines_as_tubes=True, line_width=5)

    H = Hcr @ inverse_H(Hlr)
    new_start = apply_h_transform(ax_start, H)
    new_end = apply_h_transform(ax_end, H)
    xAx = pv.Line((new_start[0, 0], new_start[1, 0], new_start[2, 0]), (new_end[0, 0], new_end[1, 0], new_end[2, 0]))
    yAx = pv.Line((new_start[0, 1], new_start[1, 1], new_start[2, 1]), (new_end[0, 1], new_end[1, 1], new_end[2, 1]))
    zAx = pv.Line((new_start[0, 2], new_start[1, 2], new_start[2, 2]), (new_end[0, 2], new_end[1, 2], new_end[2, 2]))
    pl.add_mesh(xAx, color='red', render_lines_as_tubes=True, line_width=5)
    pl.add_mesh(yAx, color='green', render_lines_as_tubes=True, line_width=5)
    pl.add_mesh(zAx, color='blue', render_lines_as_tubes=True, line_width=5)

    #H = inverse_H(Hl)
    #new_start = apply_h_transform(ax_start, H)
    #new_end = apply_h_transform(ax_end, H)
#
    #xAx = pv.Line((new_start[0, 0], new_start[1, 0], new_start[2, 0]), (new_end[0, 0], new_end[1, 0], new_end[2, 0]))
    #yAx = pv.Line((new_start[0, 1], new_start[1, 1], new_start[2, 1]), (new_end[0, 1], new_end[1, 1], new_end[2, 1]))
    #zAx = pv.Line((new_start[0, 2], new_start[1, 2], new_start[2, 2]), (new_end[0, 2], new_end[1, 2], new_end[2, 2]))
#
    #pl.add_mesh(xAx, color='red', render_lines_as_tubes=True, line_width=5)
    #pl.add_mesh(yAx, color='green', render_lines_as_tubes=True, line_width=5)
    #pl.add_mesh(zAx, color='blue', render_lines_as_tubes=True, line_width=5)
#
    #H = inverse_H(Hr) @ Hrc
    #Hcm = inverse_H(H)
    #new_start = apply_h_transform(ax_start, H)
    #new_end = apply_h_transform(ax_end, H)
#
    #xAx = pv.Line((new_start[0, 0], new_start[1, 0], new_start[2, 0]), (new_end[0, 0], new_end[1, 0], new_end[2, 0]))
    #yAx = pv.Line((new_start[0, 1], new_start[1, 1], new_start[2, 1]), (new_end[0, 1], new_end[1, 1], new_end[2, 1]))
    #zAx = pv.Line((new_start[0, 2], new_start[1, 2], new_start[2, 2]), (new_end[0, 2], new_end[1, 2], new_end[2, 2]))
#
    #pl.add_mesh(xAx, color='red', render_lines_as_tubes=True, line_width=2)
    #pl.add_mesh(yAx, color='green', render_lines_as_tubes=True, line_width=2)
    #pl.add_mesh(zAx, color='blue', render_lines_as_tubes=True, line_width=2)

    #pl.add_mesh(xAx, color='red', render_lines_as_tubes=True, line_width=5, user_matrix=inverse_H(Hrl))
    #pl.add_mesh(yAx, color='green', render_lines_as_tubes=True, line_width=5, user_matrix=inverse_H(Hrl))
    #pl.add_mesh(zAx, color='blue', render_lines_as_tubes=True, line_width=5, user_matrix=inverse_H(Hrl))

    #pl.add_mesh(xAx, color='red', render_lines_as_tubes=True, line_width=2, user_matrix=inverse_H(Hrc))
    #pl.add_mesh(yAx, color='green', render_lines_as_tubes=True, line_width=2, user_matrix=inverse_H(Hrc))
    #pl.add_mesh(zAx, color='blue', render_lines_as_tubes=True, line_width=2, user_matrix=inverse_H(Hrc))

    #pl.add_mesh(xAx, color='red', render_lines_as_tubes=True, line_width=2, user_matrix=inverse_H(Hl))
    #pl.add_mesh(yAx, color='green', render_lines_as_tubes=True, line_width=2, user_matrix=inverse_H(Hl))
    #pl.add_mesh(zAx, color='blue', render_lines_as_tubes=True, line_width=2, user_matrix=inverse_H(Hl))
    #pl.add_mesh(xAx, color='red', render_lines_as_tubes=True, line_width=2, user_matrix=inverse_H(Hr))
    #pl.add_mesh(yAx, color='green', render_lines_as_tubes=True, line_width=2, user_matrix=inverse_H(Hr))
    #pl.add_mesh(zAx, color='blue', render_lines_as_tubes=True, line_width=2, user_matrix=inverse_H(Hr))


    pl.background_color='gray'
    pl.show_axes()
    pl.show()


if __name__=="__main__":
    args = parse()
    main()

