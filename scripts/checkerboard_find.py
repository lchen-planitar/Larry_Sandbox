import cv2
import sys
import argparse
import configparser
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from skspatial.objects import Line
import pyvista as pv

from tools import apply_h_transform
from tools import inverse_H
from tools import H_to_euler

focal_length = 8 #mm
pixel_size = 0.005 #mm
grid_size = 50 #mm

def parse():
    parser = argparse.ArgumentParser('Plot Image')
    parser.add_argument('input', help='File containing camera poses')
    parser.add_argument('col', type=int, help='how many internal corners, horizontal')
    parser.add_argument('row', type=int, help='how many internal corners, vertical')
    args = parser.parse_args(None if (sys.argv[1:]) else ['-h'])
    return args

def interpolate_corners(raw_corners, col, row):
    TL = np.array([raw_corners[0, 0, 0], raw_corners[0, 0, 1]])
    TR = np.array([raw_corners[col-1, 0, 0], raw_corners[col-1, 0, 1]])
    BL = np.array([raw_corners[(row-1)*col, 0, 0], raw_corners[(row-1)*col, 0, 1]]) 
    BR = np.array([raw_corners[row*col-1, 0, 0], raw_corners[row*col-1, 0, 1]])

    int_corners = raw_corners.copy()

    for i in range (0, col):
        for j in range (0, row):
            element = j*col+i
            top = TL + i*( (TR-TL)/(col-1) )
            bottom = BL + i*( (BR-BL)/(col-1) )
            left = TL + j*( (BL-TL)/(row-1) )
            right = TR + j*( (BR-TR)/(row-1) )

            top = raw_corners[i, 0, :]
            bottom = raw_corners[(row-1)*col+i, 0, :]
            left = raw_corners[(j*col), 0, :]
            right = raw_corners[((j+1)*col-1), 0, :]

            line_a = Line.from_points(top, bottom)
            line_b = Line.from_points(left, right)
            intersect = line_a.intersect_line(line_b)
            
            int_corners[element, 0, 0] = intersect[0]
            int_corners[element, 0, 1] = intersect[1]


    return int_corners

def main():
    #code
    file = args.input
    row_num = args.row
    col_num = args.col
    print("row: %d, col: %d" % (row_num, col_num))
    np_image = plt.imread(file)
    image = cv2.imread(file)
    print("image size: %dx%d" %(np_image.shape[1], np_image.shape[0]))
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    fx = focal_length / pixel_size
    fy = focal_length / pixel_size
    cx = float(np_image.shape[1]/2)
    cy = float(np_image.shape[0]/2)
 
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (col_num,row_num), flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                               cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    plt.figure(1)
    plt.imshow(np_image)

    if ret == True:
        cv2.drawChessboardCorners(image, (col_num,row_num), corners, ret)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (col_num,row_num), (-1,-1), criteria)
        obj_points = np.zeros((corners.shape[0], 3))
        img_points = np.zeros((corners.shape[0], 2))

        element_count = 0
        for j in range(0, row_num):
            for i in range(0, col_num):
                obj_points[element_count, 0] = i*grid_size
                obj_points[element_count, 1] = j*grid_size
                element_count += 1
        img_points[:, :] = corners[:, 0, :]

        #print(obj_points)
        #print(img_points)

        cam_mtx = np.array([[fx, 0, cx,], [0, fy, cy], [0, 0, 1]], np.float32)
        dist_nom = np.array([0, 0, 0, 0, 0])

        success, rvec, tvec = cv2.solvePnP(obj_points, img_points, cam_mtx, dist_nom)
        M,_ = cv2.Rodrigues(rvec)

        H = np.zeros((4,4))
        H[:3, :3] = M
        H[:3, 3] = tvec[:, 0]
        H[3, 3] = 1

        print("Rz, Ry, Rx, Tx, Ty, Tz")
        print(H_to_euler(H))

        corners3D = apply_h_transform(obj_points.T, H)
        corners3D = corners3D.T

        cornersReproj,_ = cv2.projectPoints(obj_points, rvec, tvec, cam_mtx, dist_nom)

        #print(cornersReproj.shape)

        #print(corners)
        TL = 0
        TR = col_num-1
        BL = (row_num-1)*col_num
        BR = row_num*col_num-1

        int_corners = interpolate_corners(corners, col_num, row_num)

        delta = ( (corners[:, 0, 0]-cornersReproj[:, 0, 0])**2 + (corners[:, 0, 1]-cornersReproj[:, 0, 1])**2 )**0.5
        reprojErr_RMS = (np.sum(delta**2)/delta.shape[0])**0.5
        reprojErr_max = np.max(delta)
        #print(delta)

        for i in range(0, col_num):
            x = np.array([cornersReproj[TL+i, 0, 0], cornersReproj[BL+i, 0, 0]])
            y = np.array([cornersReproj[TL+i, 0, 1], cornersReproj[BL+i, 0, 1]])
            plt.plot(x, y, color='yellow')

        for j in range(0, row_num):
            x = np.array([cornersReproj[TL+j*col_num, 0, 0], cornersReproj[TL+(j+1)*col_num-1, 0, 0]])
            y = np.array([cornersReproj[TL+j*col_num, 0, 1], cornersReproj[TL+(j+1)*col_num-1, 0, 1]])
            plt.plot(x, y, color='yellow')

        for i in range(0, corners.shape[0]):
            x = np.array([corners[i, 0, 0], cornersReproj[i, 0, 0]])
            y = np.array([corners[i, 0, 1], cornersReproj[i, 0, 1]])
            dist = ( (x[1]-x[0])**2 + (y[1]-y[0])**2 )**0.5
            plt.plot(x, y, color='blue', lw=1)

        plt.scatter(corners[:, 0, 0], corners[:, 0, 1], c=delta, cmap='jet', s=75, alpha=0.8, zorder=10)
        plt.title("Reprojection Error, RMS: %.3fpx, Max:%.3fpx" % (reprojErr_RMS, reprojErr_max))
        plt.colorbar()
    
    plt.show()

    pl = pv.Plotter()
    pl.add_points(corners3D, color='green', render_points_as_spheres=True, point_size=25)
    pl.add_points(obj_points, color='red', render_points_as_spheres=True, point_size=25)
    pl.add_points(np.array([0,0,0], np.float32), color="blue", render_points_as_spheres=True, point_size=45)
    pl.background_color='gray'
    pl.show_axes()
    #pl.show()


if __name__=="__main__":
    args = parse()
    main()

