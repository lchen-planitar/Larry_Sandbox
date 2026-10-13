import sys
import argparse
import configparser
import numpy as np
import pandas as pd
import pyvista as pv
import math
from scipy.spatial.transform import Rotation

# Global Variable Definition
uLim = 1920
vLim = 1080
pixSize = 0.005
#sensor_width = 13.2
#sensor_height = 8.8
fl = 8
maxZ = 8000
frame_width = 6096
frame_depth = 6096
frame_height = 3048


def euler_to_H(input):
    #input has shape np.array([Rz, Ry, Rx, Tx, Ty, Tz])
    angles = input[:3]
    translation = input[3:]
    r = Rotation.from_euler("ZYX", angles, degrees=True)
    M = r.as_matrix()
    H = np.zeros((4, 4))
    H[:3, :3] = M
    H[:3, 3] = translation
    H[3, 3] = 1
    return H


def parse():
    parser = argparse.ArgumentParser('Compares 6D from 2 tools')
    parser.add_argument('input', help='File containing camera poses')
    args = parser.parse_args(None if (sys.argv[1:]) else ['-h'])
    return args


def main():
    #code
    file = args.input
    df = pd.read_csv(file)
    cam_count = df.shape[0]
    print("\ntotal number of cameras detected: %d\n" % cam_count)
    print(df)

    xMax = uLim*pixSize*maxZ/fl
    yMax = vLim*pixSize*maxZ/fl
    #xMax = sensor_width*maxZ/fl
    #yMax = sensor_height*maxZ/fl

    pointa = [xMax/2, yMax/2, maxZ]
    pointb = [-xMax/2, yMax/2, maxZ]
    pointc = [-xMax/2, -yMax/2, maxZ]
    pointd = [xMax/2, -yMax/2, maxZ]
    pointe = [0, 0, 0]
    pyramid = pv.Pyramid([pointa, pointb, pointc, pointd, pointe])

    line1 = pv.Line((0, 0, 0), (0, 0, frame_height))
    line2 = pv.Line((0, 0, frame_height), (frame_width, 0, frame_height))
    line3 = pv.Line((0, 0, frame_height), (0, frame_depth, frame_height))
    line4 = pv.Line((frame_width, 0, 0), (frame_width, 0, frame_height))
    line5 = pv.Line((frame_width, 0, frame_height), (frame_width, frame_depth, frame_height))
    line6 = pv.Line((0, frame_depth, 0), (0, frame_depth, frame_height))
    line7 = pv.Line((frame_width, frame_depth, 0), (frame_width, frame_depth, frame_height))
    line8 = pv.Line((0, frame_depth, frame_height), (frame_width, frame_depth, frame_height))

    pl1 = pv.Plotter()
    pl1.background_color='gray'
    pl1.add_mesh(line1, line_width=5, color="red", render_lines_as_tubes=True)
    pl1.add_mesh(line2, line_width=5, color="red", render_lines_as_tubes=True)
    pl1.add_mesh(line3, line_width=5, color="red", render_lines_as_tubes=True)
    pl1.add_mesh(line4, line_width=5, color="red", render_lines_as_tubes=True)
    pl1.add_mesh(line5, line_width=5, color="red", render_lines_as_tubes=True)
    pl1.add_mesh(line6, line_width=5, color="red", render_lines_as_tubes=True)
    pl1.add_mesh(line7, line_width=5, color="red", render_lines_as_tubes=True)
    pl1.add_mesh(line8, line_width=5, color="red", render_lines_as_tubes=True)

    for i in range(0, cam_count):
        pose = df.iloc[i, :].to_numpy()
        H = euler_to_H(pose)
        #print(H)
        trans_pyramid = pyramid.transform(H, inplace=False)
        pl1.add_mesh(trans_pyramid.clip(normal=(0,0,-1), origin=(0,0,0)), opacity=0.5, show_edges=True)
    
    pl1.show_axes()
    pl1.show()


if __name__=="__main__":
    args = parse()
    main()