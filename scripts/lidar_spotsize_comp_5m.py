import cv2
import sys
import argparse
import configparser
import matplotlib.pyplot as plt
import pyvista as pv
import numpy as np

def ellipse(a, b):
    # Define parameters
    h, k = 0, 0        # Center coordinates

    # Generate angles from 0 to 2*pi
    t = np.linspace(0, 2 * np.pi, 100)

    # Parametric equations for the ellipse
    x = h + a * np.cos(t)
    y = k + b * np.sin(t)
    return x, y

def parse():
    parser = argparse.ArgumentParser('Plot Image')
    parser.add_argument('input', help='ply mesh file')
    args = parser.parse_args(None if (sys.argv[1:]) else ['-h'])
    return args

def main():
    #code

    plt.figure(1)
    #JT128
    a, b = 92.8, 94.5
    x, y = ellipse(a/2, b/2)
    plt.plot(x, y, label="JT128")

    #JT64P
    a, b = 45.6, 56.1
    x, y = ellipse(a/2, b/2)
    plt.plot(x, y, label="JT64P")

    #XT32
    a, b = 28.9, 27.6
    x, y = ellipse(a/2, b/2)
    plt.plot(x, y, label="XT32")

    #M300
    a, b = 12.5, 42.0
    x, y = ellipse(a/2, b/2)
    plt.plot(x, y, label="Mid360/M300")

    #OSDome
    a, b = 35.5, 35.5
    x, y = ellipse(a/2, b/2)
    plt.plot(x, y, label="OSDome")

    #Faro
    a, b = 4.0, 4.0
    x, y = ellipse(a/2, b/2)
    plt.plot(x, y, label="Faro")

    plt.legend()
    plt.xlabel("Horizontal Spot Size @5m (mm)")
    plt.ylabel("Vertical Spot Size @5m (mm)")
    plt.axis("equal")
    #plt.xlim(-100, 100)
    plt.ylim(-100, 100)
    plt.show()
    


if __name__=="__main__":
    #args = parse()
    main()