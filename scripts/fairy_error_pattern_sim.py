import cv2
import sys
import argparse
import configparser
import matplotlib.pyplot as plt
import rawpy
import numpy as np
import imageio
import math
from matplotlib.widgets import Button, Slider


def parse():
    parser = argparse.ArgumentParser('Plot Image')
    parser.add_argument('input', help='File containing camera poses')
    args = parser.parse_args(None if (sys.argv[1:]) else ['-h'])
    return args


def hable_tonemap(x, A=0.15, B=0.5, C=0.1, D=0.2, E=0.02, F=0.3, W=1.5, exp=10):
    # Constants for John Hable's Uncharted 2 curve
    #A = 0.15 #0.15 # Shoulder Strength
    #B = 0.5 #0.50 # Linear Strength
    #C = 4 #0.10 # Linear Angle
    #D = 0.5 #0.20 # Toe Strength
    #E = 0.02 #0.02 # Toe Numerator
    #F = 0.8 #0.30 # Toe Denominator
    #W = 1.5 #11.2 # Linear White Point Value

    def curve(x):
        return ((x * (A * x + C * B) + D * E) / (x * (A * x + B * C) + D * F)) - E / F

    # Apply the curve and normalize by the white point
    exposure_bias = exp
    curr = curve(exposure_bias * x)
    white_scale = 1.0 / curve(W)
    result = curr * white_scale
    
    return result

def sigmoid_correct(x, A=0.3, B=10, C=-0.9, D=0.7):
    return ( A / (1 + np.exp(-B* (x+C))) ) + D

def main():
    #code

    dist = 1000
    x = np.linspace(-5000, 5000, 10000)
    y = x.copy()
    y[:] = dist

    r = (x**2 + y**2)**0.5
    theta = np.asin(x/r)

    error = 0.0
    error_rad = error * math.pi / 180

    theta_err = theta + error_rad
    x_err = np.sin(theta_err)*r
    y_err = np.cos(theta_err)*r


    fig1, ax1 = plt.subplots()
    #ax1.imshow(image)
    line, = ax1.plot(x, y)
    line_err, =ax1.plot(x_err, y_err)
    fig1.subplots_adjust(bottom=0.35)

    def update(val):
        dist = A_slider.val
        x = np.linspace(-5000, 5000, 10000)
        y = x.copy()
        y[:] = dist

        r = (x**2 + y**2)**0.5
        theta = np.asin(x/r)

        error = B_slider.val
        error_rad = error * math.pi / 180

        theta_err = theta + error_rad
        x_err = np.sin(theta_err)*r
        y_err = np.cos(theta_err)*r

        line_err.set_xdata(x_err)
        line_err.set_ydata(y_err)
        fig1.canvas.draw_idle()

    A_ax = fig1.add_axes([0.2, 0.25, 0.65, 0.03])
    A_slider = Slider(
        ax=A_ax,
        label='A',
        valmin=0,
        valmax=5000,
        valstep=1,
        valinit=1000,
        )

    B_ax = fig1.add_axes([0.2, 0.15, 0.65, 0.03])
    B_slider = Slider(
        ax=B_ax,
        label='B',
        valmin=-1,
        valmax=1,
        valstep=0.001,
        valinit=0,
        )
 
    A_slider.on_changed(update)
    B_slider.on_changed(update)

    ax1.set_ylim(900, 1100)
    plt.show()


if __name__=="__main__":
    #args = parse()
    main()