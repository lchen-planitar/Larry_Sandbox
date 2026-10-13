import cv2
import sys
import argparse
import configparser
import matplotlib.pyplot as plt
import rawpy
import numpy as np
import imageio
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

    x = np.linspace(0, 1, 65535)
    y1 = hable_tonemap(x)
    y2 = sigmoid_correct(x)
    y = y1 * y2

    fig1, ax1 = plt.subplots()
    #ax1.imshow(image)
    line, = ax1.plot(x, y)
    fig1.subplots_adjust(bottom=0.4)

    def update(val):
        y1 = hable_tonemap(x, A=A_slider.val, B=B_slider.val, C=C_slider.val, D=D_slider.val, E=E_slider.val, F=F_slider.val, W=W_slider.val, exp=exp_slider.val)
        y2 = sigmoid_correct(x)
        y = y1 * y2
        line.set_ydata(y)
        fig1.canvas.draw_idle()

    A_ax = fig1.add_axes([0.2, 0.35-(1*0.04), 0.65, 0.03])
    A_slider = Slider(
        ax=A_ax,
        label='A',
        valmin=0,
        valmax=1,
        valstep=0.05,
        valinit=0.15,
        )

    B_ax = fig1.add_axes([0.2, 0.35-(2*0.04), 0.65, 0.03])
    B_slider = Slider(
        ax=B_ax,
        label='B',
        valmin=0,
        valmax=1,
        valstep=0.05,
        valinit=0.5,
        )
    
    C_ax = fig1.add_axes([0.2, 0.35-(3*0.04), 0.65, 0.03])
    C_slider = Slider(
        ax=C_ax,
        label='C',
        valmin=0,
        valmax=10,
        valstep=0.1,
        valinit=0.1,
        )
    
    D_ax = fig1.add_axes([0.2, 0.35-(4*0.04), 0.65, 0.03])
    D_slider = Slider(
        ax=D_ax,
        label='D',
        valmin=0,
        valmax=1,
        valstep=0.05,
        valinit=0.2,
        )
    
    E_ax = fig1.add_axes([0.2, 0.35-(5*0.04), 0.65, 0.03])
    E_slider = Slider(
        ax=E_ax,
        label='E',
        valmin=0,
        valmax=1,
        valstep=0.02,
        valinit=0.02,
        )

    F_ax = fig1.add_axes([0.2, 0.35-(6*0.04), 0.65, 0.03])
    F_slider = Slider(
        ax=F_ax,
        label='F',
        valmin=0,
        valmax=2,
        valstep=0.05,
        valinit=0.3,
        )

    W_ax = fig1.add_axes([0.2, 0.35-(7*0.04), 0.65, 0.03])
    W_slider = Slider(
        ax=W_ax,
        label='W',
        valmin=0.1,
        valmax=2,
        valstep=0.05,
        valinit=1.5,
        )

    exp_ax = fig1.add_axes([0.2, 0.35-(8*0.04), 0.65, 0.03])
    exp_slider = Slider(
        ax=exp_ax,
        label='exp',
        valmin=0,
        valmax=30,
        valstep=0.1,
        valinit=10,
        )

    A_slider.on_changed(update)
    B_slider.on_changed(update)
    C_slider.on_changed(update)
    D_slider.on_changed(update)
    E_slider.on_changed(update)
    F_slider.on_changed(update)
    W_slider.on_changed(update)
    exp_slider.on_changed(update)

    plt.show()


if __name__=="__main__":
    #args = parse()
    main()