import cv2
import sys
import argparse
import configparser
import matplotlib.pyplot as plt
import rawpy
import numpy as np
import imageio
import math

'''
def parse():
    parser = argparse.ArgumentParser('Plot Image')
    parser.add_argument('input', help='File containing camera poses')
    args = parser.parse_args(None if (sys.argv[1:]) else ['-h'])
    return args
'''
def main():
    #code
    dz = 10 #mm
    r_min = 30 #mm
    r_max = 1000 #mm
    sim_steps = 100
    theta = np.linspace(0.1, 90, sim_steps)
    r = np.linspace(r_min, r_max, sim_steps)
    d_theta = np.zeros((int(sim_steps*sim_steps), 3))

    count = 0
    for i in range(0, sim_steps):
        print(i)
        for j in range(0, sim_steps):
            #print(j)
            d_theta[count, 0] = theta[i]
            d_theta[count, 1] = r[j]
            b = (dz**2 + r[j]**2 - 2*dz*r[j]*math.cos(theta[i]*math.pi/180))**0.5
            d_theta[count, 2] = math.acos((b**2 + r[j]**2 - dz**2)/(2*b*r[j]))*180/math.pi
            count += 1


    # Display using Matplotlib
    #plt.figure(1)
    fig = plt.figure(1)
    ax = fig.add_subplot(projection='3d')
    #output = ax.scatter(d_theta[:, 0], d_theta[:, 1], d_theta[:, 2], c=d_theta[:, 2], s=2, cmap='jet')
    output = ax.plot_trisurf(d_theta[:, 0], d_theta[:, 1], d_theta[:, 2], cmap='jet', edgecolor='none')
    fig.colorbar(output, ax=ax)
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    #plt.imshow(raw_im[::2, 1::2], cmap='gray', vmin=0, vmax=20000)
    #ax.imshow(rgb)
    #ax.add_patch(circle1)
    #ax.add_patch(circle2)
    plt.show()
    #imageio.imsave('output.jpg', rgb, quality=100)


if __name__=="__main__":
    #args = parse()
    main()