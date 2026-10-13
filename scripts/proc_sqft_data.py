import sys
import argparse
import configparser
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

def parse():
    parser = argparse.ArgumentParser('Graph sqft data')
    parser.add_argument('input', help='File containing sqft data')
    args = parser.parse_args(None if (sys.argv[1:]) else ['-h'])
    return args

def main():
    file = args.input
    df = pd.read_csv(file)
    date = df.iloc[:, 0].to_numpy().astype('datetime64')
    area = df.iloc[:, 1].to_numpy().astype('int')
    #print(area)
    filter1 = area < 12000
    date_filtered = date[filter1]
    area_filtered = area[filter1]
    area_sort = np.sort(area_filtered)
    percent = np.linspace(0, 100, num=len(area_filtered))
    fig1, ax = plt.subplots(2, 1)

    for i in range(2016, 2026):
        date1 = f"{i}-01-01"
        date2 = f"{i+1}-01-01"
        year = f"{i}"
        #print(date1, date2)

        f_yr_1 = date_filtered > np.datetime64(date1)
        date_sub = date_filtered[f_yr_1]
        area_sub = area_filtered[f_yr_1]

        f_yr_2 = date_sub < np.datetime64(date2)
        date_sub = date_sub[f_yr_2]
        area_sub = area_sub[f_yr_2]

        values, bins = np.histogram(area_sub, bins=30)
        bin_out = np.zeros(len(bins)-1)
        #print(len(bins))
        for j in range (0, len(bins)-1):
            bin_out[j] = (bins[j] + bins[j+1])/2
        
        ax[0].plot(bin_out, values, label=year)
            
    #print(values, bin_out)
    ax[1].set_xlabel("Square Footage")
    ax[0].set_ylabel("Number of Properties")
    ax[0].set_xlim(0, 12000)
    ax[0].set_title("Scanned Properties Breakdown by Square Footage")
    ax[0].legend()
    ax[1].plot(area_sort, percent)
    ax[1].set_ylabel("Percentage of Properties")
    ax[1].set_xlim(0, 12000)
    plt.grid(alpha=0.5)
    plt.show()

if __name__=="__main__":
    args = parse()
    main()