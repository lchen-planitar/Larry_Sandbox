import pandas as pd
import numpy as np
import sys
import argparse
import configparser
import matplotlib.pyplot as plt

def parse():
    parser = argparse.ArgumentParser('Plot Image')
    parser.add_argument('uv_file', help='uv file')
    parser.add_argument('d3_file', help='3d file')
    parser.add_argument('out_file', help='output file')
    args = parser.parse_args(None if (sys.argv[1:]) else ['-h'])
    return args

def main():
    #code
    uv_file = args.uv_file
    d3_file = args.d3_file

    df_uv = pd.read_csv(uv_file)
    df_3d = pd.read_csv(d3_file)

    out_matched = np.zeros((1, 6))

    for i in range(0, df_uv.shape[0]):
        id = df_uv.iloc[i, 0]
        print(id)
        for j in range(0, df_3d.shape[0]):
            if df_3d.iloc[j, 0] == id:
                temp = np.array([id, df_uv.iloc[i, 2], df_uv.iloc[i, 3], df_3d.iloc[j, 3], df_3d.iloc[j, 4], df_3d.iloc[j, 5]])
                if (not np.isnan(temp).any()):
                    out_matched = np.append(out_matched, [temp], axis=0)

    df_out = pd.DataFrame(data=out_matched[1:, :], columns=["ID", "U", "V", "X", "Y", "Z"])
    df_out.to_csv(args.out_file, index=False)
    print(df_out)

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    ax1.scatter(df_out.iloc[:, 1], df_out.iloc[:, 2])
    ax2.scatter(df_out.iloc[:, 3], df_out.iloc[:, 4], df_out.iloc[:, 5])

    ax1.set_aspect("equal")
    ax2.set_aspect("equal")

    plt.show()



if __name__=="__main__":
    args = parse()
    main()