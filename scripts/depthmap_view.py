import cv2
import sys
import argparse
import configparser
import matplotlib.pyplot as plt
import pyvista as pv
import open3d as o3d
import numpy as np
import os

def parse():
    parser = argparse.ArgumentParser('Plot Image')
    parser.add_argument('input', help='depthmap image')
    parser.add_argument('color', help='color image')
    args = parser.parse_args(None if (sys.argv[1:]) else ['-h'])
    return args

def equirectangular_depth_to_point_cloud(depth_img, color_path=None, max_depth=10.0):
    """Converts an equirectangular depth map into a 3D point cloud.

    Args:
        depth_path (str): Path to the depth map image (grayscale or raw float).
        color_path (str, optional): Path to corresponding equirectangular RGB
        image.
        max_depth (float): Maximum scale value to cap the point cloud bounds.
    """
    # 1. Load the depth map map
    # Depth Anything V2 outputs relative depth or raw metric values

    if depth_img is None:
        raise FileNotFoundError(f"Depth map not found at {depth_path}")

    # If depth is loaded as standard 8-bit or 16-bit, normalize it to float
    if depth_img.dtype == np.uint8:
        # Step A: Invert raw values so 0 is near and 255 is far
        inverted_img = 255 - depth_img
        # Step B: Normalize to a 0.0 - 1.0 baseline float
        depth = depth_img.astype(np.float32) / 255.0
    elif depth_img.dtype == np.uint16:
        # Support for high-fidelity 16-bit inverted saves
        inverted_img = 65535 - depth_img
        depth = depth_img.astype(np.float32) / 65535.0
    else:
        # Fallback for floating-point depth arrays already bounded 0-1
        depth = 1.0 - depth_img.astype(np.float32)

    # Scale depth according to max target distance
    # NOTE: If using the metric model version of Depth Anything V2,
    # skip this step and use the direct output values.
    depth = depth * max_depth

    height, width = depth.shape[:2]

    # 2. Build the spherical coordinate grid (Longitude / Latitude)
    # x corresponds to longitude [-pi, pi]
    # y corresponds to latitude [pi/2, -pi/2]
    col_indices = np.linspace(-np.pi, np.pi, width)
    row_indices = np.linspace(np.pi / 2, -np.pi / 2, height)
    lon, lat = np.meshgrid(col_indices, row_indices)

    # 3. Spherical to Cartesian Back-Projection
    # Formulas map the rays onto a sphere scaled by the calculated depth
    X = -depth * np.cos(lat) * np.sin(lon)
    Y = depth * np.sin(lat)
    Z = depth * np.cos(lat) * np.cos(lon)

    # Flatten the arrays to structure them for Open3D
    xyz = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
    print(xyz.shape)
    

    # 4. Filter out zero depth or infinitely far background points
    valid_mask = (depth.flatten() > 0.01) & (depth.flatten() < max_depth * 0.99)
    xyz = xyz[valid_mask]
    #xyz = xyz[::100, :]

    # 5. Handle Colorization (Optional)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if color_path:
        color_img = cv2.imread(color_path)
        if color_img is not None:
            # Convert BGR to RGB and normalize to [0, 1] range
            color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            colors = color_img.astype(np.float32) / 255.0
            colors = colors[:depth_img.shape[0], :depth_img.shape[1], :]
            colors = colors.reshape(-1, 3)
            colors = colors[valid_mask]
            pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

def main():
    #code
    file = args.input
    #mesh = pv.read(file)
    depthIm = plt.imread(file)

    depthIm = depthIm[:, :, 0]

    plt.figure(1)
    plt.imshow(depthIm)
    plt.show()

    pcd = equirectangular_depth_to_point_cloud(depthIm, color_path=args.color)

    o3d.visualization.draw_geometries([pcd])

    input_basename = os.path.basename(file)
    input_filename = os.path.splitext(input_basename)[0]
    output_filename = f"{input_filename}_pointCloud.pcd"

    success = o3d.io.write_point_cloud(output_filename, pcd, write_ascii=False)


    # 2. Extract points as a NumPy array
    #points = np.asarray(pcd.points)
    #point_cloud = pv.PolyData(points)
    #point_cloud["colors"] = np.asarray(pcd.colors)
#
    #scale = 0.5
    #xAx = pv.Line((0, 0, 0), (scale, 0, 0))
    #yAx = pv.Line((0, 0, 0), (0, scale, 0))
    #zAx = pv.Line((0, 0, 0), (0, 0, scale))
#
    #pl = pv.Plotter()
    #pl.add_mesh(point_cloud, rgb=False, scalars=points[:, 2], render_points_as_spheres=True, point_size=3, lighting=True, opacity=1, clim=[-2, 2])
    ##pl.add_points(np.array([0,0,0], np.float32), color="blue", render_points_as_spheres=True, point_size=20)
    #pl.add_mesh(xAx, color='red', render_lines_as_tubes=True, line_width=2)
    #pl.add_mesh(yAx, color='green', render_lines_as_tubes=True, line_width=2)
    #pl.add_mesh(zAx, color='blue', render_lines_as_tubes=True, line_width=2)
    #pl.background_color='white'
    #pl.show_axes()
    #pl.show()


if __name__=="__main__":
    args = parse()
    main()