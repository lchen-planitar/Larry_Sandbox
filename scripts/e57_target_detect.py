import cv2
import sys
import argparse
import configparser
import matplotlib.pyplot as plt
import pyvista as pv
import open3d as o3d
import numpy as np
import trimesh
import pye57
import matplotlib.cm as cm

def parse():
    parser = argparse.ArgumentParser('Plot Image')
    parser.add_argument('input', help='e57 file')
    #parser.add_argument('output', help='output file')
    args = parser.parse_args(None if (sys.argv[1:]) else ['-h'])
    return args

def main():
    #code
    file = args.input
    #outFile = args.output
    #mesh = pv.read(file)

    e57_file = pye57.E57(file)

    # 2. Extract data (Assuming the first scan in the file)
    # E57 files store scans separately; use read_scan_raw to get cartesian points
    scan_index = 0
    scan_data = e57_file.read_scan_raw(scan_index)

    # 3. Concatenate and reshape the coordinates
    x = np.array(scan_data['cartesianX'])
    y = np.array(scan_data['cartesianY'])
    z = np.array(scan_data['cartesianZ'])
    intensity = np.asarray(scan_data['intensity'], dtype=np.float64)

    # Stack coordinates into an (N x 3) numpy array and filter out invalid/NaN points
    points = np.column_stack((x, y, z))
    #points = points[~np.isnan(points).any(axis=1)]

    # Filter invalid points with ONE mask so points and intensity stay aligned
    valid = ~np.isnan(points).any(axis=1) & ~np.isnan(intensity)
    if 'cartesianInvalidState' in scan_data:          # 0 = valid in E57
        valid &= np.asarray(scan_data['cartesianInvalidState']) == 0
    points = points[valid]
    intensity = intensity[valid]

    

    # Normalise intensity to 0..1. Clip to the 1st-99th percentile so a few very
    # bright returns (retro-reflectors, specular hits) don't wash out the colours.
    lo, hi = np.percentile(intensity, [0, 100])
    normalized_i = np.clip((intensity - lo) / max(hi - lo, 1e-12), 0.0, 1.0)

    # Grayscale (looks like a typical scanner intensity view)...
    #colors = np.repeat(normalized_i[:, None], 3, axis=1)
    # ...or a colormap instead:
    colors = cm.jet(normalized_i)[:, :3]

    # 4. Convert to Open3D PointCloud format
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    new_pcd = pcd.voxel_down_sample(voxel_size=0.02)

    # Optional: Downsample if the point cloud is too dense for smooth viewing
    # pcd_down = pcd.voxel_down_sample(voxel_size=0.01)

    # 5. Visualize
    print("Displaying point cloud...")
    o3d.visualization.draw_geometries([new_pcd])
    
    
    z_coords = points[:, 2]
    z_min, z_max = z_coords.min(), z_coords.max()
    normalized_z = (z_coords - z_min) / (z_max - z_min)

    # Apply a Matplotlib colormap (e.g., 'jet', 'viridis', or 'plasma')
    # cm.jet(normalized_z) returns an (N, 4) array containing RGBA; we strip the Alpha channel
    colors = cm.jet(normalized_z)[:, :3]

    plt.figure(1)
    plt.scatter(points[:, 0], points[:, 1], c=colors)
    plt.show()
    

    #success = o3d.io.write_point_cloud(outFile, new_pcd, write_ascii=False)
    
    #if success:
    #    print(f"Successfully saved points to '{outFile}'")
    #else:
    #    print("Failed to write PCD file.")


if __name__=="__main__":
    args = parse()
    main()