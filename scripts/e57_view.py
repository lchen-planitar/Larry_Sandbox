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
    parser.add_argument('output', help='output file')
    args = parser.parse_args(None if (sys.argv[1:]) else ['-h'])
    return args

def convert_glb_primitives_to_pcd(glb_path, pcd_output_path):
    # 1. Load the GLB using trimesh
    scene = trimesh.load(glb_path)
    
    all_points = []
    all_colors = []

    # 2. Method 3 Check: Parse the scene components for explicit PointCloud geometries
    if hasattr(scene, 'geometry') and len(scene.geometry) > 0:
        for name, geom in scene.geometry.items():
            if isinstance(geom, trimesh.PointCloud):
                print(f"Extracting point cloud component: '{name}' with {len(geom.vertices)} points.")
                all_points.append(np.asarray(geom.vertices))
                
                # Check for per-point colors (glTF stores colors as RGBA or RGB normalized 0-1 or 0-255)
                if hasattr(geom, 'colors') and geom.colors is not None:
                    # Convert to standard float array (0.0 to 1.0) for Open3D compatibility
                    raw_colors = np.asarray(geom.colors)[:, :3]  # Drop Alpha channel if present
                    if raw_colors.dtype == np.uint8:
                        raw_colors = raw_colors / 255.0
                    all_colors.append(raw_colors)
                    
    # Fallback: If the GLB was loaded as a standard mesh instead of trimesh.PointCloud
    if len(all_points) == 0:
        print("No explicit PointCloud primitive found. Falling back to mesh vertices extraction.")
        mesh = scene.dump(concatenate=True) if isinstance(scene, trimesh.Scene) else scene
        all_points.append(np.asarray(mesh.vertices))

    # Combine all extracted components
    final_points = np.vstack(all_points)
    
    # 3. Use Open3D to generate the PCD file structures
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(final_points)
    
    # Map colors if they exist in the primitive structure
    if len(all_colors) > 0:
        final_colors = np.vstack(all_colors)
        o3d_pcd.colors = o3d.utility.Vector3dVector(final_colors)

    o3d.visualization.draw_geometries([o3d_pcd])
    # 4. Save to disk as PCD format
    success = o3d.io.write_point_cloud(pcd_output_path, o3d_pcd, write_ascii=False)
    
    if success:
        print(f"Successfully saved {len(final_points)} points to '{pcd_output_path}'")
    else:
        print("Failed to write PCD file.")

def main():
    #code
    file = args.input
    outFile = args.output
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

    """
    z_coords = points[:, 2]
    z_min, z_max = z_coords.min(), z_coords.max()
    normalized_z = (z_coords - z_min) / (z_max - z_min)

    # Apply a Matplotlib colormap (e.g., 'jet', 'viridis', or 'plasma')
    # cm.jet(normalized_z) returns an (N, 4) array containing RGBA; we strip the Alpha channel
    colors = cm.jet(normalized_z)[:, :3]
    """

    # Normalise intensity to 0..1. Clip to the 1st-99th percentile so a few very
    # bright returns (retro-reflectors, specular hits) don't wash out the colours.
    lo, hi = np.percentile(intensity, [0, 100])
    normalized_i = np.clip((intensity - lo) / max(hi - lo, 1e-12), 0.0, 1.0)

    # Grayscale (looks like a typical scanner intensity view)...
    #colors = np.repeat(normalized_i[:, None], 3, axis=1)
    # ...or a colormap instead:
    colors = cm.jet(normalized_i)[:, :3]

    print(points)

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

    success = o3d.io.write_point_cloud(outFile, new_pcd, write_ascii=False)
    
    if success:
        print(f"Successfully saved points to '{outFile}'")
    else:
        print("Failed to write PCD file.")


if __name__=="__main__":
    args = parse()
    main()