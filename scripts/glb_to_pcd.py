import cv2
import sys
import argparse
import configparser
import matplotlib.pyplot as plt
import pyvista as pv
import open3d as o3d
import numpy as np
import trimesh

def parse():
    parser = argparse.ArgumentParser('Plot Image')
    parser.add_argument('input', help='ply mesh file')
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

    scene = trimesh.load(file)

    # Check if the file features point primitives rather than face geometry
    if hasattr(scene, 'geometry') and len(scene.geometry) > 0:
        for name, geom in scene.geometry.items():
            # Point clouds wrapped in glTF often sit inside PointCloud geometries
            if isinstance(geom, trimesh.PointCloud):
                points = geom.vertices
                colors = geom.colors  # Extracts RGB values if present
                print(f"Found pure point cloud '{name}' with {len(points)} nodes.")

    convert_glb_primitives_to_pcd(file, outFile)


if __name__=="__main__":
    args = parse()
    main()