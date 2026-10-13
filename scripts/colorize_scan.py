#!/usr/bin/env python3
"""
colorize_scan.py

Colorizes a point cloud by projecting it onto an equirectangular camera image
and sampling pixel colours. Requires a calibration result to align the point
cloud with the image.

Usage:
    python colorize_scan.py --pcd <path> --image <path> --calibration <path>

    --pcd         : path to the .pcd file
    --image       : path to the equirectangular image
    --calibration : path to calibration.yaml from the calibration pipeline
                    (default: most recent result in results_dir)
"""
import os
import sys
import argparse
import numpy as np
import cv2
import open3d as o3d
import yaml
import pyvista as pv

from tools import apply_h_transform
from tools import inverse_H
from tools import H_to_euler
from tools import euler_to_H

def load_result(path):
    """Load a calibration result YAML. Returns (R, t, meta_dict)."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    R = np.array(data["rotation"])
    t = np.array(data["translation"])
    return R, t, data

# ── Projection ────────────────────────────────────────────────────────────────

def project_pcl_to_equirect(points, width, height, R=np.eye(3), t=np.zeros(3)):
    """
    Project point cloud points onto an equirectangular image plane.

    Parameters
    ----------
    points : (N, 3) array of 3D points
    width  : image width in pixels
    height : image height in pixels
    R      : (3, 3) rotation matrix (default: identity)
    t      : (3,) translation vector (default: zero)

    Returns
    -------
    (N, 2) array of (u, v) pixel coordinates
    """
    pts_cam = (R @ points.T).T + t
    x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    r       = np.linalg.norm(pts_cam, axis=1)
    lon     = np.arctan2(x, y)
    lat     = np.arcsin(np.clip(z / r, -1.0, 1.0))
    u       = ((lon + np.pi) / (2 * np.pi) * width) % width
    v       = (0.5 - lat / np.pi) * height
    return np.column_stack((u, v))

# ── Colorize ──────────────────────────────────────────────────────────────────

def colorize_scan(pcd_path, image_path, R, t):
    """
    Colorize a point cloud by sampling colours from an equirectangular image.

    Parameters
    ----------
    pcd_path   : path to the .pcd file
    image_path : path to the equirectangular image
    R          : (3, 3) rotation matrix from calibration
    t          : (3,) translation vector from calibration

    Returns
    -------
    colorized Open3D PointCloud
    """
    pcd = o3d.io.read_point_cloud(pcd_path)
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: could not load image {image_path}")
        sys.exit(1)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    points = np.asarray(pcd.points)
    uv     = project_pcl_to_equirect(points, w, h, R, t)

    pixel_x = np.clip(uv[:, 0].astype(int), 0, w - 1)
    pixel_y = np.clip(uv[:, 1].astype(int), 0, h - 1)

    colors      = img[pixel_y, pixel_x] / 255.0
    pcd.colors  = o3d.utility.Vector3dVector(colors)

    return pcd


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcd",         required=True, help="Path to .pcd file")
    parser.add_argument("--image",       required=True, help="Path to equirectangular image")
    parser.add_argument("--calibration", required=True, help="Path to calibration.yaml or run directory")
    parser.add_argument('--mesh', action='store_true', help='Mesh point cloud')
    args = parser.parse_args()

    cal_path = (os.path.join(args.calibration, "calibration.yaml")
                if os.path.isdir(args.calibration) else args.calibration)

    if not os.path.exists(cal_path):
        print(f"Calibration file not found: {cal_path}")
        sys.exit(1)

    R, t, _ = load_result(cal_path)
    result  = colorize_scan(args.pcd, args.image, R, t)
    points = np.asarray(result.points)
    colors = np.asarray(result.colors)
    print(points.shape)

    if (args.mesh):
        
        print("estimating normals")
        result.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=15))
        result.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
        print("normals complete")

        print("starting meshing")
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(result, depth=12)
        vertices_to_remove = densities < np.quantile(densities, 0.01)
        mesh.remove_vertices_by_mask(vertices_to_remove)

        #radii = [0.05]
        #mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        #                                        result, o3d.utility.DoubleVector(radii))
        print("meshing complete")

        o3d.visualization.draw_geometries([mesh], mesh_show_back_face=False)
        o3d.io.write_triangle_mesh("output_mesh.ply", mesh)

        #o3d.visualization.draw_geometries([result])
        #o3d.visualization.draw([result], bg_color=[0,0,0,0], show_skybox=False, raw_mode=True, point_size=8)
        
    cloud = pv.PolyData(points)
    #print(points[2000, :])
    cloud["colors"] = colors
    scale = 0.5
    xAx = pv.Line((0, 0, 0), (scale, 0, 0))
    yAx = pv.Line((0, 0, 0), (0, scale, 0))
    zAx = pv.Line((0, 0, 0), (0, 0, scale))

    #surf = cloud.delaunay_3d(alpha=0.01)

    points2 = points.copy()
    transform = [0, -25, 0, 0, 0, 0] #Rz, Ry, Rx, Tx, Ty, Tz
    H = euler_to_H(transform)
    points2 = apply_h_transform(points2.T, H).T
    filt = points2 < 0.8
    points_flat = points2[filt[:, 2]]
    filt2 = points_flat > -0.8
    points_flat = points_flat[filt2[:, 2]]
    z_copy = points_flat[:, 2].copy()
    points_flat[:, 2] = 0
    cloud_flat = pv.PolyData(points_flat)

    pl = pv.Plotter()
    #pl.camera_position = [(0.1, 0.1, 0.1), (0, 0, 0), (0, 0, 1)]
    pl.add_mesh(cloud, rgb=True, render_points_as_spheres=True, point_size=3, opacity=0.9, lighting=False)
    #pl.add_mesh(cloud_flat, scalars=z_copy, render_points_as_spheres=True, point_size=3, opacity=0.9, cmap='jet')
    #pl.add_mesh(surf, rgb=True)
    #pl.add_points(np.array([0,0,0], np.float32), color="blue", render_points_as_spheres=True, point_size=20)
    #pl.add_mesh(xAx, color='red', render_lines_as_tubes=True, line_width=2)
    #pl.add_mesh(yAx, color='green', render_lines_as_tubes=True, line_width=2)
    #pl.add_mesh(zAx, color='blue', render_lines_as_tubes=True, line_width=2)
    pl.background_color='white'
    pl.show_axes()
    pl.show()
