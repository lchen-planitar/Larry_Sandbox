import torch
import gc
import glob
import numpy as np
from depth_anything_3.api import DepthAnything3

from tools import apply_h_transform
from tools import inverse_H
from tools import H_to_euler
from tools import euler_to_H

# Load model from Hugging Face Hub
device = torch.device("cuda" if torch.cuda.is_available() else "xpu")
model = DepthAnything3.from_pretrained("depth-anything/da3-small")
model = model.to(device=device)

# Run inference on images
#images = ["image1.jpg"]  # List of image paths, PIL Images, or numpy arrays
images = glob.glob("*.jpg")
print(images)

N = len(images)
extrinsic_euler = np.zeros((N, 6))
extrinsic_matrix = np.zeros((N, 4, 4))

for i in range(0, 19):
    extrinsic_euler[i, 1] = -180 + i*20
    extrinsic_euler[i, 2] = -20
    extrinsic_matrix[i, :, :] = inverse_H(euler_to_H(extrinsic_euler[i, :]))
    noise = np.random.normal(0, 1e-7, size=(1, 3))
    extrinsic_matrix[i, :3, 3] += noise[0, :]

for i in range(19, 38):
    extrinsic_euler[i, 1] = -180 + (i-19)*20
    extrinsic_euler[i, 2] = 0
    extrinsic_matrix[i, :, :] = inverse_H(euler_to_H(extrinsic_euler[i, :]))
    noise = np.random.normal(0, 1e-7, size=(1, 3))
    extrinsic_matrix[i, :3, 3] += noise[0, :]

for i in range(38, N):
    extrinsic_euler[i, 1] = -180 + (i-38)*20
    extrinsic_euler[i, 2] = 20
    extrinsic_matrix[i, :, :] = inverse_H(euler_to_H(extrinsic_euler[i, :]))
    noise = np.random.normal(0, 1e-7, size=(1, 3))
    extrinsic_matrix[i, :3, 3] += noise[0, :]

print(extrinsic_euler)

# 3. Setup Camera Intrinsics Matrix (Example for 1920x1080 resolution)
fx = 500  # Focal length in pixels
fy = 500  # Focal length in pixels
cx = 640 / 2.0   # Optical center X
cy = 640 / 2.0   # Optical center Y

k_matrix = np.array([
    [fx,  0.0, cx],
    [0.0, fy,  cy],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

# Broadcast the calibration matrix to shape (N, 3, 3)
intrinsics_array = np.repeat(k_matrix[np.newaxis, :, :], N, axis=0)

gc.collect()
if hasattr(torch, 'xpu'):
    torch.xpu.empty_cache()  # Clears Intel G_PU VRAM cache

prediction = model.inference(
    images,
    #export_dir="output_base",
    export_dir="output_small_cond",
    intrinsics=intrinsics_array,
    extrinsics = extrinsic_matrix,
    #align_to_input_ext_scale=True,
    conf_thresh_percentile=30.0,
    export_format="glb"  # Options: glb, npz, ply, mini_npz, gs_ply, gs_video
)

# Access results
print(prediction.depth.shape)        # Depth maps: [N, H, W] float32
print(prediction.conf.shape)         # Confidence maps: [N, H, W] float32
print(prediction.extrinsics.shape)   # Camera poses (w2c): [N, 3, 4] float32
print(prediction.intrinsics.shape)   # Camera intrinsics: [N, 3, 3] float32
