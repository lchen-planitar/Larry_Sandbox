import cv2
import sys
import argparse
import configparser
import matplotlib.pyplot as plt
import rawpy
import numpy as np
from pupil_apriltags import Detector


def parse():
    parser = argparse.ArgumentParser('Plot Image')
    parser.add_argument('input', help='File containing camera poses')
    args = parser.parse_args(None if (sys.argv[1:]) else ['-h'])
    return args


def main():
    #code
    file = args.input

    # 1. Load image and convert to grayscale
    image = cv2.imread(file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Initialize the Detector
    at_detector = Detector(
        families="tag36h11",  # Replace with your actual family if different
        nthreads=4,           # Increase threads for better performance
        quad_decimate=1.0,    # CRITICAL: Set to 1.0 (disables downsampling for small/far tags)
        quad_sigma=0.8,       # Slight blur helps reduce sensor noise/aliasing
        refine_edges=1,       # Forces subpixel edge refinement
        decode_sharpening=0.5, # Enhances contrast on blurry tag borders
        debug=0,
    )

    # 3. Detect tags (optionally pass camera_params and tag_size for 6DoF pose)
    # camera_params = [fx, fy, cx, cy]
    results = at_detector.detect(gray)

    # 4. Process and organize grid results
    # Sort detections by tag_id or by spatial coordinates (center)
    grid_detections = sorted(results, key=lambda r: r.tag_id)

    centers = np.zeros((1, 2))

    for r in grid_detections:
      print(f"Tag ID: {r.tag_id}")
      print(f"  Center: {r.center}")
      print(f"  Corners: {r.corners}")
      centers = np.append(centers, [r.center], axis=0)
      # r.pose_R, r.pose_t if estimate_tag_pose=True

      # Draw bounding box and ID using OpenCV
      pts = r.corners.astype(int)
      for i in range(4):
        cv2.line(image, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 255, 0), 2)
      cv2.putText(
          image,
          str(r.tag_id),
          tuple(pts[0]),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.5,
          (0, 0, 255),
          2,
      )
    
    centers = centers[1:, :]
    print(centers)

    img_plt = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(1)
    #plt.imshow(gray, cmap='gray')
    plt.imshow(img_plt)
    plt.scatter(centers[:, 0], centers[:, 1])
    plt.show()
    


if __name__=="__main__":
    args = parse()
    main()