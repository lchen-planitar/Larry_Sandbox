import cv2
import numpy as np
from PIL import Image
from transformers import pipeline
import matplotlib.pyplot as plt

# 1. Initialize the depth estimation pipeline
# You can use "LiheYoung/depth-anything-small-hf" or "depth-anything-v2-small-hf"
print("Loading model...\n")
pipe = pipeline(task="depth-estimation", model="haodongli/DA-2")

# 2. Load your input image
print("Loading image...\n")
image_path = "001-X5_lunch1ManualEdit.jpg"
image = Image.open(image_path)

# 3. Perform depth inference
print("Inference...\n")
predictions = pipe(image)

# 4. Extract the depth map from predictions
# The pipeline returns a dictionary containing the 'depth' image (PIL format)
print("Extracting Depth...\n")
depth_pil = predictions["depth"]

# 5. Convert to OpenCV format to apply a colormap (Optional visualization step)
print("Converting Depth Map...\n")
depth_numpy = np.array(depth_pil)
depth_colorized = cv2.applyColorMap(depth_numpy, cv2.COLORMAP_INFERNO)

plt.figure(1)
plt.imshow(depth_numpy)
plt.colorbar()
plt.show()

print("Saving Output...\n")
# 6. Save the outputs
depth_pil.save("depth_grayscale.png")             # Grayscale (raw relative depth)
cv2.imwrite("depth_colorized.png", depth_colorized) # Colorized visual asset