import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the image
img = cv2.imread("shutterstock227361781--125.jpg")

# Convert to RGB for visualization
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# --- Apply transformations ---

# 1. Translation
tx, ty = 50, 30
M_translate = np.float32([[1, 0, tx], [0, 1, ty]])
translated = cv2.warpAffine(img_rgb, M_translate, (img_rgb.shape[1], img_rgb.shape[0]))

# 2. Rotation (rotate 15 degrees around center)
center = (img_rgb.shape[1] // 2, img_rgb.shape[0] // 2)
M_rotate = cv2.getRotationMatrix2D(center, 15, 1.0)
rotated = cv2.warpAffine(img_rgb, M_rotate, (img_rgb.shape[1], img_rgb.shape[0]))

# 3. Scaling (resize by 1.2x)
scaled = cv2.resize(img_rgb, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_LINEAR)

# 4. Perspective Transform
pts1 = np.float32([[50, 50], [200, 50], [50, 200], [200, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250], [220, 220]])
M_perspective = cv2.getPerspectiveTransform(pts1, pts2)
perspective = cv2.warpPerspective(img_rgb, M_perspective, (img_rgb.shape[1], img_rgb.shape[0]))

# --- Display results ---
fig, axes = plt.subplots(1, 5, figsize=(20, 6))
titles = ["Original", "Translated", "Rotated", "Scaled", "Perspective"]
images = [img_rgb, translated, rotated, scaled, perspective]

for ax, im, title in zip(axes, images, titles):
    ax.imshow(im)
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.show()
