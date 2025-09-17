"""
Module 2 - Critical Thinking Assignment
Option #1: Puppy Image Multi-Scale Representation in OpenCV

Author: Aditya Sandhu
Date: 09/16/2025

Description:
This script demonstrates multi-scale representation of an image by separating
the RGB channels of a puppy image, visualizing them individually, merging them
back into the original image, and swapping the red and green channels.
Additionally, it cleans up the puppy_files directory by filtering out only
valid image files (JPG, JPEG, PNG).
"""

import cv2
import numpy as np
import os
import shutil

# ---- Step 0: Cleanup junk files ----
script_dir = os.path.dirname(os.path.abspath(__file__))
puppy_dir = os.path.join(script_dir, "puppy_files")
clean_dir = os.path.join(script_dir, "clean_puppy_images")

os.makedirs(clean_dir, exist_ok=True)

valid_exts = {".jpg", ".jpeg", ".png"}
for fname in os.listdir(puppy_dir):
    ext = os.path.splitext(fname)[1].lower()
    if ext in valid_exts:
        src = os.path.join(puppy_dir, fname)
        dst = os.path.join(clean_dir, fname)
        shutil.copy2(src, dst)

# Pick the first valid image found
valid_images = [f for f in os.listdir(clean_dir) if os.path.splitext(f)[1].lower() in valid_exts]
if not valid_images:
    raise FileNotFoundError("No valid image found in puppy_files. Please place a JPG/PNG there.")

img_path = os.path.join(clean_dir, valid_images[0])
print(f"Using image: {img_path}")

# ---- Step 1: Load the image ----
image = cv2.imread(img_path)
if image is None:
    raise FileNotFoundError(f"Failed to load image at {img_path}")

# ---- Step 2: Split into channels ----
b_channel, g_channel, r_channel = cv2.split(image)

# ---- Step 3: Display each channel as grayscale ----
cv2.imshow("Blue Channel", b_channel)
cv2.imshow("Green Channel", g_channel)
cv2.imshow("Red Channel", r_channel)

# ---- Step 4: Merge back to original RGB image ----
original_merge = cv2.merge((b_channel, g_channel, r_channel))
cv2.imshow("Original Merged Image", original_merge)

# ---- Step 5: Swap Red and Green Channels (GRB) ----
swapped_merge = cv2.merge((b_channel, r_channel, g_channel))
cv2.imshow("Swapped Image (GRB)", swapped_merge)

# ---- Keep windows open until key press ----
cv2.waitKey(0)
cv2.destroyAllWindows()
