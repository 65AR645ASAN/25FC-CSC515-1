import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
This application utilizes morphological techniques to refine scanned images containing handwritten cursive notes, aiming to facilitate improved character recognition systems. Operations include expanding regions via dilation for connectivity, contracting via erosion to discard minor disturbances, opening for artifact removal with form retention, and closing for mending discontinuities. For optimal results, experiment with kernel shapes like elliptical for curved scripts. Provide your scanned file path below.
"""

# Define the path to the scanned document
scan_path = 'handwritten_img.png'  # Alter this to your specific scanned document location
base_scan = cv2.imread(scan_path, cv2.IMREAD_GRAYSCALE)

# Confirm successful image import
if base_scan is None:
    raise FileNotFoundError(f"Failed to import from {scan_path}. Verify the file exists and path is accurate.")

# Enhance contrast using histogram equalization prior to binarization
equalized = cv2.equalizeHist(base_scan)

# Convert to binary using adaptive thresholding to handle varying illumination
dual_img = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)  # Block size 11, constant 2 for adjustment

# Form an elliptical kernel for smoother processing on curves
morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))  # Elliptical 4x4; suitable for handwritten flow

# Apply dilation to broaden text areas
expanded = cv2.dilate(dual_img, morph_kernel, iterations=1)  # Increases stroke width to link subtle elements

# Apply erosion to narrow text areas
contracted = cv2.erode(dual_img, morph_kernel, iterations=1)  # Discards fine noise and refines boundaries

# Conduct opening to purge small irregularities
refined_open = cv2.morphologyEx(dual_img, cv2.MORPH_OPEN, morph_kernel, iterations=1)  # First contracts, then expands to clear specks

# Conduct closing to unite fragmented sections
refined_close = cv2.morphologyEx(dual_img, cv2.MORPH_CLOSE, morph_kernel, iterations=1)  # First expands, then contracts to patch interruptions

# Output shapes for debugging purposes
print(f"Base scan shape: {base_scan.shape}")
print(f"Binary shape: {dual_img.shape}")

# Prepare a horizontal figure layout for outputs
vis_fig, vis_axes = plt.subplots(1, 6, figsize=(18, 6))  # Single row, six columns for linear comparison

# Compile visuals and headings
visuals = [base_scan, dual_img, expanded, contracted, refined_open, refined_close]
headings = ['Base Scan', 'Dual Tone', 'Expanded View', 'Contracted View', 'Refined Open', 'Refined Close']

# Display each in its axis using comprehension
[ax.imshow(img, cmap='gray') or ax.set_title(head) or ax.axis('off') for ax, img, head in zip(vis_axes, visuals, headings)]

# Finalize and present the visualization
plt.tight_layout()
plt.show()