import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
This script loads a noisy image, converts it to grayscale, and applies three filters 
(Gaussian smoothing, Laplacian edge detection, and a combined Gaussian-then-Laplacian) 
"""

image_file = r"C:\Users\Aditya\Desktop\CSUDOCS\CSC515\25FC-CSC515-1\MODULE4\Mod4CT2.jpg"
original_image = cv2.imread(image_file, cv2.IMREAD_COLOR)

# Check if image loaded successfully; raise error if not
if original_image is None:
    raise FileNotFoundError(f"Image Not Found in Location ~ {image_file}.")

# Convert to grayscale for filter application
grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Define the sigma value for Gaussian operations
gaussian_sigma = 1.0

# List of kernel sizes to test
kernel_sizes = [3, 5, 7]

# Dictionary to hold filtered images for each kernel size
filtered_images = {}

# Loop through each kernel size and apply filters
for size in kernel_sizes:
    # Apply Gaussian blur for smoothing
    smoothed_image = cv2.GaussianBlur(grayscale_image, (size, size), gaussian_sigma)

    # Apply Laplacian for edge enhancement (convert to abs for display)
    edge_image = cv2.Laplacian(grayscale_image, cv2.CV_64F, ksize=size)
    edge_image = cv2.convertScaleAbs(edge_image)

    # Apply Gaussian blur followed by Laplacian for noise-reduced edges
    smoothed_edge_image = cv2.Laplacian(smoothed_image, cv2.CV_64F, ksize=size)
    smoothed_edge_image = cv2.convertScaleAbs(smoothed_edge_image)

    # Store the results as a tuple
    filtered_images[size] = (smoothed_image, edge_image, smoothed_edge_image)

# Create a 3x3 subplot grid for visualization
figure, subplots = plt.subplots(3, 3, figsize=(12, 12))
filter_types = ["Gaussian Smoothing", "Laplacian Edges", "Smoothed + Edges"]

# Populate the subplots with images
for row, size in enumerate(kernel_sizes):
    for col, type_name in enumerate(filter_types):
        subplots[row, col].imshow(filtered_images[size][col], cmap="gray")
        subplots[row, col].set_title(f"{type_name} ({size}x{size})")
        subplots[row, col].axis("off")

# Adjust layout and display the plot
plt.tight_layout()
plt.show()