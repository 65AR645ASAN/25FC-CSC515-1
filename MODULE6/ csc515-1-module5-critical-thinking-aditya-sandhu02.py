import cv2  # Library for computer vision tasks
import numpy as np  # Library for numerical operations and array handling
from matplotlib import pyplot as plt  # Library for plotting and visualizing images
import os  # Library for path handling

# Step 1: Construct absolute path to the image based on script location
# This ensures the image is loaded from the same directory as this script
image_filename = 'lllumination_scene.jpeg'  # Adjust if filename differs (e.g., check for typos like 'lllumination_scene.jpeg')
image_path = os.path.join(os.path.dirname(__file__), image_filename)

# Load the image using the absolute path
input_image = cv2.imread(image_path)

# Basic error check: Ensure image loaded successfully
if input_image is None:
    raise ValueError(f"Image file not found or unable to load. Check path: '{image_path}'")

# Step 2: Transform the color image to grayscale
# Grayscale conversion simplifies processing by reducing to one channel
grayscale_version = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

# Step 3: Apply a smoothing filter to minimize noise in the image
# Gaussian Blur uses a 5x5 kernel with standard deviation 0 for even smoothing
smoothed_image = cv2.GaussianBlur(grayscale_version, (5, 5), 0)

# Step 4: Perform adaptive thresholding to create a binary image
# This uses Gaussian adaptive method for local threshold calculation
# Parameters: max value 255, adaptive type, binary threshold, block size 15, constant 3
binary_image = cv2.adaptiveThreshold(
    smoothed_image,  # Input image after blurring
    255,  # Maximum value for pixels
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # Adaptive method using Gaussian weights
    cv2.THRESH_BINARY,  # Binary thresholding type
    15,  # Neighborhood block size
    3  # Constant subtracted from mean
)

# Step 5: Prepare lists for displaying multiple images side by side
# Titles describe each subplot for clarity
display_titles = ['Input Color Image', 'Grayscale Conversion', 'Binary Threshold Result']

# Images list holds the processed versions to visualize
display_images = [input_image, grayscale_version, binary_image]

# Step 6: Create a figure with subplots to show the images
plt.figure(figsize=(15, 5))  # Set figure size for better visibility
for index in range(3):
    plt.subplot(1, 3, index + 1)  # 1 row, 3 columns, current position
    if index == 0:
        # Convert BGR to RGB for correct color display in matplotlib, no cmap
        plt.imshow(cv2.cvtColor(display_images[index], cv2.COLOR_BGR2RGB))
    else:
        # Use grayscale colormap for non-color images
        plt.imshow(display_images[index], cmap='gray')
    plt.title(display_titles[index])  # Set title for the subplot
    plt.axis('off')  # Hide axis ticks for cleaner view

# Adjust layout to prevent overlapping
plt.tight_layout()

# Save the combined figure as a single JPG file
plt.savefig('combined_processing_stages.jpg', dpi=300)  # Higher DPI for quality

# Display the plot on screen (optional, can remove if only file is needed)
plt.show()

# Step 7: Optionally save the final binary image separately
# This writes the thresholded result as a JPG for later use
cv2.imwrite('processed_binary_output.jpg', binary_image)

# Additional notes:
# - This script assumes the input file exists in the same directory as this Python file.
# - Libraries like OpenCV and Matplotlib must be installed in the environment.
# - The adaptive thresholding is useful for images with varying illumination, as seen in the filename suggestion.
# - The saved 'combined_processing_stages.jpg' will match the side-by-side format in your screenshot.