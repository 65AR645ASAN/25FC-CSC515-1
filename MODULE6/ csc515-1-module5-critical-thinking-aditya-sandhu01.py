import cv2  # Library for computer vision tasks
import numpy as np  # Library for numerical operations and array handling
from matplotlib import pyplot as plt  # Library for plotting and visualizing images

# Step 1: Read the input image from the specified file path
# This loads the color image into memory as a NumPy array
input_image = cv2.imread('lllumination_scene.jpeg')

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
# Loop through the images and configure each subplot
for index in range(3):
    plt.subplot(1, 3, index + 1)  # 1 row, 3 columns, current position
    plt.imshow(display_images[index], cmap='gray')  # Display with grayscale colormap
    plt.title(display_titles[index])  # Set title for the subplot
    plt.axis('off')  # Hide axis ticks for cleaner view

# Adjust layout to prevent overlapping and display the plot
plt.tight_layout()  # Automatically adjusts subplot parameters
plt.show()  # Render the figure on screen

# Step 7: Save the final binary image to a file
# This writes the thresholded result as a JPG for later use
cv2.imwrite('processed_binary_output.jpg', binary_image)

# Additional notes:
# - This script assumes the input file exists in the current directory.
# - No error handling is added here for simplicity, but in production, check if image loading succeeds.
# - Libraries like OpenCV and Matplotlib must be installed in the environment.
# - The adaptive thresholding is useful for images with varying illumination, as seen in the filename suggestion.