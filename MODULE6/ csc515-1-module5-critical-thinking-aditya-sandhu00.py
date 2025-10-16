import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
img = cv2.imread('lllumination_scene.jpeg')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Adaptive thresholding using Gaussian method
adaptive_gaussian = cv2.adaptiveThreshold(
    blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 15, 3
)

# Show results using matplotlib
titles = ['Original Image', 'Grayscale', 'Adaptive Gaussian Threshold']
images = [img, gray, adaptive_gaussian]

for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()

# Save output image
cv2.imwrite('adaptive_threshold_output.jpg', adaptive_gaussian)
