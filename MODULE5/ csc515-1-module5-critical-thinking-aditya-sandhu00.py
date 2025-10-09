import cv2
# import numpy as np
import matplotlib.pyplot as plt

"""
This script demonstrates morphological operations (dilation, erosion, opening, closing) 
on a scanned image of cursive handwritten text on a sticky note using OpenCV. 
It enhances the text for better HWR/OCR by removing noise and filling gaps. 
Replace 'handwritten_sticky.jpg' with your actual image path.
"""

# Load the image in grayscale
img_path = 'handwritten_img.png'  # Update this to your scanned image path
original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if original is None:
    raise FileNotFoundError(f"Image not found at {img_path}. "
                            f"Please provide a valid path to a scanned handwritten sticky note image.")

# Apply binary thresholding to create a binary image (dark text = 0, light background = 255)
_, binary = cv2.threshold(original, 127, 255, cv2.THRESH_BINARY_INV)

# Define a structuring element (3x3 rectangle for basic enhancement)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# Apply morphological operations
dilated = cv2.dilate(binary, kernel, iterations=1)  # Expands text lines
eroded = cv2.erode(binary, kernel, iterations=1)    # Shrinks text lines, removes thin noise
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)   # Erosion then dilation: removes noise, separates characters
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # Dilation then erosion: fills gaps, connects broken parts

# Visualize results in a 2x3 subplot grid
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
images = [original, binary, dilated, eroded, opened, closed]
titles = ['Original Grayscale', 'Binary Thresholded', 'Dilation', 'Erosion', 'Opening', 'Closing']

for i, (ax, img, title) in enumerate(zip(axes.flat, images, titles)):
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.show()

# Optional: Save enhanced image (e.g., after opening + closing for combined enhancement)
enhanced = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
cv2.imwrite('enhanced_handwritten.jpg', enhanced)
print("Enhanced image saved as 'enhanced_handwritten.jpg'")