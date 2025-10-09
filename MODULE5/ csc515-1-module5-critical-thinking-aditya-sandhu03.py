import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
This software employs morphological methods to polish digitized cursive handwriting samples from notes, 
supporting enhanced readability in recognition algorithms. It expands with dilation for better linkage,
reduces with erosion for artifact clearance, opens for noise suppression with structure maintenance, and
closes for gap repair. Try different kernels, such as cross-shaped for specific noise types. Input your scan path here.
"""

# Set the document path
doc_path = 'handwritten_img.png'  # Change to your scanned file's location
raw_img = cv2.imread(doc_path, cv2.IMREAD_GRAYSCALE)

# Check for proper loading
if raw_img is None:
    raise FileNotFoundError(f"Import error from {doc_path}. Confirm file presence and correct path.")

# Apply equalization to boost contrast
balanced = cv2.equalizeHist(raw_img)

# Binarize adaptively for robust handling of light variations
binarized = cv2.adaptiveThreshold(balanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 4)  # Larger block size 15, constant 4

# Build a cross kernel for targeted processing
process_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # Cross 3x3; good for line-based text

# Dilation to enlarge areas
widened = cv2.dilate(binarized, process_kernel, iterations=3)  # Boosts width for faint script connection

# Erosion to diminish areas
narrowed = cv2.erode(binarized, process_kernel, iterations=3)  # Clears tiny disturbances and slims lines

# Opening operation for cleanup
polished_open = cv2.morphologyEx(binarized, cv2.MORPH_OPEN, process_kernel, iterations=1)  # Wipes out small flaws

# Closing operation for unification
polished_close = cv2.morphologyEx(binarized, cv2.MORPH_CLOSE, process_kernel, iterations=1)  # Mends small separations

# Debug: Print dimensions
print(f"Raw image dimensions: {raw_img.shape}")
print(f"Binarized dimensions: {binarized.shape}")

# Initialize horizontal display
disp_fig, disp_axes = plt.subplots(1, 6, figsize=(20, 5))  # Wider figure for better horizontal view
disp_fig.suptitle('Morphological Processing Stages', fontsize=16)  # Add overall figure title

# Gather displays and captions
displays = [raw_img, binarized, widened, narrowed, polished_open, polished_close]
captions = ['Raw Img', 'Binarized', 'Widened', 'Narrowed', 'Polished Open', 'Polished Close']

# Populate with images, titles, and in-plot labels
for ax, pic, cap in zip(disp_axes, displays, captions):
    ax.imshow(pic, cmap='gray')  # Render as grayscale
    ax.set_title(cap, fontsize=10)  # Title above
    ax.text(10, pic.shape[0] - 20, cap.lower(), color='red', fontsize=8, bbox=dict(facecolor='white', alpha=0.5))  # In-image label at bottom-left
    # ax.axis('off')  # Commented to show axes as in your output

# Refine layout and reveal
plt.tight_layout()
plt.show()