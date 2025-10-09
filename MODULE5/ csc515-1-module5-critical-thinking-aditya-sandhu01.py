import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
This program applies various morphological transformations to improve a digitized image of cursive script on a note pad 
for better text recognition. It uses dilation to thicken lines, erosion to eliminate small artifacts, opening to clean 
up noise while keeping shape, and closing to seal small breaks. Update the file location with your scanned document.
"""

# Specify the location of the input image
file_location = 'handwritten_img.png'  # Modify this to point to your actual scanned file
input_gray = cv2.imread(file_location, cv2.IMREAD_GRAYSCALE)

# Verify if the image was loaded correctly
if input_gray is None:
    raise FileNotFoundError(f"Unable to load image from {file_location}. "
                            f"Ensure the path is correct for the scanned note.")

# Perform thresholding to obtain a binary representation (text as foreground)
ret_val, thresh_img = cv2.threshold(input_gray, 150, 255, cv2.THRESH_BINARY_INV)
# Adjust threshold if needed for better contrast

# Create a rectangular kernel for morphological processing
struct_elem = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# Using 5x5 for slightly stronger effect; adjust as per text thickness

# Execute dilation to expand foreground regions
dilated_img = cv2.dilate(thresh_img, struct_elem, iterations=2)
# Thickens text strokes, helps in connecting faint parts

# Execute erosion to contract foreground regions
eroded_img = cv2.erode(thresh_img, struct_elem, iterations=2)  # Removes minor noise and thins out the text

# Perform opening: erosion followed by dilation to eliminate noise
opened_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, struct_elem, iterations=1)
# Cleans small specks without much loss

# Perform closing: dilation followed by erosion to bridge gaps
closed_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, struct_elem, iterations=1)
# Fills holes and connects broken lines

# Set up a figure with subplots for displaying results
plot_fig, plot_axes = plt.subplots(3, 2, figsize=(12, 15))  # 3 rows, 2 columns for vertical layout

# List of images and corresponding labels
display_imgs = [input_gray, thresh_img, dilated_img, eroded_img, opened_img, closed_img]
display_labels = ['Input Grayscale', 'Thresholded Binary', 'After Dilation', 'After Erosion', 'After Opening', 'After Closing']

# Loop to populate each subplot
for idx, (axis, image, label) in enumerate(zip(plot_axes.flat, display_imgs, display_labels)):
    axis.imshow(image, cmap='gray')  # Display in grayscale
    axis.set_title(label, fontsize=12)  # Set subplot title
    axis.axis('off')  # Hide axis ticks

# Optimize the plot layout and display
plt.tight_layout()
plt.show()

# Generate an improved version by combining opening and closing
improved = cv2.morphologyEx(opened_img, cv2.MORPH_CLOSE, struct_elem)

# Save the improved image
cv2.imwrite('improved_note.jpg', improved)
print("The processed image has been stored as 'improved_note.jpg'")