"""
Critical Thinking Module 2 Task
Choice 1: Representing Puppy Photo at Multiple Levels with OpenCV

Created by: Aditya Sandhu
Created on: 09/16/2025

Overview:
This program processes a puppy photo by first organizing valid files from a source folder,
then extracting color layers, showing them separately, recombining them, and creating
a variant with two layers exchanged. It focuses on image handling basics.
"""

import cv2
import os
import shutil


def organize_valid_photos(base_path, source_folder, target_folder):
    """Prepare a clean folder with only supported photo types."""
    full_source = os.path.join(base_path, source_folder)
    full_target = os.path.join(base_path, target_folder)
    os.makedirs(full_target, exist_ok=True)

    allowed_types = {'.jpg', '.jpeg', '.png'}
    for item in os.listdir(full_source):
        _, file_type = os.path.splitext(item)
        if file_type.lower() in allowed_types:
            origin = os.path.join(full_source, item)
            destination = os.path.join(full_target, item)
            shutil.copy(origin, destination)


# Set up paths and organize files
current_base = os.path.dirname(os.path.abspath(__file__))
organize_valid_photos(current_base, 'puppy_files', 'processed_puppy_photos')

# Find and select a photo to use
processed_path = os.path.join(current_base, 'processed_puppy_photos')
available_photos = []  # start with an empty list
for f in os.listdir(processed_path):  # loop through all items in the folder
    file_extension = os.path.splitext(f)[1].lower()  # get the file extension in lowercase
    if file_extension in {'.jpg', '.jpeg', '.png'}:  # check if it's a valid photo type
        available_photos.append(f)  # add it to the list
if not available_photos:
    raise ValueError("No suitable photo found in the source folder. Add a JPG, JPEG, or PNG file.")

selected_photo = os.path.join(processed_path, available_photos[0])
print(f"Processing this photo: {selected_photo}")

# Read the photo
photo_data = cv2.imread(selected_photo)
if photo_data is None:
    raise ValueError(f"Could not read the photo at {selected_photo}")

# Extract individual color layers
blue_layer, green_layer, red_layer = cv2.split(photo_data)

# Show each layer in gray
cv2.imshow('Blue Layer View', blue_layer)
cv2.imshow('Green Layer View', green_layer)
cv2.imshow('Red Layer View', red_layer)

# Recombine to form the initial photo
recombined_photo = cv2.merge([blue_layer, green_layer, red_layer])
cv2.imshow('Recombined Original Photo', recombined_photo)

# Create variant by exchanging green and red layers
variant_photo = cv2.merge([blue_layer, red_layer, green_layer])
cv2.imshow('Variant Photo with Layers Exchanged', variant_photo)

# Hold displays until a key is hit
cv2.waitKey(0)
cv2.destroyAllWindows()