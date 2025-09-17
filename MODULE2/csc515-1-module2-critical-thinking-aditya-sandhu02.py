"""
CSC515 - Module 2 Critical Thinking
Option #1: Multi-Scale Representation of a Puppy Image using OpenCV

Author: Aditya Sandhu
Date: 09/16/2025

Overview:
This program demonstrates basic color-channel manipulations in OpenCV.
- Filters out only valid image formats from the input directory.
- Reads a puppy image and decomposes it into its three color components.
- Displays each component in grayscale form to illustrate channel intensity.
- Reconstructs the original color image by recombining channels.
- Creates a variant with the red and green channels interchanged.

Note:
All operations are performed on NumPy arrays, where each image is represented
as a 3D tensor of pixel values in the range [0, 255].
"""

import cv2
import numpy as np
import os
import shutil


def prepare_image_directory(base_folder: str, cleaned_folder: str, extensions=None):
    """
    Copy valid images into a clean working directory.

    Parameters
    ----------
    base_folder : str
        Directory containing raw input images.
    cleaned_folder : str
        Directory where only valid image files will be stored.
    extensions : set, optional
        Allowed file extensions (default: {'.jpg', '.jpeg', '.png'}).

    Returns
    -------
    str
        Path to the first valid image found.
    """
    if extensions is None:
        extensions = {".jpg", ".jpeg", ".png"}

    os.makedirs(cleaned_folder, exist_ok=True)

    for filename in os.listdir(base_folder):
        if os.path.splitext(filename)[1].lower() in extensions:
            shutil.copy2(os.path.join(base_folder, filename),
                         os.path.join(cleaned_folder, filename))

    candidates = [f for f in os.listdir(cleaned_folder)
                  if os.path.splitext(f)[1].lower() in extensions]

    if not candidates:
        raise FileNotFoundError("No supported image file found.")

    return os.path.join(cleaned_folder, candidates[0])


def display_color_processing(image_path: str):
    """
    Perform channel separation, visualization, reconstruction,
    and color-swapping of an image.

    Parameters
    ----------
    image_path : str
        Path to the image file.
    """
    image_matrix = cv2.imread(image_path)
    if image_matrix is None:
        raise ValueError(f"Image could not be loaded: {image_path}")

    # Separate channels
    blue_layer, green_layer, red_layer = cv2.split(image_matrix)

    # Show channel intensities
    cv2.imshow("Blue Layer", blue_layer)
    cv2.imshow("Green Layer", green_layer)
    cv2.imshow("Red Layer", red_layer)

    # Reconstruct original
    reconstructed = cv2.merge((blue_layer, green_layer, red_layer))
    cv2.imshow("Reconstructed Image", reconstructed)

    # Swap red and green
    swapped = cv2.merge((blue_layer, red_layer, green_layer))
    cv2.imshow("Red-Green Swapped", swapped)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Define directories relative to script location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "puppy_files")
    working_dir = os.path.join(base_dir, "filtered_puppy_images")

    first_image_path = prepare_image_directory(input_dir, working_dir)
    print(f"Processing: {first_image_path}")
    display_color_processing(first_image_path)
