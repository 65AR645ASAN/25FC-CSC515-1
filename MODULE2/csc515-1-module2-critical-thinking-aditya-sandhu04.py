"""
CSC515 - Module 2 Critical Thinking
Option #1: Color Channel Transformations of a Puppy Image

Author: Aditya Sandhu
Date: 09/16/2025

Summary:
This program demonstrates fundamental image operations using OpenCV
and NumPy.
"""

import cv2
import numpy as np
from pathlib import Path
import shutil


def filter_valid_images(source: Path, destination: Path, extensions=None) -> Path:
    """
    Copy acceptable images into a new folder and return one path.

    Parameters
    ----------
    source : Path
        Directory with raw images.
    destination : Path
        Directory to hold only valid formats.
    extensions : set
        Valid extensions (default: {'.jpg', '.jpeg', '.png'}).

    Returns
    -------
    Path
        Path to one valid image.
    """
    if extensions is None:
        extensions = {".jpg", ".jpeg", ".png"}

    destination.mkdir(parents=True, exist_ok=True)

    for file in source.iterdir():
        if file.suffix.lower() in extensions:
            shutil.copy2(file, destination / file.name)

    candidates = [f for f in destination.iterdir() if f.suffix.lower() in extensions]
    if not candidates:
        raise RuntimeError("No supported image detected in source directory.")
    return candidates[0]


def analyze_channels(image_file: Path):

    matrix = cv2.imread(str(image_file))
    if matrix is None:
        raise ValueError(f"Image cannot be loaded: {image_file}")

    # Manual separation instead of cv2.split
    blue_plane = matrix[:, :, 0]
    green_plane = matrix[:, :, 1]
    red_plane = matrix[:, :, 2]

    # Show channel planes
    cv2.imshow("Blue Intensity Map", blue_plane)
    cv2.imshow("Green Intensity Map", green_plane)
    cv2.imshow("Red Intensity Map", red_plane)

    # Reconstruct original
    reconstructed = np.dstack([blue_plane, green_plane, red_plane])
    cv2.imshow("Reconstructed Color Image", reconstructed)

    # Swap red-green
    swapped = np.dstack([blue_plane, red_plane, green_plane])
    cv2.imshow("Red/Green Exchanged", swapped)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    src_folder = root / "puppy_files"
    dst_folder = root / "clean_puppy_images"

    chosen_image = filter_valid_images(src_folder, dst_folder)
    print(f"Working with: {chosen_image}")
    analyze_channels(chosen_image)
