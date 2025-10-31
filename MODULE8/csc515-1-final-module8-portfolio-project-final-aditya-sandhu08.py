#!/usr/bin/env python3
"""
Face-Anonymizer – Module 8 Portfolio Project
Author: Aditya Sandhu
Description: This script processes images to detect human faces using pre-trained classifiers,
validates them by checking for eyes, and applies blurring to the eye regions for anonymization purposes.
It handles multiple images in a loop, saving and displaying results.
"""

import cv2
import os

# --------------------------------------------------------------
#  Configuration Constants
# --------------------------------------------------------------
FACE_XML = 'haarcascade_frontalface_alt2.xml'
EYE_XML = 'haarcascade_eye.xml'
SRC_FOLDER = 'portfolio-images'
DEST_FOLDER = 'detected-images'

TARGET_IMAGES = [
    'animal-1.jpg',
    'group-front-standing-group-1.jpg',
    'full-body-single-person-male-1.jpg'
]


# --------------------------------------------------------------
#  Utility: Load OpenCV classifier with validation
# --------------------------------------------------------------
def get_classifier(model_name, xml_path):
    """
    Loads a CascadeClassifier from the given XML path.

    Args:
        model_name (str): Descriptive name for the model (e.g., 'frontal face').
        xml_path (str): Path to the XML file containing the trained model.

    Returns:
        cv2.CascadeClassifier: The loaded classifier object.

    Raises:
        FileNotFoundError: If the XML file does not exist.
        RuntimeError: If the classifier fails to load properly.
    """
    if not os.path.exists(xml_path):
        raise FileNotFoundError(
            "Required model '{}' not found. "
            "Download from OpenCV GitHub and place in project root.".format(model_name)
        )
    classifier = cv2.CascadeClassifier(xml_path)
    if classifier.empty():
        raise RuntimeError("Failed to initialize {} model.".format(model_name))
    return classifier


# Load required models
face_detector = get_classifier("frontal face", FACE_XML)
eye_detector = get_classifier("eye", EYE_XML)

print("\nAll detection models loaded successfully.\n")
os.makedirs(DEST_FOLDER, exist_ok=True)


# --------------------------------------------------------------
#  Utility: Safe image loader
# --------------------------------------------------------------
def load_picture(file_path):
    """
    Loads an image from the specified path using OpenCV.

    Args:
        file_path (str): Full path to the image file.

    Returns:
        numpy.ndarray: The loaded image as a NumPy array.

    Raises:
        IOError: If the image cannot be read.
    """
    picture = cv2.imread(file_path)
    if picture is None:
        raise IOError("Unable to load image from: {}".format(file_path))
    return picture


# --------------------------------------------------------------
#  Utility: Optimize image for feature extraction
# --------------------------------------------------------------
def optimize_image(input_picture):
    """
    Applies grayscale conversion, smoothing, and contrast enhancement to prepare the image for detection.

    Args:
        input_picture (numpy.ndarray): The original color image.

    Returns:
        numpy.ndarray: The optimized grayscale image.

    This function helps improve detection accuracy by reducing noise and enhancing local contrast,
    which is particularly useful for images with varying lighting conditions.
    """
    gray_version = cv2.cvtColor(input_picture, cv2.COLOR_BGR2GRAY)
    smoothed_version = cv2.GaussianBlur(gray_version, (7, 7), 0)
    contrast_enhancer = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    # Apply enhancement and return the result
    optimized = contrast_enhancer.apply(smoothed_version)
    return optimized


# --------------------------------------------------------------
#  Utility: Search and authenticate components
# --------------------------------------------------------------
def search_components(optimized_gray, base_picture):
    """
    Searches for potential face regions and authenticates them by detecting internal features (eyes).
    Applies anonymization to authenticated features.

    Args:
        optimized_gray (numpy.ndarray): Preprocessed grayscale image.
        base_picture (numpy.ndarray): Original color image to modify.

    Returns:
        tuple: (int: count of authenticated regions, numpy.ndarray: modified image)

    This function iterates over potential regions, extracts sub-areas, performs secondary detection,
    and applies transformations only if validation criteria are met, ensuring high accuracy.
    """
    suspected_areas = face_detector.detectMultiScale(
        optimized_gray,
        scaleFactor=1.05,
        minNeighbors=8,
        minSize=(40, 40),
        maxSize=(400, 400),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    authenticated = 0
    for (loc_x, loc_y, size_w, size_h) in suspected_areas:
        sub_rgb = base_picture[loc_y:loc_y + size_h, loc_x:loc_x + size_w]
        sub_mono = optimized_gray[loc_y:loc_y + size_h, loc_x:loc_x + size_w]

        component_locs = eye_detector.detectMultiScale(
            sub_mono,
            scaleFactor=1.1,
            minNeighbors=11,
            minSize=(15, 15)
        )

        # Check if sufficient components are found
        if len(component_locs) > 0:
            authenticated += 1
            # Mark the authenticated area
            cv2.rectangle(base_picture, (loc_x, loc_y), (loc_x + size_w, loc_y + size_h), (0, 0, 255), 2)

            # Process each component
            for (cx, cy, cw, ch) in component_locs:
                component_area = sub_rgb[cy:cy + ch, cx:cx + cw]
                transformed = cv2.GaussianBlur(component_area, (23, 23), 30)
                sub_rgb[cy:cy + ch, cx:cx + cw] = transformed

    return authenticated, base_picture


# --------------------------------------------------------------
#  Core: Supervise image operation
# --------------------------------------------------------------
def supervise_operation(image_label):
    """
    Supervises the entire operation for a single image: loading, optimization, detection, anonymization, saving, and display.

    Args:
        image_label (str): Filename of the image to process.

    This function serves as the main controller for each image, handling errors and coordinating utilities.
    It ensures operations are sequential and results are persisted.
    """
    access_point = os.path.join(SRC_FOLDER, image_label)

    if not os.path.isfile(access_point):
        print("[SKIP] Missing entry: {}".format(access_point))
        return

    print("Beginning operation on: {}".format(image_label))
    access_data = load_picture(access_point)

    tuned_gray = optimize_image(access_data)

    total_auth, operation_result = search_components(tuned_gray, access_data)

    print("  Supervised {} authenticated components.".format(total_auth))

    access_end = os.path.join(DEST_FOLDER, "outcome_" + image_label)
    cv2.imwrite(access_end, operation_result)
    print("  Secured outcome in: {}".format(access_end))

    cv2.imshow("Operation Outcome - {}".format(image_label), operation_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# --------------------------------------------------------------
#  Program Entry
# --------------------------------------------------------------
if __name__ == "__main__":
    print("Activating anonymization operations...\n")
    for access_file in TARGET_IMAGES:
        supervise_operation(access_file)
    print("\nOperations concluded. Inspect '{}' for outcomes.".format(DEST_FOLDER))