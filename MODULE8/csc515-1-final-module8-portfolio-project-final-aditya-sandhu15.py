#!/usr/bin/env python3
"""
Face-Anonymizer – Module 8 Portfolio Project
Author: Aditya Sandhu
Description: This script processes images to detect human faces using pre-trained classifiers
"""

import cv2
import os

#  Configuration Constants
FACE_XML = 'haarcascade_frontalface_alt2.xml'
EYE_GLASSES_XML = 'haarcascade_eye_tree_eyeglasses.xml'
SRC_FOLDER = 'portfolio-images'
DEST_FOLDER = 'detected-images'

TARGET_IMAGES = [
    'animal-1.jpg',
    'group-front-standing-group-1.jpg',
    'full-body-single-person-male-1.jpg'
]


#  Utility: Load OpenCV classifier with validation
def get_classifier(model_name, xml_path):
    """
    Loads a Cascade Classifier from the given XML path.

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
eye_glasses_detector = get_classifier("eye glasses", EYE_GLASSES_XML)

print("\nAll detection models loaded successfully.\n")
os.makedirs(DEST_FOLDER, exist_ok=True)


#  Utility: Safe image loader
def ld_pic(file_path):

    picture = cv2.imread(file_path)
    if picture is None:
        raise IOError("Unable to load image from: {}".format(file_path))
    return picture


#  Utility: Optimize image for feature extraction
def opt_img(input_picture):

    gry_vsn = cv2.cvtColor(input_picture, cv2.COLOR_BGR2GRAY)
    smth_vsn = cv2.GaussianBlur(gry_vsn, (7, 7), 0)
    ctrst_enhancer = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    # Apply enhancement and return the result
    optmzd = ctrst_enhancer.apply(smth_vsn)
    return optmzd


#  Utility: Search and authenticate components
def search_components(optmzd_gray, base_picture):
    """
    Searches for potential face regions and authenticates them by detecting internal features (eyes).
    Applies anonymization to authenticated features.

    Args:
        optmzd_gray (numpy.ndarray): Preprocessed grayscale image.
        base_picture (numpy.ndarray): Original color image to modify.

    Returns:
        tuple: (int: count of authenticated regions, numpy.ndarray: modified image)

    This function iterates over potential regions, extracts sub-areas, performs secondary detection,
    and applies transformations only if validation criteria are met, ensuring high accuracy.
    """
    suspected_areas = face_detector.detectMultiScale(
        optmzd_gray,
        scaleFactor=1.05,
        minNeighbors=8,
        minSize=(40, 40),
        maxSize=(400, 400),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    authenticated = 0
    for (loc_x, loc_y, size_w, size_h) in suspected_areas:
        sub_rgb = base_picture[loc_y:loc_y + size_h, loc_x:loc_x + size_w]
        sub_mono = optmzd_gray[loc_y:loc_y + size_h, loc_x:loc_x + size_w]

        # Additional enhancement on face ROI for better eye detection
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        sub_mono = clahe.apply(sub_mono)

        # Restrict to upper 70% of face to include more area for tilted heads
        eye_h = int(size_h * 0.7)
        sub_eye_mono = sub_mono[0:eye_h, 0:size_w]

        component_locs = eye_glasses_detector.detectMultiScale(
            sub_eye_mono,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(15, 15)
        )

        # Collect valid eyes (no further filtering needed with glasses-specific detector)
        valid_eyes = []
        for (cx, cy, cw, ch) in component_locs:
            if cy + ch <= eye_h:  # Ensure within upper region
                valid_eyes.append((cx, cy, cw, ch))

        # Check if at least one eye is found
        if len(valid_eyes) > 0:
            authenticated += 1
            # Mark the authenticated area
            cv2.rectangle(base_picture, (loc_x, loc_y), (loc_x + size_w, loc_y + size_h), (0, 0, 255), 2)

            # Process each component
            for (cx, cy, cw, ch) in valid_eyes:
                component_area = sub_rgb[cy:cy + ch, cx:cx + cw]
                transformed = cv2.GaussianBlur(component_area, (23, 23), 30)
                sub_rgb[cy:cy + ch, cx:cx + cw] = transformed

    return authenticated, base_picture


#  Core: Supervise image operation
def spvz_op(image_label):
    """
    Orchestrates the complete processing pipeline for a single input image.

    This function acts as the central coordinator, managing:
    - File existence verification
    - Image loading
    - Preprocessing and enhancement
    - Face and eye detection with validation
    - Eye region anonymization via blurring
    - Result persistence to disk
    - Visual feedback through on-screen display

    Args:
        image_label (str): The filename of the image to be processed (relative to the source directory).

    The function integrates all modular utilities into a robust, error-tolerant workflow,
    ensuring consistent execution and providing clear console feedback at each stage.
    It guarantees that only valid results are saved and displayed.
    """
    acc_pt = os.path.join(SRC_FOLDER, image_label)

    if not os.path.isfile(acc_pt):
        print("[SKIP] Missing entry: {}".format(acc_pt))
        return

    print("Beginning operation on: {}".format(image_label))
    access_data = ld_pic(acc_pt)

    tned_gray = opt_img(access_data)

    total_auth, operation_result = search_components(tned_gray, access_data)

    print("  Supervised {} authenticated components.".format(total_auth))

    access_end = os.path.join(DEST_FOLDER, "result_" + image_label)  # Changed to match report
    cv2.imwrite(access_end, operation_result)
    print("  Secured outcome in: {}".format(access_end))

    cv2.imshow("Operation Outcome - {}".format(image_label), operation_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#  Program Entry
if __name__ == "__main__":
    print("Activating anonymization operations...\n")
    for access_file in TARGET_IMAGES:
        spvz_op(access_file)
    print("\nOperations concluded. Inspect '{}' for outcomes.".format(DEST_FOLDER))