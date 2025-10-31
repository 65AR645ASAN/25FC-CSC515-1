#!/usr/bin/env python3
"""
Face-Anonymizer – Module 8 Portfolio Project
Author: Aditya Sandhu
Description: Detects human faces, confirms with eye detection, and blurs eyes.
"""

import cv2
import os

# --------------------------------------------------------------
#  Configuration Constants
# --------------------------------------------------------------
FACE_XML       = 'haarcascade_frontalface_alt2.xml'
EYE_XML        = 'haarcascade_eye.xml'
SRC_FOLDER     = 'portfolio-images'
DEST_FOLDER    = 'detected-images'

TARGET_IMAGES = [
    'animal-1.jpg',
    'group-front-standing-group-1.jpg',
    'full-body-single-person-male-1.jpg'
]

# --------------------------------------------------------------
#  Utility: Load OpenCV classifier with validation
# --------------------------------------------------------------
def get_classifier(model_name, xml_path):
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
eye_detector  = get_classifier("eye", EYE_XML)

print("\nAll detection models loaded successfully.\n")
os.makedirs(DEST_FOLDER, exist_ok=True)

# --------------------------------------------------------------
#  Utility: Safe image loader
# --------------------------------------------------------------
def load_picture(file_path):
    picture = cv2.imread(file_path)
    if picture is None:
        raise IOError("Unable to load image from: {}".format(file_path))
    return picture

# --------------------------------------------------------------
#  Utility: Enhance image for better detection
# --------------------------------------------------------------
def enhance_frame(picture_data):
    grayscale = cv2.cvtColor(picture_data, cv2.COLOR_BGR2GRAY)
    smoothed = cv2.GaussianBlur(grayscale, (7, 7), 0)
    contrast_booster = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return contrast_booster.apply(smoothed)

# --------------------------------------------------------------
#  Utility: Identify and validate facial features
# --------------------------------------------------------------
def identify_features(enhanced_grayscale, original_picture):
    potential_areas = face_detector.detectMultiScale(
        enhanced_grayscale,
        scaleFactor=1.05,
        minNeighbors=8,
        minSize=(40, 40),
        maxSize=(400, 400),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    verified_count = 0
    for (coord_x, coord_y, width, height) in potential_areas:
        area_color = original_picture[coord_y:coord_y + height, coord_x:coord_x + width]
        area_gray = enhanced_grayscale[coord_y:coord_y + height, coord_x:coord_x + width]

        feature_positions = eye_detector.detectMultiScale(
            area_gray,
            scaleFactor=1.1,
            minNeighbors=11,
            minSize=(15, 15)
        )

        if len(feature_positions) > 0:
            verified_count += 1
            cv2.rectangle(original_picture, (coord_x, coord_y), (coord_x + width, coord_y + height), (0, 0, 255), 2)

            for (fx, fy, fw, fh) in feature_positions:
                feature_zone = area_color[fy:fy + fh, fx:fx + fw]
                masked_zone = cv2.GaussianBlur(feature_zone, (23, 23), 30)
                area_color[fy:fy + fh, fx:fx + fw] = masked_zone

    return verified_count, original_picture

# --------------------------------------------------------------
#  Core: Manage image handling and output
# --------------------------------------------------------------
def manage_image_task(image_filename):
    source_location = os.path.join(SRC_FOLDER, image_filename)

    if not os.path.isfile(source_location):
        print("[SKIP] No such file: {}".format(source_location))
        return

    print("Initiating task for: {}".format(image_filename))
    input_data = load_picture(source_location)

    boosted_gray = enhance_frame(input_data)

    total_verified, final_output = identify_features(boosted_gray, input_data)

    print("  Managed {} verified items.".format(total_verified))

    save_location = os.path.join(DEST_FOLDER, "output_" + image_filename)
    cv2.imwrite(save_location, final_output)
    print("  Stored result at: {}".format(save_location))

    cv2.imshow("Final Output - {}".format(image_filename), final_output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --------------------------------------------------------------
#  Program Entry
# --------------------------------------------------------------
if __name__ == "__main__":
    print("Beginning anonymization workflow...\n")
    for target_file in TARGET_IMAGES:
        manage_image_task(target_file)
    print("\nWorkflow finished. Results stored in '{}'.".format(DEST_FOLDER))