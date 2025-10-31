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
#  Utility: Prepare image for analysis
# --------------------------------------------------------------
def prepare_input(picture_info):
    mono_tone = cv2.cvtColor(picture_info, cv2.COLOR_BGR2GRAY)
    filtered = cv2.GaussianBlur(mono_tone, (7, 7), 0)
    adjuster = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return adjuster.apply(filtered)

# --------------------------------------------------------------
#  Utility: Locate and verify key elements
# --------------------------------------------------------------
def locate_elements(adjusted_mono, source_picture):
    suspected_zones = face_detector.detectMultiScale(
        adjusted_mono,
        scaleFactor=1.05,
        minNeighbors=8,
        minSize=(40, 40),
        maxSize=(400, 400),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    approved_total = 0
    for (pos_x, pos_y, dim_w, dim_h) in suspected_zones:
        zone_rgb = source_picture[pos_y:pos_y + dim_h, pos_x:pos_x + dim_w]
        zone_mono = adjusted_mono[pos_y:pos_y + dim_h, pos_x:pos_x + dim_w]

        element_spots = eye_detector.detectMultiScale(
            zone_mono,
            scaleFactor=1.1,
            minNeighbors=11,
            minSize=(15, 15)
        )

        if len(element_spots) > 0:
            approved_total += 1
            cv2.rectangle(source_picture, (pos_x, pos_y), (pos_x + dim_w, pos_y + dim_h), (0, 0, 255), 2)

            for (sx, sy, sw, sh) in element_spots:
                element_part = zone_rgb[sy:sy + sh, sx:sx + sw]
                obscured_part = cv2.GaussianBlur(element_part, (23, 23), 30)
                zone_rgb[sy:sy + sh, sx:sx + sw] = obscured_part

    return approved_total, source_picture

# --------------------------------------------------------------
#  Core: Control image workflow
# --------------------------------------------------------------
def control_image_flow(file_label):
    entry_point = os.path.join(SRC_FOLDER, file_label)

    if not os.path.isfile(entry_point):
        print("[SKIP] Absent file: {}".format(entry_point))
        return

    print("Commencing workflow for: {}".format(file_label))
    entry_data = load_picture(entry_point)

    tuned_mono = prepare_input(entry_data)

    count_approved, end_result = locate_elements(tuned_mono, entry_data)

    print("  Handled {} approved elements.".format(count_approved))

    end_point = os.path.join(DEST_FOLDER, "result_" + file_label)
    cv2.imwrite(end_point, end_result)
    print("  Preserved outcome at: {}".format(end_point))

    cv2.imshow("End Result - {}".format(file_label), end_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --------------------------------------------------------------
#  Program Entry
# --------------------------------------------------------------
if __name__ == "__main__":
    print("Starting anonymization sequence...\n")
    for entry_file in TARGET_IMAGES:
        control_image_flow(entry_file)
    print("\nSequence ended. Check outputs in '{}'.".format(DEST_FOLDER))