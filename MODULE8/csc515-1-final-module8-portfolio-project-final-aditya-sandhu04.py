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
#  Utility: Preprocess image for detection
# --------------------------------------------------------------
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    enhancer = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return enhancer.apply(gray)

# --------------------------------------------------------------
#  Utility: Detect and confirm faces with eyes
# --------------------------------------------------------------
def detect_faces_and_eyes(preprocessed_gray, color_image):
    candidate_regions = face_detector.detectMultiScale(
        preprocessed_gray,
        scaleFactor=1.05,
        minNeighbors=8,
        minSize=(40, 40),
        maxSize=(400, 400),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    confirmed_faces = 0
    for (rx, ry, rw, rh) in candidate_regions:
        sub_color = color_image[ry:ry + rh, rx:rx + rw]
        sub_gray = preprocessed_gray[ry:ry + rh, rx:rx + rw]

        eye_locations = eye_detector.detectMultiScale(
            sub_gray,
            scaleFactor=1.1,
            minNeighbors=11,
            minSize=(15, 15)
        )

        if len(eye_locations) > 0:
            confirmed_faces += 1
            cv2.rectangle(color_image, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 2)

            for (ex, ey, ew, eh) in eye_locations:
                eye_section = sub_color[ey:ey + eh, ex:ex + ew]
                anonymized = cv2.GaussianBlur(eye_section, (23, 23), 30)
                sub_color[ey:ey + eh, ex:ex + ew] = anonymized

    return confirmed_faces, color_image

# --------------------------------------------------------------
#  Core: Handle single image processing
# --------------------------------------------------------------
def handle_image_processing(img_filename):
    input_path = os.path.join(SRC_FOLDER, img_filename)

    if not os.path.isfile(input_path):
        print("[SKIP] File does not exist: {}".format(input_path))
        return

    print("Starting process for: {}".format(img_filename))
    loaded_img = load_picture(input_path)

    enhanced_gray = preprocess_image(loaded_img)

    num_confirmed, processed_img = detect_faces_and_eyes(enhanced_gray, loaded_img)

    print("  Processed {} confirmed faces.".format(num_confirmed))

    result_path = os.path.join(DEST_FOLDER, "processed_" + img_filename)
    cv2.imwrite(result_path, processed_img)
    print("  Result saved to: {}".format(result_path))

    cv2.imshow("Processed Image - {}".format(img_filename), processed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --------------------------------------------------------------
#  Program Entry
# --------------------------------------------------------------
if __name__ == "__main__":
    print("Initiating image anonymization process...\n")
    for image_file in TARGET_IMAGES:
        handle_image_processing(image_file)
    print("\nCompleted all tasks. Outputs in '{}' folder.".format(DEST_FOLDER))