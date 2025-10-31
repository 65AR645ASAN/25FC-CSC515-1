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
#  Core: Process one image
# --------------------------------------------------------------
def anonymize_single_image(img_name):
    full_input_path = os.path.join(SRC_FOLDER, img_name)

    if not os.path.isfile(full_input_path):
        print("  [WARNING] Skipping missing file: {}".format(full_input_path))
        return

    print("\n  Processing → {}".format(img_name))
    source_img = load_picture(full_input_path)

    # --- Grayscale conversion and enhancement ---
    gray_frame = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (7, 7), 0)
    enhancer = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_frame = enhancer.apply(gray_frame)

    # --- Step 1: Candidate face regions ---
    candidate_rects = face_detector.detectMultiScale(
        gray_frame,
        scaleFactor=1.05,
        minNeighbors=8,
        minSize=(40, 40),
        maxSize=(400, 400),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    valid_face_count = 0
    for face_x, face_y, face_w, face_h in candidate_rects:
        # Extract face sub-images
        face_color_sub = source_img[face_y:face_y + face_h, face_x:face_x + face_w]
        face_gray_sub  = gray_frame[face_y:face_y + face_h, face_x:face_x + face_w]

        # --- Step 2: Eye detection within face ---
        eye_regions = eye_detector.detectMultiScale(
            face_gray_sub,
            scaleFactor=1.1,
            minNeighbors=11,
            minSize=(15, 15)
        )

        # Confirm only if at least one eye is found
        if len(eye_regions) > 0:
            valid_face_count += 1

            # Draw red boundary around confirmed face
            cv2.rectangle(
                source_img,
                (face_x, face_y),
                (face_x + face_w, face_y + face_h),
                (0, 0, 255), 2
            )

            # Blur each detected eye
            for ex, ey, ew, eh in eye_regions:
                eye_area = face_color_sub[ey:ey + eh, ex:ex + ew]
                blurred_eye = cv2.GaussianBlur(eye_area, (23, 23), 30)
                face_color_sub[ey:ey + eh, ex:ex + ew] = blurred_eye

    # --- Report and save ---
    print("   Confirmed {} human face(s) with anonymized eyes.".format(valid_face_count))

    output_filename = "detected_" + img_name
    output_path = os.path.join(DEST_FOLDER, output_filename)
    cv2.imwrite(output_path, source_img)
    print("   Output written to: {}\n".format(output_path))

    # --- Visual feedback ---
    cv2.imshow("Anonymized – {}".format(img_name), source_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --------------------------------------------------------------
#  Execution Entry Point
# --------------------------------------------------------------
if __name__ == "__main__":
    print("=== Launching Face Anonymization System ===\n")
    for current_image in TARGET_IMAGES:
        anonymize_single_image(current_image)
    print("=== Processing Complete – View results in '{}' ===\n".format(DEST_FOLDER))