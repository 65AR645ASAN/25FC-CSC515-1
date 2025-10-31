#!/usr/bin/env python3
"""
VisualPrivacyShield – Face & Eye Obfuscator
Author: Aditya Sandhu
Description: Scans input images for human faces and eyes using Haar cascades,
then conceals detected eyes with Gaussian blur to preserve privacy.
"""

import cv2
import os

# --------------------------------------------------------------
#  Setup and Directory Configuration
# --------------------------------------------------------------
FACE_MODEL = 'haarcascade_frontalface_alt2.xml'
EYE_MODEL  = 'haarcascade_eye.xml'
SOURCE_DIR = 'portfolio-images'
OUTPUT_DIR = 'processed-results'

IMAGE_QUEUE = [
    'animal-1.jpg',
    'group-front-standing-group-1.jpg',
    'full-body-single-person-male-1.jpg'
]

# --------------------------------------------------------------
#  Model Loader with Error Handling
# --------------------------------------------------------------
def load_cascade(model_tag, xml_file):
    """Loads and validates Haar cascade XML model."""
    if not os.path.exists(xml_file):
        raise FileNotFoundError(
            f"Model file missing: {xml_file}. "
            "Please ensure Haar cascade XML files are in the project root."
        )

    model = cv2.CascadeClassifier(xml_file)
    if model.empty():
        raise RuntimeError(f"Unable to initialize {model_tag} cascade.")
    return model


# Initialize classifiers
face_cascade = load_cascade("Face Detector", FACE_MODEL)
eye_cascade  = load_cascade("Eye Detector", EYE_MODEL)

print("\nModel initialization completed successfully.\n")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------------------
#  Image Import Utility
# --------------------------------------------------------------
def read_image(file_route):
    """Loads an image from file and validates the read operation."""
    frame = cv2.imread(file_route)
    if frame is None:
        raise IOError(f"Cannot read image: {file_route}")
    return frame

# --------------------------------------------------------------
#  Image Preprocessing Function
# --------------------------------------------------------------
def preprocess_frame(frame):
    """Converts to grayscale, blurs, and equalizes histogram."""
    gray_view = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    smoothed = cv2.GaussianBlur(gray_view, (7, 7), 0)
    enhancer = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return enhancer.apply(smoothed)

# --------------------------------------------------------------
#  Detection and Obfuscation Function
# --------------------------------------------------------------
def detect_and_blur(grayscale, original):
    """Detects faces and blurs eye regions within detected zones."""
    potential_faces = face_cascade.detectMultiScale(
        grayscale,
        scaleFactor=1.05,
        minNeighbors=8,
        minSize=(40, 40),
        maxSize=(400, 400),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    valid_detections = 0
    for (x, y, w, h) in potential_faces:
        face_section = original[y:y + h, x:x + w]
        gray_crop = grayscale[y:y + h, x:x + w]

        detected_eyes = eye_cascade.detectMultiScale(
            gray_crop,
            scaleFactor=1.1,
            minNeighbors=11,
            minSize=(15, 15)
        )

        if len(detected_eyes) > 0:
            valid_detections += 1
            cv2.rectangle(original, (x, y), (x + w, y + h), (255, 0, 0), 2)

            for (ex, ey, ew, eh) in detected_eyes:
                section = face_section[ey:ey + eh, ex:ex + ew]
                blurred_zone = cv2.GaussianBlur(section, (23, 23), 30)
                face_section[ey:ey + eh, ex:ex + ew] = blurred_zone

    return valid_detections, original

# --------------------------------------------------------------
#  Main Processing Routine
# --------------------------------------------------------------
def process_images(image_file):
    """Executes the privacy-preserving pipeline on selected images."""
    input_path = os.path.join(SOURCE_DIR, image_file)

    if not os.path.isfile(input_path):
        print(f"[SKIP] File missing: {input_path}")
        return

    print(f"Processing initiated for: {image_file}")
    frame_data = read_image(input_path)
    normalized = preprocess_frame(frame_data)
    num_faces, output_frame = detect_and_blur(normalized, frame_data)

    print(f"  Faces processed: {num_faces}")

    save_path = os.path.join(OUTPUT_DIR, f"secured_{image_file}")
    cv2.imwrite(save_path, output_frame)
    print(f"  Result stored at: {save_path}")

    cv2.imshow(f"Output – {image_file}", output_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --------------------------------------------------------------
#  Entry Point
# --------------------------------------------------------------
if __name__ == "__main__":
    print("Launching Visual Privacy Shield...\n")
    for img_file in IMAGE_QUEUE:
        process_images(img_file)
    print(f"\nProcess completed. Check '{OUTPUT_DIR}' for secured results.")
