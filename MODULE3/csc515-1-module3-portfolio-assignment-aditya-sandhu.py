"""
Deepfake Creation Module
Option: Face Swap Demonstration Using OpenCV

Author: Aditya Sandhu
Date: 09/24/2025

Overview:
This program blends a source face (e.g., a personal selfie) onto a target image (e.g., an online photo of someone with arms crossed) to create a simple deepfake. It detects faces, resizes, masks, and seamlessly clones for natural integration, illustrating image blending in computer vision.
"""

import cv2
import numpy as np
import os

def load_photos(source_file, target_file):
    """Load and validate the source and target photographs."""
    if not os.path.exists(source_file):
        raise ValueError(f"Source file not found: {source_file}")
    if not os.path.exists(target_file):
        raise ValueError(f"Target file not found: {target_file}")
    source_photo = cv2.imread(source_file)
    target_photo = cv2.imread(target_file)
    if source_photo is None:
        raise ValueError(f"Failed to read source photo: {source_file} (check file integrity)")
    if target_photo is None:
        raise ValueError(f"Failed to read target photo: {target_file} (check file integrity)")
    return source_photo, target_photo

def convert_to_monochrome(image):
    """Transform the image to grayscale for feature detection."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def initialize_detector():
    """Set up the face detection model using pre-trained cascade."""
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def locate_faces(gray_image, detector):
    """Identify facial regions in the grayscale image."""
    return detector.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)

def extract_and_adjust_face(photo, face_coords, new_dimensions):
    """Crop the face area and scale it to fit the new size."""
    fx, fy, fw, fh = face_coords
    face_region = photo[fy:fy + fh, fx:fx + fw]
    return cv2.resize(face_region, new_dimensions)

def generate_blend_mask(height, width):
    """Produce an elliptical mask for smooth blending."""
    blend_mask = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.ellipse(blend_mask, (width // 2, height // 2), (width // 2, height // 2), 0, 0, 360, (255, 255, 255), -1)
    return blend_mask

def perform_seamless_blend(source_face, target_photo, mask, target_center):
    """Apply advanced cloning to integrate the source face into the target."""
    return cv2.seamlessClone(source_face, target_photo, mask, target_center, cv2.NORMAL_CLONE)

def overlay_label(image, label_text):
    """Insert descriptive text onto the image."""
    cv2.putText(image, label_text, (50, image.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Define file paths with absolute paths or relative to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
personal_selfie = os.path.join(script_dir, "bearded_selfie_in_car.jpeg")  # Your selfie (corrected extension)
online_image = os.path.join(script_dir, "online_arms_crossed.jpg")      # Online image

# Print current working directory and file paths for debugging
print(f"Script directory: {script_dir}")
print(f"Source image path: {personal_selfie}")
print(f"Target image path: {online_image}")

# Execute the blending process
try:
    source_photo, target_photo = load_photos(personal_selfie, online_image)
    source_mono = convert_to_monochrome(source_photo)
    target_mono = convert_to_monochrome(target_photo)

    face_detector = initialize_detector()

    source_detections = locate_faces(source_mono, face_detector)
    target_detections = locate_faces(target_mono, face_detector)

    if len(source_detections) == 0 or len(target_detections) == 0:
        raise ValueError("Unable to detect a face in one or both photos")

    # Select primary face from each
    source_coords = source_detections[0]
    target_coords = target_detections[0]
    target_width, target_height = target_coords[2], target_coords[3]

    adjusted_source_face = extract_and_adjust_face(source_photo, source_coords, (target_width, target_height))

    integration_mask = generate_blend_mask(target_height, target_width)

    blend_center = (target_coords[0] + target_width // 2, target_coords[1] + target_height // 2)
    blended_result = perform_seamless_blend(adjusted_source_face, target_photo, integration_mask, blend_center)

    overlay_label(blended_result, "Deepfake Blend Example")

    # Store the final image
    result_file = os.path.join(script_dir, "blended_deepfake_result.jpg")
    cv2.imwrite(result_file, blended_result)

    print(f"Generated deepfake saved to {result_file}")
except Exception as e:
    print(f"Error: {e}")