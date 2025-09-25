"""
Deepfake Creation Module
Option: Face Swap Demonstration Using OpenCV

Author: Aditya Sandhu
Date: 09/24/2025

"""

import cv2
import numpy as np
import os

def fetch_images(origin_img_path, destination_img_path):
    """Retrieve and confirm the origin and destination images are accessible."""
    # Verify existence of origin image file
    if not os.path.exists(origin_img_path):
        raise ValueError(f"Origin image missing at: {origin_img_path}")
    # Verify existence of destination image file
    if not os.path.exists(destination_img_path):
        raise ValueError(f"Destination image missing at: {destination_img_path}")
    # Attempt to read origin image into memory
    origin_img = cv2.imread(origin_img_path)
    # Attempt to read destination image into memory
    destination_img = cv2.imread(destination_img_path)
    # Check if origin image loaded successfully, else error
    if origin_img is None:
        raise ValueError(f"Unable to parse origin image at: {origin_img_path} (verify format or corruption)")
    # Check if destination image loaded successfully, else error
    if destination_img is None:
        raise ValueError(f"Unable to parse destination image at: {destination_img_path} (verify format or corruption)")
    return origin_img, destination_img

def prepare_grayscale(picture):
    """Convert the provided picture to a single-channel grayscale version for analysis."""
    # Use color conversion to strip RGB and retain intensity
    return cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)

def setup_recognizer():
    """Initialize the recognition tool with a standard pretrained model for frontal views."""
    # Load the XML model file for detection patterns
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def find_facial_areas(mono_picture, recognizer):
    """Scan the monochrome picture to locate potential facial zones."""
    # Apply multi-scale scanning with specified scaling and neighbor thresholds
    return recognizer.detectMultiScale(mono_picture, scaleFactor=1.3, minNeighbors=5)

def extract_and_adjust_face(photo, face_coords, new_dimensions):
    """Crop the face area and scale it to fit the new size."""
    fx, fy, fw, fh = face_coords
    face_region = photo[fy:fy + fh, fx:fx + fw]
    return cv2.resize(face_region, new_dimensions)


def create_fusion_overlay(vertical_size, horizontal_size):
    """Build a smooth overlay shape for fusing image elements elliptically."""
    # Initialize a blank array matching the dimensions
    fusion_overlay = np.zeros((vertical_size, horizontal_size, 3), dtype=np.uint8)
    # Draw a filled ellipse to cover the area
    cv2.ellipse(fusion_overlay, (horizontal_size // 2, vertical_size // 2), (horizontal_size // 2, vertical_size // 2), 0, 0, 360, (255, 255, 255), -1)
    return fusion_overlay


def perform_seamless_blend(source_face, target_photo, mask, target_center):
    """Apply advanced cloning to integrate the source face into the target."""
    return cv2.seamlessClone(source_face, target_photo, mask, target_center, cv2.NORMAL_CLONE)


def apply_annotation(picture, annotation_content):
    """Add a text label to the picture at a fixed position."""
    # Position text near the bottom with chosen font and color
    cv2.putText(picture, annotation_content, (50, picture.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Establish paths using absolute or script-relative locations
code_location = os.path.dirname(os.path.abspath(__file__))
user_portrait = os.path.join(code_location, "bearded_selfie_in_car.jpeg")  # Path to personal portrait (updated file type)
web_portrait = os.path.join(code_location, "online_arms_crossed.jpg")     # Path to sourced web portrait

# Output the working location and image paths for verification
print(f"Code location: {code_location}")
print(f"User portrait path: {user_portrait}")
print(f"Web portrait path: {web_portrait}")


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
    source_photo, target_photo = fetch_images(personal_selfie, online_image)
    source_mono = prepare_grayscale(source_photo)
    target_mono = prepare_grayscale(target_photo)

    face_detector = setup_recognizer()

    source_detections = extract_and_adjust_face(source_mono, face_detector)
    target_detections = extract_and_adjust_face(target_mono, face_detector)

    if len(source_detections) == 0 or len(target_detections) == 0:
        raise ValueError("Unable to detect a face in one or both photos")

    # Select primary face from each
    source_coords = source_detections[0]
    target_coords = target_detections[0]
    target_width, target_height = target_coords[2], target_coords[3]

    adjusted_source_face = extract_and_adjust_face(source_photo, source_coords, (target_width, target_height))

    integration_mask = create_fusion_overlay(target_height, target_width)

    blend_center = (target_coords[0] + target_width // 2, target_coords[1] + target_height // 2)
    blended_result = perform_seamless_blend(adjusted_source_face, target_photo, integration_mask, blend_center)

    apply_annotation(blended_result, "Deepfake Blend Example")

    # Store the final image
    result_file = os.path.join(script_dir, "blended_deepfake_result.jpg")
    cv2.imwrite(result_file, blended_result)

    print(f"Generated deepfake saved to {result_file}")
except Exception as e:
    print(f"Error: {e}")