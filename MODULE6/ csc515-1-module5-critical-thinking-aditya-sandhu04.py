# Importing required modules for handling images and visualizations
import cv2 as vision_lib  # Handles image reading and transformations
import numpy as array_lib  # Supports array-based operations
import matplotlib.pyplot as plot_lib  # Enables creating and saving plots
import os  # Assists with file path management
import sys  # Allows for program exit on errors


# Function to load an image using a full path to avoid directory issues
def load_source_image(file_name):
    # Build the complete path relative to this script's location
    full_path = os.path.join(os.path.dirname(__file__), file_name)
    loaded_img = vision_lib.imread(full_path)
    # Check if loading failed and exit with a message if so
    if loaded_img is None:
        print(f"Error: Could not load the image at '{full_path}'. Verify the file exists and path is correct.")
        sys.exit(1)  # Terminate the script on failure
    return loaded_img


# Function to convert image to single-channel format for easier handling
def convert_to_monochrome(source_img):
    # Reduce color channels to one for simplified analysis
    return vision_lib.cvtColor(source_img, vision_lib.COLOR_BGR2GRAY)


# Function to reduce image artifacts with a filter
def apply_noise_reduction(mono_img):
    # Use a larger kernel for broader smoothing effect
    return vision_lib.GaussianBlur(mono_img, (7, 7), 0)


# Function to binarize the image based on local variations
def apply_local_binarization(filtered_img):
    # Adjust parameters for custom threshold behavior
    return vision_lib.adaptiveThreshold(
        filtered_img,  # Filtered input
        255,  # Peak intensity value
        vision_lib.ADAPTIVE_THRESH_GAUSSIAN_C,  # Weighted local method
        vision_lib.THRESH_BINARY,  # Output as black/white
        11,  # Local area size
        2  # Adjustment factor
    )


# Function to generate and store a combined view of processing stages
def generate_and_store_visual_summary(stages_list, labels_list, output_file):
    # Initialize a wide figure for horizontal arrangement
    plot_lib.figure(figsize=(18, 6))
    for pos in range(len(stages_list)):
        plot_lib.subplot(1, len(stages_list), pos + 1)
        if pos == 0:
            # Adjust color order for accurate rendering
            plot_lib.imshow(vision_lib.cvtColor(stages_list[pos], vision_lib.COLOR_BGR2RGB))
        else:
            # Render in monochrome for processed stages
            plot_lib.imshow(stages_list[pos], cmap='gray')
        plot_lib.title(labels_list[pos])
        plot_lib.axis('off')
    # Optimize spacing and save with high resolution
    plot_lib.tight_layout()
    plot_lib.savefig(output_file, dpi=400)
    # No on-screen display to keep script lightweight
    plot_lib.close()


# Function to store the binarized result separately
def store_final_binary(result_img, output_file):
    vision_lib.imwrite(output_file, result_img)


# Main execution block
if __name__ == "__main__":
    # Define the source file name (update if actual name differs)
    source_file = 'lllumination_scene.jpeg'

    # Load and process the image through stages
    original = load_source_image(source_file)
    mono = convert_to_monochrome(original)
    filtered = apply_noise_reduction(mono)
    binarized = apply_local_binarization(filtered)

    # Collect stages for visualization
    processing_stages = [original, mono, binarized]
    stage_labels = ['Original Scene Capture', 'Monochrome Transformation', 'Binarized Outcome']

    # Create and save the summary image
    generate_and_store_visual_summary(processing_stages, stage_labels, 'stages_summary.jpg')

    # Save the binary version alone
    store_final_binary(binarized, 'final_binary_version.jpg')

    # Custom note: This approach focuses on handling uneven lighting in scenes, ideal for variable conditions.
    # Ensure vision_lib (OpenCV) and plot_lib (Matplotlib) are available in your setup.
    # No external dependencies beyond these; runs self-contained.