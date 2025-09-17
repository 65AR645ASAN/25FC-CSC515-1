"""
CSC515 - Module 1 Portfolio Milestone
Author: Aditya Sandhu

This script demonstrates the use of OpenCV for basic
computer vision tasks. The workflow includes:
1. Locating and importing a sample image of a brain.
2. Displaying the image to verify OpenCV installation.
3. Saving a duplicate copy of the image onto the user's Desktop.

By restructuring file paths with Python's os library,
the script remains portable across operating systems.
"""

import cv2
import os

def main():
    """
    Executes the OpenCV workflow:
    - Reads a brain image from the project directory
    - Shows the image in a pop-up window
    - Writes a duplicate copy to the Desktop
    """

    # Build the relative file path to the source image
    brain_img_path = os.path.join("..", "images", "shutterstock93075775--250.jpg")

    # Load the source image into memory as a pixel matrix
    brain_matrix = cv2.imread(brain_img_path)

    # Confirm that the file was successfully read
    if brain_matrix is None:
        print(f"[ERROR] Unable to load file from: {brain_img_path}. "
              f"Verify that the path and image filename are correct.")
        return

    # Display the image in a resizable window
    cv2.imshow("N C I", brain_matrix)

    # Create an output path on the Desktop for saving a duplicate
    desktop_output = os.path.join(os.path.expanduser("~"),
                                  "Desktop",
                                  "brain_copy.jpg")

    # Write the duplicate image to the chosen location
    cv2.imwrite(desktop_output, brain_matrix)
    print(f"[INFO] A duplicate has been saved to: {desktop_output}")

    # Wait for any key press, then close the image window
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
