import cv2
import os

# Corrected path (relative to your script location)
image_path = os.path.join("..", "images", "shutterstock93075775--250.jpg")

# Read the image
image = cv2.imread(image_path)

# Check if the image was loaded
if image is None:
    print(f"Error: Could not load image from {image_path}. Check the file path.")
else:
    # Display the image in a window
    cv2.imshow("Brain Image", image)

    # Path to save the copy (update for your OS)
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "brain_copy.jpg")
    cv2.imwrite(desktop_path, image)
    print(f"Image successfully saved to: {desktop_path}")

    # Wait for a key press and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()
