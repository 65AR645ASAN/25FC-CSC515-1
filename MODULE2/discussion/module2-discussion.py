import cv2
import matplotlib.pyplot as plt

# Load the image with full path or ensure it's in the same directory
img = cv2.imread("shutterstock227361781--125.jpg")

if img is None:
    print("Error: Could not load image. Check the file path!")
else:
    # Convert BGR (default in OpenCV) to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Show the image
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.show()

    # Examine the pixel matrix
    print("Pixel matrix shape:", img.shape)  # (height, width, channels)
