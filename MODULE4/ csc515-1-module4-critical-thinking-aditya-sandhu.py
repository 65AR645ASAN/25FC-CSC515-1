import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the uploaded image
img_path = r"C:\Users\Aditya\Desktop\CSUDOCS\CSC515\25FC-CSC515-1\MODULE4\Mod4CT2.jpg"
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Chosen sigma for Gaussian
sigma = 1.0

# Define kernel sizes
kernels = [3, 5, 7]

# Store results
results = {}

for k in kernels:
    # Gaussian Blur
    gaussian = cv2.GaussianBlur(img_gray, (k, k), sigma)

    # Laplacian Filter
    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F, ksize=k)
    laplacian = cv2.convertScaleAbs(laplacian)

    # Gaussian + Laplacian
    gaussian_then_lap = cv2.Laplacian(gaussian, cv2.CV_64F, ksize=k)
    gaussian_then_lap = cv2.convertScaleAbs(gaussian_then_lap)

    results[k] = (gaussian, laplacian, gaussian_then_lap)

# Plot results in a 3x3 grid
fig, axes = plt.subplots(3, 3, figsize=(12, 12))

filter_names = ["Gaussian", "Laplacian", "Gaussian + Laplacian"]

for i, k in enumerate(kernels):
    for j, name in enumerate(filter_names):
        axes[i, j].imshow(results[k][j], cmap="gray")
        axes[i, j].set_title(f"{name} {k}x{k}")
        axes[i, j].axis("off")

plt.tight_layout()
plt.show()
