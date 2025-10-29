import cv2
import numpy as np
import matplotlib.pyplot as plt

# Create blank background
img = np.full((512, 512), 50, dtype=np.uint8)

# Draw filled white square and circle
cv2.rectangle(img, (300, 300), (400, 400), 255, -1)
cv2.circle(img, (150, 150), 50, 255, -1)

# Apply edge detectors
edges_canny = cv2.Canny(img, 100, 200)
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
edges_sobel = cv2.magnitude(sobelx, sobely)
edges_sobel = np.uint8(np.clip(edges_sobel, 0, 255))
edges_laplacian = cv2.Laplacian(img, cv2.CV_64F)
edges_laplacian = np.uint8(np.clip(np.abs(edges_laplacian), 0, 255))

# Display
titles = ['Original', 'Canny', 'Sobel', 'Laplacian']
images = [img, edges_canny, edges_sobel, edges_laplacian]

for i in range(4):
    plt.subplot(2,2,i+1), plt.imshow(images[i], cmap='gray')
    plt.title(titles[i]), plt.axis('off')
plt.show()
