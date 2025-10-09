import cv2
import matplotlib.pyplot as plt

# Load the clear image
img = cv2.imread("383A4F5F-FE44-4354-97E1-FF9C15958685.jpeg")

# Apply Gaussian Blur with different kernel sizes
blur3 = cv2.GaussianBlur(img, (3,3), 0)
blur5 = cv2.GaussianBlur(img, (5,5), 0)
blur7 = cv2.GaussianBlur(img, (7,7), 0)

# Display results side by side
titles = ["Original", "Gaussian Blur 3x3", "Gaussian Blur 5x5", "Gaussian Blur 7x7"]
images = [img, blur3, blur5, blur7]

plt.figure(figsize=(12,6))
for i in range(4):
    plt.subplot(1,4,i+1)
    plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
