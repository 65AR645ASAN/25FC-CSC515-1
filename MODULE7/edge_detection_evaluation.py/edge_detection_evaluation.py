import cv2
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Step 1: Create synthetic image
img = np.zeros((256, 256), dtype=np.uint8)
cv2.rectangle(img, (60, 60), (140, 140), 200, -1)   # filled square
cv2.circle(img, (180, 170), 35, 120, -1)            # filled circle

# Ground truth edges (for evaluation)
gt = cv2.Canny(img, 50, 150)

# Step 2: Apply edge detectors
edges_canny = cv2.Canny(img, 50, 150)
edges_sobel = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=3)
edges_sobel = cv2.convertScaleAbs(edges_sobel)
edges_laplacian = cv2.Laplacian(img, cv2.CV_64F)
edges_laplacian = cv2.convertScaleAbs(edges_laplacian)

# Step 3: Add noise and re-test
noisy = img + np.random.normal(0, 15, img.shape).astype(np.uint8)
edges_canny_noisy = cv2.Canny(noisy, 50, 150)

# Step 4: Simple evaluation (precision, recall, F1)
def evaluate(det, gt):
    det_bin = (det > 0).astype(int).flatten()
    gt_bin = (gt > 0).astype(int).flatten()
    p = precision_score(gt_bin, det_bin)
    r = recall_score(gt_bin, det_bin)
    f1 = f1_score(gt_bin, det_bin)
    return p, r, f1

p, r, f1 = evaluate(edges_canny, gt)
print(f"Canny (clean): Precision={p:.2f}, Recall={r:.2f}, F1={f1:.2f}")

p, r, f1 = evaluate(edges_canny_noisy, gt)
print(f"Canny (noisy): Precision={p:.2f}, Recall={r:.2f}, F1={f1:.2f}")

# Step 5: Display results
titles = ['Original', 'Canny', 'Sobel', 'Laplacian']
images = [img, edges_canny, edges_sobel, edges_laplacian]
for i in range(4):
    plt.subplot(1, 4, i+1), plt.imshow(images[i], cmap='gray')
    plt.title(titles[i]), plt.axis('off')
plt.show()
