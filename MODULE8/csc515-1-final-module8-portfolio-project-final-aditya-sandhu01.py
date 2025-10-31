# --------------------------------------------------------------
#  Face-Anonymizer – Module 8 Portfolio
#  Author: Aditya Sandhu
# --------------------------------------------------------------
import cv2
import os

# ---------- CONFIGURATION ----------
FACE_MODEL      = 'haarcascade_frontalface_alt2.xml'
EYE_MODEL       = 'haarcascade_eye.xml'
INPUT_DIR       = 'portfolio-images'
OUTPUT_DIR      = 'detected-images'

IMAGE_LIST = [
    'animal-1.jpg',
    'group-front-standing-group-1.jpg',
    'full-body-single-person-male-1.jpg'
]

# ---------- HELPER: load a cascade ----------
def load_classifier(name, file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            "{} missing – download from OpenCV GitHub and place in this folder."
            .format(name)
        )
    clf = cv2.CascadeClassifier(file_path)
    if clf.empty():
        raise RuntimeError("Could not load {} classifier.".format(name))
    return clf

# Load the two classifiers
face_clf = load_classifier("Frontal-face", FACE_MODEL)
eye_clf  = load_classifier("Eye", EYE_MODEL)

print("\nClassifiers are ready.\n")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- HELPER: safe image read ----------
def read_image(full_path):
    img = cv2.imread(full_path)
    if img is None:
        raise IOError("Failed to read image: " + full_path)
    return img

# ---------- MAIN ANONYMIZATION ----------
def process_and_anonymize(file_name):
    src_path = os.path.join(INPUT_DIR, file_name)

    if not os.path.isfile(src_path):
        print("  [WARN] Image not found → {}".format(src_path))
        return

    print("\n  → Working on: {}".format(file_name))
    original = read_image(src_path)

    # ---- Convert & enhance contrast ----
    gray_img = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (7, 7), 0)
    clahe_obj = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_img = clahe_obj.apply(gray_img)

    # ---- Locate possible faces ----
    possible_faces = face_clf.detectMultiScale(
        gray_img,
        scaleFactor=1.05,
        minNeighbors=8,
        minSize=(40, 40),
        maxSize=(400, 400),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    confirmed = 0
    for fx, fy, fw, fh in possible_faces:
        face_region_color = original[fy:fy+fh, fx:fx+fw]
        face_region_gray  = gray_img[fy:fy+fh, fx:fx+fw]

        # ---- Look for eyes inside this region ----
        eye_boxes = eye_clf.detectMultiScale(
            face_region_gray,
            scaleFactor=1.1,
            minNeighbors=11,
            minSize=(15, 15)
        )

        if len(eye_boxes) > 0:                     # at least one eye → real face
            confirmed += 1

            # Red rectangle around the whole face
            cv2.rectangle(original, (fx, fy), (fx+fw, fy+fh), (0, 0, 255), 2)

            # Blur every detected eye
            for ex, ey, ew, eh in eye_boxes:
                eye_patch = face_region_color[ey:ey+eh, ex:ex+ew]
                blurred_patch = cv2.GaussianBlur(eye_patch, (23, 23), 30)
                face_region_color[ey:ey+eh, ex:ex+ew] = blurred_patch

    print("   • {} face(s) confirmed & anonymized".format(confirmed))

    # ---- Write result ----
    out_file = os.path.join(OUTPUT_DIR, "detected_" + file_name)
    cv2.imwrite(out_file, original)
    print("   • Saved → {}\n".format(out_file))

    # ---- Show on screen (press any key to continue) ----
    cv2.imshow("Anonymized Result – " + file_name, original)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ---------- EXECUTION ----------
if __name__ == "__main__":
    print("=== Starting anonymization pipeline ===\n")
    for img in IMAGE_LIST:
        process_and_anonymize(img)
    print("=== All done! Check folder: {} ===".format(OUTPUT_DIR))