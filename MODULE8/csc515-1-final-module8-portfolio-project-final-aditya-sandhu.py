import cv2
import os

# ================================
# CONFIG
# ================================
FACE_CASCADE_FILE = 'haarcascade_frontalface_alt2.xml'
EYE_CASCADE_FILE  = 'haarcascade_eye.xml'
SRC_FOLDER        = 'portfolio-images'
OUT_FOLDER        = 'detected-images'

IMAGES = [
    'animal-1.jpg',
    'group-front-standing-group-1.jpg',
    'full-body-single-person-male-1.jpg'
]

# ================================
# Load cascades
# ================================
def load_cascade(name, path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found! Download from OpenCV GitHub.")
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        raise IOError(f"Failed to load {name}")
    return cascade

face_cascade = load_cascade("Face cascade", FACE_CASCADE_FILE)
eye_cascade  = load_cascade("Eye cascade",  EYE_CASCADE_FILE)

print("Cascades loaded.\n")
os.makedirs(OUT_FOLDER, exist_ok=True)

# ================================
# Face Detection + Eye Blurring
# ================================
def anonymize_image(img_name):
    path = os.path.join(SRC_FOLDER, img_name)
    if not os.path.exists(path):
        print(f"ERROR: Image not found → {path}")
        return

    print(f"Processing: {img_name}")
    img = cv2.imread(path)
    if img is None:
        print("ERROR: Failed to read image")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=8,
        minSize=(40, 40),
        maxSize=(400, 400),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    valid_count = 0
    for (x, y, w, h) in faces:
        face_roi = img[y:y+h, x:x+w]
        face_gray = gray[y:y+h, x:x+w]

        # Detect eyes
        eyes = eye_cascade.detectMultiScale(
            face_gray,
            scaleFactor=1.1,
            minNeighbors=11,
            minSize=(15, 15)
        )

        if len(eyes) >= 1:
            valid_count += 1
            # Draw red box around face
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

            # Blur each eye
            for (ex, ey, ew, eh) in eyes:
                eye = face_roi[ey:ey+eh, ex:ex+ew]
                blurred = cv2.GaussianBlur(eye, (23, 23), 30)
                face_roi[ey:ey+eh, ex:ex+ew] = blurred

    print(f"   → {valid_count} face(s) anonymized (eyes blurred)")

    out_path = os.path.join(OUT_FOLDER, f"detected_{img_name}")
    cv2.imwrite(out_path, img)
    print(f"   → Saved: {out_path}\n")

    cv2.imshow('Anonymized (Eyes Blurred)', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ================================
# Run
# ================================
if __name__ == "__main__":
    print("Starting face detection + eye blurring...\n")
    for img in IMAGES:
        anonymize_image(img)
    print(f"DONE! Check: {OUT_FOLDER}")