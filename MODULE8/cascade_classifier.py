# # cascade_classifier.py
# import cv2
# import os
#
# # ----------------------------------------------------------------------
# # CONFIG
# # ----------------------------------------------------------------------
# CASCADE_FILE   = 'haarcascade_frontalface_alt2.xml'   # must be in MODULE8
# SRC_FOLDER     = 'portfolio-images'                  # originals
# OUT_FOLDER     = 'detected-images'                   # where results go
#
# # The three images you must have in SRC_FOLDER
# IMAGES = [
#     'animal-1.jpg',                     # non-human → 0 faces
#     'group-front-standing-group-1.jpg', # many people
#     'full-body-single-person-male-1.jpg' # full-body, small face
# ]
#
# # ----------------------------------------------------------------------
# # Load cascade
# # ----------------------------------------------------------------------
# if not os.path.exists(CASCADE_FILE):
#     raise FileNotFoundError(
#         f"ERROR: '{CASCADE_FILE}' not found!\n"
#         f"Download it from:\n"
#         f"https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml\n"
#         f"And save it in: {os.getcwd()}"
#     )
#
# face_cascade = cv2.CascadeClassifier(CASCADE_FILE)
# if face_cascade.empty():
#     raise IOError(f"Failed to load cascade from '{CASCADE_FILE}'. File may be corrupted.")
#
# print(f"Loaded cascade: {CASCADE_FILE}\n")
#
# # ----------------------------------------------------------------------
# # Create output folder
# # ----------------------------------------------------------------------
# os.makedirs(OUT_FOLDER, exist_ok=True)
#
# # ----------------------------------------------------------------------
# # Helper: full path in the source folder
# # ----------------------------------------------------------------------
# def src_path(name):
#     return os.path.join(SRC_FOLDER, name)
#
# # ----------------------------------------------------------------------
# # Face detection
# # ----------------------------------------------------------------------
# def detect_and_save(image_name):
#     path = src_path(image_name)
#
#     if not os.path.exists(path):
#         print(f"ERROR: Image not found → {path}")
#         return
#
#     print(f"Processing: {image_name}")
#     img = cv2.imread(path)
#     if img is None:
#         print(f"ERROR: Could not read image → {image_name}")
#         return
#
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     # ---- preprocessing -------------------------------------------------
#     gray = cv2.GaussianBlur(gray, (5, 5), 0)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     gray = clahe.apply(gray)
#
#     # ---- detect faces --------------------------------------------------
#     faces = face_cascade.detectMultiScale(
#         gray,
#         scaleFactor=1.05,
#         minNeighbors=6,
#         minSize=(20, 20),
#         maxSize=(500, 500),
#         flags=cv2.CASCADE_SCALE_IMAGE
#     )
#     print(f"   → Found {len(faces)} face(s)")
#
#     # ---- draw red boxes ------------------------------------------------
#     for (x, y, w, h) in faces:
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
#
#     # ---- save to detected-images ---------------------------------------
#     out_name = f"detected_{image_name}"
#     out_path = os.path.join(OUT_FOLDER, out_name)
#     cv2.imwrite(out_path, img)
#     print(f"   → Saved: {out_path}\n")
#
#     # ---- show result ---------------------------------------------------
#     cv2.imshow('Detected Faces', img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#
# # ----------------------------------------------------------------------
# # Run on all images
# # ----------------------------------------------------------------------
# if __name__ == "__main__":
#     print("Starting face detection on 3 images...\n")
#     for img in IMAGES:
#         detect_and_save(img)
#     print(f"All done! Results are in folder: {OUT_FOLDER}")
#
#
# # cascade_classifier.py
# import cv2
# import os
#
# # ================================
# # CONFIG
# # ================================
# FACE_CASCADE_FILE = 'haarcascade_frontalface_alt2.xml'
# EYE_CASCADE_FILE  = 'haarcascade_eye.xml'           # NEW: for filtering false faces
# SRC_FOLDER        = 'portfolio-images'
# OUT_FOLDER        = 'detected-images'
#
# IMAGES = [
#     'animal-1.jpg',                     # non-human → 0 faces
#     'group-front-standing-group-1.jpg', # many people
#     'full-body-single-person-male-1.jpg' # small face
# ]
#
# # ================================
# # Load cascades
# # ================================
# def load_cascade(name, path):
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"{name} not found! Download from OpenCV GitHub.")
#     cascade = cv2.CascadeClassifier(path)
#     if cascade.empty():
#         raise IOError(f"Failed to load {name}")
#     return cascade
#
# face_cascade = load_cascade("Face cascade", FACE_CASCADE_FILE)
# eye_cascade  = load_cascade("Eye cascade",  EYE_CASCADE_FILE)
#
# print("Both cascades loaded successfully!\n")
# os.makedirs(OUT_FOLDER, exist_ok=True)
#
# # ================================
# # Face + Eye Validation Detection
# # ================================
# def detect_and_save(img_name):
#     path = os.path.join(SRC_FOLDER, img_name)
#     if not os.path.exists(path):
#         print(f"ERROR: Image not found → {path}")
#         return
#
#     print(f"Processing: {img_name}")
#     img = cv2.imread(path)
#     if img is None:
#         print(f"ERROR: Could not read image")
#         return
#
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray = cv2.GaussianBlur(gray, (7, 7), 0)  # Strong blur to kill animal texture
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
#     gray = clahe.apply(gray)
#
#     # --- Detect candidate faces ---
#     candidates = face_cascade.detectMultiScale(
#         gray,
#         scaleFactor=1.05,
#         minNeighbors=8,      # High = fewer false positives
#         minSize=(40, 40),    # Minimum face size (paws are smaller)
#         maxSize=(400, 400),
#         flags=cv2.CASCADE_SCALE_IMAGE
#     )
#
#     valid_faces = []
#     for (x, y, w, h) in candidates:
#         face_roi_gray = gray[y:y+h, x:x+w]
#         face_roi_color = img[y:y+h, x:x+w]
#
#         # --- Look for eyes inside the face region ---
#         eyes = eye_cascade.detectMultiScale(
#             face_roi_gray,
#             scaleFactor=1.1,
#             minNeighbors=11,
#             minSize=(15, 15)
#         )
#
#         # Only accept face if AT LEAST 1 eye is detected
#         if len(eyes) >= 1:
#             valid_faces.append((x, y, w, h))
#             # Draw red box
#             cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
#             # Optional: draw green boxes around eyes
#             for (ex, ey, ew, eh) in eyes:
#                 cv2.rectangle(face_roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 1)
#
#     print(f"   → Found {len(valid_faces)} VALID face(s) (with eyes)")
#
#     # Save result
#     out_path = os.path.join(OUT_FOLDER, f"detected_{img_name}")
#     cv2.imwrite(out_path, img)
#     print(f"   → Saved: {out_path}\n")
#
#     cv2.imshow('Detected Faces (Eyes Confirmed)', img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
# # ================================
# # Run on all images
# # ================================
# if __name__ == "__main__":
#     print("Starting SMART face detection (eyes required)...\n")
#     for img in IMAGES:
#         detect_and_save(img)
#     print(f"DONE! Results in: {OUT_FOLDER}")
#
# # cascade_classifier.py
# import cv2
# import os
#
# # ================================
# # CONFIG
# # ================================
# FACE_CASCADE_FILE = 'haarcascade_frontalface_alt2.xml'
# EYE_CASCADE_FILE  = 'haarcascade_eye.xml'  # Download this!
# SRC_FOLDER        = 'portfolio-images'
# OUT_FOLDER        = 'detected-images'
#
# IMAGES = [
#     'animal-1.jpg',                     # non-human → 0 faces
#     'group-front-standing-group-1.jpg', # many people
#     'full-body-single-person-male-1.jpg' # small face
# ]
#
# # ================================
# # Load cascades
# # ================================
# def load_cascade(name, path):
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"{name} not found! Download from OpenCV GitHub.")
#     cascade = cv2.CascadeClassifier(path)
#     if cascade.empty():
#         raise IOError(f"Failed to load {name}")
#     return cascade
#
# face_cascade = load_cascade("Face cascade", FACE_CASCADE_FILE)
# eye_cascade  = load_cascade("Eye cascade",  EYE_CASCADE_FILE)
#
# print("Both cascades loaded successfully!\n")
# os.makedirs(OUT_FOLDER, exist_ok=True)
#
# # ================================
# # Face + Eye Validation Detection
# # ================================
# def detect_and_save(img_name):
#     path = os.path.join(SRC_FOLDER, img_name)
#     if not os.path.exists(path):
#         print(f"ERROR: Image not found → {path}")
#         return
#
#     print(f"Processing: {img_name}")
#     img = cv2.imread(path)
#     if img is None:
#         print(f"ERROR: Could not read image")
#         return
#
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray = cv2.GaussianBlur(gray, (7, 7), 0)  # Strong blur to kill animal texture
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
#     gray = clahe.apply(gray)
#
#     # --- Detect candidate faces ---
#     candidates = face_cascade.detectMultiScale(
#         gray,
#         scaleFactor=1.05,
#         minNeighbors=8,      # High = fewer false positives
#         minSize=(40, 40),    # Minimum face size (paws are smaller)
#         maxSize=(400, 400),
#         flags=cv2.CASCADE_SCALE_IMAGE
#     )
#
#     valid_faces = []
#     for (x, y, w, h) in candidates:
#         face_roi_gray = gray[y:y+h, x:x+w]
#         face_roi_color = img[y:y+h, x:x+w]
#
#         # --- Look for eyes inside the face region ---
#         eyes = eye_cascade.detectMultiScale(
#             face_roi_gray,
#             scaleFactor=1.1,
#             minNeighbors=11,
#             minSize=(15, 15)
#         )
#
#         # Only accept face if AT LEAST 1 eye is detected
#         if len(eyes) >= 1:
#             valid_faces.append((x, y, w, h))
#             # Draw red box
#             cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
#             # Optional: draw green boxes around eyes (comment out if not needed)
#             # for (ex, ey, ew, eh) in eyes:
#             #     cv2.rectangle(face_roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 1)
#
#     print(f"   → Found {len(valid_faces)} VALID face(s) (with eyes)")
#
#     # Save result
#     out_path = os.path.join(OUT_FOLDER, f"detected_{img_name}")
#     cv2.imwrite(out_path, img)
#     print(f"   → Saved: {out_path}\n")
#
#     cv2.imshow('Detected Faces (Eyes Confirmed)', img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
# # ================================
# # Run on all images
# # ================================
# if __name__ == "__main__":
#     print("Starting SMART face detection (eyes required)...\n")
#     for img in IMAGES:
#         detect_and_save(img)
#     print(f"DONE! Results in: {OUT_FOLDER}")

# cascade_classifier.py - FULL ASSIGNMENT SOLUTION
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