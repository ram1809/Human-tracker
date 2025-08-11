import cv2
import numpy as np
import glob
import os

# === Settings ===
CHECKERBOARD = (8, 5)  # number of internal corners (columns, rows)
square_size = 10.0  # real-world size of a square (e.g., in mm)

# === Prepare object points ===
objp = np.zeros((CHECKERBOARD[1] * CHECKERBOARD[0], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# === Load all calibration images ===
image_folder = r"C:\Windows\System32\Dissertation\Images\intrinsic\captured_images\Cam1"
images = glob.glob(os.path.join(image_folder, "*.jpg"))  # adjust if images are PNG

#to check if the files does exist
image_folder = r"C:\Windows\System32\Dissertation\Images\intrinsic\captured_images\Cam1"
exts = ("*.jpg","*.JPG","*.jpeg","*.JPEG","*.png","*.PNG")

# Gather images
images = []
for e in exts:
    images += glob.glob(os.path.join(image_folder, e))

print(f"[INFO] Folder exists: {os.path.isdir(image_folder)}")
print(f"[INFO] Images found: {len(images)}")
if not images:
    print("[ERROR] No images matched; try copying a few files to a user folder like C:\\Users\\<you>\\Pictures\\calib and rerun.")
    raise SystemExit
for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"Could not read {fname}, skipping...")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the checkerboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        imgpoints.append(corners2)

        # Optional: draw and show the corners
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('Corners', img)
        cv2.waitKey(100)
    else:
        print(f"Checkerboard not found in {fname}")

cv2.destroyAllWindows()

# === Calibration ===
if len(objpoints) > 0:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
)