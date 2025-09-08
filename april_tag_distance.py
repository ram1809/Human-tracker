import cv2
import numpy as np
from pupil_apriltags import Detector

# === SETTINGS ===
tag_size = 0.45  # in meters

# Load intrinsic parameters (from .npz file)
intrinsics = np.load(r"C:\Windows\System32\Dissertation\Images\intrinsic\intrinsic_params_for_extrinsic\fisheye_intrinsics_Cam0.npz")  # or your calibrated .npz
K = intrinsics['K']
D = intrinsics['D']

# Initialize detector
detector = Detector(families='tagStandard41h12')

# Load an image containing AprilTag
image_path = "C:\\Windows\\System32\\Dissertation\\New_folder\\captured_images\\Camera0\\image_019.jpg"
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect tags
tags = detector.detect(gray, estimate_tag_pose=True, camera_params=(K[0, 0], K[1, 1], K[0, 2], K[1, 2]), tag_size=tag_size)

# Process first detected tag
if tags:
    tag = tags[0]
    
    # Get translation (position) vector
    tvec = tag.pose_t  # Shape: (3, 1)
    
    # Get distance from camera to tag
    distance = np.linalg.norm(tvec)
    
    print(f"Translation Vector (T):\n{tvec}")
    print(f"Distance to AprilTag: {distance:.3f} meters")
    
    # Optional: draw axis
    corners = tag.corners.astype(int)
    cv2.polylines(image, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.putText(image, f"{distance:.2f} m", tuple(corners[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show the image
    cv2.imshow("AprilTag Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("No AprilTag detected.")
