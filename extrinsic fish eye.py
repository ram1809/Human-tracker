import cv2
from pupil_apriltags import Detector
import numpy as np
import glob
import os
from datetime import datetime

# === SETTINGS ===
tag_size = 0.45  # AprilTag size in meters

intrinsic_files = {
    "Camera0": r"C:\Windows\System32\Dissertation\Images\intrinsic\intrinsic_params_for_extrinsic\fisheye_intrinsics_Cam0.npz",
    "Camera1": r"C:\Windows\System32\Dissertation\Images\intrinsic\intrinsic_params_for_extrinsic\fisheye_intrinsics_Cam1.npz"
}

camera_folders = {
    "Camera0": r"C:\Windows\System32\Dissertation\New_folder\captured_images\Camera0",
    "Camera1": r"C:\Windows\System32\Dissertation\New_folder\captured_images\Camera1"
}

output_file = r"C:\Windows\System32\Dissertation\fisheye_stereo_extrinsics.npz"
debug_dir = r"C:\Windows\System32\Dissertation\fisheye_extrinsic_debug"
os.makedirs(debug_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(debug_dir, f"fisheye_extrinsic_calibration_log_{timestamp}.txt")
def log(message):
    print(message)
    with open(log_file, 'a') as f:
        f.write(message + '\n')

log(f"\n=== FISHEYE STEREO EXTRINSIC CALIBRATION - {timestamp} ===")
log(f"Calibration performed by: {os.environ.get('USERNAME', 'ram1809')}")
log(f"Date: {timestamp}")

# Load intrinsics
intrinsics = {}
for cam_name, calib_file in intrinsic_files.items():
    try:
        data = np.load(calib_file)
        K = data['K'] if 'K' in data else data['camera_matrix']
        D = data['D'] if 'D' in data else data['dist_coeffs']
        image_size = data['image_size'] if 'image_size' in data else None
        
        intrinsics[cam_name] = {
            'K': K.astype(np.float64),
            'D': D.flatten()[:4].astype(np.float64),  # Ensure we have exactly 4 coefficients
            'image_size': image_size
        }
        log(f"Loaded fisheye intrinsic parameters for {cam_name}")
    except Exception as e:
        log(f"[ERROR] Could not load intrinsics for {cam_name}: {e}")
        exit(1)

detector = Detector(
    families="tagStandard41h12",
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=True,
    decode_sharpening=0.25
)

# Get image files
images_cam0 = sorted(glob.glob(os.path.join(camera_folders["Camera0"], "*.jpg")))
images_cam1 = sorted(glob.glob(os.path.join(camera_folders["Camera1"], "*.jpg")))

if not images_cam0:
    images_cam0 = sorted(glob.glob(os.path.join(camera_folders["Camera0"], "*.png")))
if not images_cam1:
    images_cam1 = sorted(glob.glob(os.path.join(camera_folders["Camera1"], "*.png")))

if len(images_cam0) != len(images_cam1):
    log(f"[ERROR] Image count mismatch: Camera0 ({len(images_cam0)}) vs Camera1 ({len(images_cam1)})")
    exit(1)

if not images_cam0 or not images_cam1:
    log(f"[ERROR] No images found in one or both camera folders")
    exit(1)

log(f"Found {len(images_cam0)} image pairs")

# ------------ USING FUNDAMENTAL MATRIX DIRECT APPROACH ------------
# This approach bypasses the problematic stereoCalibrate functions
# and directly calculates the fundamental matrix, which can be used 
# to derive the essential matrix and camera transformation

# Lists to store all detected points across all images
all_pts_cam0 = []
all_pts_cam1 = []

# Process each image pair
for i, (fname_cam0, fname_cam1) in enumerate(zip(images_cam0, images_cam1)):
    img_cam0 = cv2.imread(fname_cam0)
    img_cam1 = cv2.imread(fname_cam1)
    if img_cam0 is None or img_cam1 is None:
        log(f"[SKIP] Could not load image pair {i}: {fname_cam0}, {fname_cam1}")
        continue

    gray_cam0 = cv2.cvtColor(img_cam0, cv2.COLOR_BGR2GRAY)
    gray_cam1 = cv2.cvtColor(img_cam1, cv2.COLOR_BGR2GRAY)

    tags_cam0 = detector.detect(gray_cam0)
    tags_cam1 = detector.detect(gray_cam1)

    tags_cam0_dict = {tag.tag_id: tag for tag in tags_cam0}
    tags_cam1_dict = {tag.tag_id: tag for tag in tags_cam1}
    common_ids = set(tags_cam0_dict.keys()) & set(tags_cam1_dict.keys())

    if not common_ids:
        log(f"[SKIP] No common tags in image pair {i}")
        continue
    
    log(f"[FOUND] Image pair {i}: {len(common_ids)} common tags")
    
    # Collect matching corners from all detected tags
    for tag_id in common_ids:
        tag_cam0 = tags_cam0_dict[tag_id]
        tag_cam1 = tags_cam1_dict[tag_id]
        
        corners_cam0 = tag_cam0.corners
        corners_cam1 = tag_cam1.corners
        
        # Add each corner point
        for j in range(4):  # 4 corners per tag
            all_pts_cam0.append(corners_cam0[j])
            all_pts_cam1.append(corners_cam1[j])

if len(all_pts_cam0) < 8:  # Need at least 8 points for fundamental matrix
    log(f"[ERROR] Not enough matching points found. Need at least 8, got {len(all_pts_cam0)}")
    exit(1)

# Convert to numpy arrays of the correct format
all_pts_cam0 = np.array(all_pts_cam0, dtype=np.float64)
all_pts_cam1 = np.array(all_pts_cam1, dtype=np.float64)

log(f"Total matching points: {len(all_pts_cam0)}")

# Get camera matrices and distortion coefficients
K0 = intrinsics["Camera0"]["K"]
D0 = intrinsics["Camera0"]["D"]
K1 = intrinsics["Camera1"]["K"]
D1 = intrinsics["Camera1"]["D"]
img_size = tuple(int(x) for x in intrinsics["Camera0"]["image_size"])

try:
    # Step 1: Find the fundamental matrix
    fundamental_matrix, mask = cv2.findFundamentalMat(
        all_pts_cam0, 
        all_pts_cam1, 
        cv2.FM_RANSAC, 
        ransacReprojThreshold=3.0,
        confidence=0.99
    )
    
    # Keep only inlier points
    inliers_mask = mask.ravel() == 1
    inlier_pts_cam0 = all_pts_cam0[inliers_mask]
    inlier_pts_cam1 = all_pts_cam1[inliers_mask]
    
    log(f"Fundamental matrix computed with {np.sum(inliers_mask)} inliers")
    log(f"Fundamental matrix:\n{fundamental_matrix}")
    
    # Step 2: Calculate essential matrix from fundamental matrix
    essential_matrix = K1.T @ fundamental_matrix @ K0
    
    # Step 3: Recover rotation and translation from essential matrix
    retval, R, T, mask = cv2.recoverPose(essential_matrix, inlier_pts_cam0, inlier_pts_cam1, K0)
    
    log(f"Camera pose recovered successfully")
    log(f"Rotation matrix:\n{R}")
    log(f"Translation vector:\n{T}")
    log("\n=== FISHEYE EXTRINSIC CALIBRATION PROCESS COMPLETED ===")
    log(f"Log saved to: {log_file}")
    log(f"Debug images saved to: {debug_dir}")
except Exception as e:
    log(f"[ERROR] Error during calibration: {e}")
    import traceback
    log(traceback.format_exc())
    
#     # Step 4: Compute rectification transforms
#     R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
#         K0, D0.reshape(-1, 1),
#         K1, D1.reshape(-1, 1),
#         img_size, R, T,
#         flags=0,  # No flags needed
#         alpha=0.175  # No scaling
#     )
    
#     log(f"Rectification transforms computed")
    
#     # Save results
#     np.savez(output_file,
#         R=R, T=T,
#         K1=K0, D1=D0, K2=K1, D2=D1,
#         R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
#         F=fundamental_matrix, E=essential_matrix,
#         image_size=img_size
#     )
    
#     log(f"Calibration parameters saved to {output_file}")
    
#     # Create rectification test
#     try:
#         test_img_left = cv2.imread(images_cam0[0])
#         test_img_right = cv2.imread(images_cam1[0])
        
#         if test_img_left is None or test_img_right is None:
#             raise ValueError("Could not load test images")
            
#         # Standard undistortion and rectification
#         map1x, map1y = cv2.initUndistortRectifyMap(
#             K0, D0.reshape(-1, 1), R1, P1, img_size, cv2.CV_32FC1)
#         map2x, map2y = cv2.initUndistortRectifyMap(
#             K1, D1.reshape(-1, 1), R2, P2, img_size, cv2.CV_32FC1)
            
#         rect_left = cv2.remap(test_img_left, map1x, map1y, cv2.INTER_LINEAR)
#         rect_right = cv2.remap(test_img_right, map2x, map2y, cv2.INTER_LINEAR)
        
#         # Draw horizontal lines for easy verification
#         h, w = rect_left.shape[:2]
#         for y in range(0, h, 50):
#             cv2.line(rect_left, (0, y), (w, y), (0, 255, 0), 1)
#             cv2.line(rect_right, (0, y), (w, y), (0, 255, 0), 1)
            
#         # Create a side-by-side image
#         side_by_side = np.hstack((rect_left, rect_right))
        
#         # Save the images
#         cv2.imwrite(os.path.join(debug_dir, "rectified_left.jpg"), rect_left)
#         cv2.imwrite(os.path.join(debug_dir, "rectified_right.jpg"), rect_right)
#         cv2.imwrite(os.path.join(debug_dir, "rectified_side_by_side.jpg"), side_by_side)
        
#         log(f"Rectification test images saved to {debug_dir}")
        
#     except Exception as e:
#         log(f"[WARNING] Could not create rectification test: {e}")
#         import traceback
#         log(traceback.format_exc())
        
# except Exception as e:
#     log(f"[ERROR] Extrinsic calibration failed: {e}")
#     import traceback
#     log(traceback.format_exc())
#     log("Try capturing more image pairs with tags clearly visible in both cameras")

