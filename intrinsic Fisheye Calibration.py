import numpy as np
import cv2
import glob
import os
from pathlib import Path
from datetime import datetime

# === SETTINGS ===
# Checkerboard dimensions (internal corners)
CHECKERBOARD = (9, 6)  # (width, height)

# Square size in real-world units (millimeters)
square_size = 1.0  # in mm

# Camera folders
camera_folders = {
    "Cam0": r"C:\Windows\System32\Dissertation\Images\intrinsic\captured_images\Cam0",
    "Cam1": r"C:\Windows\System32\Dissertation\Images\intrinsic\captured_images\Cam1"
}

# Create output directory for essential files
output_dir = "intrinsic_params_for_extrinsic"
os.makedirs(output_dir, exist_ok=True)

# Create a log file for the calibration process
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(output_dir, f"intrinsic_calibration_log_{timestamp}.txt")

def log(message):
    print(message)
    with open(log_file, 'a') as f:
        f.write(message + '\n')

log(f"\n=== FISHEYE INTRINSIC CALIBRATION - {timestamp} ===")

# Termination criteria for corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Flags for fisheye calibration
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW

# Process each camera
for cam_name, image_folder in camera_folders.items():
    log(f"\n=== Calibrating {cam_name} ===")
    
    # Prepare object points
    objp = np.zeros((CHECKERBOARD[1] * CHECKERBOARD[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= square_size  # Scale to real-world size

    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    # Get all calibration images
    images = glob.glob(os.path.join(image_folder, "*.jpg"))
    if not images:
        images = glob.glob(os.path.join(image_folder, "*.png"))
    
    if not images:
        log(f"No images found in {image_folder}")
        continue
    
    log(f"Found {len(images)} images for {cam_name}")
    
    # Process each image - only extract corners
    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            log(f"[ERROR] Could not load image: {fname}")
            continue
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the checkerboard corners
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)
        
        # If found, refine and add points
        if ret:
            # Refine corner positions
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Store results
            objpoints.append(objp)
            imgpoints.append(corners2)
            
            log(f"[OK] {os.path.basename(fname)}")
        else:
            log(f"[MISS] {os.path.basename(fname)}")
    
    # Calibrate camera if we have enough data
    if len(objpoints) > 0:
        log(f"\nCalibrating {cam_name} with {len(objpoints)} images...")
        
        # Get image size
        h, w = gray.shape
        img_size = (w, h)
        
        # Prepare matrices for fisheye calibration
        K = np.zeros((3, 3))  # Camera matrix
        D = np.zeros((4, 1))  # Distortion coefficients
        
        # Arrays to store rotations and translations
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(len(objpoints))]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(len(objpoints))]
        
        # Reshape object points for fisheye calibration
        _objpoints = [obj.reshape(1, -1, 3).astype(np.float64) for obj in objpoints]
        _imgpoints = [img.reshape(1, -1, 2).astype(np.float64) for img in imgpoints]
        
        # Perform the fisheye calibration
        try:
            rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                _objpoints, _imgpoints, img_size, K, D, rvecs, tvecs, 
                calibration_flags, criteria
            )
            
            log(f"[SUCCESS] {cam_name} Fisheye Calibration completed")
            log(f"Camera Matrix ({cam_name}):\n{K}")
            log(f"Fisheye Distortion Coefficients ({cam_name}):\n{D.ravel()}")
            log(f"RMS error: {rms}")
            
            # # Ensure D is exactly 4x1 for compatibility with stereo calibration
            # D_fixed = D.flatten()[:4].reshape(4, 1).astype(np.float64)
            
            # # Save only the parameters needed for extrinsic calibration
            # output_file = os.path.join(output_dir, f"fisheye_intrinsics_{cam_name}.npz")
            # np.savez(output_file, 
            #          K=K.astype(np.float64),  # Ensure float64 format
            #          D=D_fixed,                # Fixed shape distortion coefficients
            #          image_size=np.array(img_size, dtype=np.int32),  # Save as numpy array
            #          rms=rms)                  # RMS error
            
            # # Also save in a format that's easy to load in other applications
            # params_file = os.path.join(output_dir, f"fisheye_intrinsics_{cam_name}.txt")
            # with open(params_file, 'w') as f:
            #     f.write(f"# Fisheye camera intrinsic parameters for {cam_name}\n")
            #     f.write(f"# Calibrated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            #     f.write(f"# Image size: {img_size}\n")
            #     f.write(f"# RMS error: {rms}\n\n")
                
            #     f.write("# Camera matrix K (3x3):\n")
            #     for row in K:
            #         f.write(" ".join([str(x) for x in row]) + "\n")
                
            #     f.write("\n# Distortion coefficients D (k1, k2, k3, k4):\n")
            #     f.write(" ".join([str(x) for x in D_fixed.ravel()]) + "\n")
            
            # log(f"Saved parameters for extrinsic calibration in {output_dir}")
            
            # # Verification - try to load the saved file to ensure it works
            # try:
            #     test_data = np.load(output_file)
            #     log(f"Verification: Successfully loaded saved parameters")
            #     log(f"K shape: {test_data['K'].shape}, D shape: {test_data['D'].shape}")
            # except Exception as e:
            #     log(f"[WARNING] Verification failed: {e}")
            
        except cv2.error as e:
            log(f"[ERROR] Fisheye calibration failed for {cam_name}: {str(e)}")
            
    else:
        log(f"[SKIP] Not enough valid detections for {cam_name}")

log("\n=== Calibration Complete ===")
log(f"Intrinsic parameters saved to {output_dir}")
log("These files contain the camera matrix (K) and distortion coefficients (D) needed for extrinsic calibration.")