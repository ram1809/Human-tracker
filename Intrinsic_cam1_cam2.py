import numpy as np
import cv2
import glob
import os
from pathlib import Path
import matplotlib.pyplot as plt

# === SETTINGS ===
# Checkerboard dimensions (internal corners)
CHECKERBOARD = (9, 6)  # (width, height)

# Square size in real-world units (millimeters)
square_size = 24.0  # in mm

# Camera folders - update these paths to your image directories
camera_folders = {
    "Cam0": r"C:\Windows\System32\Dissertation\Images\intrinsic\captured_images\Cam0",
    "Cam1": r"C:\Windows\System32\Dissertation\Images\intrinsic\captured_images\Cam1"
}

# Create output directory for visualization
output_dir = "calibration_results"
os.makedirs(output_dir, exist_ok=True)

# Termination criteria for corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Process each camera
for cam_name, image_folder in camera_folders.items():
    print(f"\n=== Calibrating {cam_name} ===")
    
    # Prepare object points: (0,0,0), (1,0,0), (2,0,0) ... (8,5,0)
    objp = np.zeros((CHECKERBOARD[1] * CHECKERBOARD[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= square_size  # Scale to real-world size

    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    # List for storing image paths where checkerboard was found
    successful_images = []
    
    # Get all calibration images
    images = glob.glob(os.path.join(image_folder, "*.jpg"))
    if not images:
        images = glob.glob(os.path.join(image_folder, "*.png"))
    
    if not images:
        print(f"No images found in {image_folder}")
        continue
    
    print(f"Found {len(images)} images for {cam_name}")
    
    # Process each image
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        if img is None:
            print(f"Could not read image {fname}. Skipping.")
            continue
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the checkerboard corners
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)
        
        # If found, refine and add points
        if ret:
            # Refine corner positions
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria
            )
            
            # Store results
            objpoints.append(objp)
            imgpoints.append(corners2)
            successful_images.append(fname)
            
            # Draw and save the corners
            cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            output_file = os.path.join(output_dir, f"{cam_name}_corners_{idx}.jpg")
            cv2.imwrite(output_file, img)
            
            print(f"[OK] {os.path.basename(fname)} - corners found")
        else:
            print(f"[MISS] {os.path.basename(fname)} - checkerboard not found")
    
    # Calibrate camera if we have enough data
    if len(objpoints) > 0:
        print(f"\nCalibrating {cam_name} with {len(objpoints)} images...")
        
        # Get image size
        h, w = gray.shape
        img_size = (w, h)
        
        # Calibrate camera
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, img_size, None, None)
        
        if ret:
            # Calculate reprojection error
            mean_error = 0
            for i in range(len(objpoints)):
                imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
                error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                mean_error += error
                
            mean_error /= len(objpoints)
            
            print(f"[SUCCESS] {cam_name} Calibration completed")
            print(f"Camera Matrix ({cam_name}):\n{camera_matrix}")
            print(f"Distortion Coefficients ({cam_name}):\n{dist_coeffs.ravel()}")
            print(f"Mean Reprojection Error: {mean_error}")
            
            # Save calibration data
            output_file = f"calibration_{cam_name}.npz"
            np.savez(output_file, 
                     camera_matrix=camera_matrix, 
                     dist_coeffs=dist_coeffs, 
                     rvecs=rvecs, 
                     tvecs=tvecs,
                     image_size=img_size,
                     reprojection_error=mean_error)
            
            print(f"✅ Saved {output_file}")
            
#             # Save a sample undistorted image
#             if successful_images:
#                 sample_img = cv2.imread(successful_images[0])
#                 h, w = sample_img.shape[:2]
                
#                 # Get optimal new camera matrix
#                 new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
#                     camera_matrix, dist_coeffs, (w, h), 1, (w, h))
                
#                 # Undistort image
#                 undistorted = cv2.undistort(sample_img, camera_matrix, dist_coeffs, None, new_camera_matrix)
                
#                 # Crop the image to ROI
#                 x, y, w, h = roi
#                 undistorted_cropped = undistorted[y:y+h, x:x+w]
                
#                 # Save results
#                 cv2.imwrite(os.path.join(output_dir, f"{cam_name}_original.jpg"), sample_img)
#                 cv2.imwrite(os.path.join(output_dir, f"{cam_name}_undistorted.jpg"), undistorted)
#                 cv2.imwrite(os.path.join(output_dir, f"{cam_name}_undistorted_cropped.jpg"), undistorted_cropped)
                
#                 # Visualize undistortion
#                 plt.figure(figsize=(14, 7))
#                 plt.subplot(1, 2, 1)
#                 plt.imshow(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))
#                 plt.title(f"{cam_name} Original Image")
#                 plt.subplot(1, 2, 2)
#                 plt.imshow(cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB))
#                 plt.title(f"{cam_name} Undistorted Image")
#                 plt.tight_layout()
#                 plt.savefig(os.path.join(output_dir, f"{cam_name}_comparison.png"))
#                 plt.close()
#         else:
#             print(f"[ERROR] Calibration failed for {cam_name}")
#     else:
#         print(f"[SKIP] Not enough valid detections for {cam_name}")

# print("\n=== Calibration Complete ===")
print(f"Results saved to {output_dir}")
print("The camera matrices and distortion coefficients are saved in the .npz files.")
print("These will be used for the extrinsic calibration with AprilTags.")

cv2.destroyAllWindows()