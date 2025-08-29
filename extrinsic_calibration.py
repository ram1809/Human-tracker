import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# ===== CONFIGURATION =====
# Path to your intrinsic calibration data
cam0_intrinsics = "C:\\Windows\\System32\\Dissertation\\calibration_cam0.npz"  # Replace with your actual path
cam1_intrinsics = "C:\\Windows\\System32\\Dissertation\\calibration_cam1.npz"  # Replace with your actual path

# Path to the stereo tag points we just collected
tag_points_file = "C:\\Windows\\System32\\Dissertation\\Images\\extrinsic\\stereo_tag_points.npz"

# Output file for extrinsic parameters
output_file = "C:\\Windows\\System32\\Dissertation\\Images\\extrinsic\\stereo_extrinsics.npz"

# Visualization settings
visualize_results = True
save_rectification_images = True

print("=== STEREO EXTRINSIC CALIBRATION ===")

# ===== LOAD INTRINSIC PARAMETERS =====
try:
    # Load camera 0 intrinsics
    cam0_data = np.load(cam0_intrinsics)
    K0 = cam0_data['mtx']
    dist0 = cam0_data['dist']
    
    # Load camera 1 intrinsics
    cam1_data = np.load(cam1_intrinsics)
    K1 = cam1_data['mtx']
    dist1 = cam1_data['dist']
    
    print("Intrinsic parameters loaded successfully")
    print(f"Camera 0 matrix:\n{K0}")
    print(f"Camera 1 matrix:\n{K1}")
    
except FileNotFoundError:
    print("ERROR: Intrinsic calibration files not found!")
    print("Make sure you have calibrated each camera individually first.")
    print("If you have the calibration data in a different format, modify this script accordingly.")
    exit(1)

# ===== LOAD TAG CORRESPONDENCES =====
try:
    tag_data = np.load(tag_points_file, allow_pickle=True)
    objpoints = tag_data['objpoints']
    imgpoints_left = tag_data['imgpoints_left'] 
    imgpoints_right = tag_data['imgpoints_right']
    
    print(f"\nLoaded {len(objpoints)} sets of tag correspondences")
    
    # Check if we have enough points
    if len(objpoints) < 5:
        print("WARNING: Very few tag correspondences found. Results may be unreliable.")
        
except FileNotFoundError:
    print(f"ERROR: Could not find tag correspondences file: {tag_points_file}")
    print("Make sure you've run the tag detection script successfully first.")
    exit(1)

# ===== STEREO CALIBRATION =====
print("\nPerforming stereo calibration...")

# Get image size from intrinsic calibration
img_size = tuple(cam0_data['image_size']) if 'image_size' in cam0_data else (1280, 720)
print(f"Using image size: {img_size}")

# Run stereo calibration
flags = cv2.CALIB_FIX_INTRINSIC  # Use the provided intrinsics, don't optimize them
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

ret, mtx, dist0, mtx2, dist1, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right, 
    K0, dist0, K1, dist1, 
    img_size, flags=flags, criteria=criteria)

print(f"Calibration RMS error: {ret}")
print(f"Rotation matrix:\n{R}")
print(f"Translation vector:\n{T}")

# ===== STEREO RECTIFICATION =====
print("\nCalculating stereo rectification parameters...")

R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
    K0, dist0, 
    K1, dist1, 
    img_size, R, T, 
    flags=cv2.CALIB_ZERO_DISPARITY, 
    alpha=0.9)  # Alpha parameter controls the amount of blank space

print("Rectification complete")
print(f"Q matrix (for disparity-to-depth):\n{Q}")

# ===== SAVE EXTRINSIC PARAMETERS =====
print("\nSaving extrinsic parameters...")
np.savez(output_file,
         R=R,                  # Rotation matrix: camera1 with respect to camera0
         T=T,                  # Translation vector: camera1 with respect to camera0
         E=E,                  # Essential matrix
         F=F,                  # Fundamental matrix
         R1=R1, R2=R2,         # Rectification transforms
         P1=P1, P2=P2,         # Projection matrices
         Q=Q,                  # Disparity-to-depth mapping matrix
         roi_left=roi_left,    # Left camera valid ROI after rectification
         roi_right=roi_right)  # Right camera valid ROI after rectification

print(f"Extrinsic parameters saved to {output_file}")

# ===== VISUALIZATION =====
if visualize_results:
    # Create empty display to visualize the stereo setup
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot camera 0 (origin)
    ax.scatter(0, 0, 0, c='r', marker='o', s=100, label='Camera 0')
    
    # Extract rotation matrix and translation vector
    rvec = cv2.Rodrigues(R)[0]
    tvec = T.flatten()
    
    # Calculate camera 1 position
    cam1_pos = -np.dot(R.T, T).flatten()
    ax.scatter(cam1_pos[0], cam1_pos[1], cam1_pos[2], c='b', marker='o', s=100, label='Camera 1')
    
    # Draw coordinate axes for camera 0
    ax.quiver(0, 0, 0, 0.1, 0, 0, color='r', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 0.1, 0, color='g', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 0, 0.1, color='b', arrow_length_ratio=0.1)
    
    # Draw coordinate axes for camera 1
    r00, r01, r02 = R[0,0], R[0,1], R[0,2]
    r10, r11, r12 = R[1,0], R[1,1], R[1,2]
    r20, r21, r22 = R[2,0], R[2,1], R[2,2]
    
    ax.quiver(cam1_pos[0], cam1_pos[1], cam1_pos[2], 0.1*r00, 0.1*r10, 0.1*r20, color='r', arrow_length_ratio=0.1)
    ax.quiver(cam1_pos[0], cam1_pos[1], cam1_pos[2], 0.1*r01, 0.1*r11, 0.1*r21, color='g', arrow_length_ratio=0.1)
    ax.quiver(cam1_pos[0], cam1_pos[1], cam1_pos[2], 0.1*r02, 0.1*r12, 0.1*r22, color='b', arrow_length_ratio=0.1)
    
    # Plot the detected tag positions in 3D
    for obj in objpoints:
        # Transform object points to camera 0 coordinates (identity transform)
        for pt in obj:
            ax.scatter(pt[0], pt[1], pt[2], c='k', marker='.', s=20)
        
        # Transform object points to camera 1 coordinates
        for pt in obj:
            pt_cam1 = np.dot(R, pt.reshape(3,1)) + T
            ax.scatter(pt_cam1[0], pt_cam1[1], pt_cam1[2], c='gray', marker='.', s=20)
            
    # Set plot limits and labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Stereo Camera Extrinsic Calibration')
    ax.legend()
    
    # Set equal aspect ratio
    max_range = np.array([
        ax.get_xlim()[1] - ax.get_xlim()[0],
        ax.get_ylim()[1] - ax.get_ylim()[0],
        ax.get_zlim()[1] - ax.get_zlim()[0]
    ]).max() / 2.0
    
    mid_x = (ax.get_xlim()[1] + ax.get_xlim()[0]) / 2
    mid_y = (ax.get_ylim()[1] + ax.get_ylim()[0]) / 2
    mid_z = (ax.get_zlim()[1] + ax.get_zlim()[0]) / 2
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Save the figure
    plt.savefig('stereo_extrinsic_visualization.png')
    plt.show()

print("\n=== EXTRINSIC CALIBRATION COMPLETE ===")
print("\nNext steps:")
print("1. Use the extrinsic parameters for stereo rectification")
print("2. Calculate disparity maps from stereo image pairs")
print("3. Generate 3D reconstructions using the Q matrix")