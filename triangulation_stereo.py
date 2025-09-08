import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def find_corresponding_points_sift(img1, img2, max_matches=100):
    """
    Find corresponding points between two images using SIFT feature matching.
    
    Args:
        img1: First image
        img2: Second image
        max_matches: Maximum number of matches to return
        
    Returns:
        points1, points2: Corresponding points in both images
    """
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Find keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    # FLANN parameters for matching
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    # Create FLANN matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Apply Lowe's ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    # Sort matches by distance and take the best ones
    good_matches = sorted(good_matches, key=lambda x: x.distance)
    good_matches = good_matches[:max_matches]
    
    # Extract matched point coordinates
    points1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    points2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    
    return points1, points2


def find_corresponding_points_orb(img1, img2, max_matches=100):
    """
    Find corresponding points between two images using ORB feature matching.
    Useful when SIFT is not available or faster computation is needed.
    
    Args:
        img1: First image
        img2: Second image
        max_matches: Maximum number of matches to return
        
    Returns:
        points1, points2: Corresponding points in both images
    """
    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=1000)
    
    # Find keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    # Create BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors
    matches = bf.match(des1, des2)
    
    # Sort by distance
    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[:max_matches]
    
    # Extract matched point coordinates
    points1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    points2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    return points1, points2


def find_corresponding_points_optical_flow(img1, img2, initial_points=None):
    """
    Find corresponding points using Lucas-Kanade optical flow.
    Useful when images are from video frames with small motion.
    
    Args:
        img1: First image (grayscale)
        img2: Second image (grayscale)
        initial_points: Initial points to track (if None, good features will be detected)
        
    Returns:
        points1, points2: Corresponding points in both images
    """
    if initial_points is None:
        # Detect good features to track
        corners = cv2.goodFeaturesToTrack(img1, maxCorners=100, qualityLevel=0.01, minDistance=10)
        initial_points = corners.reshape(-1, 2)
    
    # Calculate optical flow
    points2, status, _ = cv2.calcOpticalFlowPyrLK(img1, img2, initial_points.astype(np.float32), None)
    
    # Select good points
    good_new = points2[status == 1]
    good_old = initial_points[status == 1]
    
    return good_old, good_new


def triangulate_points(points1, points2, cam1_matrix, cam1_dist, cam2_matrix, cam2_dist, R, T, is_fisheye=True):
    # Undistort to pixel coords using fisheye or pinhole model
    if is_fisheye:
        # cam*_dist must be (4,) or (4,1) for fisheye
        p1 = cv2.fisheye.undistortPoints(points1.reshape(-1,1,2), cam1_matrix, cam1_dist.reshape(-1,1), P=cam1_matrix).reshape(-1,2)
        p2 = cv2.fisheye.undistortPoints(points2.reshape(-1,1,2), cam2_matrix, cam2_dist.reshape(-1,1), P=cam2_matrix).reshape(-1,2)
    else:
        # pinhole path (if you recalibrate that way)
        p1 = cv2.undistortPoints(points1.reshape(-1,1,2), cam1_matrix, cam1_dist, P=cam1_matrix).reshape(-1,2)
        p2 = cv2.undistortPoints(points2.reshape(-1,1,2), cam2_matrix, cam2_dist, P=cam2_matrix).reshape(-1,2)

    # Projection matrices (pixel coords)
    P1 = cam1_matrix @ np.hstack([np.eye(3), np.zeros((3,1))])
    P2 = cam2_matrix @ np.hstack([R, T.reshape(3,1)])

    X_h = cv2.triangulatePoints(P1, P2, p1.T, p2.T)
    X = (X_h[:3] / X_h[3]).T
    return X

def compute_epipolar_lines(points1, points2, fundamental_matrix, img1_shape, img2_shape):
    """
    Compute and visualize epipolar lines for corresponding points.
    Useful for verifying correct point correspondence.
    
    Args:
        points1: Points in first image
        points2: Points in second image
        fundamental_matrix: Fundamental matrix relating the two views
        img1_shape: Shape of first image (height, width)
        img2_shape: Shape of second image (height, width)
        
    Returns:
        None (displays plot)
    """
    lines1 = cv2.computeCorrespondEpilines(points2.reshape(-1, 1, 2), 2, fundamental_matrix)
    lines1 = lines1.reshape(-1, 3)
    
    lines2 = cv2.computeCorrespondEpilines(points1.reshape(-1, 1, 2), 1, fundamental_matrix)
    lines2 = lines2.reshape(-1, 3)
    
    # Create blank images to draw on
    img1_lines = np.zeros((img1_shape[0], img1_shape[1], 3), dtype=np.uint8)
    img2_lines = np.zeros((img2_shape[0], img2_shape[1], 3), dtype=np.uint8)
    
    # Draw epipolar lines and points
    for i, (pt1, pt2) in enumerate(zip(points1, points2)):
        # Colors for drawing
        color = tuple(np.random.randint(0, 255, 3).tolist())
        
        # Draw points
        x1, y1 = int(pt1[0]), int(pt1[1])
        x2, y2 = int(pt2[0]), int(pt2[1])
        cv2.circle(img1_lines, (x1, y1), 5, color, -1)
        cv2.circle(img2_lines, (x2, y2), 5, color, -1)
        
        # Draw epipolar lines
        # Line equation: ax + by + c = 0
        a, b, c = lines1[i]
        # Get two points to draw the line
        x0, y0 = 0, int(-c / b)
        x1, y1 = img1_shape[1], int(-(a * img1_shape[1] + c) / b)
        cv2.line(img1_lines, (x0, y0), (x1, y1), color, 1)
        
        a, b, c = lines2[i]
        x0, y0 = 0, int(-c / b)
        x1, y1 = img2_shape[1], int(-(a * img2_shape[1] + c) / b)
        cv2.line(img2_lines, (x0, y0), (x1, y1), color, 1)
    
    # Display epipolar lines
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(img1_lines)
    plt.title('Epipolar lines on image 1')
    plt.subplot(122)
    plt.imshow(img2_lines)
    plt.title('Epipolar lines on image 2')
    plt.tight_layout()
    plt.show()


def visualize_3d_points(points_3d):
    """
    Visualize triangulated 3D points.
    
    Args:
        points_3d: 3D points (Nx3 array)
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='b', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Triangulated 3D Points')
    plt.show()


def main():
    # 1) Load your test images
    img1 = cv2.imread(r"C:\Windows\System32\Dissertation\New_folder\captured_images\Camera0\image_019.jpg", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(r"C:\Windows\System32\Dissertation\New_folder\captured_images\Camera1\image_019.jpg", cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        print("Error loading images. Check the file paths.")
        return

    # 2) Load fisheye intrinsics you saved earlier
    data0 = np.load(r"C:\Windows\System32\Dissertation\Images\intrinsic\intrinsic_params_for_extrinsic\fisheye_intrinsics_Cam0.npz")
    data1 = np.load(r"C:\Windows\System32\Dissertation\Images\intrinsic\intrinsic_params_for_extrinsic\fisheye_intrinsics_Cam1.npz")
    K0 = (data0['K'] if 'K' in data0 else data0['camera_matrix']).astype(np.float64)
    D0 = (data0['D'] if 'D' in data0 else data0['dist_coeffs']).astype(np.float64).reshape(-1)[:4]
    K1 = (data1['K'] if 'K' in data1 else data1['camera_matrix']).astype(np.float64)
    D1 = (data1['D'] if 'D' in data1 else data1['dist_coeffs']).astype(np.float64).reshape(-1)[:4]

    # 3) Get or set extrinsics (R,T) from your extrinsic step
    # If you've saved them: load from .npz that contains R,T
    # Here, for demonstration, compute quickly from matches (or replace with your saved values)
    points1, points2 = find_corresponding_points_sift(img1, img2, max_matches=500)
    F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC, 3.0, 0.99)
    points1 = points1[mask.ravel()==1]; points2 = points2[mask.ravel()==1]

    # Essential from intrinsics
    E = K1.T @ F @ K0
    _, R, T, _ = cv2.recoverPose(E, points1, points2, K0)

    # 4) (Optional) visualize epipolar geometry
    compute_epipolar_lines(points1, points2, F, img1.shape, img2.shape)

    # 5) Triangulate using fisheye-safe path
    X = triangulate_points(points1, points2, K0, D0, K1, D1, R, T, is_fisheye=True)

    visualize_3d_points(X)
    print(f"Successfully triangulated {len(X)} 3D points.")
    return X
  
    # Example camera parameters (replace with your calibrated values)
    # Camera 1 (left camera) intrinsic parameters
    cam1_matrix = np.array([
        [340.30189619, 0, 323.09151512],
        [0, 340.34255429, 240.30179798],
        [0, 0, 1]
    ])
    cam1_dist = np.array([k1, k2, p1, p2, k3])  # Distortion coefficients
    
    # Camera 2 (right camera) intrinsic parameters
    cam2_matrix = np.array([
        [330.78263598, 0, 321.59614737],
        [0, 330.47057206, 242.61509848],
        [0, 0, 1]
    ])
    cam2_dist = np.array([k1, k2, p1, p2, k3])  # Distortion coefficients
    
    # Extrinsic parameters: rotation and translation from camera 1 to camera 2
    R = np.eye(3)  # Example rotation matrix
    T = np.array([baseline, 0, 0])  # Example translation vector (typical stereo setup)
    
    # Find corresponding points using one of the available methods
    # Method 1: SIFT feature matching
    points1, points2 = find_corresponding_points_sift(img1, img2)
    
    # Method 2: ORB feature matching (faster alternative)
    # points1, points2 = find_corresponding_points_orb(img1, img2)
    
    # Method 3: Optical flow (good for video sequences)
    # points1, points2 = find_corresponding_points_optical_flow(img1, img2)
    
    # Calculate fundamental matrix to verify correspondences
    F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC, 3, 0.99)
    
    # Filter out outlier matches using the fundamental matrix
    points1 = points1[mask.ravel() == 1]
    points2 = points2[mask.ravel() == 1]
    
    # Visualize epipolar lines to verify correct point correspondence
    compute_epipolar_lines(points1, points2, F, img1.shape, img2.shape)
    
    # Triangulate 3D points
    points_3d = triangulate_points(points1, points2, cam1_matrix, cam1_dist, 
                                  cam2_matrix, cam2_dist, R, T)
    
    # Visualize the triangulated 3D points
    visualize_3d_points(points_3d)
    
    print(f"Successfully triangulated {len(points_3d)} 3D points.")
    return points_3d


if __name__ == "__main__":
    main()