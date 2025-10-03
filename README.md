# Self-Supervised single person 3D Human Pose Estimation with Fisheye Stereo Camera

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/ram1809/3D-pose-estimator?style=social)](https://github.com/ram1809/3D-pose-estimator/stargazers)

This repository contains the code and resources for camrera calibration, adapting a portable, self-supervised single person 3D human pose estimation system from Rodriguez-Criado et al. (2024) using a custom fisheye stereo camera rig.

![System Overview](assets/system_overview.png)

## Project Overview

This project adapts the state-of-the-art, self-supervised 3D pose estimation system from Rodriguez-Criado et al. (2024) for use on a new, robust, and portable hardware platform.

**Problem:** The original research relied on a static, wall-mounted camera setup, unsuitable for dynamic applications like mobile robotics.

### Key Contributions:

- **Robust Hardware:** Designed a portable stereo rig with two 170° fisheye cameras on a fixed 3D-printed mount, ensuring a constant relative pose.

- **Fisheye Calibration:** Developed a complete intrinsic/extrinsic calibration pipeline to handle severe lens distortion using a multi-step process with AprilTags.

- **Custom Dataset:** Generated a new dataset tailored for the fisheye optics using the original self-supervised, single-person recording strategy.

- **End-to-End Validation:** Successfully trained and validated the full pipeline—from the GNN for matching to the MLP for 3D estimation—on the new hardware.

### Applications in Social Robotics:

This work provides the perception layer for mobile robots to navigate safely and intelligently around people by enabling:

- **Intent Prediction:** Anticipating human trajectories for smoother path planning.
- **Socially-Aware Movement:** Maintaining a safe distance and respecting personal space.
- **Non-Verbal Interaction:** Understanding body language for more intuitive human-robot interaction.

## Features

- **Real-time 3D Pose Estimation:** Designed for real-time performance using the high-speed trt-pose 2D detector.
- **Portable Hardware Design:** Self-contained rig is adaptable to mobile robots and dynamic environments.
- **Self-Supervised Learning:** No need for expensive motion capture setups or manual 3D annotations.
- **Fisheye Camera Support:** Includes a complete calibration pipeline for wide-angle lenses.
- **Robust to Occlusions:** The learning-based MLP can estimate complete poses even with partial occlusions.
- **Open Dataset:** Tools provided to generate a custom fisheye camera dataset for training and validation.
- **ROS Integration:** Ready-to-use ROS nodes for robotics applications.
- **Visualization Tools:** Includes scripts for both 2D and 3D pose visualization.

## Installation

```bash
# Clone the repository
git clone https://github.com/ram1809/3D-pose-estimator.git

# Navigate to the project directory
cd 3D-pose-estimator

# Install dependencies
pip install -r requirements.txt
```

## Hardware Setup

Our system uses a custom 3D-printed stereo rig with two fisheye cameras:

- 2× 2MP USB cameras with 1/2.7-inch CMOS OV2710 image sensors and 170° fisheye lenses.
- Fixed baseline on a 3D-printed mounting bracket (STL files provided in /hardware).

## Steps
all the files and folders are inside the cloned repository.

Image Acquisition: 
To caputure images for both intrinsic and extrinsic parameter
    cd camera_calibration/images.py
    
    Image Acquisition: Use the provided script to capture synchronized images of a checkerboard target from various angles and distances.
      
    To find intrinsic and extrinsic parameters using standard cv2.calibrate module
        cd camera_calibration/Intrinsic_cam1_cam2.py 
        cd camera_calibration/extrinsic_cam1_cam2.py 

    To find intrinsic and extrinsic parameters using fisheye module
        cd camera_calibration/intrinsic_fisheye_calibration.py 
        cd camera_calibration/extrinsic.py 
The camera_calibration folder also contains all .npz files required for calibration

Parameter Serialization:
The file that holds all the intrinsic and extrinsic parameters
    cd Parameter Serialization/parameters.py
    
The file that creats the Pickle file and manages transformation
cd 3D_multi_pose_estimator/calibration_receiver/SRC/specificworker.py

Dataset Generation:

The models are trained using a custom dataset generated with the self-supervised, single-person recording strategy.

    Record Sequences: Capture video sequences of a single person moving in the environment. Set the Environment variaable "CAMERA_IDENTIFIER" value based on the camera device ID whiile running the below scripts. Along with configuration file.

    Process Data: Run the 2D skeleton detector (trt-pose) on the recorded footage to generate the final .json data files.
        cd 3D_multi_pose_estimator/tracker_camera
    The file that holds the configuration is inside etc/config
    
Model Training
The training process is divided into two stages:
a) Training the Skeleton Matching Network (GNN)

cd skeleton_matching/
python3 train_skeleton_matching.py --trainset /path/to/train_paths.txt --devset /path/to/dev_paths.txt

b) Training the Pose Estimator Network (MLP)
cd pose_estimator/
python3 train_pose_estimator.py --trainset /path/to/train_paths.txt --devset /path/to/dev_paths.txt

Testing and Evaluation
a) Metrics

Evaluate the trained models on an unseen test set.

    Skeleton Matching (GNN) Metrics:

    python3 sm_metrics_without_gt.py --testfiles /path/to/test_paths.txt --modelsdir /path/to/models/

    3D Pose (MLP) Reprojection Error:

    python3 reprojection_error.py --testfiles /path/to/test_paths.txt --modelsdir /path/to/models/

b) Visualization

Visually inspect the 3D pose estimation results.

    From MLP Model:

    python3 show_results_from_model.py --testfile /path/to/single_test.json --modelsdir /path/to/models/

## Results and Evaluation

The system was evaluated on a custom test dataset generated with the fisheye stereo rig.

- **Intrinsic Calibration Accuracy:** Low RMS reprojection errors of 0.61 px and 0.89 px were achieved for the two cameras.

- **Skeleton Matching Performance:** The GNN achieved a perfect score of 1.0 across all four standard clustering metrics (Rand Index, Homogeneity, Completeness, V-measure) on the test set.

- **Final 3D Pose Estimation Accuracy:** The trained MLP model's reprojection error was compared against classic triangulation.

| Camera | Method | Mean Reprojection Error (px) |
|--------|--------|------------------------------|
| Camera 0 | MLP Estimation | 8.92 |
| Camera 0 | Triangulation | 6.42 |
| Camera 1 | MLP Estimation | 5.73 |
| Camera 1 | Triangulation | 7.57 |


## Acknowledgments
- Rodriguez-Criado et al. for the original research on self-supervised 3D pose estimation.
- The robotics lab for hardware support.
- Open-source projects: trt-pose, PyTorch, OpenCV, DGL.

## Contact
Ram Munusamy - [GitHub](https://github.com/ram1809) - ram1809@example.com
