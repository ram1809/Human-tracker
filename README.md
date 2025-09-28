# Self-Supervised Single Person 3D Human Pose Estimation

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/ram1809/3D_pose_estimator?style=social)](https://github.com/ram1809/3D_pose_estimator/stargazers)

A portable, self-supervised single person 3D human pose estimation system using a custom fisheye stereo camera rig.

![System Overview](images/system_overview.png)

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

- **Real-time 3D Pose Estimation:** Designed for real-time performance using the high-speed trt-pose 2D detector
- **Portable Hardware Design:** Self-contained rig is adaptable to mobile robots and dynamic environments
- **Self-Supervised Learning:** No need for expensive motion capture setups or manual 3D annotations
- **Fisheye Camera Support:** Includes a complete calibration pipeline for wide-angle lenses
- **Robust to Occlusions:** The learning-based MLP can estimate complete poses even with partial occlusions
- **ROS Integration:** Ready-to-use ROS nodes for robotics applications
- **Visualization Tools:** Includes scripts for both 2D and 3D pose visualization

## Hardware Setup

Our system uses a custom 3D-printed stereo rig with two fisheye cameras:

- 2× 2MP USB cameras with 1/2.7-inch CMOS OV2710 image sensors and 170° fisheye lenses
- Fixed baseline on a 3D-printed mounting bracket (STL files provided in `/hardware`)

![Hardware Setup](images/hardware_setup.png)

## Installation

```bash
# Clone the repository
git clone https://github.com/ram1809/3D_pose_estimator.git

# Navigate to the project directory
cd 3D_pose_estimator

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Calibration

```python
# Run camera calibration
python scripts/calibrate_cameras.py --config configs/camera_config.yaml

# Verify calibration results
python scripts/verify_calibration.py --config configs/camera_config.yaml
```

### Training

```python
# Generate self-supervised dataset
python scripts/generate_dataset.py --config configs/dataset_config.yaml

# Train the GNN for keypoint matching
python scripts/train_gnn.py --config configs/train_config.yaml

# Train the MLP for 3D pose estimation
python scripts/train_mlp.py --config configs/train_config.yaml
```

### Inference

```python
# Run real-time inference
python scripts/run_inference.py --config configs/inference_config.yaml
```

## Results and Evaluation

The system was evaluated on a custom test dataset generated with the fisheye stereo rig.

- **Intrinsic Calibration Accuracy:** Low RMS reprojection errors of 0.61 px and 0.89 px were achieved for the two cameras.

- **Skeleton Matching Performance:** The GNN achieved a perfect score of 1.0 across all four standard clustering metrics (Rand Index, Homogeneity, Completeness, V-measure) on the test set.

- **Final 3D Pose Estimation Accuracy:** The trained MLP model's reprojection error was compared against classic triangulation.

| Camera | Method | Mean Reprojection Error (px) |
|--------|--------|------------------------------|
| Camera 0 | MLP Estimation | 8.92 |
| Camera 0 | Triangulation | 6.42 |
| Camera 2 | MLP Estimation | 5.73 |
| Camera 2 | Triangulation | 7.57 |

## Documentation

For detailed documentation, please visit the [Wiki](https://github.com/ram1809/3D_pose_estimator/wiki).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the (LICENSE) file for details.

## Acknowledgments

- Rodriguez-Criado et al. for the original research on self-supervised 3D pose estimation
- The robotics lab at Aston University for hardware support
- Open-source projects: trt-pose, PyTorch, OpenCV, DGL

## Contact

- Ram Munusamy - [GitHub](https://github.com/ram1809) - [your.email@example.com]
