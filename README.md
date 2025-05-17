# Visual Odometry Pipeline

## Overview

This Python script implements a monocular visual odometry (VO) pipeline using OpenCV, NumPy, and Matplotlib. It estimates a camera's trajectory from a sequence of images by matching keypoints, computing relative poses, and plotting the 2D trajectory (x-z plane). Suitable for datasets like KITTI or custom image sequences.

---

## Requirements

- **Python**: 3.6+
- **Dependencies**: Install with:
  ```bash
  pip install opencv-python numpy matplotlib tqdm
  ```
- **Data**:
  - Image directory with `.png`, `.jpg`, or `.jpeg` files (e.g., `frame001.png`).
  - Calibration file (`MultiMatrix.npz`) with `camMatrix` (3x3 intrinsic matrix).

---

## Pipeline

The visual odometry pipeline consists of the following steps, implemented in the `VisualOdometry` class and orchestrated by the `main` function:

1. **Setup**:
   - Load camera intrinsics (`K`) and images (grayscale, up to 125).
   - Initialize ORB detector (3000 keypoints) and FLANN matcher.

2. **Keypoint Matching**:
   - Detect ORB keypoints and descriptors for consecutive frames.
   - Match using FLANN with Lowe’s ratio test (0.8).
   - Require ≥10 matches; optionally visualize.

3. **Pose Estimation**:
   - Compute essential matrix (`E`) with RANSAC (threshold 0.5).
   - Decompose `E` into rotation (`R`) and translation (`t`).
   - Select valid solution by triangulating points.
   - Form 4x4 transformation matrix.

4. **Trajectory**:
   - Chain transformations from identity pose.
   - Extract x, z coordinates for visualization.
   - Skip invalid frames.

5. **Output**:
   - Plot trajectory as `trajectory.png`.


---

## Usage

1. Place images in `frames/` and `MultiMatrix.npz` in `calibration/`.
2. Run:
   ```bash
   python visual_odometry.py
   ```
   Or:
   ```python
   from visual_odometry import main
   main(data_dir="frames", calib_dir="calibration", visualize_matches=True)
   ```
3. **Options**:
   - `data_dir`: Image directory.
   - `calib_dir`: Calibration directory.
   - `visualize_matches`: Show keypoint matches (default: `False`).

4. **Output**: `frames/trajectory.png`.

---
## Potential Improvements

1. **Robust Matching**: Use geometric constraints (e.g., epipolar geometry) to filter outliers.
2. **Drift Correction**: Implement bundle adjustment or loop closure for long sequences.
3. **3D Visualization**: Plot the trajectory in 3D (x, y, z) using Matplotlib or Plotly.
4. **Performance**: Parallelize image loading or use GPU-accelerated feature detection.
---

## Limitations

- Scale ambiguity in monocular VO.
- Drift over long sequences.
- Hardcoded parameters (e.g., 125 images, 10 matches).

---
