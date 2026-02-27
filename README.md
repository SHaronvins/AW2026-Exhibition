# AW2026-Exhibition
An algorithm for tracking welding paths using vision-based techniques

## 🔧 Overview

This project is developed for AW2026 Exhibition.

It integrates:
- Industrial Robot Control (UR, Fairino support planned)
- 3D Vision (Mech-Eye Camera)
- Object Detection (YOLO)
- 6D Pose Estimation
- Eye-in-Hand Calibration

The system detects objects using a camera, estimates their 6D pose, and controls the robot to interact with the object.

## ⚙️ Pipeline

1. Capture RGB / Depth / Point Cloud from Mech-Eye camera
2. Detect object using YOLO
3. Estimate object pose (6D Pose Estimation)
4. Convert camera coordinate to robot base coordinate
5. Generate robot motion path
6. Execute robot motion

## 📦 Requirements

This project requires the following models:

- YOLO (for object detection)
- FoundationPose (for 6D pose estimation)

YOLO is used to detect objects from RGB images, providing bounding boxes for the target objects.

FoundationPose takes the detected objects along with RGB-D data and corresponding 3D models to estimate accurate 6D poses.

These components are essential for enabling precise robot manipulation based on visual perception.

```txt
numpy==1.26.4
scipy==1.12.0

opencv-python==4.9.0.80
opencv-contrib-python==4.9.0.80
open3d==0.18.0
pillow==12.0.0

torch==2.8.0+cu129
torchvision==0.23.0+cu129
torchaudio==2.8.0+cu129

ultralytics==8.4.9

pytorch3d==0.7.9
kaolin==0.18.0
nvdiffrast==0.4.0
trimesh==4.2.2

MechEyeAPI==2.5.4
ur_rtde==1.6.2

tqdm==4.67.1
matplotlib==3.10.8
PyYAML==6.0.1
```

<h2>📂 Project Structure</h2>

<pre>
AW2026-Exhibition/
│
├── base_coordi.py
│   └── Camera → Robot Base coordinate transformation
│
├── camera.py
│   └── Mech-Eye camera interface (RGB / Depth / Point Cloud)
│
├── circle_move.py
│   └── Generate circular robot motion
│
├── circle_move_test.py
│   └── Test script for circular motion
│
├── vertical_move.py
│   └── Z-axis movement (vertical motion, just test for orientation)
│
├── pose_est_cal_auto.py
│   └── auto
│
├── pose_est_cal_key.py
│   └── Key-based
│
├── pose_save.py
│   └── Save robot poses (for cap & ready pose)
│
├── estimater.py
│   └── 6D pose estimation logic
│
├── dataset_cap_for_yolo.py
│   └── Capture dataset for YOLO training and YOLO test
│
├── main.py
│   └── Main
│
├── robot_poses.txt
│   └── Stored robot pose data
│
├── best.pt
│   └── YOLO model weights (need pt pth file)
│
├── model_ex1.obj
├── model_ex2.obj
│   └── 3D models for pose estimation
│
├── TexturedPointCloud.ply
│   └── Sample point cloud data
│
├── estimated_pose.png
├── estimated_pose_rotated.png
│   └── Visualization Point result
│
├── splash_.png
├── splash_cropped.png
│   └── UI / visualization images
│
├── debug_live_mechmind/
│   └── Camera debug data
│
├── debug_pose_est/
│   └── Pose estimation debug data
│
├── fairino/
│   └── Fairino robot control SDK (in progress)
│
├── mycpp/
│   └── C++ modules
│
├── object_pose/
│   └── Object pose utilities
│
├── obj_images/
│   └── Object images / dataset
│
├── weights/
│   └── Model weights
│
├── yolo/
│   └── YOLO related code
│
├── __pycache__/
│   └── Python cache files
│
├── README.md
│   └── Project description
│
└── .gitignore
    └── Ignored files configuration
</pre>

<h2>🚀 Execution</h2>

<p>
The entire pipeline can be executed by running the main script.
</p>

<pre><code>python main.py</code></pre>

<p>
This script integrates camera input, object detection, pose estimation, and robot control into a single workflow.
</p>

---

<h2>📊 Result</h2>

<p>
The system performs the following steps:
</p>

<ul>
  <li>Capture RGB, Depth, and Point Cloud data from the camera</li>
  <li>Detect objects using YOLO</li>
  <li>Estimate 6D pose using FoundationPose</li>
  <li>Transform coordinates from camera to robot base</li>
  <li><b>Move pre-taught robot points based on the estimated pose</b></li>
  <li>Execute robot movement</li>
</ul>

