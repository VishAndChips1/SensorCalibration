# Calibration from `updated_bags/` (5x7 checkerboard, 58 mm squares)

Plain checkerboard (4x6 inner corners), RGB camera + LiDAR. Read `CALIBRATION_REPORT.md` first: results, held-out test, limits.

Run order (from this folder; needs ROS 2 Jazzy for the bag reader, paths at the top of each script point at `~/SensorCalibration/updated_bags` and `~/checkerboard_analysis`, edit them for your machine):
1. `extract.py`: bags to arrays. 2. `detect_final.py`: board corners. 3. `intrinsics.py`: lens + stability checks.
4. `lidar_board.py` / `extrinsic.py`: LiDAR to camera. 5. `evaluate.py`: held-out overlap test.

`intrinsics.yaml` and `extrinsics.yaml` are the values now used by the PitFusion3D pipeline.
