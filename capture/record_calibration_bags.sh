#!/usr/bin/env bash
# Walks through the 5-pose capture protocol (close/far/left/right/centre),
# recording one short bag per pose with the topics the calibration scripts
# need. Prompts before each recording so you have time to move the board
# and check the live preview if you want.
#
# Usage: ./record_calibration_bags.sh [output_dir] [duration_seconds]
set -e

OUT_DIR="${1:-/home/vish/SensorCalibration/bags}"
DURATION="${2:-6}"
# 6s is enough frames for detection + plane fitting (RGB ~4Hz=24 frames,
# thermal ~20Hz=120 frames, lidar ~10Hz=60 scans) while keeping each bag
# small enough to comfortably live in a git repo (GitHub hard-limits any
# single file to 100MB). If you want longer captures, consider Git LFS
# for the bags/ directory instead of raising this much past ~15s.
TOPICS="/landfill/rgb/image_raw /landfill/thermal/image_raw /landfill/livox/points /landfill/rgb/camera_info /landfill/thermal/camera_info"

mkdir -p "$OUT_DIR"
source /opt/ros/jazzy/setup.bash
source /home/vish/ros2_ws/install/setup.bash 2>/dev/null || true
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST

POSES="close far left right centre"

echo "This will record $DURATION seconds per pose for: $POSES"
echo "Recommended distances/positions (adjust to your room):"
echo "  close  -- board ~1.5-2m from the sensor rig"
echo "  far    -- board ~4-5m from the sensor rig"
echo "  left   -- board ~3m, offset left of centre, angled slightly toward the sensor"
echo "  right  -- board ~3m, offset right of centre, angled slightly toward the sensor"
echo "  centre -- board ~3m, directly centred and facing the sensor"
echo ""
echo "Varying distance AND angle (not just left-right position) matters for the LiDAR-camera"
echo "solve -- see calibration/common/plane_extrinsic_solver.py's docstring for why."
echo ""

for pose in $POSES; do
  bag_path="$OUT_DIR/$pose"
  if [ -d "$bag_path" ]; then
    echo "WARNING: $bag_path already exists -- delete it first if you want to re-record this pose."
    read -p "Skip this pose and continue? [y/N] " skip
    [ "$skip" = "y" ] && continue
    rm -rf "$bag_path"
  fi
  echo ""
  echo "=== Pose: $pose ==="
  read -p "Position the board, hold it steady, then press Enter to start recording ($DURATION s)... " _
  ros2 bag record -s mcap -o "$bag_path" $TOPICS < /dev/null &
  REC_PID=$!
  sleep "$DURATION"
  # Escalating stop: SIGINT alone was found NOT to reliably stop this
  # recorder when its stdin isn't a real terminal (it logs "keyboard
  # handling disabled" in that case, and its normal rclpy SIGINT handler
  # doesn't seem to engage either) -- a first real recording ran for over
  # 3 minutes and wrote 3.5GB before this was caught. Never trust a single
  # signal to actually stop it; always verify and escalate.
  kill -INT $REC_PID 2>/dev/null || true
  for i in 1 2; do
    kill -0 $REC_PID 2>/dev/null || break
    sleep 1
  done
  if kill -0 $REC_PID 2>/dev/null; then
    # Confirmed by testing: SIGINT reliably does NOT stop this recorder in
    # this backgrounded/non-tty context, but SIGTERM reliably DOES (stops
    # cleanly, flushes properly) -- this isn't a last-resort path, it's the
    # normal one. The short SIGINT wait above is just in case that changes
    # on a different ros2bag version.
    echo "  (sending SIGTERM...)"
    kill -TERM $REC_PID 2>/dev/null || true
    sleep 2
  fi
  if kill -0 $REC_PID 2>/dev/null; then
    echo "  (still alive, forcing with SIGKILL -- this bag's trailing seconds may be unflushed)"
    kill -9 $REC_PID 2>/dev/null || true
    sleep 1
  fi
  wait $REC_PID 2>/dev/null || true
  size=$(du -sh "$bag_path" 2>/dev/null | cut -f1)
  echo "Recorded $bag_path ($size)"
  if [ -n "$size" ]; then
    size_mb=$(du -sm "$bag_path" 2>/dev/null | cut -f1)
    if [ "$size_mb" -gt 200 ]; then
      echo "  WARNING: $bag_path is ${size}, much bigger than a $DURATION second capture should be" \
           "-- the stop likely failed again. Check it before trusting this pose's data."
    fi
  fi
done

echo ""
echo "Done. Bags in $OUT_DIR:"
ls -la "$OUT_DIR"
echo ""
echo "Next steps (see README.md for the full sequence):"
echo "  1. calibrate RGB intrinsics from one bag (e.g. centre)"
echo "  2. calibrate thermal intrinsics the same way (if check_thermal_contrast.py looked good)"
echo "  3. run lidar_camera_extrinsic.py with all 5 bags"
echo "  4. run rgb_thermal_stereo.py with one bag"
echo "  5. run compose_lidar_thermal.py to get lidar->thermal"
