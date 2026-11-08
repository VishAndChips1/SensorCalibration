#!/usr/bin/env python3
"""One-shot live diagnostic: grab a single RGB frame, thermal frame, and
LiDAR scan from the live pipeline, and report what's actually in front of
the sensors right now -- board distance/visibility in each modality --
without needing a human to look at anything. Meant for unattended runs
where no one can confirm framing by eye.

Usage: live_scene_check.py <out_dir> [--wait 10]
  --rgb-topic /landfill/rgb/image_raw
  --thermal-topic /landfill/thermal/image_raw
  --lidar-topic /landfill/livox/points

Writes <out_dir>/rgb.png, thermal_raw.png (normalized for viewing),
thermal_raw.npy (actual uint16 values), lidar_xyz.npy, and prints a
distance histogram + ArUco detection attempt on the RGB frame.
"""
import argparse
import os
import sys

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'calibration', 'common'))
from bag_reader import image_msg_to_numpy, pointcloud2_to_xyz, normalize_to_u8  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out_dir')
    ap.add_argument('--wait', type=float, default=10.0)
    ap.add_argument('--rgb-topic', default='/landfill/rgb/image_raw')
    ap.add_argument('--thermal-topic', default='/landfill/thermal/image_raw')
    ap.add_argument('--lidar-topic', default='/landfill/livox/points')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rclpy.init()
    node = Node('live_scene_check')
    got = {}

    def mk_cb(key):
        def cb(msg):
            got.setdefault(key, msg)
        return cb

    node.create_subscription(Image, args.rgb_topic, mk_cb('rgb'), qos_profile_sensor_data)
    node.create_subscription(Image, args.thermal_topic, mk_cb('thermal'), qos_profile_sensor_data)
    node.create_subscription(PointCloud2, args.lidar_topic, mk_cb('lidar'), qos_profile_sensor_data)

    import time
    t0 = time.monotonic()
    while len(got) < 3 and time.monotonic() - t0 < args.wait and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.2)
    node.destroy_node()
    rclpy.shutdown()

    missing = [k for k in ('rgb', 'thermal', 'lidar') if k not in got]
    if missing:
        print(f'FAILED: no message received for {missing} in {args.wait}s -- is the pipeline running?')
        sys.exit(1)

    rgb = image_msg_to_numpy(got['rgb'])
    cv2.imwrite(os.path.join(args.out_dir, 'rgb.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                if rgb.ndim == 3 else rgb)

    thermal_raw = image_msg_to_numpy(got['thermal'])
    np.save(os.path.join(args.out_dir, 'thermal_raw.npy'), thermal_raw)
    cv2.imwrite(os.path.join(args.out_dir, 'thermal_view.png'), normalize_to_u8(thermal_raw))
    t = thermal_raw.astype(np.float64)
    print(f'Thermal: {t.shape}, raw range [{t.min():.0f},{t.max():.0f}], '
          f'mean={t.mean():.1f}, std={t.std():.1f} (AGC units, not Kelvin)')

    xyz = pointcloud2_to_xyz(got['lidar'])
    np.save(os.path.join(args.out_dir, 'lidar_xyz.npy'), xyz)
    if xyz.shape[0] == 0:
        print('LiDAR: 0 points in this single scan.')
    else:
        d = np.linalg.norm(xyz, axis=1)
        print(f'LiDAR: {xyz.shape[0]} points this scan. Distance histogram (m):')
        for lo in np.arange(0.0, 3.01, 0.25):
            hi = lo + 0.25
            n = int(((d >= lo) & (d < hi)).sum())
            if n:
                print(f'  [{lo:.2f},{hi:.2f}): {n}')
        n_close = int((d < 1.0).sum())
        print(f'  Total points <1.0m (LiDAR blind-zone boundary): {n_close}')

    gray = rgb if rgb.ndim == 2 else cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters_create() if hasattr(cv2.aruco, 'DetectorParameters_create') \
        else cv2.aruco.DetectorParameters()
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
    n_markers = 0 if ids is None else len(ids)
    print(f'RGB: {gray.shape[1]}x{gray.shape[0]}, {n_markers} ArUco markers decoded this frame.')
    if n_markers:
        sizes = [cv2.contourArea(c.reshape(4, 2).astype(np.float32)) for c in corners]
        print(f'  Marker pixel areas: min={min(sizes):.0f} max={max(sizes):.0f} '
              f'(need roughly >=400-900px^2 / ~20-30px side to decode reliably)')

    print(f'Saved outputs to {args.out_dir}/')


if __name__ == '__main__':
    main()
