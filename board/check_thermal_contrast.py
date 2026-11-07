#!/usr/bin/env python3
"""Quick live check: is the board showing enough thermal contrast to be
worth a full capture session? Grabs a few frames from the live thermal
topic, tries ChArUco detection directly on them, and reports plainly
whether it's working -- run this BEFORE setting up a 5-pose capture, not
after, so a weak-contrast board doesn't waste a whole session.

Usage: check_thermal_contrast.py [--topic /landfill/thermal/image_raw] [--duration 10]
Needs the pipeline running live (this reads from a ROS2 topic directly, not
a bag -- there's nothing to calibrate against yet at this stage).
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'calibration', 'common'))
import yaml  # noqa: E402
from charuco_compat import load_board_and_dictionary, detect_charuco  # noqa: E402

BOARD_CFG = os.path.join(os.path.dirname(__file__), '..', 'config', 'board_config.yaml')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--topic', default='/landfill/thermal/image_raw')
    ap.add_argument('--duration', type=float, default=10.0)
    ap.add_argument('--save', default=os.path.join(os.path.dirname(__file__), 'thermal_contrast_check.png'))
    args = ap.parse_args()

    with open(BOARD_CFG) as f:
        cfg = yaml.safe_load(f)
    board, dictionary = load_board_and_dictionary(cfg)

    rclpy.init()
    node = Node('check_thermal_contrast')
    frames = []

    def cb(msg):
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        img = arr.reshape(msg.height, msg.width, -1)
        gray = img[:, :, 0] if img.ndim == 3 else img
        frames.append(gray)

    node.create_subscription(Image, args.topic, cb, qos_profile_sensor_data)

    print(f'Listening on {args.topic} for {args.duration}s -- point the board at the camera now...')
    t_end = time.monotonic() + args.duration
    while time.monotonic() < t_end and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.2)

    node.destroy_node()
    rclpy.shutdown()

    if not frames:
        print(f'FAILED: no frames received on {args.topic} -- is the pipeline running?')
        sys.exit(1)

    best = None
    for gray in frames:
        contrast = float(gray.std())
        corners, ids, marker_corners, marker_ids = detect_charuco(gray, board, dictionary)
        n_markers = 0 if marker_ids is None else len(marker_ids)
        n_corners = 0 if ids is None else len(ids)
        if best is None or n_corners > best[2]:
            best = (gray, contrast, n_corners, n_markers)

    gray, contrast, n_corners, n_markers = best
    print(f'\n{len(frames)} frames checked. Best frame: pixel std-dev {contrast:.1f}/255, '
          f'{n_markers} ArUco markers detected, {n_corners} ChArUco corners detected.')

    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(args.save, vis)
    print(f'Saved best frame to {args.save} -- look at it directly, numbers alone can mislead.')

    if n_corners >= 6:
        print('LOOKS GOOD: enough corners detected to be usable for calibration. '
              'Proceed with the full 5-pose capture.')
    elif n_markers >= 3:
        print('MARGINAL: markers are visible but few/no checkerboard corners interpolated -- '
              'contrast may be too weak or uneven. Try warming the board more, or switch to '
              'the foil-tape approach (see generate_charuco_board.py).')
    else:
        print('NOT WORKING YET: no markers detected at all. The board likely has little to no '
              'real thermal contrast right now -- this is the expected result for a plain '
              'room-temperature paper print. Try warming it (near a warm surface / brief sun '
              'exposure) and re-run this check immediately after, or go straight to the '
              'foil-tape approach for a reliable fix.')


if __name__ == '__main__':
    main()
