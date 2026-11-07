#!/usr/bin/env python3
"""Grab a single frame from a live ROS2 image topic and save it as PNG --
for getting a quick photo of your physical board to run through
identify_board_dictionary.py, without needing a bag or any other tooling.

Usage: grab_one_frame.py <topic> <output.png> [--wait 10]
"""
import argparse
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from PIL import Image as PILImage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('topic')
    ap.add_argument('output')
    ap.add_argument('--wait', type=float, default=10.0)
    args = ap.parse_args()

    rclpy.init()
    node = Node('grab_one_frame')
    got = {}

    def cb(msg):
        if 'img' in got:
            return
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        img = arr.reshape(msg.height, msg.width, -1)
        got['img'] = img[:, :, :3] if img.shape[2] >= 3 else img[:, :, 0]

    node.create_subscription(Image, args.topic, cb, qos_profile_sensor_data)

    print(f'Waiting up to {args.wait}s for a frame on {args.topic} -- hold the board in view now.')
    t_end = args.wait
    import time
    t0 = time.monotonic()
    while 'img' not in got and time.monotonic() - t0 < t_end and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.2)

    node.destroy_node()
    rclpy.shutdown()

    if 'img' not in got:
        print(f'FAILED: no frame received on {args.topic} in {args.wait}s -- is the pipeline running?')
        sys.exit(1)

    img = got['img']
    if img.ndim == 3:
        PILImage.fromarray(img).save(args.output)
    else:
        PILImage.fromarray(img, mode='L').save(args.output)
    print(f'Saved {args.output} ({img.shape[1]}x{img.shape[0]})')


if __name__ == '__main__':
    main()
