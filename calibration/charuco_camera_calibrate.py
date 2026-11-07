#!/usr/bin/env python3
"""Intrinsic calibration for ONE camera (RGB or thermal -- same code path;
see the module docstring in common/charuco_compat.py and this project's
README for why that works) from a set of ChArUco board detections across a
rosbag2 recording.

This is standard, fully generic camera calibration: works for any camera
resolution and any ChArUco board, as long as config/board_config.yaml
matches what was printed. The thermal case is the only non-obvious part --
it works IF (and only if) the board shows real pixel contrast in the
thermal image, which depends on getting a temperature differential onto it
before capture (see board/generate_charuco_board.py's docstring). This
script tells you plainly if detection isn't finding enough frames, rather
than silently calibrating against too little data.

Usage:
    charuco_camera_calibrate.py <bag_path> <image_topic> <output.yaml>
        [--board config/board_config.yaml] [--max-frames 60] [--stride 5]
        [--preview out_dir]   # saves annotated detections for a visual check

Output YAML: camera_matrix (3x3), distortion_coefficients, image_width/
height, reprojection_error_px, n_frames_used, n_frames_total.
"""
import argparse
import os
import sys

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'common'))
from charuco_compat import load_board_and_dictionary, detect_charuco, calibrate_camera_charuco  # noqa: E402
from bag_reader import read_bag, image_msg_to_numpy  # noqa: E402


def to_gray(img):
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag_path')
    ap.add_argument('image_topic')
    ap.add_argument('output_yaml')
    ap.add_argument('--board', default=os.path.join(os.path.dirname(__file__), '..', 'config', 'board_config.yaml'))
    ap.add_argument('--max-frames', type=int, default=60,
                     help='cap how many detected frames feed the solver -- more than ~60 well-'
                          'spread detections rarely improves accuracy and does slow the solve down')
    ap.add_argument('--stride', type=int, default=1,
                     help='only attempt detection on every Nth message (bags are often much '
                          'denser than needed for calibration frame diversity)')
    ap.add_argument('--min-corners', type=int, default=6,
                     help='a detection with fewer than this many ChArUco corners found is '
                          'discarded -- too few points make that frame\'s pose poorly constrained')
    ap.add_argument('--preview', default=None,
                     help='directory to save annotated detection images for a visual sanity check')
    args = ap.parse_args()

    with open(args.board) as f:
        cfg = yaml.safe_load(f)
    board, dictionary = load_board_and_dictionary(cfg)

    if args.preview:
        os.makedirs(args.preview, exist_ok=True)

    all_corners, all_ids = [], []
    image_size = None
    n_total = 0
    n_detected = 0

    for i, (_topic, msg, _t) in enumerate(read_bag(args.bag_path, topics=[args.image_topic])):
        if i % args.stride != 0:
            continue
        n_total += 1
        img = image_msg_to_numpy(msg)
        gray = to_gray(img)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])

        corners, ids, marker_corners, marker_ids = detect_charuco(gray, board, dictionary)
        has_enough = corners is not None and ids is not None and len(ids) >= args.min_corners

        if args.preview and (has_enough or n_total <= 5):
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            if marker_ids is not None and len(marker_ids) > 0:
                cv2.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)
            if has_enough:
                cv2.aruco.drawDetectedCornersCharuco(vis, corners, ids)
            cv2.imwrite(os.path.join(args.preview, f'frame_{n_total:04d}.png'), vis)

        if not has_enough:
            continue
        n_detected += 1
        all_corners.append(corners)
        all_ids.append(ids)
        if n_detected >= args.max_frames:
            break

    print(f'{args.image_topic}: {n_detected}/{n_total} sampled frames had >= {args.min_corners} '
          f'ChArUco corners detected.')

    if n_detected < 4:
        print(f'FAILED: only {n_detected} usable detections -- camera calibration needs the '
              'board to be seen clearly from several different angles/distances/positions in '
              'the frame. Check --preview output if this was unexpected (especially for '
              'thermal: see board/generate_charuco_board.py for how to get thermal contrast).')
        sys.exit(1)

    reproj_err, camera_matrix, dist_coeffs, _rvecs, _tvecs = calibrate_camera_charuco(
        all_corners, all_ids, board, image_size)

    result = {
        'image_width': image_size[0],
        'image_height': image_size[1],
        'camera_matrix': camera_matrix.tolist(),
        'distortion_coefficients': dist_coeffs.flatten().tolist(),
        'reprojection_error_px': float(reproj_err),
        'n_frames_used': n_detected,
        'n_frames_sampled': n_total,
        'source_bag': os.path.abspath(args.bag_path),
        'source_topic': args.image_topic,
    }
    with open(args.output_yaml, 'w') as f:
        yaml.safe_dump(result, f, default_flow_style=False)

    print(f'Reprojection error: {reproj_err:.3f} px (good calibrations are typically well under '
          '1px; anything over ~1px is worth investigating before trusting the result -- loose '
          'mounting, a warped board, or motion blur during capture are the usual causes)')
    print(f'fx={camera_matrix[0,0]:.1f} fy={camera_matrix[1,1]:.1f} '
          f'cx={camera_matrix[0,2]:.1f} cy={camera_matrix[1,2]:.1f}')
    print(f'Wrote {args.output_yaml}')


if __name__ == '__main__':
    main()
