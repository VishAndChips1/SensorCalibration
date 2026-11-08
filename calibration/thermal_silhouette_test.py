#!/usr/bin/env python3
"""Tests whether the board's whole-silhouette can be localized in THERMAL
frames (no internal pattern needed -- see results/STATUS.md on why the
checkerboard pattern itself isn't visible in thermal) well enough to be
useful, and cross-checks the resulting plane orientation against the
same-moment RGB ChArUco ground truth.

Two things are reported, kept separate on purpose:
1. Detection-only (no intrinsics needed): how often a single big
   plausible quadrilateral blob is found, and whether its aspect ratio
   is close to the board's known physical ratio. This part doesn't
   depend on having a calibrated thermal camera at all.
2. A rough 3D normal-direction cross-check against RGB ground truth,
   USING PitFusion3D's placeholder thermal intrinsics (fx=fy=919, not a
   real calibration -- thermal intrinsics calibration is still blocked,
   see STATUS.md). This part's absolute accuracy is not trustworthy; it
   only checks "is the sign/rough direction plausible," not "how many mm."

Usage: thermal_silhouette_test.py <output.yaml> [--bags bags/close ...]
"""
import argparse
import os
import sys

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'common'))
from charuco_compat import load_board_and_dictionary, detect_charuco, board_object_points  # noqa: E402
from bag_reader import read_bag, image_msg_to_numpy, normalize_to_u8  # noqa: E402
from silhouette_plane_pose import find_board_quad, solve_quad_pose  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('output_yaml')
    ap.add_argument('--bags', nargs='+', default=['bags/close', 'bags/far', 'bags/left',
                                                    'bags/right', 'bags/centre'])
    ap.add_argument('--rgb-topic', default='/landfill/rgb/image_raw')
    ap.add_argument('--thermal-topic', default='/landfill/thermal/image_raw')
    ap.add_argument('--board', default=os.path.join(os.path.dirname(__file__), '..', 'config', 'board_config.yaml'))
    ap.add_argument('--rgb-intrinsics', default=os.path.join(os.path.dirname(__file__), '..', 'results',
                                                               'rgb_intrinsics.yaml'))
    ap.add_argument('--max-time-skew-ms', type=float, default=60.0)
    args = ap.parse_args()

    with open(args.board) as f:
        cfg = yaml.safe_load(f)
    board, dictionary = load_board_and_dictionary(cfg)
    expected_aspect = max(cfg['squares_y'], cfg['squares_x']) / min(cfg['squares_y'], cfg['squares_x'])
    board_w, board_h = cfg['squares_x'] * cfg['square_length_m'], cfg['squares_y'] * cfg['square_length_m']

    with open(args.rgb_intrinsics) as f:
        rgb_intr = yaml.safe_load(f)
    K_rgb = np.array(rgb_intr['camera_matrix'])
    D_rgb = np.array(rgb_intr['distortion_coefficients'])
    # PitFusion3D's placeholder thermal intrinsics (config/calibration/intrinsics.yaml in
    # PitFusion3D_repo) -- NOT a real calibration, see module docstring.
    K_th = np.array([[919.0, 0, 320.0], [0, 919.0, 256.0], [0, 0, 1]])
    D_th = np.zeros(5)

    n_thermal_frames = 0
    n_quad_found = 0
    aspect_ratios = []
    n_pairs_crosschecked = 0
    normal_angle_errs = []

    for bag_path in args.bags:
        rgb_buf, th_buf = [], []
        for topic, msg, t in read_bag(bag_path, topics=[args.rgb_topic, args.thermal_topic]):
            if topic == args.thermal_topic:
                n_thermal_frames += 1
                th_gray = normalize_to_u8(image_msg_to_numpy(msg))
                quad = find_board_quad(th_gray, expect_bright=True)
                th_buf.append((t, th_gray, quad))
            else:
                rgb_buf.append((t, image_msg_to_numpy(msg)))

        for t_th, th_gray, quad in th_buf:
            if quad is None:
                continue
            n_quad_found += 1
            corners, aspect = quad
            aspect_ratios.append(aspect)

            best = min(rgb_buf, key=lambda r: abs(r[0] - t_th), default=None)
            if best is None or abs(best[0] - t_th) / 1e6 > args.max_time_skew_ms:
                continue
            _t_rgb, rgb_img = best
            rgb_gray = rgb_img if rgb_img.ndim == 2 else cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
            rc, rids, _mc, _mi = detect_charuco(rgb_gray, board, dictionary)
            if rc is None or rids is None or len(rids) < 20:
                continue
            obj_pts = board_object_points(board, rids)
            ok, rvec_rgb, _tvec_rgb = cv2.solvePnP(obj_pts, rc, K_rgb, D_rgb)
            if not ok:
                continue
            R_rgb, _ = cv2.Rodrigues(rvec_rgb)
            n_rgb = R_rgb @ np.array([0.0, 0.0, 1.0])

            pose = solve_quad_pose(corners, board_w, board_h, K_th, D_th)
            if pose is None:
                continue
            _rvec_th, _tvec_th, n_th = pose
            cos_a = np.clip(np.dot(n_rgb, n_th) / (np.linalg.norm(n_rgb) * np.linalg.norm(n_th)), -1, 1)
            angle = float(np.degrees(np.arccos(cos_a)))
            # The board-vs-camera facing convention can come out flipped
            # (normal pointing away vs towards the camera) depending on
            # corner winding -- both conventions are "correct" information,
            # so take whichever flip is closer to RGB's convention.
            angle = min(angle, 180 - angle)
            normal_angle_errs.append(angle)
            n_pairs_crosschecked += 1

    print(f'Thermal frames seen: {n_thermal_frames}. Plausible board-silhouette quad found in: '
          f'{n_quad_found} ({100 * n_quad_found / max(1, n_thermal_frames):.1f}%).')
    if aspect_ratios:
        a = np.array(aspect_ratios)
        print(f'Detected quad aspect ratio: mean={a.mean():.2f}, median={np.median(a):.2f} '
              f'(board\'s real aspect ratio is {expected_aspect:.2f} -- closer is better; a hand or '
              'background clutter being picked up instead of the board would show as a mismatch here)')
    if normal_angle_errs:
        e = np.array(normal_angle_errs)
        print(f'Cross-checked against time-paired RGB ground truth in {n_pairs_crosschecked} frames: '
              f'normal-direction error mean={e.mean():.1f} deg, median={np.median(e):.1f} deg, '
              f'p90={np.percentile(e, 90):.1f} deg. (Uses an UNCALIBRATED thermal-intrinsics placeholder '
              '-- see module docstring -- so treat this as "is the rough direction plausible," not a '
              'real accuracy number.)')
    else:
        print('No thermal-silhouette/RGB-ground-truth pairs could be cross-checked (either no quad '
              'found with a time-close well-detected RGB frame, or solvePnP failed on all of them).')

    with open(args.output_yaml, 'w') as f:
        yaml.safe_dump({
            'n_thermal_frames': n_thermal_frames, 'n_quad_found': n_quad_found,
            'detection_rate': n_quad_found / max(1, n_thermal_frames),
            'aspect_ratio_mean': float(np.mean(aspect_ratios)) if aspect_ratios else None,
            'expected_aspect_ratio': expected_aspect,
            'n_pairs_crosschecked': n_pairs_crosschecked,
            'normal_angle_err_deg_mean': float(np.mean(normal_angle_errs)) if normal_angle_errs else None,
            'normal_angle_err_deg_p90': float(np.percentile(normal_angle_errs, 90)) if normal_angle_errs else None,
            'note': 'cross-check uses an uncalibrated thermal-intrinsics placeholder (fx=fy=919, '
            'PitFusion3D_repo\'s own pre-existing rough guess) -- not a real accuracy number, see docstring.',
        }, f, default_flow_style=False)
    print(f'Wrote {args.output_yaml}')


if __name__ == '__main__':
    main()
