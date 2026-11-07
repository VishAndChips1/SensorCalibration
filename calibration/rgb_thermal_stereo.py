#!/usr/bin/env python3
"""RGB<->thermal extrinsic calibration via standard OpenCV stereo
calibration. This only needs BOTH cameras to detect the ChArUco board in
the SAME synchronized frame pair -- once that works, it's textbook stereo
calibration (cv2.stereoCalibrate), no custom geometry needed. This is the
easy half of the two sensor-to-sensor calibrations this project needs; see
lidar_camera_extrinsic.py for the harder one.

Why do this instead of calibrating thermal-to-LiDAR directly: once you have
LiDAR->RGB (from lidar_camera_extrinsic.py) and RGB->thermal (from this
script), LiDAR->thermal is just the composition of the two -- one fewer
hard (LiDAR-involving) calibration to get right. See README.md.

Usage:
    rgb_thermal_stereo.py <rgb_intrinsics.yaml> <thermal_intrinsics.yaml> \\
        <output.yaml> --bag some_bag --rgb-topic ... --thermal-topic ...
        [--board config/board_config.yaml] [--stride 1] [--max-frames 40]
        [--board-type charuco|custom] [--custom-layout config/custom_board_layout.yaml]

Needs a bag where RGB and thermal both saw the board at the same time --
any of the 5 capture-protocol bags work, doesn't need a dedicated capture.
"""
import argparse
import os
import sys

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'common'))
from charuco_compat import load_board_and_dictionary, detect_charuco, get_chessboard_corners  # noqa: E402
from bag_reader import read_bag, image_msg_to_numpy, normalize_to_u8  # noqa: E402
import custom_board  # noqa: E402


def to_gray(img):
    return normalize_to_u8(img) if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('rgb_intrinsics_yaml')
    ap.add_argument('thermal_intrinsics_yaml')
    ap.add_argument('output_yaml')
    ap.add_argument('--bag', required=True)
    ap.add_argument('--rgb-topic', required=True)
    ap.add_argument('--thermal-topic', required=True)
    ap.add_argument('--board', default=os.path.join(os.path.dirname(__file__), '..', 'config', 'board_config.yaml'))
    ap.add_argument('--board-type', choices=['charuco', 'custom'], default='charuco')
    ap.add_argument('--custom-layout',
                     default=os.path.join(os.path.dirname(__file__), '..', 'config', 'custom_board_layout.yaml'))
    ap.add_argument('--max-pairs', type=int, default=40)
    ap.add_argument('--max-time-skew-ms', type=float, default=50.0,
                     help='max gap between an RGB and thermal message to treat them as the '
                          '"same frame" -- these topics are published independently, not '
                          'synchronized at the driver level, so some skew is expected')
    args = ap.parse_args()

    if args.board_type == 'charuco':
        with open(args.board) as f:
            cfg = yaml.safe_load(f)
        board, dictionary = load_board_and_dictionary(cfg)
        board_object_points = get_chessboard_corners(board)
    else:
        dictionary, layout = custom_board.load_layout(args.custom_layout)

    with open(args.rgb_intrinsics_yaml) as f:
        rgb_intr = yaml.safe_load(f)
    with open(args.thermal_intrinsics_yaml) as f:
        th_intr = yaml.safe_load(f)
    K_rgb = np.array(rgb_intr['camera_matrix'])
    D_rgb = np.array(rgb_intr['distortion_coefficients'])
    K_th = np.array(th_intr['camera_matrix'])
    D_th = np.array(th_intr['distortion_coefficients'])
    rgb_size = (rgb_intr['image_width'], rgb_intr['image_height'])

    # Buffer recent messages per topic, pair by nearest timestamp within
    # the skew tolerance (these two topics are not hardware-synchronized).
    rgb_buf, th_buf = [], []
    obj_points, rgb_points, th_points = [], [], []
    n_pairs_tried = 0

    def try_pairs():
        nonlocal rgb_buf, th_buf
        while rgb_buf and th_buf:
            rt, rimg = rgb_buf[0]
            tt, timg = th_buf[0]
            skew_ms = abs(rt - tt) / 1e6
            if skew_ms > args.max_time_skew_ms:
                if rt < tt:
                    rgb_buf.pop(0)
                else:
                    th_buf.pop(0)
                continue
            yield rimg, timg
            rgb_buf.pop(0)
            th_buf.pop(0)

    for topic, msg, t in read_bag(args.bag, topics=[args.rgb_topic, args.thermal_topic]):
        if len(obj_points) >= args.max_pairs:
            break
        if topic == args.rgb_topic:
            rgb_buf.append((t, image_msg_to_numpy(msg)))
        else:
            th_buf.append((t, image_msg_to_numpy(msg)))

        for rimg, timg in list(try_pairs()):
            n_pairs_tried += 1
            rgray, tgray = to_gray(rimg), to_gray(timg)

            if args.board_type == 'charuco':
                rc, rids, _, _ = detect_charuco(rgray, board, dictionary)
                tc, tids, _, _ = detect_charuco(tgray, board, dictionary)
                if rc is None or tc is None or len(rids) < 6 or len(tids) < 6:
                    continue
                common_ids = np.intersect1d(rids.flatten(), tids.flatten())
                if len(common_ids) < 6:
                    continue
                r_map = {i: c for i, c in zip(rids.flatten(), rc.reshape(-1, 2))}
                t_map = {i: c for i, c in zip(tids.flatten(), tc.reshape(-1, 2))}
                obj_points.append(board_object_points[common_ids].astype(np.float32))
                rgb_points.append(np.array([r_map[i] for i in common_ids], dtype=np.float32))
                th_points.append(np.array([t_map[i] for i in common_ids], dtype=np.float32))
            else:
                rc, rids = custom_board.detect_markers(rgray, dictionary)
                tc, tids = custom_board.detect_markers(tgray, dictionary)
                if rids is None or tids is None:
                    continue
                common_ids = [i for i in np.intersect1d(rids.flatten(), tids.flatten()) if i in layout]
                if len(common_ids) < 2:  # 2 markers = 8 corner correspondences, enough
                    continue
                r_map = {i: c.reshape(4, 2) for i, c in zip(rids.flatten(), rc)}
                t_map = {i: c.reshape(4, 2) for i, c in zip(tids.flatten(), tc)}
                obj_points.append(np.concatenate([layout[i] for i in common_ids], axis=0).astype(np.float32))
                rgb_points.append(np.concatenate([r_map[i] for i in common_ids], axis=0).astype(np.float32))
                th_points.append(np.concatenate([t_map[i] for i in common_ids], axis=0).astype(np.float32))

    print(f'{n_pairs_tried} time-synced RGB/thermal frame pairs tried, '
          f'{len(obj_points)} had enough shared ChArUco corners on BOTH cameras.')
    if len(obj_points) < 4:
        print('FAILED: not enough dual-visible detections. If thermal detection is the '
              'bottleneck, this is the expected failure mode when the board has no real '
              'thermal contrast -- see board/generate_charuco_board.py\'s docstring on how '
              'to get a temperature differential onto it before capture.')
        sys.exit(1)

    reproj_err, _K1, _D1, _K2, _D2, R, T, _E, _F = cv2.stereoCalibrate(
        obj_points, rgb_points, th_points, K_rgb, D_rgb, K_th, D_th, rgb_size,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )
    roll, pitch, yaw = cv2.RQDecomp3x3(R)[0]

    result = {
        'rotation_matrix': R.tolist(),
        'translation_m': T.flatten().tolist(),
        'rotation_rpy_deg': [float(roll), float(pitch), float(yaw)],
        'reprojection_error_px': float(reproj_err),
        'n_frame_pairs_used': len(obj_points),
        'note': 'Maps an RGB-frame point to the thermal frame via p_thermal = R @ p_rgb + T.',
    }
    with open(args.output_yaml, 'w') as f:
        yaml.safe_dump(result, f, default_flow_style=False)

    print(f'Stereo reprojection error: {reproj_err:.3f}px')
    print(f'Translation (m): {np.array(result["translation_m"]).round(4)}')
    print(f'Rotation (deg rpy): {roll:.2f}, {pitch:.2f}, {yaw:.2f}')
    print(f'Wrote {args.output_yaml}')


if __name__ == '__main__':
    main()
