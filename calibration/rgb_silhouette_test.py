#!/usr/bin/env python3
"""Same silhouette-detection test as thermal_silhouette_test.py, but on
RGB: can the board's outer rectangle be found by plain contour/threshold
detection (no ArUco decoding), and does it land in roughly the right
place? Checked against the SAME frame's full ChArUco detection (no time
pairing needed, unlike the thermal case) by comparing the silhouette
quad's centroid to the mean of all detected ArUco marker centers.

Usage: rgb_silhouette_test.py <output.yaml> [--bags bags/close ...]
"""
import argparse
import os
import sys

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'common'))
from charuco_compat import load_board_and_dictionary, detect_charuco  # noqa: E402
from bag_reader import read_bag, image_msg_to_numpy, normalize_to_u8  # noqa: E402
from silhouette_plane_pose import find_board_quad  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('output_yaml')
    ap.add_argument('--bags', nargs='+', default=['bags/close', 'bags/far', 'bags/left',
                                                    'bags/right', 'bags/centre'])
    ap.add_argument('--rgb-topic', default='/landfill/rgb/image_raw')
    ap.add_argument('--board', default=os.path.join(os.path.dirname(__file__), '..', 'config', 'board_config.yaml'))
    args = ap.parse_args()

    with open(args.board) as f:
        cfg = yaml.safe_load(f)
    board, dictionary = load_board_and_dictionary(cfg)
    expected_aspect = max(cfg['squares_y'], cfg['squares_x']) / min(cfg['squares_y'], cfg['squares_x'])

    n_frames = 0
    n_quad_found = 0
    n_markers_available = 0
    centroid_errs_px = []
    aspect_ratios = []

    for bag_path in args.bags:
        for topic, msg, _t in read_bag(bag_path, topics=[args.rgb_topic]):
            n_frames += 1
            img = image_msg_to_numpy(msg)
            gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            quad = None
            for polarity in (True, False):
                q = find_board_quad(gray, expect_bright=polarity, min_area_frac=0.05)
                if q is not None:
                    quad = q
                    break
            if quad is None:
                continue
            n_quad_found += 1
            corners, aspect = quad
            aspect_ratios.append(aspect)
            quad_centroid = corners.mean(axis=0)

            _rc, _rids, marker_corners, marker_ids = detect_charuco(gray, board, dictionary)
            if marker_ids is None or len(marker_ids) < 4:
                continue
            n_markers_available += 1
            all_marker_pts = np.concatenate([c.reshape(4, 2) for c in marker_corners], axis=0)
            board_centroid = all_marker_pts.mean(axis=0)
            centroid_errs_px.append(float(np.linalg.norm(quad_centroid - board_centroid)))

    print(f'RGB frames: {n_frames}. Plausible silhouette quad found: {n_quad_found} '
          f'({100 * n_quad_found / max(1, n_frames):.1f}%).')
    if aspect_ratios:
        a = np.array(aspect_ratios)
        print(f'Quad aspect ratio: mean={a.mean():.2f} (board real ratio {expected_aspect:.2f})')
    if centroid_errs_px:
        c = np.array(centroid_errs_px)
        print(f'Silhouette-centroid vs. ArUco-centroid offset: mean={c.mean():.1f}px, '
              f'median={np.median(c):.1f}px, p90={np.percentile(c, 90):.1f}px, out of '
              f'{n_markers_available} frames with >=4 markers also decoded for comparison.')

    with open(args.output_yaml, 'w') as f:
        yaml.safe_dump({
            'n_frames': n_frames, 'n_quad_found': n_quad_found,
            'detection_rate': n_quad_found / max(1, n_frames),
            'aspect_ratio_mean': float(np.mean(aspect_ratios)) if aspect_ratios else None,
            'expected_aspect_ratio': expected_aspect,
            'centroid_offset_px_mean': float(np.mean(centroid_errs_px)) if centroid_errs_px else None,
            'centroid_offset_px_p90': float(np.percentile(centroid_errs_px, 90)) if centroid_errs_px else None,
        }, f, default_flow_style=False)
    print(f'Wrote {args.output_yaml}')


if __name__ == '__main__':
    main()
