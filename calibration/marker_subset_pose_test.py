#!/usr/bin/env python3
"""Tests whether a COARSE board-pose estimate (just a handful of ArUco
markers, or even just the board's outer silhouette -- not a full ChArUco
detection) is accurate enough to be useful for extrinsic calibration, as
an alternative to needing the full 35-marker/48-corner detection that
limits usable range to ~25-40cm for this board.

Method: for every frame where the FULL ChArUco board detects (ground
truth, uses every corner found), also solve the pose from just a random
subset of N of that same frame's detected ArUco markers (N=1,2,4,8,all),
using only those markers' own corners -- never the interpolated
chessboard corners. Compare the subset pose to the full-detection pose
(same frame, same camera model, so this measures relative degradation
from less information, not absolute accuracy against the real world).
Repeats several random subsets per N per frame for robustness.

This does NOT need a live board or new captures -- runs entirely against
the already-recorded close/far/left/right/centre bags.

Usage: marker_subset_pose_test.py <output.yaml> [--bags bags/close ...]
    [--rgb-topic /landfill/rgb/image_raw] [--intrinsics results/rgb_intrinsics.yaml]
    [--trials-per-n 8] [--min-full-corners 20]
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


def to_gray(img):
    return normalize_to_u8(img) if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def pose_normal_and_t(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    n = R @ np.array([0.0, 0.0, 1.0])
    return n, tvec.flatten()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('output_yaml')
    ap.add_argument('--bags', nargs='+', default=['bags/close', 'bags/far', 'bags/left',
                                                    'bags/right', 'bags/centre'])
    ap.add_argument('--rgb-topic', default='/landfill/rgb/image_raw')
    ap.add_argument('--board', default=os.path.join(os.path.dirname(__file__), '..', 'config', 'board_config.yaml'))
    ap.add_argument('--intrinsics', default=os.path.join(os.path.dirname(__file__), '..', 'results',
                                                           'rgb_intrinsics.yaml'))
    ap.add_argument('--trials-per-n', type=int, default=8)
    ap.add_argument('--min-full-corners', type=int, default=20,
                     help='only use frames this well-detected as "ground truth" -- a noisy full '
                          'detection would make a poor reference to compare subsets against')
    ap.add_argument('--n-values', type=int, nargs='+', default=[1, 2, 4, 8])
    args = ap.parse_args()

    with open(args.board) as f:
        cfg = yaml.safe_load(f)
    board, dictionary = load_board_and_dictionary(cfg)

    with open(args.intrinsics) as f:
        intr = yaml.safe_load(f)
    K = np.array(intr['camera_matrix'])
    D = np.array(intr['distortion_coefficients'])

    rng = np.random.default_rng(0)
    results = {n: {'angle_deg': [], 'trans_m': []} for n in args.n_values}
    results['all'] = {'angle_deg': [], 'trans_m': []}
    n_frames_used = 0
    n_frames_seen = 0

    for bag_path in args.bags:
        for _topic, msg, _t in read_bag(bag_path, topics=[args.rgb_topic]):
            n_frames_seen += 1
            gray = to_gray(image_msg_to_numpy(msg))
            corners, ids, marker_corners, marker_ids = detect_charuco(gray, board, dictionary)
            if corners is None or ids is None or len(ids) < args.min_full_corners:
                continue
            if marker_ids is None or len(marker_ids) < 2:
                continue
            n_frames_used += 1

            obj_pts = board_object_points(board, ids)
            ok, rvec_full, tvec_full = cv2.solvePnP(obj_pts, corners, K, D)
            if not ok:
                continue
            n_full, t_full = pose_normal_and_t(rvec_full, tvec_full)

            m_ids = marker_ids.flatten()
            n_markers = len(m_ids)
            board_ids_list = list(board.ids.flatten()) if hasattr(board, 'ids') else list(range(len(board.objPoints)))

            def marker_obj_corners(marker_id):
                idx = board_ids_list.index(int(marker_id))
                return np.array(board.objPoints[idx], dtype=np.float32)

            for n_use in list(args.n_values) + ['all']:
                n_actual = n_markers if n_use == 'all' else n_use
                if n_actual > n_markers:
                    continue
                trials = 1 if n_use == 'all' else args.trials_per_n
                for _trial in range(trials):
                    chosen = m_ids if n_use == 'all' else rng.choice(m_ids, size=n_actual, replace=False)
                    obj_sub = np.concatenate([marker_obj_corners(mid) for mid in chosen], axis=0)
                    img_sub = np.concatenate(
                        [marker_corners[list(m_ids).index(mid)].reshape(4, 2) for mid in chosen], axis=0
                    ).astype(np.float32)
                    if len(obj_sub) < 4:
                        continue
                    ok2, rvec_s, tvec_s = cv2.solvePnP(obj_sub, img_sub, K, D)
                    if not ok2:
                        continue
                    n_s, t_s = pose_normal_and_t(rvec_s, tvec_s)
                    cos_angle = np.clip(np.dot(n_full, n_s) / (np.linalg.norm(n_full) * np.linalg.norm(n_s)), -1, 1)
                    angle_deg = float(np.degrees(np.arccos(cos_angle)))
                    trans_err = float(np.linalg.norm(t_s - t_full))
                    results[n_use]['angle_deg'].append(angle_deg)
                    results[n_use]['trans_m'].append(trans_err)

    print(f'{n_frames_used}/{n_frames_seen} frames had a full ground-truth detection '
          f'(>= {args.min_full_corners} corners) and >= 2 markers to subsample from.')

    summary = {}
    for key, vals in results.items():
        if not vals['angle_deg']:
            summary[str(key)] = {'n_samples': 0}
            continue
        a = np.array(vals['angle_deg'])
        t = np.array(vals['trans_m'])
        summary[str(key)] = {
            'n_samples': int(len(a)),
            'angle_deg_mean': float(a.mean()), 'angle_deg_median': float(np.median(a)),
            'angle_deg_p90': float(np.percentile(a, 90)),
            'trans_m_mean': float(t.mean()), 'trans_m_median': float(np.median(t)),
            'trans_m_p90': float(np.percentile(t, 90)),
        }
        print(f'N={key}: n={len(a)}, normal-angle err mean={a.mean():.2f} deg (p90={np.percentile(a,90):.2f}), '
              f'translation err mean={t.mean()*1000:.1f}mm (p90={np.percentile(t,90)*1000:.1f}mm)')

    with open(args.output_yaml, 'w') as f:
        yaml.safe_dump({
            'n_frames_used': n_frames_used, 'n_frames_seen': n_frames_seen,
            'by_n_markers': summary, 'note': 'angle/translation error of a pose solved from only N '
            'randomly-chosen markers, vs. the same-frame full-detection pose (relative degradation, '
            'not absolute ground truth -- both share this run\'s intrinsics, including its '
            'distortion-coefficient uncertainty, see results/STATUS.md).',
        }, f, default_flow_style=False)
    print(f'Wrote {args.output_yaml}')


if __name__ == '__main__':
    main()
