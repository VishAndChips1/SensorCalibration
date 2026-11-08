#!/usr/bin/env python3
"""LiDAR-to-camera extrinsic calibration via plane correspondences (Zhang &
Pless 2004 -- see common/plane_extrinsic_solver.py for the method and a
from-scratch derivation/verification of the math). Takes one bag PER BOARD
POSE (the close/far/left/right/centre protocol this repo's capture script
produces is exactly the right shape of input -- different positions
naturally vary the board's distance and angle, which is what the solver
needs to be well-constrained) and solves for the single rigid transform
that's consistent across all of them.

Usage:
    lidar_camera_extrinsic.py <rgb_intrinsics.yaml> <output.yaml> \\
        --bag close.mcap:/landfill/rgb/image_raw:/landfill/livox/points \\
        --bag far.mcap:/landfill/rgb/image_raw:/landfill/livox/points \\
        --bag left.mcap:... --bag right.mcap:... --bag centre.mcap:...
        [--board config/board_config.yaml]
        [--board-type charuco|custom] [--custom-layout config/custom_board_layout.yaml]
        [--lidar-min-range 1.0] [--lidar-max-range 6.0]
        [--preview out_dir]

Each --bag is "path:rgb_topic:lidar_topic". Needs >=3 bags; 5 (the default
capture protocol) gives a better-constrained, more robust solve than the
minimum.
"""
import argparse
import os
import sys

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'common'))
from charuco_compat import load_board_and_dictionary, estimate_board_pose  # noqa: E402
from bag_reader import read_bag, image_msg_to_numpy, pointcloud2_to_xyz, normalize_to_u8  # noqa: E402
from plane_extrinsic_solver import fit_plane_ransac, solve_extrinsic_from_plane_correspondences  # noqa: E402
import custom_board  # noqa: E402


def board_pose_from_image_charuco(gray, board, dictionary, camera_matrix, dist_coeffs):
    """Returns (normal_cam, d_cam, rvec, tvec, method) for the board's plane
    in the camera frame, from the first good detection found (full ChArUco,
    or a >=8-marker-only fallback for longer range -- see
    charuco_compat.estimate_board_pose), or None if nothing usable was
    detected."""
    result = estimate_board_pose(gray, board, dictionary, camera_matrix, dist_coeffs)
    if result is None:
        return None
    rvec, tvec, method, _n = result
    R_board_to_cam, _ = cv2.Rodrigues(rvec)
    normal_cam = R_board_to_cam @ np.array([0, 0, 1.0])  # board's local +Z, into camera frame
    tvec = tvec.flatten()
    d_cam = normal_cam @ tvec
    return normal_cam, d_cam, rvec, tvec, method


def board_pose_from_image_custom(gray, dictionary, layout, camera_matrix, dist_coeffs):
    corners, ids = custom_board.detect_markers(gray, dictionary)
    obj_pts, img_pts = custom_board.correspondences_for_frame(corners, ids, layout)
    if len(obj_pts) < 6:
        return None
    result = custom_board.board_plane_pose(obj_pts, img_pts, camera_matrix, dist_coeffs)
    return result


def lidar_plane_in_roi(points, min_range, max_range, board_area_m2, rng):
    """Crop to a distance ROI, RANSAC-fit the dominant plane, and sanity-
    check its inlier footprint is plausibly board-sized (not e.g. a wall
    that happened to be in range) -- warns rather than silently trusting
    an implausible result."""
    ranges = np.linalg.norm(points, axis=1)
    roi = points[(ranges > min_range) & (ranges < max_range)]
    if len(roi) < 10:
        raise ValueError(f'only {len(roi)} LiDAR points in the {min_range}-{max_range}m ROI -- '
                          'widen the range or check the board was actually in the scene for '
                          'this capture')
    normal, d, inliers = fit_plane_ransac(roi, inlier_threshold_m=0.02, rng=rng)
    inlier_pts = roi[inliers]
    extent = inlier_pts.max(axis=0) - inlier_pts.min(axis=0)
    footprint_estimate = np.sort(extent)[-1] * np.sort(extent)[-2]  # two largest extents
    if footprint_estimate > board_area_m2 * 6 or footprint_estimate < board_area_m2 * 0.15:
        print(f'  WARNING: dominant-plane inlier footprint ~{footprint_estimate:.2f}m^2 is a poor '
              f'match for the board\'s ~{board_area_m2:.2f}m^2 -- this may have fit a wall/floor '
              'instead of the board. Tighten --lidar-min-range/--lidar-max-range for this pose, '
              'or check --preview.')
    return normal, d, inlier_pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('rgb_intrinsics_yaml')
    ap.add_argument('output_yaml')
    ap.add_argument('--bag', action='append', required=True,
                     help='path:rgb_topic:lidar_topic, repeat for each pose (>=3 required)')
    ap.add_argument('--board', default=os.path.join(os.path.dirname(__file__), '..', 'config', 'board_config.yaml'))
    ap.add_argument('--board-type', choices=['charuco', 'custom'], default='charuco')
    ap.add_argument('--custom-layout',
                     default=os.path.join(os.path.dirname(__file__), '..', 'config', 'custom_board_layout.yaml'))
    ap.add_argument('--lidar-min-range', type=float, default=None,
                     help='fixed global ROI instead of the default per-pose auto-window '
                          '(see --lidar-range-margin) -- use this if auto-windowing picks the '
                          'wrong surface for every pose')
    ap.add_argument('--lidar-max-range', type=float, default=None)
    ap.add_argument('--lidar-range-margin', type=float, default=0.2,
                     help='default mode: crop LiDAR points to [camera_distance-margin, '
                          'camera_distance+margin] PER POSE, using the distance already solved '
                          'from the RGB image -- much tighter than a single global range, and '
                          'confirmed necessary on real data: a global 0.3-1.5m range still let '
                          'RANSAC fit a wall instead of a small (~12x18cm) board every single '
                          'time, even after accumulating all scans in the bag (2500-3700 '
                          'points) -- the wall simply has more points than the board anywhere '
                          'the wall is also in range.')
    ap.add_argument('--preview', default=None, help='directory to save per-pose plane-fit plots')
    args = ap.parse_args()

    if args.board_type == 'charuco':
        with open(args.board) as f:
            board_cfg = yaml.safe_load(f)
        board, dictionary = load_board_and_dictionary(board_cfg)
        board_area_m2 = (board_cfg['squares_x'] * board_cfg['square_length_m']) * \
                        (board_cfg['squares_y'] * board_cfg['square_length_m'])
    else:
        dictionary, layout = custom_board.load_layout(args.custom_layout)
        # Board extent for the LiDAR-plane sanity check, from the layout's own points
        # rather than a separate squares_x/squares_y (custom layouts don't have those).
        all_pts = np.concatenate(list(layout.values()), axis=0)
        extent = all_pts.max(axis=0) - all_pts.min(axis=0)
        board_area_m2 = float(np.sort(extent)[-1] * np.sort(extent)[-2])

    with open(args.rgb_intrinsics_yaml) as f:
        intr = yaml.safe_load(f)
    camera_matrix = np.array(intr['camera_matrix'])
    dist_coeffs = np.array(intr['distortion_coefficients'])

    if len(args.bag) < 3:
        print(f'FAILED: need >=3 poses, got {len(args.bag)}')
        sys.exit(1)

    rng = np.random.default_rng(0)
    normals_lidar, d_lidar, normals_cam, d_cam, pose_names = [], [], [], [], []

    if args.preview:
        os.makedirs(args.preview, exist_ok=True)

    for spec in args.bag:
        bag_path, rgb_topic, lidar_topic = spec.split(':', 2)
        pose_name = os.path.basename(bag_path.rstrip('/'))
        print(f'-- pose "{pose_name}" ({bag_path}) --')

        cam_result = None
        lidar_scans = []
        for topic, msg, _t in read_bag(bag_path, topics=[rgb_topic, lidar_topic]):
            if topic == rgb_topic and cam_result is None:
                img = image_msg_to_numpy(msg)
                gray = normalize_to_u8(img) if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if args.board_type == 'charuco':
                    cam_result = board_pose_from_image_charuco(gray, board, dictionary, camera_matrix, dist_coeffs)
                else:
                    cam_result = board_pose_from_image_custom(gray, dictionary, layout, camera_matrix, dist_coeffs)
            elif topic == lidar_topic:
                # Accumulate EVERY scan in the bag, not just the first.
                # Livox's non-repetitive scan pattern means a single ~100ms
                # scan only sparsely covers a small target -- confirmed on
                # real data: using just the first scan gave 35-60 points
                # spread in a shape that clearly wasn't the board (see
                # results/lidar_preview/*.png from that run) and solved to
                # a physically impossible multi-metre translation. The
                # board is assumed stationary during the ~6s capture
                # (that's the whole point of "hold it steady"), so scans
                # accumulate onto the same real surface rather than
                # smearing it.
                lidar_scans.append(pointcloud2_to_xyz(msg))

        lidar_points = np.concatenate(lidar_scans, axis=0) if lidar_scans else None

        if cam_result is None:
            print(f'  SKIPPED: no usable board pose on {rgb_topic} in this bag '
                  '(neither a full ChArUco detection nor >=8 decoded markers)')
            continue
        if lidar_points is None:
            print(f'  SKIPPED: no LiDAR message found on {lidar_topic} in this bag')
            continue

        nc, dc, rvec, tvec, cam_method = cam_result
        cam_distance = float(np.linalg.norm(tvec))
        if args.lidar_min_range is not None and args.lidar_max_range is not None:
            min_r, max_r = args.lidar_min_range, args.lidar_max_range
        else:
            min_r = max(0.05, cam_distance - args.lidar_range_margin)
            max_r = cam_distance + args.lidar_range_margin
        try:
            nl, dl, inlier_pts = lidar_plane_in_roi(
                lidar_points, min_r, max_r, board_area_m2, rng)
        except ValueError as e:
            print(f'  SKIPPED: {e}')
            continue

        if args.preview:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(6, 5))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(inlier_pts[:, 0], inlier_pts[:, 1], inlier_pts[:, 2], s=2)
            ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_zlabel('z (m)')
            plt.savefig(os.path.join(args.preview, f'{pose_name}_lidar_plane.png'), dpi=150)
            plt.close()

        print(f'  camera distance to board: {cam_distance:.2f}m via {cam_method} '
              f'(LiDAR ROI: {min_r:.2f}-{max_r:.2f}m), LiDAR plane: {len(inlier_pts)} inliers')
        normals_cam.append(nc); d_cam.append(dc)
        normals_lidar.append(nl); d_lidar.append(dl)
        pose_names.append(pose_name)

    if len(normals_lidar) < 3:
        print(f'FAILED: only {len(normals_lidar)} usable poses out of {len(args.bag)} -- need >=3')
        sys.exit(1)

    R, t, angular_residuals_deg = solve_extrinsic_from_plane_correspondences(
        normals_lidar, d_lidar, normals_cam, d_cam)

    rvec_out, _ = cv2.Rodrigues(R)
    roll, pitch, yaw = cv2.RQDecomp3x3(R)[0]  # degrees

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    result = {
        'rotation_matrix': R.tolist(),
        'translation_m': t.tolist(),
        'transform_4x4': T.tolist(),
        'rotation_rpy_deg': [float(roll), float(pitch), float(yaw)],
        'n_poses_used': len(pose_names),
        'poses_used': pose_names,
        'angular_residuals_deg': [float(x) for x in angular_residuals_deg],
        'note': 'Maps a LiDAR-frame point p_L to the camera frame via p_C = R @ p_L + t. '
                'angular_residuals_deg is how well each pose\'s plane normal aligned after '
                'solving -- large outliers there point at which specific pose had a bad '
                'detection or plane fit, worth re-checking with --preview before trusting '
                'the result.',
    }
    with open(args.output_yaml, 'w') as f:
        yaml.safe_dump(result, f, default_flow_style=False)

    print(f'\nUsed {len(pose_names)}/{len(args.bag)} poses: {pose_names}')
    print(f'Angular residuals (deg): {[round(float(x),2) for x in angular_residuals_deg]} '
          f'-- large values flag a specific bad pose, not necessarily a bad overall result')
    print(f'Translation (m): {t.round(4)}')
    print(f'Rotation (deg, roll/pitch/yaw): {roll:.2f}, {pitch:.2f}, {yaw:.2f}')
    print(f'Wrote {args.output_yaml}')


if __name__ == '__main__':
    main()
