#!/usr/bin/env python3
"""Compose LiDAR->RGB and RGB->thermal into LiDAR->thermal, instead of
running a third independent (and harder -- LiDAR+thermal is the least
reliable pair to calibrate directly, since thermal contrast is already the
shakiest part of this whole pipeline) calibration. Pure matrix algebra, no
new data needed.

p_rgb = R_lr @ p_lidar + t_lr        (from lidar_camera_extrinsic.py)
p_thermal = R_rt @ p_rgb + t_rt       (from rgb_thermal_stereo.py)
=> p_thermal = (R_rt @ R_lr) @ p_lidar + (R_rt @ t_lr + t_rt)

Usage: compose_lidar_thermal.py lidar_rgb.yaml rgb_thermal.yaml output.yaml
"""
import sys

import numpy as np
import yaml


def main():
    lidar_rgb_path, rgb_thermal_path, out_path = sys.argv[1:4]
    with open(lidar_rgb_path) as f:
        lr = yaml.safe_load(f)
    with open(rgb_thermal_path) as f:
        rt = yaml.safe_load(f)

    R_lr = np.array(lr['rotation_matrix'])
    t_lr = np.array(lr['translation_m'])
    R_rt = np.array(rt['rotation_matrix'])
    t_rt = np.array(rt['translation_m'])

    R_lt = R_rt @ R_lr
    t_lt = R_rt @ t_lr + t_rt
    import cv2
    roll, pitch, yaw = cv2.RQDecomp3x3(R_lt)[0]

    result = {
        'rotation_matrix': R_lt.tolist(),
        'translation_m': t_lt.tolist(),
        'rotation_rpy_deg': [float(roll), float(pitch), float(yaw)],
        'note': 'p_thermal = R @ p_lidar + t. Composed from lidar->rgb and rgb->thermal, not '
                'independently solved -- its error is the sum of both inputs\' errors, so if '
                'this looks off, check lidar_rgb.yaml and rgb_thermal.yaml\'s own residuals '
                'first.',
        'composed_from': {'lidar_rgb': lidar_rgb_path, 'rgb_thermal': rgb_thermal_path},
    }
    with open(out_path, 'w') as f:
        yaml.safe_dump(result, f, default_flow_style=False)
    print(f'Translation (m): {t_lt.round(4)}')
    print(f'Rotation (deg rpy): {roll:.2f}, {pitch:.2f}, {yaw:.2f}')
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
