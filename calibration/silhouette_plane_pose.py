"""Whole-board silhouette detection: find the board as a single bright (or
dark) quadrilateral blob, without decoding any ArUco marker or chessboard
corner. Meant for cases where the fine pattern isn't resolvable (too far
for RGB, or thermal where the pattern never had internal contrast -- see
results/STATUS.md) but the board's outer rectangle might still be.

Returns None if no plausible quadrilateral blob is found.
"""
import cv2
import numpy as np


def find_board_quad(gray_u8, expect_bright=True, min_area_frac=0.03):
    """gray_u8: normalized 8-bit image. expect_bright: True if the board is
    brighter than its background (thermal: heated board; RGB: light board
    on a dark background -- check both ways if unsure which applies).
    Returns (quad_corners 4x2 float32 ordered TL,TR,BR,BL, aspect_ratio) or None.
    """
    blurred = cv2.GaussianBlur(gray_u8, (5, 5), 0)
    _thresh, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if not expect_bright:
        binary = 255 - binary

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    img_area = gray_u8.shape[0] * gray_u8.shape[1]
    candidates = [c for c in contours if cv2.contourArea(c) >= min_area_frac * img_area]
    if not candidates:
        return None
    largest = max(candidates, key=cv2.contourArea)

    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)  # 4x2, not necessarily in a consistent winding
    (w, h) = rect[1]
    if min(w, h) < 1:
        return None
    aspect = max(w, h) / min(w, h)

    # Order as TL, TR, BR, BL by summing/differencing coordinates.
    s = box.sum(axis=1)
    diff = np.diff(box, axis=1).flatten()
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = box[np.argmin(s)]       # top-left: smallest x+y
    ordered[2] = box[np.argmax(s)]       # bottom-right: largest x+y
    ordered[1] = box[np.argmin(diff)]    # top-right: smallest y-x
    ordered[3] = box[np.argmax(diff)]    # bottom-left: largest y-x
    return ordered, float(aspect)


def solve_quad_pose(quad_corners, board_width_m, board_height_m, K, D):
    """quad_corners: 4x2 image points, TL/TR/BR/BL order. board_width/height_m:
    the real-world size the quad is assumed to represent (board's full outer
    rectangle, not just the inner ArUco grid -- see caller for which this is).
    Returns (rvec, tvec, normal_in_camera_frame) or None if solvePnP fails."""
    w, h = board_width_m, board_height_m
    obj_pts = np.array([
        [-w / 2, -h / 2, 0], [w / 2, -h / 2, 0], [w / 2, h / 2, 0], [-w / 2, h / 2, 0],
    ], dtype=np.float32)
    ok, rvec, tvec = cv2.solvePnP(obj_pts, quad_corners.astype(np.float32), K, D)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    normal = R @ np.array([0.0, 0.0, 1.0])
    return rvec, tvec, normal
