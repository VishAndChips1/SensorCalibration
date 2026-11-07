"""Compatibility shim over cv2.aruco's two incompatible APIs.

OpenCV replaced the ChArUco API between versions: pre-4.7 ("legacy") uses
free functions (cv2.aruco.CharucoBoard_create, cv2.aruco.detectMarkers,
cv2.aruco.interpolateCornersCharuco, cv2.aruco.calibrateCameraCharuco);
4.7+ ("new") wraps the same functionality in cv2.aruco.CharucoBoard /
cv2.aruco.CharucoDetector objects. "Anyone" running this code will have
whatever OpenCV version their own system ships -- this module picks
whichever API is present at import time so the rest of the codebase doesn't
need to care. Verified against OpenCV 4.6.0 (legacy API) on the Pi this was
developed on; the new-API branch is written from OpenCV's own documented
4.7+ interface but has NOT been run against a real 4.7+ install -- if you
hit an error on the new-API path, that's the first place to look.
"""
import cv2
import numpy as np

USING_NEW_API = hasattr(cv2.aruco, 'CharucoDetector')


def load_board_and_dictionary(cfg):
    """cfg is the parsed board_config.yaml dict. Returns (board, dictionary)
    in whichever form this OpenCV version's API wants."""
    dict_id = getattr(cv2.aruco, cfg['dictionary'])
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    if USING_NEW_API:
        board = cv2.aruco.CharucoBoard(
            (cfg['squares_x'], cfg['squares_y']),
            cfg['square_length_m'], cfg['marker_length_m'], dictionary,
        )
    else:
        board = cv2.aruco.CharucoBoard_create(
            cfg['squares_x'], cfg['squares_y'],
            cfg['square_length_m'], cfg['marker_length_m'], dictionary,
        )
    return board, dictionary


def detect_charuco(gray_image, board, dictionary):
    """Detect the ChArUco board in a single grayscale (or AGC-normalized
    8-bit) image. Returns (charuco_corners, charuco_ids, marker_corners,
    marker_ids) -- any of these may be None/empty if nothing was found.
    charuco_corners is Nx1x2 float32 pixel coordinates; charuco_ids is
    Nx1 int32, each id indexing into board.getChessboardCorners()."""
    if USING_NEW_API:
        detector = cv2.aruco.CharucoDetector(board)
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray_image)
        return charuco_corners, charuco_ids, marker_corners, marker_ids

    params = cv2.aruco.DetectorParameters_create()
    marker_corners, marker_ids, _rejected = cv2.aruco.detectMarkers(
        gray_image, dictionary, parameters=params)
    if marker_ids is None or len(marker_ids) == 0:
        return None, None, marker_corners, marker_ids
    _n, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners, marker_ids, gray_image, board)
    return charuco_corners, charuco_ids, marker_corners, marker_ids


def calibrate_camera_charuco(all_charuco_corners, all_charuco_ids, board, image_size):
    """Runs the ChArUco-specific camera calibration (uses the known 3D
    board geometry directly, more accurate than feeding generic 3D-2D
    point pairs through cv2.calibrateCamera). Returns
    (reproj_error, camera_matrix, dist_coeffs, rvecs, tvecs)."""
    if USING_NEW_API:
        return cv2.aruco.calibrateCameraCharuco(
            all_charuco_corners, all_charuco_ids, board, image_size, None, None)
    return cv2.aruco.calibrateCameraCharuco(
        all_charuco_corners, all_charuco_ids, board, image_size, None, None)


def board_object_points(board, charuco_ids):
    """3D board-frame coordinates (z=0 plane) for a set of detected
    charuco corner ids, in the same order as charuco_ids."""
    all_corners = np.array(board.getChessboardCorners())
    return all_corners[charuco_ids.flatten()]
