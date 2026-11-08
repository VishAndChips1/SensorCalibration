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


def get_chessboard_corners(board):
    """All of the board's interior chessboard-corner 3D points (z=0 plane),
    indexed by charuco corner id. OpenCV 4.6 (legacy, verified against real
    data) exposes this as a `.chessboardCorners` attribute, not a method --
    `board.getChessboardCorners()` looks like the right name by analogy
    with the rest of this API but doesn't exist and raises AttributeError.
    Not verified which form 4.7+'s new API uses; falls back to trying the
    method form too."""
    if hasattr(board, 'chessboardCorners'):
        return np.array(board.chessboardCorners)
    return np.array(board.getChessboardCorners())


def board_object_points(board, charuco_ids):
    """3D board-frame coordinates (z=0 plane) for a set of detected
    charuco corner ids, in the same order as charuco_ids."""
    all_corners = get_chessboard_corners(board)
    return all_corners[charuco_ids.flatten()]


def estimate_board_pose(gray_image, board, dictionary, camera_matrix, dist_coeffs,
                         min_full_corners=6, min_markers_fallback=8):
    """Board pose (rvec, tvec) in the camera frame, trying the accurate
    full-ChArUco path first and falling back to a marker-only PnP when
    too few markers were seen close enough together to interpolate any
    chessboard corner (common at longer range, where markers decode
    sparsely/non-contiguously instead of all failing together).

    The min_markers_fallback=8 default is not a guess: a 2026-11-08 test
    (see SensorCalibration/calibration/marker_subset_pose_test.py and
    results/CALIBRATION_METHODS_TEST.md) measured normal-direction/
    translation error from N-marker-only PnP against full-detection
    ground truth, on this project's real captured data: 1 marker ~16deg/
    237mm (unusable), 4 markers ~10deg/24mm (marginal), 8 markers ~3deg/
    8mm (usable for plane-correspondence extrinsics, which only needs a
    decent normal+distance, not survey-grade pose), full detection
    ~1deg/6mm. 8 is the lowest N that test found consistently usable.

    Returns (rvec, tvec, method, n_markers_or_corners) or None if neither
    path has enough data. method is 'full_charuco' or 'marker_fallback'.
    """
    corners, ids, marker_corners, marker_ids = detect_charuco(gray_image, board, dictionary)
    if corners is not None and ids is not None and len(ids) >= min_full_corners:
        obj_pts = board_object_points(board, ids)
        ok, rvec, tvec = cv2.solvePnP(obj_pts, corners, camera_matrix, dist_coeffs)
        if ok:
            return rvec, tvec, 'full_charuco', int(len(ids))

    if marker_ids is not None and len(marker_ids) >= min_markers_fallback:
        board_ids_list = list(np.array(board.ids).flatten()) if hasattr(board, 'ids') \
            else list(range(len(board.objPoints)))
        obj_pts, img_pts = [], []
        for mid, mc in zip(marker_ids.flatten(), marker_corners):
            if int(mid) not in board_ids_list:
                continue
            idx = board_ids_list.index(int(mid))
            obj_pts.append(np.array(board.objPoints[idx], dtype=np.float32))
            img_pts.append(mc.reshape(4, 2).astype(np.float32))
        if len(obj_pts) >= min_markers_fallback:
            obj_pts = np.concatenate(obj_pts, axis=0)
            img_pts = np.concatenate(img_pts, axis=0)
            ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, camera_matrix, dist_coeffs)
            if ok:
                return rvec, tvec, 'marker_fallback', int(len(marker_ids))

    return None
