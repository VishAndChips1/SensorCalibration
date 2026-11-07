"""Core math for plane-correspondence LiDAR-camera extrinsic calibration
(Zhang & Pless, "Camera and range sensor calibration using a single
publicly available target", 2004 -- a well-established, widely used
method, not something invented for this project). Chosen over trying to
match individual ChArUco corners in the LiDAR cloud: at a few metres'
range, sparse LiDAR data essentially never resolves individual 3-4cm
chessboard squares, but it resolves the BOARD'S FLAT PLANE very reliably,
and a plane is all this method needs.

The physics: a rigid transform (R, t) taking LiDAR-frame points to
camera-frame points rotates a plane's normal (n_cam = R @ n_lidar) and
shifts its origin-distance in a way that only depends on R and t, not on
which point on the plane you picked. Specifically, if a plane satisfies
n . p = d in each frame:

    n_cam . t = d_cam - d_lidar      (derived in this project's own notes;
                                       see README.md's calibration section)

Given >=3 board poses with non-parallel normals (exactly what the 5-pose
close/far/left/right/centre capture protocol gives you -- different
positions naturally give different plane orientations relative to both
sensors), this is enough to solve for R (by aligning the two sets of
normal vectors with the same SVD/Kabsch procedure used for rigid point-
cloud registration) and then t (by least-squares over the linear equation
above).
"""
import numpy as np


def fit_plane_ransac(points, n_iterations=500, inlier_threshold_m=0.02, rng=None):
    """RANSAC plane fit. points: Nx3. Returns (normal (unit, 3,), d, inlier_mask)
    for the plane n.p = d with the most inliers, or raises ValueError if
    fewer than 3 points are given."""
    if len(points) < 3:
        raise ValueError(f'need >=3 points to fit a plane, got {len(points)}')
    rng = rng or np.random.default_rng(0)
    best_inliers = None
    best_normal, best_d = None, None
    n = len(points)
    for _ in range(n_iterations):
        idx = rng.choice(n, 3, replace=False)
        p0, p1, p2 = points[idx]
        v1, v2 = p1 - p0, p2 - p0
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm
        d = normal @ p0
        residuals = np.abs(points @ normal - d)
        inliers = residuals < inlier_threshold_m
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers, best_normal, best_d = inliers, normal, d

    if best_inliers is None or best_inliers.sum() < 3:
        raise ValueError('RANSAC found no plane with enough inliers -- check '
                          'inlier_threshold_m and the input ROI crop')

    # Refine with a least-squares fit (SVD) over the inlier set, more
    # accurate than the minimal 3-point sample RANSAC picked.
    pts = points[best_inliers]
    centroid = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - centroid)
    normal = vt[-1]
    # Keep the normal pointing back toward the sensor origin (consistent
    # sign convention -- without this, half your poses could come back
    # with a flipped normal and corrupt the correspondence solve below).
    if normal @ centroid > 0:
        normal = -normal
    d = normal @ centroid
    return normal, d, best_inliers


def solve_extrinsic_from_plane_correspondences(normals_lidar, d_lidar, normals_cam, d_cam):
    """normals_lidar, normals_cam: lists/arrays of unit 3-vectors, one per
    pose, same order. d_lidar, d_cam: matching lists of scalars. Returns
    (R (3x3), t (3,)) such that a LiDAR-frame point p_L maps to the
    camera frame as R @ p_L + t.

    Needs >=3 poses with normals that aren't all near-parallel (otherwise
    the rotation isn't fully constrained) -- this is checked and raises
    ValueError rather than silently returning a degenerate answer.
    """
    Nl = np.asarray(normals_lidar, dtype=float)
    Nc = np.asarray(normals_cam, dtype=float)
    dl = np.asarray(d_lidar, dtype=float)
    dc = np.asarray(d_cam, dtype=float)
    n_poses = len(Nl)
    if n_poses < 3:
        raise ValueError(f'need >=3 pose correspondences, got {n_poses}')

    # Rotation: align two sets of unit vectors (same procedure as rigid
    # point-cloud registration / Kabsch, applied to normals instead of
    # points -- both centroids are implicitly the origin since these are
    # unit directions, not positions, so no centring step is needed here).
    H = Nl.T @ Nc
    U, _S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        # Reflection, not a rotation -- standard Kabsch fix-up.
        Vt_fixed = Vt.copy()
        Vt_fixed[-1, :] *= -1
        R = Vt_fixed.T @ U.T

    # Sanity: how well did the rotation actually align the normals? Large
    # residuals here mean something is wrong upstream (bad plane fits, a
    # mislabelled pose, or poses that are too close to parallel to
    # constrain rotation well) -- surfaced to the caller, not hidden.
    aligned = (R @ Nl.T).T
    angular_residuals_deg = np.degrees(np.arccos(np.clip(np.sum(aligned * Nc, axis=1), -1, 1)))

    # Translation: n_cam . t = d_cam - d_lidar, one row per pose, least
    # squares over all poses (see module docstring for the derivation).
    A = Nc
    b = dc - dl
    t, residuals, rank, _sv = np.linalg.lstsq(A, b, rcond=None)
    if rank < 3:
        raise ValueError('translation system is rank-deficient -- the plane normals across '
                          'your poses are too close to parallel (e.g. only left/right with no '
                          'depth variation) to fully constrain translation. Use poses that '
                          'vary the board\'s distance AND angle, not just left-right position.')

    return R, t, angular_residuals_deg
