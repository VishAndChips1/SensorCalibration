# Calibration status (real run against the 5 captured bags)

**RGB intrinsics: WORKING, BUT NOT YET TRUSTWORTHY -- unstable under
pooling (found 2026-11-08).** `rgb_intrinsics.yaml` -- 27/27 `centre`-bag
frames detected (100%), reprojection error 0.327px (well under the ~1px
"trust this" threshold). fx=2108.9, fy=2090.0, cx=613.5, cy=250.5.

`charuco_camera_calibrate.py` was extended to accept multiple bags (it
only ever used `centre` alone before). Pooling in the other 4 bags to
check robustness, as camera-calibration best practice recommends, exposed
a real problem: fx swings from ~970 to ~2640 depending on which single
extra bag is added to `centre` -- a >2.5x range -- while the reported
reprojection error *stays low throughout* (0.3-0.9px for every
combination tried). Low reprojection error is therefore not a reliable
trust signal here on its own: this is the classic signature of a
poorly-conditioned calibration (a scale/focal-length ambiguity the solver
can trade off against board pose and still fit the 2D points well),
**not a code bug** -- the solver is behaving correctly on insufficiently-
diverse input.

Root cause, most likely: the same board-size problem already identified
for LiDAR extrinsics below. This board is small (12.6cm x 18cm, 9mm
markers) and has to be captured within ~25-40cm for RGB to detect it at
all, which leaves too little real depth/tilt parallax across the 5 poses
to pin down focal length independent of board pose -- a known degenerate
regime for Zhang-style calibration, worse the smaller/closer the target.
**The single bigger board fix recommended below for LiDAR would very
likely fix this too** -- a ~75cm board viewable at 1.5-2m gives much more
real parallax across poses.

**Until this is resolved: do not treat any fx/fy/cx/cy number from this
board as final, and do not write it into a live pipeline's camera_info.**
The `centre`-only number above is kept as the best available single-bag
result (useful for a rough sanity check, e.g. confirming the real lens
has a much narrower FOV than the placeholder assumed), but it is not
verified stable and should not replace PitFusion3D's placeholder
intrinsics yet. This is this project's first-ever REAL camera calibration
attempt (everything before this was a placeholder, see the 2026-10-26
thesis campaign's inventory.md) -- real detection and a real solver, just
not yet a trustworthy number.

**Thermal intrinsics: NOT WORKING YET.** The board shows up in thermal as a
uniform warm rectangle with no internal checkerboard/marker pattern visible
at all (see `thermal_preview/frame_0003.png`) -- heating the whole board
makes it visible as a silhouette but doesn't create contrast *between* its
black and white regions specifically. This is the exact failure mode
`board/generate_charuco_board.py`'s docstring warned about. **Fix needed**:
the foil-tape approach (cover plain squares with foil, not the heating
approach) -- see that doc, or the README's thermal section.

**LiDAR-RGB extrinsics: NOT WORKING, hardware constraint found.** Tracing
through the real data: the LiDAR returns essentially zero points below
~1.0m range (0 points at 0.27-0.67m and 0.59-0.99m windows, 1 point at
0.64-1.04m), only becoming usable above ~1.0-1.2m (10 points at 1.06m, 157
at 1.19m). This is the Livox Avia's minimum-range blind zone -- a real
sensor characteristic, not a bug. It directly conflicts with what RGB
needs: this board (12.6cm x 18cm, 9mm markers) only detects reliably
within ~25-40cm of the camera (measured: 100% detection at 0.47m in the
`centre` bag; 0% detection at the ~1.5m+ distances used in the first
capture attempt). There is no distance where both sensors currently get
good data with this board.

**Fix needed for LiDAR extrinsics specifically**: a bigger calibration
target. Rough sizing: RGB needs roughly 20-30px per marker to decode
reliably; this board's 9mm markers only clear that bar within ~25-40cm.
To get comfortable RGB detection at ~1.5-2m (solidly inside the LiDAR's
good range), the board would need to be roughly 6x larger in each
dimension -- around 75cm wide, with proportionally larger squares/markers.
This is a different, bigger board than the current one, not a software
fix.

**RGB-thermal stereo: blocked** on thermal intrinsics working.
**Composed LiDAR-thermal: blocked** on both of the above.

## What's actually usable right now
Nothing from this board is a finished, trustworthy number yet --
`results/rgb_intrinsics.yaml` looked good in isolation (low reprojection
error) but turned out unstable once checked against more than one bag
(see above); thermal and LiDAR extrinsics are blocked outright. **One
board fix (a bigger, ~75cm board, viewable at 1.5-2m) is the common root
cause behind the RGB-intrinsics instability AND the LiDAR-extrinsics
blind-zone conflict** -- that is the single highest-leverage next step,
ahead of the foil-tape thermal fix.

The *code*, independent of this board's results, is built and tested
against real data, and real bugs were found and fixed while running it
this session (see git log): a `mono16` thermal image encoding that wasn't
handled, a wrong OpenCV ChArUco API method name (`getChessboardCorners`
vs the real `chessboardCorners` attribute), the LiDAR extrinsic script
only reading the first scan in a bag instead of accumulating all of them,
and (2026-11-08) `charuco_camera_calibrate.py` only ever supporting one
bag at a time, which is what let the RGB-intrinsics instability go
unnoticed until bags were pooled. All are fixed. None of this makes the
current board's calibration numbers trustworthy -- that needs new data
from a bigger board, not another code fix.
