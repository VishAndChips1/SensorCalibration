# Calibration status (real run against the 5 captured bags)

**RGB intrinsics: WORKING.** `rgb_intrinsics.yaml` -- 27/27 frames detected
(100%), reprojection error 0.327px (well under the ~1px "trust this"
threshold). fx=2108.9, fy=2090.0, cx=613.5, cy=250.5. This is this
project's first-ever REAL camera calibration (everything before this was a
placeholder, see the 2026-10-26 thesis campaign's inventory.md).

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
`results/rgb_intrinsics.yaml` is a real, good calibration and can be used
as-is. Everything else needs either the foil-tape board fix (thermal) or a
larger board (LiDAR extrinsics) before it'll produce real numbers -- the
code itself is built, tested against real data, and three real bugs were
found and fixed while running it this session (see git log): a `mono16`
thermal image encoding that wasn't handled, a wrong OpenCV ChArUco API
method name (`getChessboardCorners` vs the real `chessboardCorners`
attribute), and the LiDAR extrinsic script only reading the first scan in
a bag instead of accumulating all of them. All three are fixed and the
fixes are the reason RGB intrinsics now works cleanly.
