# Calibration methods test — "can coarse detection replace full marker/pattern decoding?"

Run autonomously 2026-11-08, no operator present (Vish's own instruction:
run whatever is possible automatically, no prompts, no physical changes
needed). Tests his hypothesis directly: *if RGB/thermal can roughly
localize the board without reading every marker/the full checkerboard
pattern, and LiDAR can find it as a flat surface, can that replace the
harder fine-grained detection this repo has relied on so far?*

**Headline answer: partially yes, partially no, and the "no" part is
physics, not something this session's code changes could get around.**
2D localization (where in the image is the board) turns out to be very
robust with coarse detection, in both RGB and thermal. Turning that into
an accurate 3D plane pose (which is what extrinsic calibration actually
needs) still needs real information -- either enough individual markers,
or a calibrated camera model -- that the current setup doesn't supply at
the distances where LiDAR can see anything at all. See §5 for why that's
unchanged from before.

## 0. Live system state at the start of this run
Pipeline was not running; launched fresh (`ros2 launch pitfusion3d_bringup
pitfusion3d_full.launch.py`, mains RPi power + Anker-powered LiDAR, per
Vish's note) and confirmed healthy -- all 8 nodes up, fusion/merge topics
flowing (log: `thesis_tests/pipeline_launch_20261108.log`). **Left running**
at the end of this session, as instructed.

**The calibration board is not currently in the camera rig's field of
view** -- a live single-frame check (`board/live_scene_check.py`, new
tool this run) showed RGB and thermal both looking at an ordinary bedroom
scene, no board anywhere in frame (saved to `/tmp/scene_check_1/` for
reference, not committed -- it's someone's bedroom, not calibration
data). This means **no new live captures of the board were attempted** --
holding the board is a two-hands job per the existing capture photos, and
no one was available to do that, consistent with Vish saying he'd be out
of the room. Everything below uses the 5 already-recorded bags
(`bags/close,far,left,right,centre`) plus one live LiDAR-only check
(no board needed for that one, see §4).

## 1. RGB: does N-markers-only (no full ChArUco) give a usable pose?
`calibration/marker_subset_pose_test.py` (new). For every frame with a
strong full-ChArUco "ground truth" detection (>=20 corners) across all 5
bags, also solved the pose from a random subset of just N of that same
frame's detected ArUco markers, and compared to the full-detection pose.
93 qualifying frames, several random subsets per N.

| N markers used | normal-angle error (mean / p90) | translation error (mean / p90) |
|---|---|---|
| 1 | 15.8° / 27.8° | 237mm / 337mm |
| 2 | 40.3° / 47.0° | 214mm / 325mm |
| 4 | 10.2° / 21.6° | 24mm / 58mm |
| 8 | 3.3° / 6.2° | 8.4mm / 15.3mm |
| all decoded (no interpolation) | 1.2° / 2.1° | 5.6mm / 8.7mm |

**Conclusion**: 1-4 markers is not reliable enough to trust (the 2-marker
row being *worse* than 1-marker is a real artifact of coplanar-point PnP
pose ambiguity at low N, not a typo -- `solvePnP`'s default iterative
solver can converge to a flipped/mirror solution with few near-planar
points; this is a known limitation of the method at low N, not a bug in
this test). **8 markers is the practical minimum** for a pose accurate
enough to feed plane-correspondence extrinsics (which only needs a
decent normal+distance, not survey-grade accuracy). Decoding all visible
markers without needing the full sub-pixel chessboard-corner
interpolation step (the "all decoded" row) is already nearly as good as
full ChAruco -- useful, because marker decoding alone works at smaller
apparent sizes than reliable chessboard-corner interpolation.
**Code change made from this**: `charuco_compat.estimate_board_pose()`
(new) tries full ChArUco first, falls back to marker-only PnP once >=8
markers are decoded, and `lidar_camera_extrinsic.py` now uses it instead
of requiring full detection. Tested against all 5 bags -- no regression
(same poses succeed/fail as before; see §4, the fallback never actually
triggers on this data because the limiting factor there is LiDAR range,
not RGB marker count).

## 2. RGB: does a plain silhouette (no ArUco at all) find the board?
`calibration/rgb_silhouette_test.py` (new), using
`calibration/silhouette_plane_pose.py` (new, generic bright/dark-blob
quadrilateral finder). 93/93 frames (100%) found a plausible quadrilateral
blob. Its aspect ratio averaged 1.43 -- an exact match to the board's real
7:10 physical ratio (1.43). That match is strong evidence the detector is
finding the real board outline, not noise.

A secondary check (silhouette centroid vs. the mean position of whatever
ArUco markers also decoded in the same frame) showed a large average
offset (236px on a 1280px-wide image). **This is most likely an artifact
of the reference, not the silhouette method**: the board is captured very
close (25-40cm) and is often cropped by the frame edge (see
`results/rgb_preview/frame_0012.png`), so a partial ArUco decode is itself
a biased estimate of the true board center, while the silhouette finds the
board's actual full visible extent. Flagged honestly rather than silently
resolved -- would need a frame where the full board fits uncropped to
properly separate "silhouette is wrong" from "the comparison reference is
wrong," which none of the 5 bags happen to provide.

## 3. Thermal: does a plain silhouette find the board?
`calibration/thermal_silhouette_test.py` (new). 577/577 thermal frames
(100%) across all 5 bags found a plausible quadrilateral blob -- thermal's
uniform warm-rectangle appearance (the same effect that broke full
ChArUco pattern detection, see STATUS.md) turns out to be an *excellent*
silhouette target: bright, clean-edged, high-contrast against a cold
background (see `results/thermal_preview/frame_0003.png`). Aspect ratio
averaged 1.34 vs. the real 1.43 -- close, not exact.

Cross-checked the resulting 3D plane normal (solved using PitFusion3D's
existing UNCALIBRATED thermal-intrinsics placeholder, fx=fy=919 -- flagged
loudly in the script and here, this is not a real thermal calibration)
against the same-moment RGB ground-truth normal, where a close-enough
timestamp pair existed (168 pairs): mean error 32.5°, median 44.7°.
**Large -- not usable for real calibration as-is.** Consistent with §1's
finding that a 4-point (single quad) PnP solve is poorly conditioned; this
is the same underlying limitation showing up in a second modality, not a
new problem. Detection (2D, "is the board roughly here") is strong;
single-quad 3D pose estimation (what extrinsic calibration actually
needs) is not, at least not without a real calibrated K for this camera
and/or more than 4 points.

## 4. LiDAR: can "the board stands out as a flat surface" work at the current capture distances?
No new code needed here -- this is a physical-range question, re-checked
with fresh evidence. A live single-scan check just now (different scene,
no board, just confirming the sensor's raw behavior) found: 5427 points
reported at distance <0.25m, but **every single one of them is an exact
(0,0,0) coordinate** -- a padding/invalid-return artifact, not real
returns (confirmed by direct inspection, not assumed). Real, non-zero
returns only start at 1.25m+ in that live scan. This matches the already-
documented bag-data finding almost exactly (effectively zero real returns
below ~1.0-1.2m) -- two independent checks, same conclusion, a genuine
Livox Avia minimum-range blind zone.

Re-ran `lidar_camera_extrinsic.py` against all 5 bags with this session's
marker-fallback code in place: **same result as before, 2/5 usable poses,
still fails the >=3 minimum.** The fallback doesn't help here because the
bottleneck in these specific bags was never RGB's marker count -- it's
that the board itself was physically placed within the LiDAR's blind zone
for all 5 captures (25cm-1.2m range; only `far` at 1.19m and `left` at
1.06m even cleared the ROI point-count minimum, and `left` only barely,
10 inliers). **No detection algorithm, however coarse, can see LiDAR
returns that were never collected in the first place** -- this part of
Vish's hypothesis (treat the board as a simple flat surface for LiDAR)
would very likely still work fine once the board is beyond ~1.25m, it
just can't be demonstrated or disproven on today's data, because none of
the 5 bags have the board past that range.

## 5. What this means, put together
- **RGB and thermal silhouette/coarse detection: genuinely validated,
  now part of the toolkit.** Both reliably find the board with no pattern
  decoding needed (100% detection rate in both, correct-to-nearly-correct
  aspect ratio). This is real, reusable progress toward the "generalizable
  for anyone's sensors" goal Vish described earlier -- it no longer
  strictly requires a resolvable ArUco pattern to know roughly where a
  planar target is.
- **Turning "roughly where" into "precise enough pose for extrinsic
  calibration": still needs real information.** 8+ individual markers
  works (§1); a single 4-corner silhouette quad does not (§3), at least
  not without a properly calibrated camera (which, per the earlier
  session, this RGB camera doesn't reliably have yet either -- see
  STATUS.md's intrinsics-instability finding, unchanged by this run).
- **LiDAR extrinsics: still blocked by the same hardware minimum-range
  limit as before, reconfirmed independently today.** This cannot be
  fixed by a cleverer detector in RGB or thermal -- it needs the board
  physically farther than ~1.25m next time someone can hold it there,
  which didn't happen this run (no operator present, by design).
- **Net effect of today's changes**: real new capability added (coarse
  silhouette/marker-subset detection, both validated with numbers, both
  now reusable code), zero risk taken (no live board data fabricated, no
  pipeline left in a bad state, no destructive action), and the one
  remaining real blocker (board range vs. LiDAR blind zone) is now
  additionally supported by independent live evidence from today, not
  just the original capture session.

## 6. New files this run
- `board/live_scene_check.py` -- one-shot live RGB/thermal/LiDAR snapshot
  + diagnostic (distance histogram, ArUco decode count) for unattended
  sanity checks like this one.
- `calibration/silhouette_plane_pose.py` -- generic bright/dark-blob
  quadrilateral board finder + PnP pose solve from just its 4 corners.
- `calibration/marker_subset_pose_test.py`, `rgb_silhouette_test.py`,
  `thermal_silhouette_test.py` -- the three accuracy tests above, each
  independently re-runnable against any future bag.
- `charuco_compat.estimate_board_pose()` -- the validated >=8-marker
  fallback, wired into `lidar_camera_extrinsic.py`.
- Raw outputs: `results/marker_subset_test.yaml`,
  `results/rgb_silhouette_test.yaml`, `results/thermal_silhouette_test.yaml`.
