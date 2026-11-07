# SensorCalibration

Automatic intrinsic + extrinsic calibration for a LiDAR + RGB camera +
thermal camera rig, from ROS2 bag recordings of one ChArUco calibration
board seen at several poses. Built to be usable by anyone with this sensor
combination, not tied to one specific rig -- see "How generalizable is
this, really?" below for the honest answer on that.

## Status

Every script here runs and has been checked against synthetic/round-trip
data (ChArUco detection, PointCloud2 parsing, and the plane-correspondence
extrinsic solver are all individually verified -- see each module's
docstring / the commit history for how). **Nothing here has been run
against real captured sensor data yet** -- there was no board available to
capture with when this was written. The 5 bags in `bags/` (once recorded)
are the first real-world test of the whole pipeline; treat the first run
through `calibration/*.py` on them as a validation pass, not an assumed-
working black box. If something looks wrong, the per-script diagnostics
(reprojection error, angular residuals, detection counts, `--preview`
images) are there to help find out what.

## Is a fully generic, "anyone plugs in their sensors" calibration model possible?

Mostly yes, with one honest exception:

- **RGB intrinsics**: a fully solved problem. OpenCV's ChArUco calibration
  is completely generic -- works for any camera resolution, any board
  size, as long as `config/board_config.yaml` matches what you printed.
  Nothing rig-specific here at all.
- **LiDAR<->RGB extrinsics**: generalizable using a well-established method
  (plane correspondences across multiple board poses -- see
  `calibration/common/plane_extrinsic_solver.py`), not a rig-specific
  hack. Works for any LiDAR that returns point clouds in its own frame and
  any camera with known intrinsics. The one real sensitivity: it needs the
  board's plane to be the *dominant* thing in the LiDAR's region-of-
  interest crop, which is a capture-protocol concern (keep the background
  behind the board reasonably clear), not a fundamental limit.
- **RGB<->thermal extrinsics**: fully generic *if* the thermal camera can
  see the board at all -- once it can, it's textbook stereo calibration,
  no custom geometry.
- **Thermal intrinsics/detection**: this is the real exception. A thermal
  camera doesn't see printed ink patterns the way an RGB camera does --
  it needs *some* material or temperature trick to make the board's
  pattern show up as contrast at all (see `board/generate_charuco_board.py`
  and `board/check_thermal_contrast.py`). The calibration *math* is
  identical once you have detections; getting the board to be *detectable*
  in thermal is the part that depends on your specific board/camera and
  needs a quick live check before you trust it, not something that can be
  guaranteed to "just work" sight unseen the way the RGB side can.

So: "a model anyone can run" is realistic for 3 of the 4 calibration steps
with zero rig-specific assumptions, and realistic for the 4th (thermal)
*conditional on* a 30-second live contrast check passing first. That's
about as close to "fully generic" as this problem actually gets without
specialized (expensive) hardware -- see the thermal section below for what
the non-DIY alternative looks like.

## Why one board instead of three different targets

A ChArUco board (chessboard + embedded ArUco markers) is the standard
choice because it combines the two easiest-to-detect calibration patterns:
ArUco markers are robust to angle/partial occlusion and give you instant
board identification, while the interpolated chessboard corners between
them give sub-pixel-accurate 2D points (much more precise than marker
corners alone) for the actual camera calibration math. One board, one
capture session, used three different ways:
`config/board_config.yaml` is the single source of truth for its geometry
so the printed board and every script agree.

## Why 5 poses (close/far/left/right/centre)

Camera intrinsic calibration (RGB, thermal) wants the board seen at
multiple distances and angles so the solver isn't fitting against a flat,
single-plane-facing dataset -- the 5-pose set already does this.

LiDAR-camera extrinsic calibration specifically needs the board's plane
*normal* to point in different directions across poses (not just different
positions in the image) -- a purely left/right sweep without distance or
angle variation would under-constrain the rotation solve. The close/far
(different distance) and left/right (naturally different angle if you tilt
the board slightly toward the sensor as you move it, not just translate
it) combination is deliberately chosen for this; see
`record_calibration_bags.sh`'s on-screen guidance.

## Calibration order (what to actually run)

```
# 1. Generate and print the board (do this once)
python3 board/generate_charuco_board.py

# 2. Quick thermal contrast check BEFORE a full capture session
python3 board/check_thermal_contrast.py

# 3. Capture the 5 poses (needs the pipeline running live)
./capture/record_calibration_bags.sh bags/ 15

# 4. RGB intrinsics (pick any one bag, centre is a reasonable default)
python3 calibration/charuco_camera_calibrate.py \
    bags/centre /landfill/rgb/image_raw results/rgb_intrinsics.yaml

# 5. Thermal intrinsics -- same script, different topic
python3 calibration/charuco_camera_calibrate.py \
    bags/centre /landfill/thermal/image_raw results/thermal_intrinsics.yaml

# 6. LiDAR -> RGB extrinsics, using all 5 poses
python3 calibration/lidar_camera_extrinsic.py \
    results/rgb_intrinsics.yaml results/lidar_rgb_extrinsics.yaml \
    --bag bags/close:/landfill/rgb/image_raw:/landfill/livox/points \
    --bag bags/far:/landfill/rgb/image_raw:/landfill/livox/points \
    --bag bags/left:/landfill/rgb/image_raw:/landfill/livox/points \
    --bag bags/right:/landfill/rgb/image_raw:/landfill/livox/points \
    --bag bags/centre:/landfill/rgb/image_raw:/landfill/livox/points \
    --preview results/lidar_preview

# 7. RGB -> thermal extrinsics (any single bag where both saw the board)
python3 calibration/rgb_thermal_stereo.py \
    results/rgb_intrinsics.yaml results/thermal_intrinsics.yaml \
    results/rgb_thermal_extrinsics.yaml \
    --bag bags/centre --rgb-topic /landfill/rgb/image_raw \
    --thermal-topic /landfill/thermal/image_raw

# 8. LiDAR -> thermal, composed (not independently solved -- see the script)
python3 calibration/compose_lidar_thermal.py \
    results/lidar_rgb_extrinsics.yaml results/rgb_thermal_extrinsics.yaml \
    results/lidar_thermal_extrinsics.yaml
```

Every script prints a sanity-check number (reprojection error in pixels
for camera calibration; angular residuals in degrees for the LiDAR solve)
-- read those before trusting the output, not just the fact that it ran
without crashing.

## Calibrating the thermal camera -- the actual hard part

A thermal camera sees apparent temperature (true temperature combined
with emissivity, not reflected light), so a plain printed black-and-white
board has only a small, often marginal, contrast at room temperature.
Two practical fixes, in `board/generate_charuco_board.py`'s docstring in
full:

1. **Quick, no-prep test**: warm the board (near a warm surface, or brief
   direct sun) right before capture. Cheap to try; the contrast fades as
   it re-cools, so you're working against the clock during a session.
2. **Reliable, one-time craft (~15min)**: cover every plain white square
   (never the ones with an ArUco marker in them) with aluminum foil tape.
   Foil's emissivity is far lower than paper/toner, giving strong contrast
   with no heating needed and no fading over a session.

`board/check_thermal_contrast.py` tells you in ~10 seconds whether either
approach is working, before you commit to a full 5-pose capture.

If neither DIY approach gives reliable detection on your specific thermal
module, the non-DIY fallback used in professional LiDAR-thermal rigs is a
board with alternating anodized/polished metal squares (anodized =
high emissivity, polished = very low emissivity + reflects ambient IR) --
much more reliable, but needs fabrication, not just printing. Not built
here since the DIY options are worth trying first.

## Using a board you already have (not generated by this repo)

Every script above also accepts `--board-type custom` (default is
`charuco`), which reads `config/custom_board_layout.yaml` instead of
`config/board_config.yaml` and calibrates from plain ArUco marker corners
rather than interpolated ChArUco checkerboard corners -- see
`docs/BUILDING_A_CUSTOM_BOARD_LAYOUT.md` for why, and exactly how to build
that file for your specific board (it's a short manual step, deliberately
not automated -- see that doc for the reasoning).

## Repo layout
```
board/            board generation + thermal contrast check
config/           board_config.yaml -- the one shared source of truth for board geometry
calibration/       the actual calibration scripts (see "Calibration order" above)
  common/          shared code: bag reading, ChArUco API compatibility, the plane-correspondence solver
capture/           the 5-pose bag recording script
bags/              recorded calibration bags go here (close/ far/ left/ right/ centre/)
results/           calibration outputs land here (created by the scripts, not pre-populated)
```

## Known limitations / what's not handled

- **Not tested on real sensor data** (see Status above) -- this is the
  main thing to validate first.
- **Non-ROS (e.g. a laptop without ROS2) bag reading is designed but not
  wired up** -- see `requirements.txt` and `calibration/common/
  bag_reader.py`'s docstring for exactly what's needed to finish this.
- **LiDAR plane ROI cropping uses simple distance-range filtering**
  (`--lidar-min-range`/`--lidar-max-range` in
  `lidar_camera_extrinsic.py`), not a tighter angular/spatial crop. This
  is deliberately simple to stay generic across different rig geometries,
  with a built-in sanity check (the fitted plane's footprint is compared
  against the board's known size, and it warns loudly if they don't
  match) rather than silently trusting whatever RANSAC's biggest plane
  happens to be. If your scene has other large flat surfaces (walls,
  floors) within the same range window as the board, tighten the range or
  add a proper spatial crop.
- **OpenCV's new (4.7+) ChArUco API path in `charuco_compat.py` is written
  from documented behaviour but not run against a real 4.7+ install** --
  this repo was built against OpenCV 4.6.0 (legacy API), which is the
  path that's actually been exercised.
- **Camera-to-camera time sync** (`rgb_thermal_stereo.py`) uses a simple
  nearest-timestamp pairing with a configurable skew tolerance, not
  hardware sync -- fine for a mostly-static calibration target, would need
  rework for a moving scene.
