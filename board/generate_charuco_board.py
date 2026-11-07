#!/usr/bin/env python3
"""Generate a printable ChArUco board image from config/board_config.yaml --
the ONE place board geometry is defined, so what you print always matches
what the calibration code expects.

Usage: generate_charuco_board.py [--out board.png] [--dpi 300]

Print at 100% scale (disable "fit to page" / "shrink to fit" in your PDF
viewer or printer driver -- that's the #1 way people accidentally break
ChArUco calibration: the print comes out smaller than board_config.yaml
says, and every downstream measurement is wrong by that same factor,
silently). Mount it on something flat and rigid (foam board / clipboard) --
any physical warp bends the "known flat plane" assumption every step below
relies on.

THERMAL CONTRAST (needed for calibrating the thermal camera off this same
board -- see README.md): plain printed paper has only a small emissivity
difference between black toner and white paper, so contrast at room
temperature is often too weak to detect reliably. Two options, cheapest
first:
  1. Quick test, no prep: warm the board for a minute (near, not on, a
     warm surface, or ~30s direct sun) right before capture, then check
     with board/check_thermal_contrast.py. Heating amplifies the small
     toner-vs-paper emissivity gap into a bigger apparent-temperature
     signal, but it fades as the board re-cools, limiting how long you
     have per pose.
  2. Reliable, ~15min one-time craft: cover every PLAIN WHITE square
     (never the squares containing an ArUco marker -- covering those
     kills marker detection) with aluminum foil tape or foil + glue.
     Foil's emissivity (~0.05) vs paper/toner (~0.9) is a far bigger gap
     than toner-vs-paper alone, it needs no heating, and it doesn't fade
     over a capture session.
Check either approach with board/check_thermal_contrast.py before
committing to a full 5-pose capture.
"""
import argparse
import os
import sys

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'calibration', 'common'))
from charuco_compat import load_board_and_dictionary  # noqa: E402

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'board_config.yaml')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'charuco_board.png'))
    ap.add_argument('--dpi', type=int, default=300)
    args = ap.parse_args()

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    board, _dict = load_board_and_dictionary(cfg)

    px_per_m = args.dpi / 0.0254
    width_px = int(round(cfg['squares_x'] * cfg['square_length_m'] * px_per_m))
    height_px = int(round(cfg['squares_y'] * cfg['square_length_m'] * px_per_m))

    if hasattr(board, 'generateImage'):
        img = board.generateImage((width_px, height_px))
    else:
        img = board.draw((width_px, height_px))

    cv2.imwrite(args.out, img)
    w_m = cfg['squares_x'] * cfg['square_length_m']
    h_m = cfg['squares_y'] * cfg['square_length_m']
    print(f'Wrote {args.out}: {width_px}x{height_px}px @ {args.dpi}dpi')
    print(f'Physical size when printed at 100% scale: {w_m*100:.1f}cm x {h_m*100:.1f}cm')
    print(f'Dictionary: {cfg["dictionary"]}, {cfg["squares_x"]}x{cfg["squares_y"]} squares, '
          f'square={cfg["square_length_m"]*1000:.0f}mm, marker={cfg["marker_length_m"]*1000:.0f}mm')
    print('After printing: measure one printed square with a ruler and confirm it matches '
          f'{cfg["square_length_m"]*1000:.0f}mm before using this board for anything.')


if __name__ == '__main__':
    main()
