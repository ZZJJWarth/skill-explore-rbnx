# SPDX-License-Identifier: MulanPSL-2.0
"""Real-time map optimization of Liu et al., Sci Rep 15, 12261 (2025),
doi:10.1038/s41598-025-97231-9, section "Map optimization".

Far from the lidar the beams thin out, and the free space they leave
between them stays unknown. Every such hole is ringed by frontier cells,
so explored rooms keep producing frontiers the robot goes back for. The
paper removes them before frontier detection:

1. Grey image: occupied 0, unknown 205, free 255.
2. Bilateral filter (Eq. 1-4) over a (2r+1)^2 window

       G_s(||p-q||) = exp(-((i-x)^2 + (j-y)^2) / (2 sigma_s^2))
       G_r(|I(p)-I(q)|) = exp(-(I(p)-I(q))^2 / (2 sigma_r^2))
       W_p = sum_q G_s G_r,   I'(p) = sum_q G_s G_r I(q) / W_p

   A step to an obstacle (205 or 255 against 0) weighs almost nothing, so
   walls keep their edges; unknown against free (205 against 255) is a
   small step and blurs, so thin unknown streaks inside free space turn
   free.
3. Dilation of the explored region, with a template as wide as the
   spatial window, to close the unknown specks the filter leaves.

Only the frontier search sees the result. Navigation plans on its own
costmap, and coverage is measured on the raw map.
"""
from __future__ import annotations

import numpy as np

from .frontier import OCC_THRESH, GridView

UNKNOWN_GREY, OCC_GREY, FREE_GREY = 205.0, 0.0, 255.0

# The paper sets sigma_s from the widest gap sparse beams leave next to a
# real obstacle, and sigma_r from the grey step between known and unknown
# (255 - 205 = 50). On a 0.05 m map the gaps are one or two cells.
WINDOW_RADIUS_CELLS = 2
SIGMA_S_CELLS = 1.5
SIGMA_R = 50.0
# Filtered grey at or above this is free: halfway from unknown to free.
FREE_CUT = (UNKNOWN_GREY + FREE_GREY) / 2.0


def bilateral(img: np.ndarray, r: int = WINDOW_RADIUS_CELLS,
              sigma_s: float = SIGMA_S_CELLS,
              sigma_r: float = SIGMA_R) -> np.ndarray:
    """Eq. 1-4 on a float image; the border is edge-padded."""
    h, w = img.shape
    pad = np.pad(img, r, mode="edge")
    num = np.zeros_like(img)
    den = np.zeros_like(img)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            q = pad[r + dy:r + dy + h, r + dx:r + dx + w]
            g = (np.exp(-(dx * dx + dy * dy) / (2.0 * sigma_s ** 2))
                 * np.exp(-((img - q) ** 2) / (2.0 * sigma_r ** 2)))
            num += g * q
            den += g
    return num / den


def optimize(gv: GridView, r: int = WINDOW_RADIUS_CELLS) -> GridView:
    """`gv` after bilateral filtering and dilation of the explored region.

    Occupied cells are kept as they are. The dilation grows free space one
    4-neighbour step at a time, r steps, into unknown cells only, so it
    cannot jump a one-cell wall into the room behind it."""
    g = gv.data
    occ = g >= OCC_THRESH
    img = np.where(g == -1, UNKNOWN_GREY, np.where(occ, OCC_GREY, FREE_GREY))
    free = (bilateral(img.astype(np.float64), r) >= FREE_CUT) & ~occ
    for _ in range(r):
        grow = free.copy()
        grow[1:, :] |= free[:-1, :]
        grow[:-1, :] |= free[1:, :]
        grow[:, 1:] |= free[:, :-1]
        grow[:, :-1] |= free[:, 1:]
        free = grow & ~occ
    out = np.where(occ, g, np.where(free, 0, -1)).astype(np.int8)
    return GridView(data=out, resolution=gv.resolution,
                    origin_x=gv.origin_x, origin_y=gv.origin_y,
                    width=gv.width, height=gv.height)
