"""Bridge uncertain team evidence from the confident evidence around it (the "+bridge" option).

The team model is confident on some frames (|log-odds| at least log(p/(1-p)) for
p = BRIDGE_CONF_P) and unsure on others, mostly inside ball gaps and duels. For each unsure frame
we look at the nearest confident frame before it and after it, in the same half:

  - both name the same team, and neither is more than BRIDGE_MAX_S away: the frame gets their
    evidence, weighted by closeness, and capped at the confidence threshold, so a bridged frame
    never counts for more than a confident one
  - they name different teams: the frame is left alone. A hand-over happens somewhere in
    between, and the decoder places it where the evidence turns
  - no confident frame on one side, or too far away: left alone

Fixed, explained values in config.py; kept only if it wins the leave-one-match-out comparison on
IDSSE (team_model.py). No I/O, so it is tested on synthetic input.
"""

import numpy as np

import config as C
from features import FPS


def _nearest(conf):
    """Index of the last confident frame at or before each frame, and of the next one at or
    after it; -1 where there is none."""
    n = len(conf)
    idx = np.arange(n)
    prev = np.maximum.accumulate(np.where(conf, idx, -1))
    nxt = np.where(conf, idx, n)
    nxt = np.minimum.accumulate(nxt[::-1])[::-1]
    nxt = np.where(nxt >= n, -1, nxt)
    return prev, nxt


def apply(logit, half):
    """Bridged log-odds (home has the ball). `logit` is calibrated; `half` is 1 / 2 / other."""
    logit = np.asarray(logit, dtype=np.float64)
    out = logit.copy()
    cap = float(np.log(C.BRIDGE_CONF_P) - np.log1p(-C.BRIDGE_CONF_P))
    max_frames = C.BRIDGE_MAX_S * FPS
    for h in (1, 2):
        rows = np.flatnonzero(np.asarray(half) == h)
        if len(rows) < 2:
            continue
        seg = logit[rows]
        conf = np.abs(seg) >= cap
        prev, nxt = _nearest(conf)
        k = np.arange(len(seg))
        ok = ~conf & (prev >= 0) & (nxt >= 0)
        ok &= ((k - prev) <= max_frames) & ((nxt - k) <= max_frames)
        lp = seg[np.maximum(prev, 0)]
        ln = seg[np.maximum(nxt, 0)]
        ok &= np.sign(lp) == np.sign(ln)
        dp = (k - prev).astype(np.float64) / FPS
        dn = (nxt - k).astype(np.float64) / FPS
        wp, wn = 1.0 / (1.0 + dp), 1.0 / (1.0 + dn)
        bridged = np.clip((wp * lp + wn * ln) / (wp + wn), -cap, cap)
        seg_out = np.where(ok, bridged, seg)
        out[rows] = seg_out
    return out
