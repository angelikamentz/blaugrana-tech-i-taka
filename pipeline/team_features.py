"""Inputs of the learned team model (p_home), built from a feature table (features.compute).

Only tracking-derived columns go in, because the model is trained on other matches (IDSSE) that
have nothing else. Smart Tagging is added later, outside the model, with v1's fixed weight.

Groups:
  - who is at the ball: nearest home / away distance and their difference, players near the ball
  - where the ball is and where it goes: x from the centre (+x = home attacks), y, signed x
    velocity, speed
  - team shape: each team's centre relative to the ball, team speeds
  - ball gaps (the ball is missing on 41% of our frames): how long the gap is, and the "who is
    nearer" signal at the last frame before and the first frame after the gap
  - context: the "who is nearer" signal averaged over short and long windows centred on the frame,
    and over the past and the future separately. Offline labelling, so looking ahead is allowed.

Home/away symmetry: `swap(cols)` gives the same match with the teams exchanged (pitch rotated so
the new home side still attacks +x). Training uses both, and prediction averages
p(home | cols) with 1 - p(home | swap(cols)), so the model can't prefer "home" for its own sake.

No I/O: works on the column dict, so it is tested on synthetic input.
"""

import numpy as np

import config as C
from features import FPS, PITCH_LENGTH, PITCH_WIDTH, moving_nanmean

# Pairs of columns that exchange when the teams are swapped.
_PAIRS = (("near_H_m", "near_A_m"), ("n_H_near", "n_A_near"), ("H_cx_m", "A_cx_m"),
          ("speed_H", "speed_A"), ("n_H", "n_A"))
# Columns measured along x or y, mirrored by the half-turn of the pitch.
_X = ("ball_x_m", "H_cx_m", "A_cx_m")
_Y = ("ball_y_m",)


def swap(cols):
    """The same columns with home and away exchanged and the pitch turned half a circle."""
    out = dict(cols)
    for a, b in _PAIRS:
        if a in cols and b in cols:
            out[a], out[b] = cols[b], cols[a]
    for k in _X:
        if k in out:
            out[k] = (PITCH_LENGTH - np.asarray(out[k], dtype=np.float32)).astype(np.float32)
    for k in _Y:
        if k in out:
            out[k] = (PITCH_WIDTH - np.asarray(out[k], dtype=np.float32)).astype(np.float32)
    return out


def _ffill(values):
    """Last finite value carried forward; NaN before the first one."""
    values = np.asarray(values, dtype=np.float32)
    ok = np.isfinite(values)
    idx = np.where(ok, np.arange(len(values)), -1)
    last = np.maximum.accumulate(idx)
    out = np.where(last >= 0, values[np.maximum(last, 0)], np.nan)
    return out.astype(np.float32)


def _bfill(values):
    return _ffill(np.asarray(values)[::-1])[::-1].copy()


def _past_mean(values, frames):
    """Mean over the `frames` frames ending at this one (NaN-aware)."""
    centred = moving_nanmean(values, frames)
    shift = frames // 2
    out = np.full(len(values), np.nan, dtype=np.float32)
    if shift and len(values) > shift:
        out[shift:] = centred[:-shift]
    else:
        out[:] = centred
    return out


def _future_mean(values, frames):
    return _past_mean(np.asarray(values)[::-1], frames)[::-1].copy()


def matrix(cols):
    """(X (n, k) float32, feature names). NaN marks a missing value; the model handles it."""
    have = np.asarray(cols["ball_filled"], dtype=bool)
    near_h = np.where(have, cols["near_H_m"], np.nan).astype(np.float32)
    near_a = np.where(have, cols["near_A_m"], np.nan).astype(np.float32)
    # Positive: home nearer. Capped so a team with nobody tracked doesn't dominate.
    cap = C.TEAM_NO_PLAYER_M
    diff = (np.fmin(near_a, cap) - np.fmin(near_h, cap)).astype(np.float32)

    bx = np.where(have, cols["ball_x_m"], np.nan).astype(np.float32)
    by = np.where(have, cols["ball_y_m"], np.nan).astype(np.float32)
    vx = np.full(len(bx), np.nan, dtype=np.float32)
    vx[1:] = np.diff(bx) * FPS
    vx[np.abs(vx) > C.BALL_SPEED_MAX_MS] = np.nan
    vx = moving_nanmean(vx, C.BALL_SPEED_SMOOTH_FRAMES)
    vx[~have] = np.nan

    feats = {
        "near_H_m": near_h,
        "near_A_m": near_a,
        "near_diff_m": diff,
        "n_H_near": np.where(have, cols["n_H_near"], np.nan).astype(np.float32),
        "n_A_near": np.where(have, cols["n_A_near"], np.nan).astype(np.float32),
        "ball_x_c_m": (bx - PITCH_LENGTH / 2.0).astype(np.float32),
        "ball_y_c_m": (by - PITCH_WIDTH / 2.0).astype(np.float32),
        "ball_vx_ms": vx,
        "ball_speed_ms": np.asarray(cols["ball_speed_ms"], dtype=np.float32),
        "H_cx_rel_m": (np.asarray(cols["H_cx_m"], dtype=np.float32) - bx),
        "A_cx_rel_m": (np.asarray(cols["A_cx_m"], dtype=np.float32) - bx),
        "H_cx_c_m": np.asarray(cols["H_cx_m"], dtype=np.float32) - PITCH_LENGTH / 2.0,
        "A_cx_c_m": np.asarray(cols["A_cx_m"], dtype=np.float32) - PITCH_LENGTH / 2.0,
        "speed_H": np.asarray(cols["speed_H"], dtype=np.float32),
        "speed_A": np.asarray(cols["speed_A"], dtype=np.float32),
        "ball_seen": np.asarray(cols["ball_seen"], dtype=np.float32),
        "log_ball_gap_s": np.log1p(np.asarray(cols["ball_gap_len_s"], dtype=np.float32)),
        "log_ball_since_s": np.log1p(np.asarray(cols["ball_since_seen_s"], dtype=np.float32)),
        "log_ball_until_s": np.log1p(np.asarray(cols["ball_until_seen_s"], dtype=np.float32)),
        # who was nearer at the edges of the current ball gap
        "diff_last_seen_m": _ffill(diff),
        "diff_next_seen_m": _bfill(diff),
        "ball_x_last_seen_c_m": _ffill(bx) - PITCH_LENGTH / 2.0,
        "ball_x_next_seen_c_m": _bfill(bx) - PITCH_LENGTH / 2.0,
    }
    for w in C.TEAM_WINDOWS_S:
        frames = max(int(round(w * FPS)), 1)
        feats["near_diff_w{:g}".format(w)] = moving_nanmean(diff, frames)
        feats["near_diff_past{:g}".format(w)] = _past_mean(diff, frames)
        feats["near_diff_future{:g}".format(w)] = _future_mean(diff, frames)
        feats["ball_x_c_w{:g}".format(w)] = moving_nanmean(feats["ball_x_c_m"], frames)
        feats["ball_vx_w{:g}".format(w)] = moving_nanmean(vx, frames)
    names = list(feats)
    X = np.stack([np.asarray(feats[k], dtype=np.float32) for k in names], axis=1)
    return X, names
