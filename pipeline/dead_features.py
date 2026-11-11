"""The inputs of the v2 dead-ball model, built from a feature table (features.compute).

Only tracking-derived columns go in, because the model is trained on other matches that have
nothing else. Smart Tagging is added later, outside the model (config.DEAD_TAG_PRIOR).

Each base signal is used as is and averaged over short and long windows centred on the frame,
plus a past-vs-future comparison of team speed (slowing down into a stoppage, speeding up out of
a restart). Offline labelling, so looking ahead is allowed.
"""

import numpy as np

import config as C
from features import FPS, moving_nanmean


def ball_still_s(cols):
    """Seconds the ball has been seen and nearly still, counted along each still run."""
    still = cols["ball_filled"] & (np.nan_to_num(cols["ball_speed_ms"], nan=99.0) < C.BALL_STILL_MS)
    out = np.zeros(len(still), dtype=np.float32)
    padded = np.concatenate(([False], still, [False])).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    for a, z in zip(edges[0::2], edges[1::2]):
        out[a:z] = np.arange(1, z - a + 1) / float(FPS)
    return out


def _shift(values, frames):
    """values moved later by `frames` (positive) or earlier (negative); NaN where it runs out."""
    out = np.full(len(values), np.nan, dtype=np.float32)
    if frames > 0:
        out[frames:] = values[:-frames]
    elif frames < 0:
        out[:frames] = values[-frames:]
    else:
        out[:] = values
    return out


def matrix(cols):
    """(X (n, k) float32, feature names). NaN marks a missing value; the model handles it."""
    base = {
        "ball_seen": cols["ball_seen"].astype(np.float32),
        "ball_out": cols["ball_out"].astype(np.float32),
        "ball_line_dist_m": cols["ball_line_dist_m"],
        "ball_speed_ms": cols["ball_speed_ms"],
        "ball_still_s": ball_still_s(cols),
        "log_ball_gap_s": np.log1p(cols["ball_gap_len_s"]).astype(np.float32),
        "log_ball_since_s": np.log1p(cols["ball_since_seen_s"]).astype(np.float32),
        "log_ball_until_s": np.log1p(cols["ball_until_seen_s"]).astype(np.float32),
        "speed_all": cols["speed_all"],
        "frac_slow": cols["frac_slow"],
        "spread_x_m": cols["spread_x_m"],
        "spread_y_m": cols["spread_y_m"],
        "n_crowd": cols["n_crowd"].astype(np.float32),
        "near_any_m": np.fmin(cols["near_H_m"], cols["near_A_m"]),
    }
    windowed = ("ball_seen", "ball_out", "ball_speed_ms", "speed_all", "frac_slow",
                "spread_x_m", "n_crowd", "near_any_m")
    feats = dict(base)
    for w in C.DEAD_WINDOWS_S:
        frames = int(round(w * FPS))
        for name in windowed:
            feats["{}_w{:g}".format(name, w)] = moving_nanmean(base[name], frames)
    shift = int(round(C.DEAD_SHIFT_S * FPS))
    smooth = moving_nanmean(base["speed_all"], shift)
    past = _shift(smooth, shift // 2)              # mean over the last ~3 s
    future = _shift(smooth, -(shift // 2))         # mean over the next ~3 s
    feats["speed_future_minus_past"] = future - past
    names = list(feats)
    X = np.stack([np.asarray(feats[k], dtype=np.float32) for k in names], axis=1)
    return X, names
