"""Stage 2: per-frame evidence from the feature table, as probabilities.

    p_dead  P(play is stopped)
    p_home  P(home has the ball | play is live)

Each rule adds or removes log-odds; the weights and their reasons are in config.py. No I/O here:
`evidence()` takes the feature columns and returns arrays, so it is tested on synthetic input.
"""

import numpy as np

import config as C
from features import FPS, seconds_until_next, SETPIECE_TYPES


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def setpiece_window(cols):
    """True from DEAD_SETPIECE_BEFORE_S before a set-piece tag start to DEAD_SETPIECE_AFTER_S
    after it."""
    since = cols["tag_setpiece_since_s"]
    starts = np.isfinite(since) & (since == 0)
    until = seconds_until_next(starts)
    after = np.isfinite(since) & (since <= C.DEAD_SETPIECE_AFTER_S)
    before = np.isfinite(until) & (until <= C.DEAD_SETPIECE_BEFORE_S)
    return after | before


def dead_score(cols):
    """Log-odds that play is stopped."""
    n = len(cols["frame"])
    s = np.full(n, C.DEAD_BIAS, dtype=np.float64)
    s += C.DEAD_BALL_OUT * cols["ball_out"]
    s += C.DEAD_SETPIECE * setpiece_window(cols)
    speed = np.nan_to_num(cols["speed_all"], nan=C.DEAD_SLOW_MS)  # no players: neutral-ish
    s += C.DEAD_SLOW * (speed < C.DEAD_SLOW_MS)
    s += C.DEAD_VERY_SLOW * (speed < C.DEAD_VERY_SLOW_MS)
    s += C.DEAD_LONG_GAP * (cols["ball_gap_len_s"] > C.DEAD_LONG_GAP_S)
    ball_speed = np.nan_to_num(cols["ball_speed_ms"], nan=0.0)
    s += C.LIVE_BALL_FAST * (ball_speed > C.LIVE_BALL_FAST_MS)
    s += C.LIVE_TEAM_FAST * (speed > C.LIVE_TEAM_FAST_MS)
    return s


def team_score(cols):
    """Log-odds that home has the ball, given play is live. 0 = no information."""
    have = cols["ball_filled"]
    near_h = np.where(np.isfinite(cols["near_H_m"]), cols["near_H_m"], C.TEAM_NO_PLAYER_M)
    near_a = np.where(np.isfinite(cols["near_A_m"]), cols["near_A_m"], C.TEAM_NO_PLAYER_M)
    control = np.where(np.minimum(near_h, near_a) < C.TEAM_CONTROL_RADIUS_M,
                       1.0, C.TEAM_LOOSE_FACTOR)
    near = C.TEAM_NEAR_GAIN * np.tanh((near_a - near_h) / C.TEAM_NEAR_SCALE_M) * control
    s = np.where(have, near, 0.0)
    votes = cols["tag_H_votes"].astype(np.float64) - cols["tag_A_votes"].astype(np.float64)
    s += C.TEAM_TAG_GAIN * votes
    return s


def first_kickoff_tag_start(cols, half):
    """First frame of the first 'Kick Off (Start)' set-piece tag in a half, or None.

    Everything in a graded half before it is the wait for kick-off, so it is dead. Found from the
    tags, not written in by hand.
    """
    kick = SETPIECE_TYPES.index("Kick Off (Start)")
    rows = np.flatnonzero((cols["half"] == half) & (cols["tag_setpiece_type"] == kick))
    return int(rows[0]) if len(rows) else None


def evidence(cols):
    """(p_dead, p_home, forced_dead) arrays of length n."""
    lo, hi = C.PROB_CLIP, 1.0 - C.PROB_CLIP
    p_dead = np.clip(sigmoid(dead_score(cols)), lo, hi)
    p_home = np.clip(sigmoid(team_score(cols)), lo, hi)
    forced = np.zeros(len(p_dead), dtype=bool)
    for half in (1, 2):
        start = first_kickoff_tag_start(cols, half)
        if start is not None:
            forced[(cols["half"] == half) & (cols["frame"] < start)] = True
    return p_dead, p_home, forced
