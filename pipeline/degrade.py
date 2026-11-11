"""Make clean tracking from other matches look like BAR-ATH's automatic tracking.

IDSSE's tracking is official and corrected: the ball is almost always there and all 22 players
are tracked. Ours loses the ball on 41% of graded frames, in long gaps, and misses ~7% of player
positions. A model trained on clean data would lean on the ball and fail on ours, so the training
data is degraded first: the ball is hidden in random gaps, players vanish for random stretches.

Gap placement is random and independent of the labels, on purpose: we don't know how strongly a
missing ball goes with a stoppage in BAR-ATH without looking at its answers, so the model is not
taught that link. All draws are seeded, so the same input always gives the same output.
"""

import numpy as np

import config as C

FPS = 25


def _lognormal_sigma(median, p90):
    """Sigma of a lognormal with this median and 90th percentile."""
    return float(np.log(p90 / median) / 1.2815515655446004)


def hide_ball(ball, graded, rng, target=None, median_s=None, p90_s=None):
    """Copy of `ball` with NaN gaps until `target` of graded frames have no ball."""
    target = C.DEGRADE_BALL_MISSING if target is None else target
    median_s = median_s or C.DEGRADE_BALL_GAP_MEDIAN_S
    p90_s = p90_s or C.DEGRADE_BALL_GAP_P90_S
    ball = ball.copy()
    missing = ~np.isfinite(ball[:, 0])
    idx = np.flatnonzero(graded)
    if not len(idx):
        return ball
    sigma = _lognormal_sigma(median_s, p90_s)
    goal = int(target * len(idx))
    hidden = int(missing[idx].sum())
    for _ in range(100000):
        if hidden >= goal:
            break
        start = int(rng.choice(idx))
        length = max(1, int(round(rng.lognormal(np.log(median_s * FPS), sigma))))
        stop = min(len(ball), start + length)
        new = graded[start:stop] & ~missing[start:stop]
        hidden += int(new.sum())
        missing[start:stop] = True
    ball[missing] = np.nan
    return ball


def drop_players(players, rng, target=None, median_s=None):
    """Copy of `players` with random per-player stretches set to NaN until `target` of the
    present positions are gone."""
    target = C.DEGRADE_PLAYER_MISSING if target is None else target
    median_s = median_s or C.DEGRADE_PLAYER_GAP_MEDIAN_S
    players = players.copy()
    present = np.isfinite(players[:, :, 0])
    total = int(present.sum())
    if not total:
        return players
    goal = int(target * total)
    removed = 0
    n, s = present.shape
    candidates = np.flatnonzero(present.any(axis=0))
    for _ in range(1000000):
        if removed >= goal:
            break
        slot = int(rng.choice(candidates))
        start = int(rng.integers(0, n))
        length = max(1, int(round(rng.lognormal(np.log(median_s * FPS), 0.8))))
        stop = min(n, start + length)
        removed += int(present[start:stop, slot].sum())
        present[start:stop, slot] = False
    players[~present] = np.nan
    return players


def degrade(players, ball, graded, seed):
    """Both degradations with one seeded generator. Returns (players, ball)."""
    rng = np.random.default_rng(seed)
    return drop_players(players, rng), hide_ball(ball, graded, rng)
