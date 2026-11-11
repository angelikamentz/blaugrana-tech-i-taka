"""Stage 3: turn per-frame evidence into a clean sequence of H / A / D (Viterbi).

Three states. Each frame scores every state by how well it fits the evidence:

    D:  log p_dead
    H:  log (1 - p_dead) + log p_home
    A:  log (1 - p_dead) + log (1 - p_home)

and every change of state costs a fixed amount (config.SWITCH_COST_*). Viterbi finds the
sequence with the best total, looking at the whole half at once, so a change is placed where the
evidence actually turns, not a smoothing window later. Frames with no team evidence (p_home = 0.5)
favour neither team, so the current owner carries on through ball gaps.

Each graded half is decoded on its own. Ungraded frames are written as X.
"""

import numpy as np

import config as C

H, A, D = 0, 1, 2
LABELS = np.array(["H", "A", "D"])


def emissions(p_dead, p_home, forced_dead):
    """(n, 3) log-likelihood of each state per frame."""
    live = np.log1p(-p_dead)
    out = np.stack([live + np.log(p_home), live + np.log1p(-p_home), np.log(p_dead)], axis=1)
    out[forced_dead, H] = -np.inf
    out[forced_dead, A] = -np.inf
    out[forced_dead, D] = 0.0
    return out


def transition_cost():
    """(3, 3) cost of going from state i (row) to state j (column)."""
    t, d = C.SWITCH_COST_TEAM, C.SWITCH_COST_DEAD
    return np.array([[0.0, t, d],
                     [t, 0.0, d],
                     [d, d, 0.0]])


def viterbi(emit, cost):
    """Best state path for one stretch of frames."""
    n = len(emit)
    back = np.zeros((n, 3), dtype=np.int8)
    score = emit[0].copy()
    for t in range(1, n):
        cand = score[:, None] - cost           # cand[i, j]: come from i into j
        back[t] = np.argmax(cand, axis=0)
        score = cand[back[t], np.arange(3)] + emit[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(cols, p_dead, p_home, forced_dead):
    """Labels for every frame: H / A / D on graded frames, X elsewhere."""
    labels = np.full(len(p_dead), "X", dtype="<U1")
    emit = emissions(p_dead, p_home, forced_dead)
    cost = transition_cost()
    for half in (1, 2):
        rows = np.flatnonzero((cols["half"] == half) & cols["graded"])
        if len(rows):
            labels[rows] = LABELS[viterbi(emit[rows], cost)]
    return labels
