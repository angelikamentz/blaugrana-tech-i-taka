"""Quick check of the team model's code on made-up data. Reads no match data.

    python pipeline/test_team_model.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import team_features as TF  # noqa: E402
import team_model as TM  # noqa: E402


def synthetic(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    home = (np.arange(n) // 500) % 2 == 0          # possession swaps every 20 s
    seen = rng.random(n) > 0.4
    near_h = np.where(home, 1.0, 6.0) + rng.random(n)
    near_a = np.where(home, 6.0, 1.0) + rng.random(n)
    bx = np.where(home, 70.0, 35.0) + rng.normal(0, 3, n)
    cols = {
        "ball_filled": seen, "ball_seen": seen,
        "near_H_m": np.where(seen, near_h, np.nan).astype(np.float32),
        "near_A_m": np.where(seen, near_a, np.nan).astype(np.float32),
        "n_H_near": rng.integers(0, 4, n).astype(np.int8),
        "n_A_near": rng.integers(0, 4, n).astype(np.int8),
        "n_H": np.full(n, 10, np.int8), "n_A": np.full(n, 10, np.int8),
        "ball_x_m": np.where(seen, bx, np.nan).astype(np.float32),
        "ball_y_m": np.where(seen, 34.0, np.nan).astype(np.float32),
        "ball_speed_ms": rng.random(n).astype(np.float32) * 5,
        "H_cx_m": np.where(home, 60.0, 40.0).astype(np.float32),
        "A_cx_m": np.where(home, 45.0, 65.0).astype(np.float32),
        "speed_H": rng.random(n).astype(np.float32) * 3,
        "speed_A": rng.random(n).astype(np.float32) * 3,
        "ball_gap_len_s": np.zeros(n, np.float32),
        "ball_since_seen_s": np.zeros(n, np.float32),
        "ball_until_seen_s": np.zeros(n, np.float32),
        "tag_H_votes": np.zeros(n, np.int8), "tag_A_votes": np.zeros(n, np.int8),
    }
    return cols, home


def main():
    cols, home = synthetic()
    X, names = TF.matrix(cols)
    Xs, names_s = TF.matrix(TF.swap(cols))
    assert X.shape == Xs.shape and names == names_s, "shape mismatch"
    k = names.index("near_diff_m")
    ok = np.isfinite(X[:, k])
    assert np.allclose(X[ok, k], -Xs[ok, k]), "swap must flip who is nearer"
    k = names.index("ball_x_c_m")
    ok = np.isfinite(X[:, k])
    assert np.allclose(X[ok, k], -Xs[ok, k], atol=1e-3), "swap must mirror the ball"
    print("features: {} columns, swap symmetric".format(len(names)))

    # v2team+: team movement inputs, and their behaviour under the swap
    Xm, Xms, names_m = TM.inputs(cols, True)
    assert len(names_m) > len(names), "movement must add inputs"
    k = names_m.index("cx_vx_diff_ms")
    ok = np.isfinite(Xm[:, k]) & np.isfinite(Xms[:, k])
    assert ok.any() and np.allclose(Xm[ok, k], Xms[ok, k], atol=1e-3), \
        "swap: (home - away) centre velocity is unchanged by the half-turn"
    print("movement: {} more columns".format(len(names_m) - len(names)))

    m = {"XT": {"v2team": X, "v2team+": Xm}, "XT_swap": {"v2team": Xs, "v2team+": Xms},
         "live": np.ones(len(home), bool), "labels": np.where(home, "H", "A")}
    for v, (a, b) in (("v2team", (X, Xs)), ("v2team+", (Xm, Xms))):
        model = TM.fit([m], v)
        logit = TM.raw_logit(model, a, b)
        p = TM.to_p_home(logit, 1.0, cols, use_tags=True)
        acc = float(((p > 0.5) == home).mean())
        t, _, _ = TM.best_temperature(logit, home)
        print("toy {}: right on {:.1f}% of frames, temperature {:.2f}".format(v, 100 * acc, t))
        assert acc > 0.9, "the toy model should separate the teams"
    bundle = {"model": model, "movement": True, "temperature": 1.0}
    assert TM.predict_home(bundle, cols).shape == home.shape
    print("OK")


if __name__ == "__main__":
    main()
