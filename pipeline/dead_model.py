"""v2 dead-ball model: learned on the IDSSE matches, checked one held-out match at a time.

    python pipeline/idsse.py         # first: download + convert the IDSSE matches
    python pipeline/dead_model.py    # leave-one-match-out check, then the final model

What it does:
  1. each IDSSE match is degraded to look like our tracking (degrade.py), turned into the same
     feature table as BAR-ATH (features.compute) and the model inputs (dead_features.matrix)
  2. leave-one-match-out: train on six matches, predict the seventh, and score the held-out
     match with the OFFICIAL metrics (evaluation/possession_metrics.py) -- once with v1's
     dead-ball rules and once with the model, both through the same team rules and decoder.
     This is the legal way to compare versions: other games, real labels, never BAR-ATH.
  3. trains the final model on all matches and saves it under the data directory
     (<DATA_DIR>/cache/models/dead_v2.joblib)

Prints aggregates only. IDSSE is open data about other games, but the printout stays in the same
style as everything else here.
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import config as C  # noqa: E402
import features as F  # noqa: E402
import dead_features  # noqa: E402
import decode  # noqa: E402
import degrade  # noqa: E402
import idsse  # noqa: E402
import rules  # noqa: E402
import games  # noqa: E402

sys.path.insert(0, os.path.join(F.ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

L = F.L
MODEL_PATH = os.path.join(games.DATA_DIR, "cache", "models", "dead_v2.joblib")


def prepare(match_id, seed):
    """Degraded feature table, model inputs and labels for one IDSSE match."""
    a = idsse.load(match_id)
    graded = np.isin(a["phase"], L.GRADED_PHASES)
    players, ball = degrade.degrade(a["players"], a["ball"], graded, seed)
    cols, _ = F.compute(players, ball, a["side"], a["phase"], F.empty_tags(),
                        pitch=tuple(float(v) for v in a["pitch"]))
    X, names = dead_features.matrix(cols)
    labels = a["labels"]
    use = graded & (labels != "X")
    return {"cols": cols, "X": X, "names": names, "labels": labels, "use": use}


def new_model():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(**C.DEAD_MODEL_PARAMS)


def fit(matches):
    Xs, ys = [], []
    for m in matches:
        rows = np.flatnonzero(m["use"])[::C.DEAD_TRAIN_STEP]
        Xs.append(m["X"][rows])
        ys.append(m["labels"][rows] == "D")
    model = new_model()
    model.fit(np.concatenate(Xs), np.concatenate(ys))
    return model


def combine(p_model, cols, use_tags):
    """Model probability, plus the Smart Tagging set-piece prior when the match has tags."""
    p = np.clip(p_model, 1e-4, 1 - 1e-4)
    logit = np.log(p) - np.log1p(-p)
    if use_tags:
        logit = logit + C.DEAD_TAG_PRIOR * rules.setpiece_window(cols)
    return np.clip(rules.sigmoid(logit), C.PROB_CLIP, 1 - C.PROB_CLIP)


def predict_dead(model, cols, use_tags=True):
    X, _ = dead_features.matrix(cols)
    return combine(model.predict_proba(X)[:, 1], cols, use_tags)


def labels_with(cols, p_dead):
    """Full pipeline labels given a p_dead: v1 team rules + decoder."""
    _, p_home, forced = rules.evidence(cols)
    return decode.decode(cols, p_dead, p_home, forced)


def score(gt, pred):
    r = pm.evaluate(pm.label_codes(gt), pm.label_codes(pred))
    return {k: r[k] for k in ("spell_f1_mean", "macro_f1", "alive_dead_f1", "n_gt_spells",
                              "n_pred_spells")} | {"D_f1": r["per_class"]["D"]["f1"]}


def cross_validate(ids, matches):
    line = "  {:<8} {:>7} {:>7} {:>7} {:>7} {:>11} | {:>7} {:>7} {:>7} {:>7} {:>11}"
    print("\nleave-one-match-out, official metrics on the held-out match (degraded tracking)")
    print(line.format("match", "spellF1", "macroF1", "aliveF1", "D F1", "poss pr/gt",
                      "spellF1", "macroF1", "aliveF1", "D F1", "poss pr/gt"))
    print("  {:<8} {:^51} | {:^51}".format("", "v1 dead-ball rules", "v2 dead-ball model"))
    rows = {"v1": [], "v2": []}
    for k, held in enumerate(ids):
        t0 = time.time()
        model = fit([m for j, m in enumerate(matches) if j != k])
        m = matches[k]
        gt = np.where(m["use"], m["labels"], "X")
        p_v1 = np.clip(rules.sigmoid(rules.dead_score(m["cols"])), C.PROB_CLIP, 1 - C.PROB_CLIP)
        p_v2 = combine(model.predict_proba(m["X"])[:, 1], m["cols"], use_tags=False)
        out = []
        for name, p in (("v1", p_v1), ("v2", p_v2)):
            s = score(gt, labels_with(m["cols"], p))
            rows[name].append(s)
            out += [s["spell_f1_mean"], s["macro_f1"], s["alive_dead_f1"], s["D_f1"],
                    "{}/{}".format(s["n_pred_spells"], s["n_gt_spells"])]
        print(line.format(held, *["{:.3f}".format(v) if isinstance(v, float) else v for v in out])
              + "   ({:.0f} s)".format(time.time() - t0))
    for name in ("v1", "v2"):
        keys = ("spell_f1_mean", "macro_f1", "alive_dead_f1", "D_f1")
        means = [np.mean([r[k] for r in rows[name]]) for k in keys]
        print("  mean {}: spellF1 {:.3f}  macroF1 {:.3f}  aliveF1 {:.3f}  D F1 {:.3f}".format(
            name, *means))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-cv", action="store_true", help="skip the leave-one-match-out check")
    args = parser.parse_args()
    import joblib
    import sklearn

    ids = idsse.available()
    if len(ids) < 2:
        sys.exit("need at least 2 converted IDSSE matches (python pipeline/idsse.py); have {}"
                 .format(len(ids)))
    print("IDSSE matches: {}".format(", ".join(ids)))
    matches = []
    for k, match_id in enumerate(ids):
        m = prepare(match_id, C.DEGRADE_SEED + k)
        print("  {}  frames {}  graded {}  dead {:.1f}%  ball after degrading {:.1f}%".format(
            match_id, len(m["labels"]), int(m["use"].sum()),
            100.0 * float((m["labels"][m["use"]] == "D").mean()),
            100.0 * float(m["cols"]["ball_seen"][m["use"]].mean())))
        matches.append(m)

    if not args.no_cv:
        cross_validate(ids, matches)

    model = fit(matches)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": model, "features": matches[0]["names"], "matches": ids,
                 "sklearn": sklearn.__version__, "params": C.DEAD_MODEL_PARAMS}, MODEL_PATH)
    print("\nfinal model on {} matches -> <DATA_DIR>/cache/models/dead_v2.joblib".format(len(ids)))


def load_model():
    import joblib
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["features"]


if __name__ == "__main__":
    main()
