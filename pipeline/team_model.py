"""Learned team model (p_home): trained on the IDSSE matches, checked one held-out match at a time.

    python pipeline/idsse.py         # first: download + convert the IDSSE matches
    python pipeline/team_model.py    # leave-one-match-out check, then the final model

What it does:
  1. each IDSSE match is degraded exactly as for the dead-ball model (same seed per match), turned
     into the feature table and into the team model's inputs (team_features.matrix)
  2. leave-one-match-out: for each match, the dead-ball model AND the team model are trained on
     the other six. The held-out match is labelled through the full pipeline twice:
        v2      dead-ball model + v1 team rules (nearest player) + decoder
        v2team  dead-ball model + learned team model          + decoder
     and both are scored with the OFFICIAL metrics against IDSSE's own labels. BAR-ATH is never
     used to choose anything.
  3. one temperature (a single number) rescales the team model's log-odds so its probabilities
     are honest for the decoder. It is fitted on the pooled held-out predictions of step 2.
  4. trains the final model on all matches and saves it, with the temperature, under the data
     directory (<DATA_DIR>/cache/models/team_v1.joblib)

Decision rule, agreed before running: v2team replaces v2 only if its mean spellF1 is higher AND it
wins on at least 5 of the 7 matches. The script prints the verdict.

Prints aggregates only.
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
import dead_model  # noqa: E402
import decode  # noqa: E402
import idsse  # noqa: E402
import rules  # noqa: E402
import team_features  # noqa: E402
import games  # noqa: E402

MODEL_PATH = os.path.join(games.DATA_DIR, "cache", "models", "team_v1.joblib")
TEMPERATURES = np.round(np.arange(0.5, 4.01, 0.1), 2)


def prepare(match_id, seed):
    """Dead-ball model inputs (dead_model.prepare) plus the team model's inputs, both sides."""
    m = dead_model.prepare(match_id, seed)
    m["XT"], m["team_names"] = team_features.matrix(m["cols"])
    m["XT_swap"], _ = team_features.matrix(team_features.swap(m["cols"]))
    m["live"] = m["use"] & np.isin(m["labels"], ("H", "A"))
    return m


def new_model():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(**C.TEAM_MODEL_PARAMS)


def fit(matches):
    """Trained on live frames, every TEAM_TRAIN_STEP-th, each also with the teams swapped."""
    Xs, ys = [], []
    for m in matches:
        rows = np.flatnonzero(m["live"])[::C.TEAM_TRAIN_STEP]
        home = m["labels"][rows] == "H"
        Xs += [m["XT"][rows], m["XT_swap"][rows]]
        ys += [home, ~home]
    model = new_model()
    model.fit(np.concatenate(Xs), np.concatenate(ys))
    return model


def raw_logit(model, X, X_swap):
    """Symmetric log-odds that home has the ball: averages the match and its swapped copy."""
    p = 0.5 * (model.predict_proba(X)[:, 1] + (1.0 - model.predict_proba(X_swap)[:, 1]))
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p) - np.log1p(-p)


def to_p_home(logit, temperature, cols, use_tags):
    """Calibrated p_home, plus v1's fixed-weight Smart Tagging team votes when there are tags."""
    s = logit / temperature
    if use_tags:
        votes = cols["tag_H_votes"].astype(np.float64) - cols["tag_A_votes"].astype(np.float64)
        s = s + C.TEAM_TAG_GAIN * votes
    return np.clip(rules.sigmoid(s), C.PROB_CLIP, 1 - C.PROB_CLIP)


def best_temperature(logits, targets):
    """The single temperature with the lowest log-loss on the pooled held-out live frames."""
    losses = []
    for t in TEMPERATURES:
        p = np.clip(rules.sigmoid(logits / t), 1e-6, 1 - 1e-6)
        losses.append(-np.mean(np.where(targets, np.log(p), np.log1p(-p))))
    k = int(np.argmin(losses))
    return float(TEMPERATURES[k]), float(losses[k]), float(losses[list(TEMPERATURES).index(1.0)])


def cross_validate(ids, matches):
    # Pass 1: out-of-fold dead-ball probabilities and team log-odds for every match.
    oof = []
    for k, held in enumerate(ids):
        t0 = time.time()
        others = [m for j, m in enumerate(matches) if j != k]
        dead = dead_model.fit(others)
        team = fit(others)
        m = matches[k]
        p_dead = dead_model.combine(dead.predict_proba(m["X"])[:, 1], m["cols"], use_tags=False)
        logit = raw_logit(team, m["XT"], m["XT_swap"])
        oof.append({"p_dead": p_dead, "logit": logit})
        rows = m["live"]
        acc = float(((logit[rows] > 0) == (m["labels"][rows] == "H")).mean())
        print("  {}  trained on the other {} ({:.0f} s)  team right on live frames {:.1f}%".format(
            held, len(others), time.time() - t0, 100.0 * acc))

    # One temperature from the pooled held-out predictions.
    pooled = np.concatenate([o["logit"][m["live"]] for o, m in zip(oof, matches)])
    target = np.concatenate([m["labels"][m["live"]] == "H" for m in matches])
    temperature, loss, loss_raw = best_temperature(pooled, target)
    print("\ntemperature {:.2f}  (held-out log-loss {:.4f}, {:.4f} without it)".format(
        temperature, loss, loss_raw))

    # Pass 2: full pipeline on each held-out match, v2 vs v2team, official metrics.
    line = "  {:<8} {:>7} {:>7} {:>11} | {:>7} {:>7} {:>11}"
    print("\nleave-one-match-out, official metrics on the held-out match (degraded tracking)")
    print("  {:<8} {:^27} | {:^27}".format("", "v2 (team rules)", "v2team (team model)"))
    print(line.format("match", "spellF1", "macroF1", "poss pr/gt", "spellF1", "macroF1",
                      "poss pr/gt"))
    rows = {"v2": [], "v2team": []}
    for k, held in enumerate(ids):
        m, o = matches[k], oof[k]
        gt = np.where(m["use"], m["labels"], "X")
        _, p_home_rules, forced = rules.evidence(m["cols"])
        p_home_model = to_p_home(o["logit"], temperature, m["cols"], use_tags=False)
        out = []
        for name, p_home in (("v2", p_home_rules), ("v2team", p_home_model)):
            s = dead_model.score(gt, decode.decode(m["cols"], o["p_dead"], p_home, forced))
            rows[name].append(s)
            out += [s["spell_f1_mean"], s["macro_f1"],
                    "{}/{}".format(s["n_pred_spells"], s["n_gt_spells"])]
        print(line.format(held, *["{:.3f}".format(v) if isinstance(v, float) else v
                                  for v in out]))
    means = {}
    for name in ("v2", "v2team"):
        means[name] = [np.mean([r[k] for r in rows[name]]) for k in ("spell_f1_mean", "macro_f1")]
        print("  mean {:<7} spellF1 {:.3f}  macroF1 {:.3f}".format(name, *means[name]))
    wins = sum(b["spell_f1_mean"] > a["spell_f1_mean"] for a, b in zip(rows["v2"], rows["v2team"]))
    adopt = means["v2team"][0] > means["v2"][0] and wins >= 5
    print("\nv2team wins on {} of {} matches.  VERDICT: {}".format(
        wins, len(ids), "ADOPT v2team" if adopt else "KEEP v2"))
    return temperature


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-cv", action="store_true",
                        help="skip the check and use temperature 1 (not for the submission)")
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
        m = prepare(match_id, C.DEGRADE_SEED + k)    # same degradation as dead_model.py
        print("  {}  live frames {}  home share {:.1f}%".format(
            match_id, int(m["live"].sum()),
            100.0 * float((m["labels"][m["live"]] == "H").mean())))
        matches.append(m)

    temperature = 1.0 if args.no_cv else cross_validate(ids, matches)

    model = fit(matches)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": model, "features": matches[0]["team_names"], "matches": ids,
                 "temperature": temperature, "sklearn": sklearn.__version__,
                 "params": C.TEAM_MODEL_PARAMS}, MODEL_PATH)
    print("\nfinal team model on {} matches, temperature {:.2f} -> "
          "<DATA_DIR>/cache/models/team_v1.joblib".format(len(ids), temperature))


def load_model():
    import joblib
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["features"], bundle["temperature"]


def predict_home(model, temperature, cols, use_tags=True):
    X, _ = team_features.matrix(cols)
    X_swap, _ = team_features.matrix(team_features.swap(cols))
    return to_p_home(raw_logit(model, X, X_swap), temperature, cols, use_tags)


if __name__ == "__main__":
    main()
