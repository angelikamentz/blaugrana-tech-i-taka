"""Learned team model (p_home): trained on the IDSSE matches, checked one held-out match at a time.

    python pipeline/idsse.py         # first: download + convert the IDSSE matches
    python pipeline/team_model.py    # leave-one-match-out check, then the final model

What it does:
  1. each IDSSE match is degraded exactly as for the dead-ball model (same seed per match), turned
     into the feature table and into the team model's inputs (team_features.matrix)
  2. leave-one-match-out: for each match, the dead-ball model AND every team-model variant are
     trained on the other six. The held-out match is labelled through the full pipeline as
        v2        dead-ball model + v1 team rules (nearest player)        + decoder
        v2team    dead-ball model + learned team model                    + decoder
        v2team+   dead-ball model + learned team model + team movement    + decoder
        ...+bridge  the same, with unsure team evidence filled from the confident evidence on
                    both sides when both name the same team (bridge.py)
     and all are scored with the OFFICIAL metrics against IDSSE's own labels. BAR-ATH is never
     used to choose anything.
  3. one temperature per variant (a single number) rescales the model's log-odds so its
     probabilities are honest for the decoder, fitted on that variant's pooled held-out
     predictions from step 2
  4. picks the variant with the agreed rule, trains it on all matches and saves it, with its
     temperature, under the data directory (<DATA_DIR>/cache/models/team_bridge.joblib)

Decision rule, agreed before running: a version replaces the one before it only if its mean
spellF1 is higher AND it wins on at least 5 of the 7 matches. v2team is judged against v2,
v2team+ against v2team, and +bridge against the team variant chosen before it. The script prints
each verdict.

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
import bridge  # noqa: E402
import games  # noqa: E402

MODEL_PATH = os.path.join(games.DATA_DIR, "cache", "models", "team_bridge.joblib")
TEMPERATURES = np.round(np.arange(0.5, 4.01, 0.1), 2)
VARIANTS = {"v2team": False, "v2team+": True}       # name -> team movement inputs on?
MIN_WINS = 5


def inputs(cols, movement):
    """(X, X with the teams swapped, names)."""
    X, names = team_features.matrix(cols, movement=movement)
    X_swap, _ = team_features.matrix(team_features.swap(cols), movement=movement)
    return X, X_swap, names


def prepare(match_id, seed, variants=VARIANTS):
    """Dead-ball model inputs (dead_model.prepare) plus each variant's team inputs."""
    m = dead_model.prepare(match_id, seed)
    m["XT"], m["XT_swap"], m["team_names"] = {}, {}, {}
    for v, movement in variants.items():
        m["XT"][v], m["XT_swap"][v], m["team_names"][v] = inputs(m["cols"], movement)
    m["live"] = m["use"] & np.isin(m["labels"], ("H", "A"))
    return m


def new_model():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(**C.TEAM_MODEL_PARAMS)


def fit(matches, variant):
    """Trained on live frames, every TEAM_TRAIN_STEP-th, each also with the teams swapped."""
    Xs, ys = [], []
    for m in matches:
        rows = np.flatnonzero(m["live"])[::C.TEAM_TRAIN_STEP]
        home = m["labels"][rows] == "H"
        Xs += [m["XT"][variant][rows], m["XT_swap"][variant][rows]]
        ys += [home, ~home]
    model = new_model()
    model.fit(np.concatenate(Xs), np.concatenate(ys))
    return model


def raw_logit(model, X, X_swap):
    """Symmetric log-odds that home has the ball: averages the match and its swapped copy."""
    p = 0.5 * (model.predict_proba(X)[:, 1] + (1.0 - model.predict_proba(X_swap)[:, 1]))
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p) - np.log1p(-p)


def to_p_home(logit, temperature, cols, use_tags, bridged=False):
    """Calibrated p_home, optionally bridged across unsure stretches (bridge.py), plus v1's
    fixed-weight Smart Tagging team votes when there are tags."""
    s = logit / temperature
    if bridged:
        s = bridge.apply(s, cols["half"])
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


def verdict(rows, new, old):
    wins = sum(b["spell_f1_mean"] > a["spell_f1_mean"] for a, b in zip(rows[old], rows[new]))
    gain = np.mean([r["spell_f1_mean"] for r in rows[new]]) - \
        np.mean([r["spell_f1_mean"] for r in rows[old]])
    adopt = gain > 0 and wins >= MIN_WINS
    print("  {} vs {}: wins {} of {}, mean spellF1 {:+.3f}  ->  {}".format(
        new, old, wins, len(rows[old]), gain, "ADOPT " + new if adopt else "KEEP " + old))
    return adopt


def cross_validate(ids, matches):
    names = ["v2"] + list(VARIANTS) + [v + "+bridge" for v in VARIANTS]
    # Pass 1: out-of-fold dead-ball probabilities and team log-odds for every match.
    oof = []
    for k, held in enumerate(ids):
        t0 = time.time()
        others = [m for j, m in enumerate(matches) if j != k]
        dead = dead_model.fit(others)
        m = matches[k]
        o = {"p_dead": dead_model.combine(dead.predict_proba(m["X"])[:, 1], m["cols"],
                                          use_tags=False), "logit": {}}
        msg = []
        for v in VARIANTS:
            team = fit(others, v)
            o["logit"][v] = raw_logit(team, m["XT"][v], m["XT_swap"][v])
            rows = m["live"]
            acc = float(((o["logit"][v][rows] > 0) == (m["labels"][rows] == "H")).mean())
            msg.append("{} {:.1f}%".format(v, 100.0 * acc))
        oof.append(o)
        print("  {}  trained on the other {} ({:.0f} s)  team right on live frames: {}".format(
            held, len(others), time.time() - t0, ", ".join(msg)))

    # One temperature per variant from its pooled held-out predictions.
    target = np.concatenate([m["labels"][m["live"]] == "H" for m in matches])
    temps = {}
    print()
    for v in VARIANTS:
        pooled = np.concatenate([o["logit"][v][m["live"]] for o, m in zip(oof, matches)])
        temps[v], loss, loss_raw = best_temperature(pooled, target)
        print("{:<8} temperature {:.2f}  (held-out log-loss {:.4f}, {:.4f} without it)".format(
            v, temps[v], loss, loss_raw))

    # Pass 2: full pipeline on each held-out match, official metrics.
    print("\nleave-one-match-out, official metrics on the held-out match (degraded tracking)")
    print("  {:<8}".format("match") + "".join(" | {:^25}".format(n) for n in names))
    print("  {:<8}".format("") + " | spellF1 macroF1 poss pr/gt" * len(names))
    rows = {n: [] for n in names}
    for k, held in enumerate(ids):
        m, o = matches[k], oof[k]
        gt = np.where(m["use"], m["labels"], "X")
        _, p_home_rules, forced = rules.evidence(m["cols"])
        line = "  {:<8}".format(held)
        for n in names:
            if n == "v2":
                p_home = p_home_rules
            else:
                v = n.replace("+bridge", "")
                p_home = to_p_home(o["logit"][v], temps[v], m["cols"], use_tags=False,
                                   bridged=n.endswith("+bridge"))
            s = dead_model.score(gt, decode.decode(m["cols"], o["p_dead"], p_home, forced))
            rows[n].append(s)
            line += " | {:>7.3f} {:>7.3f} {:>10}".format(
                s["spell_f1_mean"], s["macro_f1"],
                "{}/{}".format(s["n_pred_spells"], s["n_gt_spells"]))
        print(line)
    print()
    for n in names:
        print("  mean {:<8} spellF1 {:.3f}  macroF1 {:.3f}  possessions {:.0f} / {:.0f}".format(
            n, np.mean([r["spell_f1_mean"] for r in rows[n]]),
            np.mean([r["macro_f1"] for r in rows[n]]),
            np.mean([r["n_pred_spells"] for r in rows[n]]),
            np.mean([r["n_gt_spells"] for r in rows[n]])))

    print("\nverdicts (rule: higher mean spellF1 and at least {} of {} wins)".format(
        MIN_WINS, len(ids)))
    team_ok = verdict(rows, "v2team", "v2")
    plus_ok = verdict(rows, "v2team+", "v2team")
    chosen = "v2team+" if plus_ok else "v2team"
    bridge_ok = verdict(rows, chosen + "+bridge", chosen)
    print("  VERDICT: team model {}; saved for the pipeline: {}{}".format(
        "ADOPTED" if team_ok else "NOT ADOPTED (use --team rules)", chosen,
        " with +bridge" if bridge_ok else ", no bridge"))
    return chosen, temps[chosen], bridge_ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-cv", action="store_true",
                        help="skip the check and use temperature 1 (not for the submission)")
    parser.add_argument("--variant", choices=tuple(VARIANTS), default=None,
                        help="with --no-cv only: which variant to train for a quick test")
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

    if args.no_cv:
        chosen, temperature, bridged = args.variant or "v2team", 1.0, False
    else:
        if args.variant:
            sys.exit("--variant only goes with --no-cv; the check picks the variant itself")
        chosen, temperature, bridged = cross_validate(ids, matches)

    model = fit(matches, chosen)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": model, "features": matches[0]["team_names"][chosen],
                 "variant": chosen, "movement": VARIANTS[chosen], "bridge": bridged,
                 "matches": ids,
                 "temperature": temperature, "sklearn": sklearn.__version__,
                 "params": C.TEAM_MODEL_PARAMS}, MODEL_PATH)
    print("\nfinal team model ({}{}) on {} matches, temperature {:.2f} -> "
          "<DATA_DIR>/cache/models/team_bridge.joblib".format(
              chosen, " +bridge" if bridged else "", len(ids), temperature))


def load_model():
    """The saved bundle: model, features, variant, movement, temperature, ..."""
    import joblib
    return joblib.load(MODEL_PATH)


def predict_home(bundle, cols, use_tags=True):
    X, X_swap, _ = inputs(cols, bundle.get("movement", False))
    return to_p_home(raw_logit(bundle["model"], X, X_swap), bundle["temperature"], cols, use_tags,
                     bridged=bundle.get("bridge", False))


if __name__ == "__main__":
    main()
