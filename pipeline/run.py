"""The whole pipeline, raw files to a submission, in one command.

    python pipeline/run.py                 # writes evaluation/results/v1.csv
    python pipeline/run.py --name v1b      # another file name
    python pipeline/run.py --rebuild       # rebuild the feature cache first

Run it in your own terminal: it reads the match data. It prints aggregates of its own output only
(label shares, possession counts and lengths), which are safe to paste back to Claude.

Stages: features.py (feature table) -> rules.py (per-frame evidence) -> decode.py (Viterbi) ->
CSV. Deterministic: same data, same file.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import features as F  # noqa: E402  (sets up the rest of sys.path)
import rules  # noqa: E402
import decode  # noqa: E402

RESULTS_DIR = os.path.join(F.ROOT, "evaluation", "results")


def write_submission(labels, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))


def stretches(labels):
    """[(label, length)] of maximal runs."""
    if not len(labels):
        return []
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(labels)]))
    return [(labels[a], b - a) for a, b in zip(starts, ends)]


def report(cols, labels, p_dead, p_home, forced):
    line = "  {:<44} {}"
    fps = float(F.FPS)
    print("=" * 90)
    print("PIPELINE -- what the submission says (aggregates of our own output only)")
    print("=" * 90)
    g = cols["graded"]
    print(line.format("graded frames", int(g.sum())))
    print(line.format("forced dead before 2nd-half kick-off tag", int(forced.sum())))
    print(line.format("mean p_dead / frames with p_dead > 0.5 %", "{:.3f} / {:.1f}".format(
        p_dead[g].mean(), 100.0 * (p_dead[g] > 0.5).mean())))
    print(line.format("frames with no team evidence %", "{:.1f}".format(
        100.0 * (np.abs(p_home[g] - 0.5) < 1e-6).mean())))
    total_pos = 0
    for half in (1, 2):
        lab = labels[(cols["half"] == half) & g]
        runs = stretches(lab)
        pos = [n for l, n in runs if l in ("H", "A")]
        dead = [n for l, n in runs if l == "D"]
        total_pos += len(pos)
        print("\n## half {}".format(half))
        print(line.format("H / A / D %", " / ".join(
            "{:.1f}".format(100.0 * (lab == k).mean()) for k in ("H", "A", "D"))))
        print(line.format("possessions (H / A)", "{} ({} / {})".format(
            len(pos), sum(1 for l, _ in runs if l == "H"), sum(1 for l, _ in runs if l == "A"))))
        if pos:
            print(line.format("possession length s: p10 / p50 / p90", " / ".join(
                "{:.1f}".format(v / fps) for v in np.percentile(pos, (10, 50, 90)))))
        print(line.format("dead stretches", len(dead)))
        if dead:
            print(line.format("dead length s: p10 / p50 / p90", " / ".join(
                "{:.1f}".format(v / fps) for v in np.percentile(dead, (10, 50, 90)))))
        direct = sum(1 for (l1, _), (l2, _) in zip(runs, runs[1:])
                     if {l1, l2} == {"H", "A"})
        print(line.format("direct H<->A hand-overs", direct))
    print("\n" + line.format("possessions, whole match", total_pos))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dead", choices=("rules", "model"), default="rules",
                        help="dead-ball evidence: v1 rules, or the v2 model trained on IDSSE "
                             "(python pipeline/dead_model.py first)")
    parser.add_argument("--name", default=None,
                        help="output file name, without .csv (default: v1 / v2)")
    parser.add_argument("--rebuild", action="store_true", help="rebuild the feature cache")
    args = parser.parse_args()
    name = args.name or ("v2" if args.dead == "model" else "v1")

    cols, _ = F.build(use_cache=not args.rebuild)
    p_dead, p_home, forced = rules.evidence(cols)
    if args.dead == "model":
        import dead_model
        model, names = dead_model.load_model()
        _, current = dead_model.dead_features.matrix(cols)
        if list(names) != list(current):
            sys.exit("the saved model was trained on different features; rerun "
                     "python pipeline/dead_model.py")
        p_dead = dead_model.predict_dead(model, cols, use_tags=True)
    labels = decode.decode(cols, p_dead, p_home, forced)
    args.name = name
    path = os.path.join(RESULTS_DIR, args.name + ".csv")
    write_submission(labels, path)
    report(cols, labels, p_dead, p_home, forced)
    print("\nwrote evaluation/results/{}.csv".format(args.name))
    print("next: python evaluation/validate_submission.py evaluation/results/{}.csv".format(
        args.name))


if __name__ == "__main__":
    main()
