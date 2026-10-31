"""Where a submission goes wrong: a list of timestamped problems to go and look at.

    python evaluation/problems.py                     # every CSV in evaluation/results/
    python evaluation/problems.py results/mine.csv    # one file

Writes evaluation/reports/<name>_problems.csv (gitignored) and prints a summary per category.
The dashboard's Problems tab shows the same list.

A problem is a stretch of graded frames where the submission disagrees with the ground truth,
named by what kind of mistake it is -- the label changed too late, a possession was split by a
flicker, a stoppage was missed -- because the kind says what to fix. Categories are judged on
the abstention-filled prediction, the way spellF1 sees it; the stretches themselves on the raw
one, the way macroF1 sees it.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import possession_metrics as pm  # noqa: E402

GT_DIR = os.path.join(HERE, "gt")
RESULTS_DIR = os.path.join(HERE, "results")
REPORTS_DIR = os.path.join(HERE, "reports")
FPS = pm.FPS
LIVE = (pm.H, pm.A)

# Category -> what to do about it. Order is the order the summary prints in.
CATEGORIES = {
    "Late handover": "possession changed team, the submission switched late",
    "Early handover": "the submission switched team before the truth did",
    "Late stoppage": "ball went dead, the submission kept a team on it",
    "Early stoppage": "the submission called dead ball before it was",
    "Late restart": "play restarted, the submission stayed on D",
    "Early restart": "the submission restarted play before it was",
    "Flicker: possession split": "switch to the other team and back, inside one true possession",
    "Flicker: phantom stoppage": "D and back to the same team, inside live play",
    "Flicker: phantom play": "a team label and back to D, inside a stoppage",
    "Missed whole possession": "a true possession that never appears",
    "Missed whole stoppage": "a true stoppage that never appears",
    "Wrong team": "live ball, other team",
    "Missed stoppage": "truth is D, the submission says a team",
    "Phantom stoppage": "truth is live, the submission says D",
    "No answer (X)": "abstained; free on spellF1 because X repeats the previous label",
}


def change_kind(before, after):
    if before in LIVE and after in LIVE:
        return "handover"
    return "stoppage" if after == pm.D else "restart"


def video_time(frame):
    """h:mm:ss.ss in the tactical video -- what to type into a video player to seek."""
    seconds = frame / float(FPS)
    return "{:d}:{:02d}:{:05.2f}".format(int(seconds // 3600), int(seconds % 3600 // 60),
                                         seconds % 60)


def sequence(codes):
    """'H' or 'H → D' -- the labels a stretch passes through, in order."""
    keep = [codes[0]] + [c for p, c in zip(codes[:-1], codes[1:]) if c != p]
    return " → ".join(pm.LABELS[c] for c in keep)


def classify(gt, filled, a, z, start, stop):
    """Category of the wrong stretch [a, z) inside the graded run [start, stop)."""
    g, p = gt[a:z], filled[a:z]
    if np.array_equal(g, p):
        return "No answer (X)"
    g0 = int(g[0])
    constant = (g == g0).all()
    before = int(gt[a - 1]) if a > start else None
    after = int(gt[z]) if z < stop else None
    p_before = int(filled[a - 1]) if a > start else None
    p_after = int(filled[z]) if z < stop else None

    if constant:
        if before != g0 and after != g0 and (p != g0).all():
            return "Missed whole " + ("stoppage" if g0 == pm.D else "possession")
        if before is not None and before != g0 and (p == before).all():
            return "Late " + change_kind(before, g0)
        if after is not None and after != g0 and (p == after).all():
            return "Early " + change_kind(g0, after)
        if before == g0 == after and p_before == g0 == p_after:
            if g0 == pm.D:
                return "Flicker: phantom play"
            return "Flicker: phantom stoppage" if (p == pm.D).any() else \
                "Flicker: possession split"

    wrong_team = int(np.sum(np.isin(g, LIVE) & np.isin(p, LIVE) & (g != p)))
    missed_dead = int(np.sum((g == pm.D) & np.isin(p, LIVE)))
    phantom_dead = int(np.sum(np.isin(g, LIVE) & (p == pm.D)))
    counts = {"Wrong team": wrong_team, "Missed stoppage": missed_dead,
              "Phantom stoppage": phantom_dead}
    return max(counts, key=counts.get) if any(counts.values()) else "No answer (X)"


def find_problems(gt_codes, pred_codes):
    """DataFrame of every wrong stretch on graded frames, in time order."""
    runs = pm.valid_runs(gt_codes)
    filled = pm.fill_abstentions(pred_codes, runs)
    rows = []
    for start, stop in runs:
        wrong = (pred_codes[start:stop] != gt_codes[start:stop]).astype(np.int8)
        edges = np.diff(np.concatenate(([0], wrong, [0])))
        for a, z in zip(np.flatnonzero(edges == 1) + start, np.flatnonzero(edges == -1) + start):
            a, z = int(a), int(z)
            rows.append({
                "video_time": video_time(a),
                "start_frame": a,
                "end_frame": z - 1,
                "duration_s": round((z - a) / float(FPS), 2),
                "problem": classify(gt_codes, filled, a, z, start, stop),
                "truth": sequence(gt_codes[a:z]),
                "said": sequence(pred_codes[a:z]),
            })
    columns = ["video_time", "start_frame", "end_frame", "duration_s", "problem", "truth", "said"]
    return pd.DataFrame(rows, columns=columns)


def summary(problems):
    """Per category: how many stretches and how many seconds, biggest first."""
    if problems.empty:
        return pd.DataFrame(columns=["problem", "count", "seconds", "meaning"])
    out = (problems.groupby("problem")["duration_s"].agg(["count", "sum"])
           .rename(columns={"sum": "seconds"}).reset_index())
    out["seconds"] = out["seconds"].round(1)
    out["meaning"] = out["problem"].map(CATEGORIES)
    return out.sort_values("seconds", ascending=False, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("submissions", nargs="*", help="CSV files (default: evaluation/results)")
    parser.add_argument("--gt", default=None, help="ground truth CSV (default: the public one)")
    args = parser.parse_args()

    gt_path = args.gt or sorted(glob.glob(os.path.join(GT_DIR, "*_public.csv")))[0]
    gt_codes = pm.label_codes(pm.read_label_csv(gt_path))
    paths = args.submissions or sorted(glob.glob(os.path.join(RESULTS_DIR, "*.csv")))
    os.makedirs(REPORTS_DIR, exist_ok=True)

    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        table = pm.read_label_csv(path)
        reason = pm.validate_submission(table, len(gt_codes))
        if reason:
            print("{}: skipped, {}".format(name, reason))
            continue
        problems = find_problems(gt_codes, pm.label_codes(table))
        out = os.path.join(REPORTS_DIR, name + "_problems.csv")
        problems.to_csv(out, index=False)
        print("\n{}  {} problems, {:.1f} s wrong  -> {}".format(
            name, len(problems), problems["duration_s"].sum(), os.path.relpath(out)))
        print(summary(problems).to_string(index=False))


if __name__ == "__main__":
    main()
