"""Download the IDSSE open matches and convert each to the same arrays BAR-ATH uses.

    python pipeline/idsse.py                 # every match not converted yet
    python pipeline/idsse.py --only J03WPY   # one match

IDSSE: 7 Bundesliga 2022/23 matches by the DFL, CC-BY 4.0 (Bassek et al., Scientific Data 2025,
doi 10.1038/s41597-025-04505-y; data on Hugging Face, pysport/idsse-data). Official tracking at
25 fps, with a per-frame ball status (alive / dead) and the team in possession: the same kind of
label as our ground truth. It is about other games, so the rules allow training on it.

Downloads go through kloppy (requirements-pipeline.txt). Each match is a large XML; expect a few
hundred MB per match and a few minutes to parse. Converted arrays are written under the data
directory (<DATA_DIR>/external/idsse/<match>.npz) to keep all large files in one place, outside
the repository.

Per match the arrays are:
  players (n, S, 3)  x, y normalised 0-1, speed m/s (computed from positions)
  ball    (n, 3)     x, y normalised 0-1, NaN
  side    (S,)       0 home / 1 away
  phase   (n,)       loaders.FIRST / HALFTIME (one separator frame) / SECOND
  labels  (n,)       H / A / D, X on the separator
  pitch   (2,)       length, width in metres
Halves are concatenated with one separator frame between them, so nothing temporal runs across.
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import features as F  # noqa: E402
import games  # noqa: E402

L = F.L
OUT_DIR = os.path.join(games.DATA_DIR, "external", "idsse")
# kloppy's own match map (kloppy master, _providers/sportec.py). The dataset card lists two
# different ids; any id kloppy refuses is reported and skipped.
MATCH_IDS = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
SPEED_SMOOTH_FRAMES = 5


def _speed_from_positions(x_m, y_m, fps):
    """Per-player speed in m/s from positions, smoothed over a few frames."""
    v = np.full(x_m.shape, np.nan, dtype=np.float32)
    v[1:] = np.hypot(np.diff(x_m, axis=0), np.diff(y_m, axis=0)) * fps
    for s in range(v.shape[1]):
        v[:, s] = F.moving_nanmean(v[:, s], SPEED_SMOOTH_FRAMES)
    return v


def convert(dataset):
    """kloppy TrackingDataset -> dict of arrays (see module docstring), plus a small report."""
    meta = dataset.metadata
    home_team, away_team = meta.teams[0], meta.teams[1]
    roster = [(p, 0) for p in home_team.players] + [(p, 1) for p in away_team.players]
    index = {p.player_id: i for i, (p, _) in enumerate(roster)}
    side = np.array([s for _, s in roster], dtype=np.int8)

    dims = meta.pitch_dimensions
    x_lo, x_hi = dims.x_dim.min, dims.x_dim.max
    y_lo, y_hi = dims.y_dim.min, dims.y_dim.max
    length = getattr(dims, "pitch_length", None) or (x_hi - x_lo)
    width = getattr(dims, "pitch_width", None) or (y_hi - y_lo)
    fps = float(dataset.frame_rate or 25)

    frames = [f for f in dataset.frames if f.period is not None and f.period.id in (1, 2)]
    frames.sort(key=lambda f: (f.period.id, f.frame_id))
    n1 = sum(1 for f in frames if f.period.id == 1)
    n = len(frames) + 1                                    # + one separator frame
    S = len(roster)
    players = np.full((n, S, 3), np.nan, dtype=np.float32)
    ball = np.full((n, 3), np.nan, dtype=np.float32)
    labels = np.full(n, "X", dtype="<U1")
    phase = np.full(n, L.HALFTIME, dtype=np.int8)

    unknown_state = no_owner = 0
    last_team = "H"
    for k, frame in enumerate(frames):
        i = k if k < n1 else k + 1
        phase[i] = L.FIRST if frame.period.id == 1 else L.SECOND
        for player, pdata in frame.players_data.items():
            j = index.get(player.player_id)
            if j is None or pdata.coordinates is None:
                continue
            players[i, j, 0] = (pdata.coordinates.x - x_lo) / (x_hi - x_lo)
            players[i, j, 1] = (pdata.coordinates.y - y_lo) / (y_hi - y_lo)
        b = frame.ball_coordinates
        if b is not None and b.x is not None:
            ball[i, 0] = (b.x - x_lo) / (x_hi - x_lo)
            ball[i, 1] = (b.y - y_lo) / (y_hi - y_lo)
        state = str(getattr(frame.ball_state, "value", frame.ball_state)).lower()
        if state == "dead":
            labels[i] = "D"
        elif state == "alive":
            owner = frame.ball_owning_team
            if owner is None:                      # keep the last team, and count it
                no_owner += 1
            else:
                last_team = "H" if owner == home_team else "A"
            labels[i] = last_team
        else:
            unknown_state += 1

    speed = _speed_from_positions(players[:, :, 0] * length, players[:, :, 1] * width, fps)
    players[:, :, 2] = speed

    live = np.isin(labels, ("H", "A", "D"))
    report = {
        "frames": int(n - 1), "fps": fps, "pitch_m": (round(float(length), 1), round(float(width), 1)),
        "players in roster": S,
        "mean players tracked": float(np.isfinite(players[live, :, 0]).sum(axis=1).mean()),
        "ball present %": 100.0 * float(np.isfinite(ball[live, 0]).mean()),
        "player x 0-1 p0.1/p99.9": tuple(np.round(np.nanpercentile(players[live, :, 0], (0.1, 99.9)), 3)),
        "labels H/A/D %": tuple(round(100.0 * float((labels[live] == c).mean()), 1) for c in "HAD"),
        "frames with unknown ball state": unknown_state,
        "live frames with no owning team": no_owner,
        "median speed m/s": float(np.nanmedian(speed[live])),
    }
    arrays = {"players": players, "ball": ball, "side": side, "phase": phase, "labels": labels,
              "pitch": np.array([length, width], dtype=np.float32)}
    return arrays, report


def path_for(match_id):
    return os.path.join(OUT_DIR, "{}.npz".format(match_id))


def load(match_id):
    with np.load(path_for(match_id), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def available():
    """Match ids already converted."""
    return [m for m in MATCH_IDS if os.path.isfile(path_for(m))]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="*", help="match ids (default: all)")
    parser.add_argument("--force", action="store_true", help="convert again even if present")
    args = parser.parse_args()
    from kloppy import sportec

    os.makedirs(OUT_DIR, exist_ok=True)
    for match_id in args.only or MATCH_IDS:
        if os.path.isfile(path_for(match_id)) and not args.force:
            print("{}  already converted".format(match_id))
            continue
        t0 = time.time()
        print("{}  downloading + parsing ...".format(match_id), flush=True)
        try:
            dataset = sportec.load_open_tracking_data(match_id=match_id, only_alive=False)
        except Exception as error:
            print("{}  SKIPPED: {}: {}".format(match_id, type(error).__name__, error))
            continue
        arrays, report = convert(dataset)
        del dataset
        np.savez_compressed(path_for(match_id), **arrays)
        print("{}  done in {:.0f} s".format(match_id, time.time() - t0))
        for key, value in report.items():
            print("      {:<34} {}".format(key, value))
    print("\nconverted: {}".format(", ".join(available()) or "none"))


if __name__ == "__main__":
    main()
