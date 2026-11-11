"""Download the IDSSE open matches and convert each to the same arrays BAR-ATH uses.

    python pipeline/idsse.py                 # every match not converted yet
    python pipeline/idsse.py --only J03WPY   # one match

IDSSE: 7 Bundesliga 2022/23 matches by the DFL, CC-BY 4.0 (Bassek et al., Scientific Data 2025,
doi 10.1038/s41597-025-04505-y; data on Hugging Face, pysport/idsse-data). Official tracking at
25 fps, with a per-frame ball status (alive / dead) and the team in possession: the same kind of
label as our ground truth. It is about other games, so the rules allow training on it.

Each match is two files on Hugging Face: a small match-information XML and a 350-420 MB tracking
XML (about 2.6 GB for all seven). They are downloaded once, directly, into
<DATA_DIR>/external/idsse/raw/, then parsed by kloppy (requirements-pipeline.txt) from the local
copies. kloppy's own downloader returned empty files, so it is not used. Converted arrays go to
<DATA_DIR>/external/idsse/<match>.npz. Everything stays in the data directory, outside the
repository. Delete raw/ to free the space once all matches are converted.

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
RAW_DIR = os.path.join(OUT_DIR, "raw")
# The seven matches and their competition, as listed in the Hugging Face dataset (checked
# against its file listing): 1st division = DFL-COM-000001, 2nd division = DFL-COM-000002.
MATCHES = {
    "J03WMX": "DFL-COM-000001", "J03WN1": "DFL-COM-000001",
    "J03WOH": "DFL-COM-000002", "J03WOY": "DFL-COM-000002", "J03WPY": "DFL-COM-000002",
    "J03WQQ": "DFL-COM-000002", "J03WR9": "DFL-COM-000002",
}
MATCH_IDS = tuple(MATCHES)
BASE_URL = "https://huggingface.co/datasets/pysport/idsse-data/resolve/main/"
META_FILE = "DFL_02_01_matchinformation_{comp}_DFL-MAT-{match}.xml"
TRACKING_FILE = "DFL_04_03_positions_raw_observed_{comp}_DFL-MAT-{match}.xml"
SPEED_SMOOTH_FRAMES = 5
USER_AGENT = "mtah-pipeline/1.0 (python urllib)"


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


def download(filename):
    """Fetch one dataset file into RAW_DIR, once. Follows Hugging Face's redirect, writes to a
    temporary name and renames at the end, so an interrupted download is never mistaken for a
    complete one. Returns the local path."""
    import urllib.request

    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, filename)
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return path
    request = urllib.request.Request(BASE_URL + filename, headers={"User-Agent": USER_AGENT})
    part = path + ".part"
    with urllib.request.urlopen(request, timeout=60) as response, open(part, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done, last = 0, -1
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total and done * 10 // total != last:
                last = done * 10 // total
                print("      {:>3}% of {:.0f} MB".format(last * 10, total / 1e6), flush=True)
    if total and os.path.getsize(part) != total:
        os.remove(part)
        raise IOError("{}: got {} of {} bytes".format(filename, done, total))
    if os.path.getsize(part) == 0:
        os.remove(part)
        raise IOError("{}: empty download".format(filename))
    os.replace(part, path)
    return path


def load_dataset(match_id):
    """kloppy TrackingDataset for one match, from the local copies of its two files."""
    from kloppy import sportec

    comp = MATCHES[match_id]
    meta = download(META_FILE.format(comp=comp, match=match_id))
    raw = download(TRACKING_FILE.format(comp=comp, match=match_id))
    return sportec.load_tracking(meta_data=meta, raw_data=raw, only_alive=False)


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

    os.makedirs(OUT_DIR, exist_ok=True)
    for match_id in args.only or MATCH_IDS:
        if match_id not in MATCHES:
            print("{}  SKIPPED: not one of {}".format(match_id, ", ".join(MATCH_IDS)))
            continue
        if os.path.isfile(path_for(match_id)) and not args.force:
            print("{}  already converted".format(match_id))
            continue
        t0 = time.time()
        print("{}  downloading (once) + parsing ...".format(match_id), flush=True)
        try:
            dataset = load_dataset(match_id)
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
