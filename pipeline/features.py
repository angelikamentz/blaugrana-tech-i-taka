"""Stage 1 of the pipeline: one feature table, one row per video frame of the tactical recording.

    python pipeline/features.py             # build (or load from cache) and print a summary
    python pipeline/features.py --rebuild   # ignore the cache

Run it in your own terminal: it reads the match data. The summary prints aggregates only
(counts, percentages, medians), so it is safe to paste back to Claude. Never add a print of rows,
coordinates or per-frame values here.

What it does, in order:
  1. load the parsed tracking and the Smart Tagging through inspection/loaders.py (cached arrays)
  2. flip one half so the home team always attacks towards +x, and work in metres
  3. fill short ball gaps by straight lines; measure the long ones
  4. compute per-frame clues: ball out, ball speed, distance from the ball to each team, team
     speeds and centres
  5. turn the Smart Tagging intervals into per-frame columns

The table is saved under the data directory (<DATA_DIR>/cache/BAR-ATH/features.npz), never in the
repository: it is an extract of the data.

`compute()` takes arrays and a tag table and does no I/O, so it can be tested on synthetic input.
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _path in (ROOT, os.path.join(ROOT, "inspection"), HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import config as C  # noqa: E402
import games  # noqa: E402
import loaders as L  # noqa: E402

FPS = L.FPS
PITCH_LENGTH, PITCH_WIDTH = L.PITCH_LENGTH, L.PITCH_WIDTH
HOME, AWAY = L.HOME, L.AWAY
CACHE_PATH = os.path.join(games.DATA_DIR, "cache", games.GAME, "features.npz")

# Set-piece Type values (DATA.md), stored as a small integer per frame; -1 = no set piece.
SETPIECE_TYPES = ("Throw-in", "Corner Kick", "Goal Kick", "Free Kick", "Kick Off (After Goal)",
                  "Kick Off (Start)")


# -- small array helpers ------------------------------------------------------------------------

def moving_nanmean(values, window):
    """Centred moving mean that ignores NaN. NaN only where the whole window is NaN."""
    values = np.asarray(values, dtype=np.float64)
    if window <= 1:
        return values.astype(np.float32)
    ok = np.isfinite(values)
    filled = np.where(ok, values, 0.0)
    csum = np.concatenate(([0.0], np.cumsum(filled)))
    ccnt = np.concatenate(([0], np.cumsum(ok)))
    half = window // 2
    idx = np.arange(len(values))
    lo = np.clip(idx - half, 0, len(values))
    hi = np.clip(idx + window - half, 0, len(values))
    total = csum[hi] - csum[lo]
    count = ccnt[hi] - ccnt[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return out.astype(np.float32)


def seconds_since_last(mask):
    """Seconds since the most recent frame where mask is True (0 on such frames). NaN before the
    first one."""
    n = len(mask)
    idx = np.where(mask, np.arange(n), -1)
    last = np.maximum.accumulate(idx)
    out = (np.arange(n) - last) / float(FPS)
    out[last < 0] = np.nan
    return out.astype(np.float32)


def seconds_until_next(mask):
    """Seconds until the next frame where mask is True (0 on such frames). NaN after the last."""
    return seconds_since_last(np.asarray(mask)[::-1])[::-1].copy()


def interpolate_short_gaps(xy, max_frames):
    """Fill NaN runs of at most max_frames, with real values on both sides, by straight lines.

    Returns (filled copy, bool mask of the frames that were filled).
    """
    xy = xy.copy()
    missing = np.isnan(xy[:, 0])
    filled = np.zeros(len(xy), dtype=bool)
    for start, stop in L.runs(missing):
        length = stop - start
        if start == 0 or stop >= len(xy) or length > max_frames:
            continue
        a, b = xy[start - 1], xy[stop]
        t = (np.arange(1, length + 1) / float(length + 1))[:, None]
        xy[start:stop] = a + (b - a) * t
        filled[start:stop] = True
    return xy, filled


# -- the feature computation --------------------------------------------------------------------

def attack_flip(players_x_m, slot_side, phase):
    """Per frame: True where x must be mirrored so the home team attacks towards +x.

    Decided per half from the data: a team's average position lies in its own half, so if home
    sits at higher x than away, home defends the +x goal and that half gets flipped. Ungraded
    frames take the orientation of the nearest half (pre-match and halftime: 1st, post: 2nd).
    Returns (flip per frame, {half: flipped?}).
    """
    home = slot_side == HOME
    away = slot_side == AWAY
    decided = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for half, ph in ((1, L.FIRST), (2, L.SECOND)):
            rows = phase == ph
            hx = np.nanmean(players_x_m[rows][:, home])
            ax = np.nanmean(players_x_m[rows][:, away])
            decided[str(half)] = bool(hx > ax)   # str keys: the dict goes through JSON
    flip = np.where(np.isin(phase, (L.SECOND, L.POST)), decided["2"], decided["1"])
    return flip, decided


def compute(players, ball, slot_side, phase, tags, pitch=None):
    """Feature table from arrays.

    players   (n, S, 3) float: x, y (0-1), speed m/s; NaN where empty
    ball      (n, >=2) float: x, y (0-1); NaN where missing
    slot_side (S,) int: HOME / AWAY per slot
    phase     (n,) int: loaders.PRE / FIRST / HALFTIME / SECOND / POST
    tags      DataFrame with start_frame, end_frame (half-open), code, possession_side, Type
              (may be empty: other matches have no Smart Tagging)
    pitch     (length, width) in metres; BAR-ATH's measured size by default
    Returns (columns dict of length-n arrays, info dict of match-level facts).
    """
    PITCH_LENGTH, PITCH_WIDTH = pitch or (globals()["PITCH_LENGTH"], globals()["PITCH_WIDTH"])
    n = len(players)
    home = slot_side == HOME
    away = slot_side == AWAY
    out = {"frame": np.arange(n, dtype=np.int32), "phase": phase.astype(np.int8)}
    out["graded"] = np.isin(phase, L.GRADED_PHASES)
    out["half"] = np.select([phase == L.FIRST, phase == L.SECOND], [1, 2], 0).astype(np.int8)

    # Metres, with the home team attacking +x everywhere. Mirroring x and y together is a
    # rotation, so left and right stay consistent.
    px = players[:, :, 0].astype(np.float32) * PITCH_LENGTH
    py = players[:, :, 1].astype(np.float32) * PITCH_WIDTH
    flip, decided = attack_flip(px, slot_side, phase)
    px = np.where(flip[:, None], PITCH_LENGTH - px, px)
    py = np.where(flip[:, None], PITCH_WIDTH - py, py)
    speed = players[:, :, 2].astype(np.float32)
    present = np.isfinite(px)
    out["n_H"] = present[:, home].sum(axis=1).astype(np.int8)
    out["n_A"] = present[:, away].sum(axis=1).astype(np.int8)

    # Ball: metres, same flip, short gaps filled.
    bxy = np.stack([ball[:, 0] * PITCH_LENGTH, ball[:, 1] * PITCH_WIDTH], axis=1).astype(np.float32)
    bxy[:, 0] = np.where(flip, PITCH_LENGTH - bxy[:, 0], bxy[:, 0])
    bxy[:, 1] = np.where(flip, PITCH_WIDTH - bxy[:, 1], bxy[:, 1])
    seen = np.isfinite(bxy[:, 0])
    bxy, interp = interpolate_short_gaps(bxy, C.BALL_INTERP_MAX_FRAMES)
    have = np.isfinite(bxy[:, 0])
    out["ball_seen"] = seen
    out["ball_filled"] = have
    out["ball_interp"] = interp
    out["ball_x_m"], out["ball_y_m"] = bxy[:, 0], bxy[:, 1]

    since = seconds_since_last(seen)
    until = seconds_until_next(seen)
    out["ball_since_seen_s"] = since
    out["ball_until_seen_s"] = until
    out["ball_gap_len_s"] = np.where(seen, 0.0, since + until).astype(np.float32)

    m = C.BALL_OUT_MARGIN_M
    out["ball_out"] = have & ((bxy[:, 0] < -m) | (bxy[:, 0] > PITCH_LENGTH + m)
                              | (bxy[:, 1] < -m) | (bxy[:, 1] > PITCH_WIDTH + m))
    # Signed distance to the nearest line: positive inside the pitch, negative outside. Restarts
    # (throw-ins, corners, goal kicks) happen at or near a line.
    inside = np.minimum.reduce([bxy[:, 0], PITCH_LENGTH - bxy[:, 0],
                                bxy[:, 1], PITCH_WIDTH - bxy[:, 1]])
    out["ball_line_dist_m"] = inside.astype(np.float32)

    step = np.full(n, np.nan, dtype=np.float32)
    step[1:] = np.hypot(np.diff(bxy[:, 0]), np.diff(bxy[:, 1])) * FPS
    step[step > C.BALL_SPEED_MAX_MS] = np.nan
    out["ball_speed_ms"] = moving_nanmean(step, C.BALL_SPEED_SMOOTH_FRAMES)
    out["ball_speed_ms"][~have] = np.nan

    # Distance from the ball to each team, in chunks to keep memory flat.
    near = {side: np.full(n, np.nan, dtype=np.float32) for side in ("H", "A")}
    count = {side: np.zeros(n, dtype=np.int8) for side in ("H", "A")}
    rows = np.flatnonzero(have)
    for start in range(0, len(rows), 20000):
        r = rows[start:start + 20000]
        d = np.hypot(px[r] - bxy[r, None, 0], py[r] - bxy[r, None, 1])
        for side, mask in (("H", home), ("A", away)):
            ds = d[:, mask]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                near[side][r] = np.nanmin(ds, axis=1)
            count[side][r] = (ds < C.NEAR_BALL_RADIUS_M).sum(axis=1)
    out["near_H_m"], out["near_A_m"] = near["H"], near["A"]
    out["n_H_near"], out["n_A_near"] = count["H"], count["A"]
    # Everyone within a wider radius: a crowd round a still ball is a free kick or a corner.
    crowd = np.zeros(n, dtype=np.int8)
    for start in range(0, len(rows), 20000):
        r = rows[start:start + 20000]
        d = np.hypot(px[r] - bxy[r, None, 0], py[r] - bxy[r, None, 1])
        crowd[r] = (d < C.CROWD_RADIUS_M).sum(axis=1)
    out["n_crowd"] = crowd

    # Team movement and shape.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        raw_all = np.nanmean(speed, axis=1)
        raw_h = np.nanmean(speed[:, home], axis=1)
        raw_a = np.nanmean(speed[:, away], axis=1)
        out["H_cx_m"] = np.nanmean(px[:, home], axis=1).astype(np.float32)
        out["A_cx_m"] = np.nanmean(px[:, away], axis=1).astype(np.float32)
        # Share of tracked players walking or standing, and how spread out everyone is.
        slow = np.where(present, speed < C.SLOW_PLAYER_MS, False).sum(axis=1)
        raw_slow = np.where(present.any(axis=1), slow / np.maximum(present.sum(axis=1), 1),
                            np.nan)
        out["spread_x_m"] = np.nanstd(px, axis=1).astype(np.float32)
        out["spread_y_m"] = np.nanstd(py, axis=1).astype(np.float32)
    w = C.TEAM_SPEED_SMOOTH_FRAMES
    out["speed_all"] = moving_nanmean(raw_all, w)
    out["speed_H"] = moving_nanmean(raw_h, w)
    out["speed_A"] = moving_nanmean(raw_a, w)
    out["frac_slow"] = moving_nanmean(raw_slow, w)

    # Smart Tagging -> per-frame columns.
    sp_since = np.full(n, np.nan, dtype=np.float32)
    sp_type = np.full(n, -1, dtype=np.int8)
    votes = {"H": np.zeros(n, dtype=np.int8), "A": np.zeros(n, dtype=np.int8)}
    team_codes = {c.upper() for c in C.TAG_TEAM_CODES}
    for row in tags.itertuples(index=False):
        a, z = max(0, int(row.start_frame)), min(n, int(row.end_frame))
        if z <= a:
            continue
        code = str(row.code).strip().upper()
        if code == C.TAG_SETPIECE_CODE:
            s = (np.arange(a, z) - a) / float(FPS)
            newer = ~(sp_since[a:z] <= s)          # NaN or an older start -> take this one
            sp_since[a:z] = np.where(newer, s, sp_since[a:z])
            kind = str(getattr(row, "Type", "") or "").strip().lower()
            types = [t.lower() for t in SETPIECE_TYPES]
            if kind in types:
                sp_type[a:z] = np.where(newer, types.index(kind), sp_type[a:z])
        if code in team_codes and row.possession_side in votes:
            votes[row.possession_side][a:z] += 1
    out["tag_setpiece_since_s"] = sp_since
    out["tag_setpiece_type"] = sp_type
    out["tag_H_votes"], out["tag_A_votes"] = votes["H"], votes["A"]

    info = {"flipped_half": decided, "n_frames": n}
    return out, info


def empty_tags():
    """A tag table with no rows, for matches that have no Smart Tagging."""
    import pandas as pd
    return pd.DataFrame(columns=["start_frame", "end_frame", "code", "possession_side", "Type"])


# -- cache --------------------------------------------------------------------------------------

def _key():
    raw = games.asset_paths()["atd_raw"]
    tag = games.asset_paths()["smart_tagging"]
    return "{}|{}|{}".format(C.FEATURE_VERSION, L._file_key(raw), L._file_key(tag))


def build(use_cache=True, verbose=True):
    """Feature table for BAR-ATH, from the cache when nothing has changed. Returns (cols, info)."""
    key = _key()
    if use_cache and os.path.isfile(CACHE_PATH):
        try:
            with np.load(CACHE_PATH, allow_pickle=False) as z:
                if str(z["__key"]) == key:
                    cols = {k: z[k] for k in z.files if not k.startswith("__")}
                    return cols, json.loads(str(z["__info"]))
        except Exception:
            pass  # stale or unreadable: rebuild
    if verbose:
        print("building features ...", flush=True)
    tr = L.load_tracking(use_cache=True, verbose=verbose)
    tags, _ = L.load_tagging()
    phase = L.phase_of_frames(len(tr["players"]))
    cols, info = compute(tr["players"], tr["ball"], tr["slot_side"], phase, tags)
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    np.savez_compressed(CACHE_PATH, __key=key, __info=json.dumps(info), **cols)
    return cols, info


# -- summary (aggregates only) ------------------------------------------------------------------

def _pct(mask):
    mask = np.asarray(mask, dtype=bool)
    return 100.0 * mask.mean() if len(mask) else float("nan")


def _q(values, qs=(10, 50, 90)):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return "no finite values"
    return " / ".join("{:.2f}".format(v) for v in np.percentile(values, qs))


def summary(cols, info):
    """Print aggregates over the graded frames. Never rows or per-frame values."""
    g = cols["graded"]
    line = "  {:<44} {}"
    print("=" * 90)
    print("FEATURES -- summary over graded frames (aggregates only)")
    print("=" * 90)
    print(line.format("frames total / graded", "{} / {}".format(len(g), int(g.sum()))))
    print(line.format("columns", len(cols)))
    print(line.format("half flipped so home attacks +x (1st, 2nd)",
                      "{}, {}".format(info["flipped_half"]["1"], info["flipped_half"]["2"])))

    print("\n## ball")
    print(line.format("seen % (raw)", "{:.2f}".format(_pct(cols["ball_seen"][g]))))
    print(line.format("seen or filled % (after short-gap fill)",
                      "{:.2f}".format(_pct(cols["ball_filled"][g]))))
    print(line.format("frames in gaps > 2 s %",
                      "{:.2f}".format(_pct(cols["ball_gap_len_s"][g] > 2))))
    print(line.format("gap length (s) where missing: p10/p50/p90",
                      _q(cols["ball_gap_len_s"][g & ~cols["ball_seen"]])))
    have = g & cols["ball_filled"]
    print(line.format("ball out % (of frames with a ball)", "{:.2f}".format(_pct(cols["ball_out"][have]))))
    print(line.format("ball speed m/s: p10/p50/p90", _q(cols["ball_speed_ms"][have])))
    print(line.format("ball x m (home attacks +x): p10/p50/p90", _q(cols["ball_x_m"][have])))

    print("\n## players")
    print(line.format("players present H: p10/p50/p90", _q(cols["n_H"][g])))
    print(line.format("players present A: p10/p50/p90", _q(cols["n_A"][g])))
    print(line.format("nearest H to ball m: p10/p50/p90", _q(cols["near_H_m"][have])))
    print(line.format("nearest A to ball m: p10/p50/p90", _q(cols["near_A_m"][have])))
    closer_h = cols["near_H_m"][have] < cols["near_A_m"][have]
    print(line.format("H nearer than A % (ball frames)", "{:.2f}".format(_pct(closer_h))))
    print(line.format("players within 5 m H / A: mean",
                      "{:.2f} / {:.2f}".format(cols["n_H_near"][have].mean(),
                                               cols["n_A_near"][have].mean())))
    print(line.format("team speed all m/s: p10/p50/p90", _q(cols["speed_all"][g])))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for half in (1, 2):
            rows = cols["half"] == half
            print(line.format("mean centre x m, half {}: H / A".format(half),
                              "{:.1f} / {:.1f}".format(np.nanmean(cols["H_cx_m"][rows]),
                                                       np.nanmean(cols["A_cx_m"][rows]))))

    print("\n## Smart Tagging")
    sp = cols["tag_setpiece_since_s"]
    in_sp = g & np.isfinite(sp)
    print(line.format("graded frames inside a set-piece tag %", "{:.2f}".format(_pct(in_sp[g]))))
    print(line.format("team speed m/s, set piece first 10 s / rest",
                      "{} | {}".format(_q(cols["speed_all"][in_sp & (sp < 10)], (50,)),
                                       _q(cols["speed_all"][g & ~(in_sp & (sp < 10))], (50,)))))
    print(line.format("ball seen %, set piece first 10 s / rest",
                      "{:.1f} / {:.1f}".format(_pct(cols["ball_seen"][in_sp & (sp < 10)]),
                                               _pct(cols["ball_seen"][g & ~(in_sp & (sp < 10))]))))
    types = cols["tag_setpiece_type"][in_sp]
    counts = {SETPIECE_TYPES[t]: int((types == t).sum()) for t in range(len(SETPIECE_TYPES))}
    counts["(no type)"] = int((types < 0).sum())
    print(line.format("set-piece frames by type", ", ".join(
        "{}: {}".format(k, v) for k, v in counts.items())))
    vh, va = cols["tag_H_votes"][g], cols["tag_A_votes"][g]
    any_hint = (vh + va) > 0
    print(line.format("graded frames with a team hint %", "{:.2f}".format(_pct(any_hint))))
    print(line.format("of those: H only / A only / both %", "{:.1f} / {:.1f} / {:.1f}".format(
        _pct(((vh > 0) & (va == 0))[any_hint]), _pct(((va > 0) & (vh == 0))[any_hint]),
        _pct(((vh > 0) & (va > 0))[any_hint]))))
    with np.errstate(invalid="ignore"):
        nearer_h = (cols["near_H_m"] < cols["near_A_m"])[g]
    usable = (vh != va) & cols["ball_filled"][g]
    agree = ((vh > va) == nearer_h)[usable]
    print(line.format("hint agrees with nearer team % (ball frames)", "{:.1f}".format(_pct(agree))))

    print("\n## missing values (graded frames, % NaN)")
    for name in ("ball_x_m", "ball_speed_ms", "near_H_m", "near_A_m", "H_cx_m", "A_cx_m",
                 "speed_all", "ball_since_seen_s", "ball_until_seen_s"):
        print(line.format(name, "{:.2f}".format(_pct(~np.isfinite(cols[name][g])))))
    print("\ncache: <DATA_DIR>/cache/{}/features.npz".format(games.GAME))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rebuild", action="store_true", help="ignore the cache")
    args = parser.parse_args()
    cols, info = build(use_cache=not args.rebuild)
    summary(cols, info)


if __name__ == "__main__":
    main()
