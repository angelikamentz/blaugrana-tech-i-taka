"""Shared loaders for the data-inspection tools: one parse of each file, cached, in plain arrays.

Used by inspection/inspect_data.py (the text report) and inspection/dashboard.py (the pictures).
Everything is read through games.asset_paths(), so MTAH_DATA_DIR decides where the data is, and
everything this module writes goes under the data directory too (``<DATA_DIR>/_inspection_cache``),
never into the repository: a parsed copy of the tracking is still the tracking.

Formats are taken from the starter code, which is the documented reference, and reused rather
than re-derived wherever the starter already has a function:

    baseline.team_of_each_track / team_names   slot -> home/away, club names (first <Team> = home)
    load_homography.load / project             per-frame 3x3 matrices, pitch (0-1) -> pixels
    load_smart_tagging.load                    tagged intervals, seconds -> frames

The one thing parsed here from scratch is the raw tracking text, because kloppy takes minutes and
gigabytes for it and the dashboard needs it as arrays anyway. That parser is defensive: it counts
every way a line can differ from the documented layout instead of assuming, and those counts are
what inspect_data.py prints as "format facts".

PRIVACY OF ERRORS. Parse errors never quote the offending text -- only line numbers and field
counts -- so a traceback can be shared without leaking a single coordinate.
"""

import collections
import difflib
import json
import os
import sys
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _path in (ROOT, os.path.join(ROOT, "starter")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import games  # noqa: E402
from games import GAME, GAMES, SOURCES, DEFAULT_SOURCE, asset_paths, half_video_blocks  # noqa: E402
import baseline as starter_baseline  # noqa: E402
import load_homography as starter_homography  # noqa: E402
import load_smart_tagging as starter_tagging  # noqa: E402

# Params
FPS = GAMES[GAME]["fps"]
PITCH_LENGTH, PITCH_WIDTH = GAMES[GAME]["pitch"]   # metres, measured (games.py)
CACHE_VERSION = 1
CACHE_DIR = os.path.join(games.DATA_DIR, "_inspection_cache")
SAMPLE_EVERY = 997          # lines between detailed format samples (prime, so no aliasing)
SIDES = ("H", "A")
HOME, AWAY, NOBODY = 0, 1, -1

# Match phases of the TACTICAL video, from the published graded halves (DATA.md).
PHASES = ("pre-match", "1st half", "halftime", "2nd half", "post-match")
PRE, FIRST, HALFTIME, SECOND, POST = range(5)
GRADED_PHASES = (FIRST, SECOND)


class FormatError(Exception):
    """A file does not look like its documented format. Messages never contain file content."""


# -- small helpers -----------------------------------------------------------------------------

def available_assets(source=DEFAULT_SOURCE):
    """{asset: (path, exists, size_bytes)} for every file games.py expects for this source."""
    out = {}
    for name, path in asset_paths(source=source).items():
        exists = os.path.isfile(path)
        out[name] = (path, exists, os.path.getsize(path) if exists else 0)
    return out


def has(source, *names):
    assets = available_assets(source)
    return all(assets.get(n, (None, False))[1] for n in names)


def n_frames(source=DEFAULT_SOURCE):
    return SOURCES[source]["n_video_frames"]


def phase_of_frames(n, source=DEFAULT_SOURCE):
    """int8 phase per video frame (PHASES). Broadcast has its own timeline: all -1 (unknown)."""
    if source != "tactical":
        return np.full(n, -1, dtype=np.int8)
    (_, a1, z1), (_, a2, z2) = half_video_blocks(GAME)
    out = np.full(n, PRE, dtype=np.int8)
    out[a1:z1 + 1] = FIRST
    out[z1 + 1:a2] = HALFTIME
    out[a2:z2 + 1] = SECOND
    out[z2 + 1:] = POST
    return out


def clock(frame, source=DEFAULT_SOURCE):
    """'1H 12:34' inside a graded half, otherwise 'video 01:02:03'."""
    if source == "tactical":
        for half, first, last in half_video_blocks(GAME):
            if first <= frame <= last:
                s = (frame - first) / float(FPS)
                return "{}H {:02d}:{:02d}".format(half, int(s // 60), int(s % 60))
    s = frame / float(FPS)
    return "video {:d}:{:02d}:{:02d}".format(int(s // 3600), int(s % 3600 // 60), int(s % 60))


def runs(mask):
    """[(start, stop)] half-open runs where a boolean array is True."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def per_second(values, how="mean"):
    """Aggregate a per-frame array into per-second bins (FPS frames). Returns (x_frames, y)."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values) // FPS * FPS
    blocks = values[:n].reshape(-1, FPS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if how == "mean":
            y = np.nanmean(blocks, axis=1)
        elif how == "max":
            y = np.nanmax(blocks, axis=1)
        elif how == "min":
            y = np.nanmin(blocks, axis=1)
        else:
            raise ValueError(how)
    return np.arange(0, n, FPS) + FPS // 2, y


def _file_key(path):
    st = os.stat(path)
    return "{}:{}:{}".format(CACHE_VERSION, st.st_size, int(st.st_mtime))


# -- tracking metadata -------------------------------------------------------------------------

def _local(tag):
    return tag.rsplit("}", 1)[-1]


def load_meta(source=DEFAULT_SOURCE):
    """Everything inspection needs from the ATD metadata XML.

    Returns a dict:
      teams        [(team_id, name)] in file order; the first is home (FIFA/EPTS, baseline.py)
      slot_side    int8 per slot, HOME/AWAY, from starter baseline.team_of_each_track
      slot_names   the Player <Name> per slot ("Track 1"...)
      paths        Counter of element paths (namespace-free), for the format-facts block
      attributes   {path: sorted attribute names}
      numbers      {path: value} for small numeric leaf elements (frame rate, field size, ...)
      slot_order   whether the raw-line player order in DataFormatSpecifications matches the
                   <Player> order baseline.py assumes (True / False / None when not stated)
    """
    path = asset_paths(source=source)["atd_meta"]
    root = ET.parse(path).getroot()

    paths = collections.Counter()
    attributes = collections.defaultdict(set)
    numbers = {}

    def walk(node, prefix):
        here = prefix + "/" + _local(node.tag)
        paths[here] += 1
        attributes[here].update(node.attrib.keys())
        text = (node.text or "").strip()
        if len(node) == 0 and text and len(text) < 20:
            try:
                numbers.setdefault(here, float(text))
            except ValueError:
                pass
        for child in node:
            walk(child, here)

    walk(root, "")

    teams = [(t.get("id"), t.findtext("Name")) for t in root.iter() if _local(t.tag) == "Team"]
    players = [p for p in root.iter() if _local(p.tag) == "Player"]
    slot_names = [p.findtext("Name") or "slot {}".format(i + 1) for i, p in enumerate(players)]
    slot_side = starter_baseline.team_of_each_track(path)

    # Order of players in the raw line, if the spec states it: PlayerChannelRef -> PlayerChannel
    # -> playerId. Compared with the <Player> order, which is what baseline.py relies on.
    channel_player = {c.get("id"): c.get("playerId") for c in root.iter()
                      if _local(c.tag) == "PlayerChannel"}
    ref_order = []
    for ref in root.iter():
        if _local(ref.tag) == "PlayerChannelRef":
            pid = channel_player.get(ref.get("playerChannelId"))
            if pid is not None and (not ref_order or ref_order[-1] != pid):
                ref_order.append(pid)
    player_ids = [p.get("id") for p in players]
    slot_order = None
    if ref_order:
        deduped = list(dict.fromkeys(ref_order))
        slot_order = deduped == player_ids

    return {
        "teams": teams,
        "slot_side": slot_side,
        "slot_names": slot_names,
        "n_slots": len(players),
        "paths": paths,
        "attributes": {k: sorted(v) for k, v in attributes.items()},
        "numbers": numbers,
        "slot_order": slot_order,
        "n_spec_player_refs": len(ref_order),
    }


def team_names(source=DEFAULT_SOURCE):
    return starter_baseline.team_names(asset_paths(source=source)["atd_meta"])


# -- raw tracking ------------------------------------------------------------------------------

def _parse_tracking(raw_path, n, n_slots):
    """Stream the raw ATD into arrays. Returns (players, ball, seen, stats).

    players (n, n_slots, 3) float32: x, y (0-1), speed  -- NaN where the slot is empty
    ball    (n, 3) float32: the ball group's first three fields (x, y, third field if any)
    seen    (n,) bool: a record exists for this video frame
    """
    players = np.full((n, n_slots, 3), np.nan, dtype=np.float32)
    ball = np.full((n, 3), np.nan, dtype=np.float32)
    seen = np.zeros(n, dtype=bool)

    stats = {
        "lines": 0, "blank_lines": 0, "parsed": 0, "duplicates": 0, "out_of_range": 0,
        "bad_frame_id": 0, "unparseable": 0, "unparseable_first_lines": [],
        "parts_per_line": collections.Counter(),
        "groups_per_line": collections.Counter(),
        "fields_per_player_group": collections.Counter(),    # sampled lines
        "fields_per_missing_player_group": collections.Counter(),  # sampled lines
        "fields_per_ball_group": collections.Counter(),
        "fields_per_missing_ball_group": collections.Counter(),
        "first_frame_id": None, "last_frame_id": None, "frame_ids_increasing": True,
    }
    last_id = None

    with open(raw_path, "r") as handle:
        for lineno, line in enumerate(handle, 1):
            stats["lines"] += 1
            line = line.strip()
            if not line:
                stats["blank_lines"] += 1
                continue
            parts = line.split(":")
            stats["parts_per_line"][len(parts)] += 1
            if len(parts) != 3:
                continue
            try:
                frame_id = int(parts[0])
            except ValueError:
                stats["bad_frame_id"] += 1
                continue
            if stats["first_frame_id"] is None:
                stats["first_frame_id"] = frame_id
            if last_id is not None and frame_id <= last_id:
                stats["frame_ids_increasing"] = False
            last_id = frame_id
            stats["last_frame_id"] = frame_id

            f = frame_id - 1
            if not 0 <= f < n:
                stats["out_of_range"] += 1
                continue
            if seen[f]:
                stats["duplicates"] += 1
                continue

            groups = parts[1].split(";")
            stats["groups_per_line"][len(groups)] += 1
            sample = lineno % SAMPLE_EVERY == 1

            try:
                # Fast path: every group is x,y,speed.
                flat = parts[1].replace(";", ",").split(",")
                if len(flat) == 3 * len(groups):
                    values = np.array(flat, dtype=np.float32).reshape(-1, 3)
                else:
                    # Slow path: some group is spelled differently (e.g. a bare "NaN").
                    values = np.full((len(groups), 3), np.nan, dtype=np.float32)
                    for i, group in enumerate(groups):
                        fields = group.split(",")
                        if group[:1] in ("N", "n") or len(fields) < 2:
                            continue
                        values[i, :min(3, len(fields))] = np.array(fields[:3], dtype=np.float32)
                k = min(len(groups), n_slots)
                players[f, :k] = values[:k]

                ball_fields = parts[2].split(",")
                stats["fields_per_ball_group"][len(ball_fields)] += 1
                if parts[2][:1] in ("N", "n"):
                    stats["fields_per_missing_ball_group"][len(ball_fields)] += 1
                else:
                    b = np.array(ball_fields[:3], dtype=np.float32)
                    ball[f, :len(b)] = b
            except (ValueError, IndexError):
                stats["unparseable"] += 1
                if len(stats["unparseable_first_lines"]) < 5:
                    stats["unparseable_first_lines"].append(lineno)
                continue

            if sample:
                for group in groups:
                    count = group.count(",") + 1
                    stats["fields_per_player_group"][count] += 1
                    if group[:1] in ("N", "n"):
                        stats["fields_per_missing_player_group"][count] += 1
            seen[f] = True
            stats["parsed"] += 1

    # JSON-friendly
    for key, value in list(stats.items()):
        if isinstance(value, collections.Counter):
            stats[key] = {str(k): v for k, v in sorted(value.items())}
    return players, ball, seen, stats


def load_tracking(source=DEFAULT_SOURCE, use_cache=True, verbose=False):
    """Parsed tracking for one source, from the cache when the raw file has not changed.

    Returns a dict: players (n, S, 3), ball (n, 3), seen (n,), stats, slot_side (S,), meta.
    """
    paths = asset_paths(source=source)
    raw_path = paths["atd_raw"]
    if not os.path.isfile(raw_path):
        raise FormatError("tracking file for '{}' not found (see DATA.md)".format(source))
    meta = load_meta(source)
    n = n_frames(source)
    key = _file_key(raw_path)
    cache = os.path.join(CACHE_DIR, "{}_tracking.npz".format(SOURCES[source]["stem"]))

    if use_cache and os.path.isfile(cache):
        try:
            with np.load(cache, allow_pickle=False) as z:
                if str(z["key"]) == key:
                    return {"players": z["players"], "ball": z["ball"], "seen": z["seen"],
                            "stats": json.loads(str(z["stats"])), "slot_side": meta["slot_side"],
                            "meta": meta}
        except Exception:
            pass  # stale or unreadable cache: rebuild below

    if verbose:
        print("parsing {} tracking (first run only, ~1-3 min) ...".format(source), flush=True)
    players, ball, seen, stats = _parse_tracking(raw_path, n, max(meta["n_slots"], 1))
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez(cache, players=players, ball=ball, seen=seen, stats=json.dumps(stats), key=key)
    except OSError as error:
        if verbose:
            print("  (could not write cache: {})".format(type(error).__name__))
    return {"players": players, "ball": ball, "seen": seen, "stats": stats,
            "slot_side": meta["slot_side"], "meta": meta}


def derived(tr, source=DEFAULT_SOURCE):
    """Per-frame signals used by both the report and the dashboard. All length n."""
    players, ball, side = tr["players"], tr["ball"], tr["slot_side"]
    n = len(players)
    present = ~np.isnan(players[:, :, 0])
    home = side == HOME
    away = side == AWAY
    ball_seen = ~np.isnan(ball[:, 0])

    x_m = players[:, :, 0] * PITCH_LENGTH
    y_m = players[:, :, 1] * PITCH_WIDTH
    speed = players[:, :, 2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        out = {
            "phase": phase_of_frames(n, source),
            "n_present": present.sum(axis=1),
            "n_home": present[:, home].sum(axis=1),
            "n_away": present[:, away].sum(axis=1),
            "ball_seen": ball_seen,
            "ball_x_m": ball[:, 0] * PITCH_LENGTH,
            "ball_y_m": ball[:, 1] * PITCH_WIDTH,
            "home_cx": np.nanmean(x_m[:, home], axis=1),
            "away_cx": np.nanmean(x_m[:, away], axis=1),
            "home_speed": np.nanmean(speed[:, home], axis=1),
            "away_speed": np.nanmean(speed[:, away], axis=1),
            "all_speed": np.nanmean(speed, axis=1),
        }
    out["nearest_side"] = nearest_side(tr)
    return out


def nearest_side(tr):
    """Per frame: side of the tracked player closest to the ball (in metres), NOBODY if no ball."""
    players, ball, side = tr["players"], tr["ball"], tr["slot_side"]
    out = np.full(len(players), NOBODY, dtype=np.int8)
    rows = np.flatnonzero(~np.isnan(ball[:, 0]))
    chunk = 20000
    for start in range(0, len(rows), chunk):
        r = rows[start:start + chunk]
        dx = (players[r, :, 0] - ball[r, None, 0]) * PITCH_LENGTH
        dy = (players[r, :, 1] - ball[r, None, 1]) * PITCH_WIDTH
        d = dx * dx + dy * dy
        d = np.where(np.isnan(d), np.inf, d)
        best = np.argmin(d, axis=1)
        ok = np.isfinite(d[np.arange(len(r)), best])
        out[r[ok]] = side[best[ok]]
    return out


def slot_segments(tr):
    """Per slot: number of contiguous presence runs and their lengths (frames)."""
    present = ~np.isnan(tr["players"][:, :, 0])
    return [[z - a for a, z in runs(present[:, s])] for s in range(present.shape[1])]


# -- homography --------------------------------------------------------------------------------

def load_homography(source=DEFAULT_SOURCE):
    """(matrices (n,3,3), reprojection_error (n,), valid (n,) bool, alignment report)."""
    matrices, error, valid = starter_homography.load(source=source)
    report = starter_homography.check_alignment(source=source)
    return matrices, error, valid, report


project = starter_homography.project


# -- smart tagging -----------------------------------------------------------------------------

def load_tagging():
    """Smart Tagging as a DataFrame: one row per instance, one column per tag group.

    Adds `side` (H / A / N/A / ?) by matching the Team tag to the club names in the tracking
    metadata, case-insensitively and fuzzily, because the two files spell the away club
    differently ("Atletic Club" in the tags). N/A stays N/A: it is not a team.
    """
    events = starter_tagging.load()
    rows = []
    for i, event in enumerate(events):
        row = {"id": i, "code": event["code"], "start_s": event["start_s"],
               "end_s": event["end_s"], "start_frame": event["start_frame"],
               "end_frame": event["end_frame"]}
        for group, value in event["tags"].items():
            row[group] = value
        rows.append(row)
    table = pd.DataFrame(rows)
    table["duration_s"] = table["end_s"] - table["start_s"]
    mapping = team_side_mapping(table["Team"].dropna().unique() if "Team" in table else [])
    table["side"] = table["Team"].map(mapping).fillna("?") if "Team" in table else "?"
    # The defending codes name the team WITHOUT the ball: the first inspection run showed each
    # one covering exactly the same frames as an attacking code (DEFENDING IN MIDDLE THIRD ==
    # PROGRESSION, DEFENDING IN ATTACKING THIRD == BUILD UP, ...) with the opposite Team tag.
    # `possession_side` is the side the tag implies has the ball.
    flip = table["code"].map(is_defending_code)
    opposite = {"H": "A", "A": "H"}
    table["possession_side"] = np.where(flip, table["side"].map(opposite).fillna(table["side"]),
                                        table["side"])
    return table, mapping


def is_defending_code(code):
    code = str(code).strip().upper()
    return code.startswith("DEFENDING") or code == "DEFENSIVE TRANSITION"


def team_side_mapping(values, source="tactical"):
    """{Team tag value: 'H' | 'A' | 'N/A' | '?'} using the metadata club names."""
    home, away = (name or "" for name in team_names(source))
    out = {}
    for value in values:
        v = str(value).strip()
        if v.lower() in ("n/a", "na", "none", ""):
            out[value] = "N/A"
            continue
        score_h = difflib.SequenceMatcher(None, v.lower(), home.lower()).ratio()
        score_a = difflib.SequenceMatcher(None, v.lower(), away.lower()).ratio()
        best = max(score_h, score_a)
        out[value] = "?" if best < 0.5 else ("H" if score_h >= score_a else "A")
    return out


def tagging_check_time_base():
    return starter_tagging.check_time_base(starter_tagging.load())


def event_profile(starts, signal, before_s=20, after_s=45):
    """Mean of a per-frame signal around a set of event frames (an event-triggered average).

    Returns (offsets in seconds, mean per offset). Events too close to either end are skipped.
    """
    signal = np.asarray(signal, dtype=np.float64)
    a, z = int(before_s * FPS), int(after_s * FPS)
    starts = [int(s) for s in starts if s - a >= 0 and s + z < len(signal)]
    if not starts:
        return np.array([]), np.array([])
    stack = np.stack([signal[s - a:s + z] for s in starts])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.arange(-a, z) / float(FPS), np.nanmean(stack, axis=0)


def tag_agreement(table, nearest):
    """Per code: how often the nearest tracked player to the ball is on the side that, by the
    tag, has the ball (`possession_side`: the Team tag, flipped for the defending codes).

    Frame-weighted over the frames inside each code's intervals that have a ball. Only intervals
    tagged H or A count towards the agreement; N/A intervals count towards the ball share only.
    """
    n = len(nearest)
    rows = []
    for code, group in table.groupby("code"):
        frames = with_ball = sided_ball = agree = 0
        for a, z, side in zip(group["start_frame"], group["end_frame"],
                              group["possession_side"]):
            a, z = max(0, int(a)), min(n, int(z))
            if z <= a:
                continue
            seg = nearest[a:z]
            has_ball = seg != NOBODY
            frames += z - a
            with_ball += int(has_ball.sum())
            if side in SIDES:
                want = HOME if side == "H" else AWAY
                sided_ball += int(has_ball.sum())
                agree += int((seg[has_ball] == want).sum())
        rows.append({
            "code": code,
            "Team tag names": "defenders" if is_defending_code(code) else "attackers",
            "instances": len(group),
            "sided instances": int(group["side"].isin(SIDES).sum()),
            "frames": frames,
            "ball visible %": round(100.0 * with_ball / frames, 1) if frames else None,
            "nearest player on possessing side %": round(100.0 * agree / sided_ball, 1)
            if sided_ball else None,
        })
    table = pd.DataFrame(rows, columns=["code", "Team tag names", "instances", "sided instances",
                                        "frames", "ball visible %",
                                        "nearest player on possessing side %"])
    return table.sort_values("instances", ascending=False)


# -- video -------------------------------------------------------------------------------------

def video_info(source=DEFAULT_SOURCE):
    """{frames, fps, width, height} from OpenCV, or None when OpenCV or the file is missing."""
    try:
        import cv2
    except ImportError:
        return None
    path = asset_paths(source=source)["video"]
    if not os.path.isfile(path):
        return None
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return None
        return {"frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "fps": float(cap.get(cv2.CAP_PROP_FPS)),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    finally:
        cap.release()
