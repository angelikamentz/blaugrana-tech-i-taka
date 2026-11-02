"""Inspect every provided file and print a report of aggregates: what is there, what it looks like,
and whether it matches what DATA.md says it should be.

    pip install -r requirements-inspection.txt
    python inspection/inspect_data.py                    # tactical (the reference recording)
    python inspection/inspect_data.py --source broadcast
    python inspection/inspect_data.py --no-cache         # re-parse the tracking from scratch

Run it in your own terminal, not inside Claude Code (games.asset_paths() refuses to run there).

What it prints is safe to share: counts, percentages, medians, category names, booleans. No rows,
no coordinates, no frame-level values. That is deliberate -- the output is meant to be pasted back
into a conversation with an LLM that may not see the data itself. Keep it that way if you edit it.

Sections
--------
  files         which of the expected files exist, and their sizes
  metadata      teams, slots per team, the XML's structure (element paths, no values)
  tracking      format facts (how lines and groups are spelled), completeness, ball visibility,
                tracked objects per phase, coordinate ranges, speed units, direction of play,
                ball gaps, slot lifetimes
  homography    validity, reprojection error per phase, whether projections land in the image
  smart tagging codes, tag groups, durations, overlaps, time base, team -> side mapping
  cross-checks  tags vs tracking: does the tagged team usually have the nearest player to the ball,
                and do set pieces look like dead balls (slow players)?
  video         frame count, fps and size from the file header (needs OpenCV)

Each check with a published value shows `expected` and OK / DIFF. A DIFF is a question to answer,
not necessarily a bug -- some published numbers are rounded.

A JSON copy of the report goes to <DATA_DIR>/_inspection_cache/, next to the parsed-tracking cache.
"""

import argparse
import collections
import json
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import loaders as L  # noqa: E402

# Params
# Published numbers (DATA.md, __training_pipeline.md). Tolerances are for rounded figures.
EXPECTED = {
    "tactical": {
        "frames": 199319, "slots": 46, "ball_frames": 113488, "ball_pct": 56.9,
        "ball_graded_frames": 94214, "ball_graded_pct": 59.3, "objects_graded_mean": 20.5,
        "homography_valid_pct": 100.0, "reproj_median_play": 0.24, "reproj_median_halftime": 1.5,
        "tags": 638, "team_na": 36, "video_fps": 25, "video_size": (1920, 1080),
        "codes": {
            "PLAYERS IN THE BOX": 97, "BALL IN FINAL THIRD": 93, "BALL IN THE BOX": 72,
            "PROGRESSION": 66, "DEFENDING IN MIDDLE THIRD": 66, "SET PIECES": 59,
            "CREATING CHANCES": 30, "DEFENDING IN DEFENSIVE THIRD": 30, "LONG BALL": 29,
            "ATTACKING TRANSITION": 24, "DEFENSIVE TRANSITION": 24, "BUILD UP": 22,
            "DEFENDING IN ATTACKING THIRD": 22, "GOALS": 4,
        },
        "types": {"Throw-in": 26, "Corner Kick": 12, "Goal Kick": 10, "Free Kick": 5,
                  "Kick Off (After Goal)": 4, "Kick Off (Start)": 2},
    },
    "broadcast": {
        "frames": 164488, "slots": 51, "ball_pct": 50.5, "homography_valid_pct": 65.8,
        "video_size": (1920, 1080),
    },
}
OUT_OF_PITCH_MARGIN = 0.02      # 0-1 units beyond the touchline before a position counts as "out"
SPEED_SAMPLE_STEP = 10          # every n-th frame for the speed-unit check
LONG_GAP_S = 2.0


# -- report plumbing ---------------------------------------------------------------------------

class Report:
    def __init__(self, source):
        self.source = source
        self.sections = []

    def section(self, title):
        self.sections.append({"title": title, "rows": [], "notes": []})

    def row(self, check, got, expected=None, tol=None, rel=None):
        """One line. `tol` absolute or `rel` relative tolerance for numeric comparisons."""
        ok = None
        if expected is not None:
            if isinstance(expected, (int, float)) and isinstance(got, (int, float, np.number)):
                if rel is not None:
                    ok = abs(float(got) - expected) <= rel * abs(expected)
                else:
                    ok = abs(float(got) - expected) <= (tol or 0)
            else:
                ok = got == expected
        self.sections[-1]["rows"].append({"check": check, "got": _plain(got),
                                          "expected": _plain(expected), "ok": ok})

    def note(self, text):
        self.sections[-1]["notes"].append(text)

    def print(self):
        print("=" * 90)
        print("DATA INSPECTION -- source: {}".format(self.source))
        print("=" * 90)
        for section in self.sections:
            print("\n## {}".format(section["title"]))
            for r in section["rows"]:
                mark = "" if r["ok"] is None else ("  OK" if r["ok"] else "  DIFF")
                exp = "" if r["expected"] is None else "   (expected {})".format(_fmt(r["expected"]))
                print("  {:<52} {}{}{}".format(r["check"], _fmt(r["got"]), exp, mark))
            for text in section["notes"]:
                print("  - {}".format(text))
        diffs = [(s["title"], r["check"]) for s in self.sections for r in s["rows"]
                 if r["ok"] is False]
        print("\n## Summary")
        print("  {} checks against published numbers, {} differ{}".format(
            sum(r["ok"] is not None for s in self.sections for r in s["rows"]), len(diffs),
            ":" if diffs else ""))
        for title, check in diffs:
            print("    {} / {}".format(title, check))

    def to_json(self):
        return {"source": self.source, "sections": self.sections}


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    return value


def _fmt(value):
    if isinstance(value, float):
        return "{:.4g}".format(value)
    if isinstance(value, dict):
        return ", ".join("{}: {}".format(k, _fmt(v)) for k, v in value.items()) or "{}"
    if isinstance(value, list):
        return "[" + ", ".join(_fmt(v) for v in value) + "]"
    return str(value)


def pct(a, b):
    return round(100.0 * a / b, 2) if b else float("nan")


def q(values, p):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return round(float(np.percentile(values, p)), 4) if len(values) else float("nan")


# -- sections ----------------------------------------------------------------------------------

def files_section(rep, source):
    rep.section("files")
    for name, (path, exists, size) in L.available_assets(source).items():
        rep.row("{:<14} {}".format(name, os.path.basename(path)),
                "{:.1f} MB".format(size / 1e6) if exists else "MISSING")


def metadata_section(rep, source, exp):
    rep.section("metadata (tracking XML)")
    meta = L.load_meta(source)
    rep.row("teams", len(meta["teams"]), 2)
    home, away = L.team_names(source)
    rep.row("home (H) = first <Team>", home)
    rep.row("away (A)", away)
    rep.row("track slots", meta["n_slots"], exp.get("slots"))
    rep.row("slots per side", {"H": int((meta["slot_side"] == L.HOME).sum()),
                               "A": int((meta["slot_side"] == L.AWAY).sum())})
    rep.row("raw player order stated in spec", meta["n_spec_player_refs"] > 0)
    rep.row("spec order == <Player> order (baseline assumes)", meta["slot_order"],
            True if meta["slot_order"] is not None else None)
    interesting = {k: v for k, v in meta["numbers"].items()
                   if any(w in k.lower() for w in ("rate", "size", "width", "height", "length",
                                                   "fps"))}
    rep.row("numeric settings", interesting)
    rep.note("format facts -- element paths (count) [attribute names]:")
    for path, count in sorted(meta["paths"].items()):
        attrs = meta["attributes"].get(path) or []
        rep.note("  {} ({}){}".format(path, count, " [{}]".format(", ".join(attrs)) if attrs else ""))
    return meta


def tracking_section(rep, source, exp, use_cache):
    rep.section("tracking (raw ATD)")
    tr = L.load_tracking(source, use_cache=use_cache, verbose=True)
    st = tr["stats"]
    n = L.n_frames(source)
    seen = tr["seen"]

    rep.row("lines in file", st["lines"], exp.get("frames"))
    rep.row("records parsed", st["parsed"], exp.get("frames"))
    rep.row("blank / bad frame id / unparseable lines",
            [st["blank_lines"], st["bad_frame_id"], st["unparseable"]], [0, 0, 0])
    if st["unparseable_first_lines"]:
        rep.row("first unparseable line numbers", st["unparseable_first_lines"])
    rep.row("duplicate / out-of-range frame ids", [st["duplicates"], st["out_of_range"]], [0, 0])
    rep.row("frame_id first, last", [st["first_frame_id"], st["last_frame_id"]], [1, n])
    rep.row("frame ids strictly increasing", st["frame_ids_increasing"], True)
    rep.row("every video frame has a record", bool(seen.all()), True)

    rep.note("format facts:")
    rep.note("  ':'-separated parts per line: {}".format(_fmt(st["parts_per_line"])))
    rep.note("  ';'-separated player groups per line: {}".format(_fmt(st["groups_per_line"])))
    rep.note("  fields per player group (sampled): {}".format(_fmt(st["fields_per_player_group"])))
    rep.note("  fields per MISSING player group (sampled): {}".format(
        _fmt(st["fields_per_missing_player_group"])))
    rep.note("  fields per ball group: {}".format(_fmt(st["fields_per_ball_group"])))
    rep.note("  fields per MISSING ball group: {}".format(_fmt(st["fields_per_missing_ball_group"])))

    d = L.derived(tr, source)
    phase = d["phase"]
    graded = np.isin(phase, L.GRADED_PHASES)
    ball = d["ball_seen"]

    rep.row("ball visible, all frames", int(ball.sum()), exp.get("ball_frames"))
    rep.row("ball visible, all frames %", pct(ball.sum(), n), exp.get("ball_pct"), tol=0.15)
    if source == "tactical":
        rep.row("ball visible, graded frames", int(ball[graded].sum()), exp.get("ball_graded_frames"))
        rep.row("ball visible, graded frames %", pct(ball[graded].sum(), graded.sum()),
                exp.get("ball_graded_pct"), tol=0.15)
        rep.row("mean tracked objects, graded", round(float(d["n_present"][graded].mean()), 2),
                exp.get("objects_graded_mean"), tol=0.1)
        rep.row("graded frames with > 22 objects %", pct((d["n_present"][graded] > 22).sum(),
                                                         graded.sum()))
        rep.note("per phase -- frames | ball visible % | mean objects (H / A):")
        for p, name in enumerate(L.PHASES):
            m = phase == p
            if m.any():
                rep.note("  {:<11} {:>7} | {:5.1f}% | {:5.2f} ({:5.2f} / {:5.2f})".format(
                    name, int(m.sum()), pct(ball[m].sum(), m.sum()), d["n_present"][m].mean(),
                    d["n_home"][m].mean(), d["n_away"][m].mean()))
    rep.row("max tracked objects in a frame", int(d["n_present"].max()))

    # Coordinates
    players = tr["players"]
    px, py = players[:, :, 0], players[:, :, 1]
    finite = np.isfinite(px)
    lo, hi = -OUT_OF_PITCH_MARGIN, 1 + OUT_OF_PITCH_MARGIN
    out = finite & ((px < lo) | (px > hi) | (py < lo) | (py > hi))
    rep.row("player x range (0-1 units, p0.1 .. p99.9)", [q(px[finite], 0.1), q(px[finite], 99.9)])
    rep.row("player y range (0-1 units, p0.1 .. p99.9)", [q(py[finite], 0.1), q(py[finite], 99.9)])
    rep.row("player positions beyond the pitch +2% %", pct(out.sum(), finite.sum()))
    bx, by = tr["ball"][:, 0], tr["ball"][:, 1]
    bf = np.isfinite(bx)
    rep.row("ball positions beyond the pitch +2% %",
            pct((bf & ((bx < lo) | (bx > hi) | (by < lo) | (by > hi))).sum(), bf.sum()))
    third = tr["ball"][:, 2]
    rep.row("ball 3rd field: finite share when ball seen %", pct(np.isfinite(third[bf]).sum(), bf.sum()))
    if np.isfinite(third[bf]).any():
        rep.row("ball 3rd field: p1 / median / p99", [q(third, 1), q(third, 50), q(third, 99)])
        rep.note("ball 3rd field: near 0 with occasional metres = height; m/s-like = speed")

    # Speed units: provided vs computed from positions
    speed = players[:, :, 2]
    rep.row("provided player speed: median / p99 / max",
            [q(speed, 50), q(speed, 99), q(speed, 100)])
    idx = np.arange(0, n - 1, SPEED_SAMPLE_STEP)
    dx = (players[idx + 1, :, 0] - players[idx, :, 0]) * L.PITCH_LENGTH
    dy = (players[idx + 1, :, 1] - players[idx, :, 1]) * L.PITCH_WIDTH
    computed = np.sqrt(dx * dx + dy * dy) * L.FPS
    provided = speed[idx]
    m = np.isfinite(computed) & np.isfinite(provided) & (computed > 0.5) & (computed < 12)
    if m.any():
        rep.row("computed speed from positions: median (m/s)", q(computed[m], 50))
        rep.row("provided / computed speed: median ratio", q(provided[m] / computed[m], 50), 1.0,
                tol=0.15)
        rep.note("ratio ~1 -> speed is m/s at 25 fps; ~3.6 -> km/h; far off -> check the units")
    rep.row("position jumps > 12 m/s between frames (sampled) %",
            pct((np.isfinite(computed) & (computed > 12)).sum(), np.isfinite(computed).sum()))

    # Direction of play
    if source == "tactical":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            h1 = float(np.nanmean(d["home_cx"][phase == L.FIRST]))
            a1 = float(np.nanmean(d["away_cx"][phase == L.FIRST]))
            h2 = float(np.nanmean(d["home_cx"][phase == L.SECOND]))
            a2 = float(np.nanmean(d["away_cx"][phase == L.SECOND]))
        rep.row("mean x (m) 1st half: home / away", [round(h1, 1), round(a1, 1)])
        rep.row("mean x (m) 2nd half: home / away", [round(h2, 1), round(a2, 1)])
        rep.row("teams switch ends at halftime in the raw coords", (h1 < a1) != (h2 < a2))
        rep.note("if True, coordinates are NOT normalised to a fixed attacking direction")
        rep.note("y orientation (is y=0 the near or far touchline?) can't be read from the "
                 "numbers -- check it on the dashboard's video overlay")

    # Ball gaps on graded frames
    if source == "tactical":
        gaps = [z - a for a, z in L.runs(~ball & graded)]
        if gaps:
            g = np.array(gaps) / float(L.FPS)
            rep.row("ball gaps in graded frames: count", len(gaps))
            rep.row("ball gap length (s): median / p90 / max", [q(g, 50), q(g, 90), q(g, 100)])
            rep.row("graded frames in gaps > {:.0f} s %".format(LONG_GAP_S),
                    pct(sum(x for x in gaps if x / L.FPS > LONG_GAP_S), graded.sum()))

    # Slot lifetimes
    segs = L.slot_segments(tr)
    used = [s for s in segs if s]
    lengths = np.array([x for s in used for x in s], dtype=float) / L.FPS
    rep.row("slots ever used / total", [len(used), len(segs)])
    if len(lengths):
        rep.row("presence runs per used slot: median / max",
                [int(np.median([len(s) for s in used])), max(len(s) for s in used)])
        rep.row("presence run length (s): median / p90 / max",
                [q(lengths, 50), q(lengths, 90), q(lengths, 100)])

    # Nearest player to the ball
    ns = d["nearest_side"]
    m = graded if source == "tactical" else np.ones(n, dtype=bool)
    m = m & (ns != L.NOBODY)
    rep.row("nearest player to ball is H / A (% of ball frames)",
            [pct((ns[m] == L.HOME).sum(), m.sum()), pct((ns[m] == L.AWAY).sum(), m.sum())])
    return tr, d


def homography_section(rep, source, exp, phase):
    rep.section("homography")
    matrices, error, valid, report = L.load_homography(source)
    rep.row("rows", report["n_rows"], exp.get("frames"))
    rep.row("one row per frame from 0", report["agrees"], True)
    rep.row("valid %", pct(valid.sum(), len(valid)), exp.get("homography_valid_pct"), tol=0.1)
    finite = np.isfinite(matrices.reshape(len(matrices), -1)).all(axis=1)
    rep.row("matrix finite wherever valid", bool(finite[valid].all()), True)
    rep.row("matrix NaN wherever invalid", bool((~finite[~valid]).all()) if (~valid).any() else True,
            True)
    rep.row("reprojection error (valid): median / p99 / max",
            [q(error[valid], 50), q(error[valid], 99), q(error[valid], 100)])
    if source == "tactical" and phase is not None:
        play = np.isin(phase, L.GRADED_PHASES) & valid
        rep.row("reprojection error median, graded halves", q(error[play], 50),
                exp.get("reproj_median_play"), rel=0.25)
        rep.row("reprojection error median, halftime", q(error[(phase == L.HALFTIME) & valid], 50),
                exp.get("reproj_median_halftime"), rel=0.35)
        rep.note("first run: halftime median 0.39, not DATA.md's ~1.5 -- the published figure "
                 "is not reproduced over frames 90652-110807; see the Timeline tab")

    width, height = exp.get("video_size", (1920, 1080))
    sample = np.flatnonzero(valid)[::250]
    if len(sample):
        inside = 0
        sane = 0
        for f in sample:
            cx, cy = L.project(matrices[f], 0.5, 0.5)
            if 0 <= cx <= width and 0 <= cy <= height:
                inside += 1
            xs, ys = L.project(matrices[f], np.array([0., 1., 1., 0.]), np.array([0., 0., 1., 1.]))
            if np.all(np.isfinite(xs)) and np.all(np.isfinite(ys)):
                sane += 1
        rep.row("pitch centre projects inside the image % (sampled)", pct(inside, len(sample)))
        rep.row("pitch corners project to finite pixels % (sampled)", pct(sane, len(sample)), 100.0,
                tol=0.5)
        rep.note("image size assumed {}x{} (DATA.md); a low 'inside' share during play means the "
                 "matrix direction or the coordinate convention is not what we think".format(
                     width, height))


def tagging_section(rep, exp):
    rep.section("smart tagging")
    table, mapping = L.load_tagging()
    rep.row("instances", len(table), exp.get("tags"))
    counts = table["code"].value_counts().to_dict()
    for code, expected in exp.get("codes", {}).items():
        rep.row("code {}".format(code), int(counts.get(code, 0)), expected)
    extra = sorted(set(counts) - set(exp.get("codes", {})))
    if extra:
        rep.row("codes not in DATA.md", extra)
    groups = [c for c in table.columns if c not in ("id", "code", "start_s", "end_s", "start_frame",
                                                    "end_frame", "duration_s", "side")]
    rep.row("tag groups", groups)
    for group in groups:
        vc = table[group].value_counts(dropna=True)
        shown = dict(list(vc.items())[:12])
        rep.row("  {} values ({} distinct)".format(group, len(vc)), shown)
    if "Team" in table:
        rep.row("Team = N/A", int((table["side"] == "N/A").sum()), exp.get("team_na"))
    if "Type" in table:
        tc = table["Type"].value_counts().to_dict()
        for name, expected in exp.get("types", {}).items():
            rep.row("Type {}".format(name), int(tc.get(name, 0)), expected)
    rep.row("Team tag -> side mapping", mapping)
    rep.row("intervals with end < start", int((table["end_s"] < table["start_s"]).sum()), 0)
    n = L.n_frames("tactical")
    rep.row("intervals outside the video", int(((table["start_frame"] < 0)
                                                 | (table["end_frame"] > n)).sum()), 0)
    rep.row("duration (s): median / p90 / max",
            [q(table["duration_s"], 50), q(table["duration_s"], 90), q(table["duration_s"], 100)])
    rep.note("median duration (s) per code:")
    for code, dur in table.groupby("code")["duration_s"].median().sort_values().items():
        rep.note("  {:<32} {:6.1f}".format(code, dur))

    cover = np.zeros(n + 1, dtype=np.int32)
    for a, z in zip(table["start_frame"].clip(0, n), table["end_frame"].clip(0, n)):
        cover[a] += 1
        cover[z] -= 1
    concurrent = np.cumsum(cover)[:n]
    phase = L.phase_of_frames(n, "tactical")
    graded = np.isin(phase, L.GRADED_PHASES)
    rep.row("graded frames covered by >= 1 tag %", pct((concurrent[graded] > 0).sum(), graded.sum()))
    rep.row("concurrent tags on covered graded frames: mean / max",
            [round(float(concurrent[graded & (concurrent > 0)].mean()), 2)
             if (graded & (concurrent > 0)).any() else 0, int(concurrent.max())])
    tb = L.tagging_check_time_base()
    rep.row("time base: tagged kickoffs contain real ones", tb["agrees"], True)
    return table


def cross_section(rep, table, d):
    rep.section("cross-checks: tags vs tracking")
    n = len(d["phase"])
    graded = np.isin(d["phase"], L.GRADED_PHASES)
    rep.note("per code, over the frames inside its tagged intervals. Possessing side = the Team "
             "tag, flipped for the defending codes:")
    rep.note("  code | Team tag names | instances (H/A-tagged) | frames | ball visible % | "
             "nearest player on possessing side %")
    for _, r in L.tag_agreement(table, d["nearest_side"]).iterrows():
        agree = r["nearest player on possessing side %"]
        rep.note("  {:<32} {:<9} {:>4} ({:>4}) | {:>7} | {:>5} | {}".format(
            r["code"], r["Team tag names"], r["instances"], r["sided instances"], r["frames"],
            r["ball visible %"], "  -  " if agree is None or agree != agree else agree))
    rep.note("high = the tag is a usable possession prior for that code; ~50% = it says little. "
             "The nearest player is itself a noisy proxy, so compare codes rather than read "
             "absolute values")

    # Where in a set-piece interval is the restart? Event-triggered averages around its start.
    sp = table[table["code"].str.upper() == "SET PIECES"]
    starts = [s for s in sp["start_frame"] if graded[min(max(int(s), 0), n - 1)]]
    if starts:
        offsets, speed = L.event_profile(starts, d["all_speed"])
        _, ball = L.event_profile(starts, d["ball_seen"].astype(float))
        rep.note("around SET PIECES starts ({} in graded play), 5-s bins: seconds from start | "
                 "mean player speed m/s | ball visible %".format(len(starts)))
        for lo in range(-20, 45, 5):
            m = (offsets >= lo) & (offsets < lo + 5)
            if m.any():
                rep.note("  {:>+4}..{:<+4} | {:4.2f} | {:5.1f}".format(
                    lo, lo + 5, float(np.nanmean(speed[m])), 100 * float(np.nanmean(ball[m]))))
        rep.note("a dip in speed marks the stoppage; where it sits relative to 0 tells you where "
                 "the restart is inside the tagged interval")

    # Do set pieces look like dead balls?
    sp = table[table["code"].str.upper() == "SET PIECES"]
    if len(sp):
        inside = np.zeros(n, dtype=bool)
        for a, z in zip(sp["start_frame"].clip(0, n), sp["end_frame"].clip(0, n)):
            inside[a:z] = True
        speed = d["all_speed"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            rep.row("mean player speed (m/s): set pieces / other graded",
                    [round(float(np.nanmean(speed[inside & graded])), 2),
                     round(float(np.nanmean(speed[~inside & graded])), 2)])
        rep.row("ball visible %: set pieces / other graded",
                [pct(d["ball_seen"][inside & graded].sum(), (inside & graded).sum()),
                 pct(d["ball_seen"][~inside & graded].sum(), (~inside & graded).sum())])


def video_section(rep, source, exp):
    rep.section("video")
    info = L.video_info(source)
    if info is None:
        rep.note("skipped: OpenCV not installed (pip install opencv-python-headless) or no video")
        return
    rep.row("frames (container header)", info["frames"], exp.get("frames"))
    rep.row("fps", round(info["fps"], 3), exp.get("video_fps"), tol=0.01)
    rep.row("size", [info["width"], info["height"]], list(exp.get("video_size", (1920, 1080))))
    rep.note("the header frame count can be off by a few in some mp4s; the dashboard's video tab "
             "checks seeking frame by frame")


# -- main --------------------------------------------------------------------------------------

def build_report(source="tactical", use_cache=True, sections=None):
    exp = EXPECTED.get(source, {})
    rep = Report(source)
    wanted = set(sections or ("files", "metadata", "tracking", "homography", "tagging", "cross",
                              "video"))

    def guarded(name, fn, *args):
        """Run one section; a failure is reported by type only, never with file content."""
        if name not in wanted:
            return None
        try:
            return fn(*args)
        except Exception as error:  # noqa: BLE001 -- one broken file must not hide the others
            rep.section("{} -- FAILED".format(name))
            rep.note("{}: {}".format(type(error).__name__, _safe_message(error)))
            return None

    guarded("files", files_section, rep, source)
    if L.has(source, "atd_meta"):
        guarded("metadata", metadata_section, rep, source, exp)
    td = None
    if L.has(source, "atd_meta", "atd_raw"):
        td = guarded("tracking", tracking_section, rep, source, exp, use_cache)
    phase = L.phase_of_frames(L.n_frames(source), source) if source == "tactical" else None
    if L.has(source, "homography"):
        guarded("homography", homography_section, rep, source, exp, phase)
    table = None
    if source == "tactical" and L.has(source, "smart_tagging"):
        table = guarded("tagging", tagging_section, rep, exp)
    if table is not None and td is not None:
        guarded("cross", cross_section, rep, table, td[1])
    if L.has(source, "video"):
        guarded("video", video_section, rep, source, exp)
    return rep


def _safe_message(error):
    """Exception text with anything that looks like data removed: keep it short and digit-light."""
    text = str(error).splitlines()[0] if str(error) else ""
    if isinstance(error, L.FormatError):
        return text
    if sum(ch.isdigit() for ch in text) > 12 or len(text) > 160:
        return "(message withheld: it may quote file content)"
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default="tactical", choices=sorted(L.SOURCES))
    parser.add_argument("--no-cache", action="store_true", help="re-parse the raw tracking")
    args = parser.parse_args()

    report = build_report(args.source, use_cache=not args.no_cache)
    report.print()
    try:
        os.makedirs(L.CACHE_DIR, exist_ok=True)
        path = os.path.join(L.CACHE_DIR, "report_{}.json".format(args.source))
        with open(path, "w") as handle:
            json.dump(report.to_json(), handle, indent=1, default=str)
        print("\nJSON copy: <DATA_DIR>/_inspection_cache/{}".format(os.path.basename(path)))
    except OSError as error:
        print("\n(could not write the JSON copy: {})".format(type(error).__name__))
