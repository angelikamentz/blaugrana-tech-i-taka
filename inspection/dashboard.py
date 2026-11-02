"""Data inspection, drawn: look at every provided file and at how they line up.

    pip install -r requirements-inspection.txt
    streamlit run inspection/dashboard.py

The companion of inspection/inspect_data.py (the same checks, as text). Run it in your own
terminal -- inside Claude Code games.asset_paths() refuses to run. Unlike the text report, these
pages show the data itself (positions, frames of video): they are for your eyes. Don't paste
screenshots of them into an LLM conversation; the terms treat them as extracts of the data.

Tabs
----
  Overview          file inventory + every check from inspect_data.py with expected values
  Timeline          the whole match per second: ball visibility, players per team, speed, team
                    positions, nearest player to the ball, homography error, Smart Tagging,
                    and label CSVs from evaluation/ for comparison
  Pitch + video     the selected frame on a pitch next to the video with the tracking projected
                    onto it through the homography -- one frame, or both animated together
  Tracking quality  ball gaps, players per frame, slot lifetimes, speeds, positional heatmaps
  Smart Tagging     the tags as a table, how well each code's Team tag agrees with the
                    tracking, and what happens on average around a tag
  Guide             inspection/README.md

Every plot has its legend and its "how to read" note directly above it.
"""
import base64
import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import loaders as L  # noqa: E402
import inspect_data as I  # noqa: E402

# Params
COLORS = {"H": "#2F6BD8", "A": "#D64541", "D": "#A3A8AE", "X": "#D5D9DD", "N/A": "#8E44AD",
          "?": "#5B6B7A"}
LABEL_NAMES = {"H": "H · home in possession", "A": "A · away in possession", "D": "D · dead ball",
               "X": "X · no label", "N/A": "N/A · tag has no team", "?": "? · unknown team"}
PHASE_COLORS = {"pre-match": "#B0B7BF", "1st half": "#27AE60", "halftime": "#F39C12",
                "2nd half": "#16A085", "post-match": "#7F8C8D", "all": "#5B6B7A"}
BGR = {"H": (216, 107, 47), "A": (65, 69, 214), "ball": (255, 255, 255), "line": (255, 255, 255)}
GRASS, LINE = "#3f8f4f", "#ffffff"
SELECTED = "#F1C40F"        # the selected-frame marker, everywhere
GRADED = "#2ECC71"          # graded-halves shading, everywhere
FPS = L.FPS
LEN, WID = L.PITCH_LENGTH, L.PITCH_WIDTH
GT_DIR = os.path.join(ROOT, "evaluation", "gt")
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
MAX_ANIMATION_S = 30
VIDEO_WIDTHS = (640, 960, 1280)
CLIP_DIR = os.path.join(L.CACHE_DIR, "clips")   # rendered clips are data: under the data dir
KEEP_CLIPS = 12
GRASS_BGR, PITCH_BG_BGR = (79, 143, 63), (58, 107, 45)

st.set_page_config(page_title="Data inspection", page_icon="🔎", layout="wide")


# -- cached loading ----------------------------------------------------------------------------
# The parsed tracking is ~110 MB; cache_resource keeps one copy instead of re-pickling it per
# rerun. `key` (file size + mtime) invalidates the caches when a file changes.

def file_key(source, asset):
    path, exists, size = L.available_assets(source)[asset]
    return "{}:{}".format(size, int(os.path.getmtime(path))) if exists else "missing"


@st.cache_resource(show_spinner="Parsing the tracking (first run: a few minutes; then cached)…")
def get_tracking(source, key):
    return L.load_tracking(source)


@st.cache_resource(show_spinner="Computing per-frame signals…")
def get_signals(source, key):
    return L.derived(get_tracking(source, key), source)


@st.cache_resource(show_spinner="Reading the homography…")
def get_homography(source, key):
    return L.load_homography(source)


@st.cache_data(show_spinner="Reading Smart Tagging…")
def get_tagging(key):
    return L.load_tagging()


@st.cache_data(show_spinner="Running the checks…")
def get_report(source, keys):
    return I.build_report(source).to_json()


@st.cache_data(show_spinner=False)
def get_labels(path, mtime):
    return pd.read_csv(path, dtype={"label": "object"})["label"].fillna("X").to_numpy()


@st.cache_resource
def get_capture(path):
    import cv2
    return cv2.VideoCapture(path)


def cv2_or_none():
    try:
        import cv2
        return cv2
    except ImportError:
        return None


# -- small helpers -----------------------------------------------------------------------------

def show(fig):
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def set_frame(frame):
    st.session_state["frame"] = int(min(max(frame, 0), N - 1))


def guarded(fn):
    """Run one tab; a failure shows its type and message here instead of killing the page."""
    try:
        fn()
    except Exception as error:  # noqa: BLE001
        st.error("This tab failed: {}: {}".format(type(error).__name__, error))


SYMBOLS = {"box": "■", "line": "━", "dot": "●", "ring": "○", "dash": "┅", "band": "▮"}


def legend(items, prefix=""):
    """A legend line written directly above a plot: items = [(color, label, symbol)]."""
    parts = ["<span style='color:{};font-size:1.15em'>{}</span>&nbsp;{}".format(
        color, SYMBOLS[symbol], label) for color, label, symbol in items]
    st.markdown((prefix + " " if prefix else "") + " &nbsp;&nbsp; ".join(parts),
                unsafe_allow_html=True)


def plot_header(title, note=None, items=None):
    """Title, then the legend, then the how-to-read note -- all above the plot they describe."""
    st.markdown("**{}**".format(title))
    if items:
        legend(items)
    if note:
        st.caption(note)


def side_items(sides=("H", "A")):
    return [(COLORS[s], NAMES.get(s, LABEL_NAMES.get(s, s)), "box") for s in sides]


def colored(text, color, symbol="line"):
    """Inline legend for plotly subplot titles (they accept a small subset of HTML)."""
    return '<span style="color:{}">{}</span> {}'.format(color, SYMBOLS[symbol], text)


def shade_halves(fig, lo, hi, subplots=False):
    if SOURCE != "tactical":
        return
    for _, a, z in L.half_video_blocks(L.GAME):
        if z < lo or a > hi:
            continue
        kwargs = {"row": "all", "col": 1} if subplots else {}
        fig.add_vrect(x0=max(a, lo), x1=min(z, hi), fillcolor=GRADED, opacity=0.08,
                      line_width=0, **kwargs)


def mark_selected(fig, subplots=False):
    kwargs = {"row": "all", "col": 1} if subplots else {}
    fig.add_vline(x=FRAME, line_color=SELECTED, line_width=2, **kwargs)


def frame_ticks(fig, lo, hi, n_ticks=9, **kwargs):
    ticks = np.linspace(lo, hi, n_ticks).astype(int).tolist()
    fig.update_xaxes(tickvals=ticks, ticktext=[L.clock(t, SOURCE) for t in ticks], **kwargs)


def time_legend_items():
    items = [(SELECTED, "<b>selected frame</b> (sidebar)", "line")]
    if SOURCE == "tactical":
        items.insert(0, (GRADED, "graded halves (scored)", "band"))
    return items


def label_runs(letters, lo, hi):
    """Runs of the per-second majority label, in frames. Per second keeps flickery files light."""
    sub = letters[lo:hi]
    n = len(sub) // FPS * FPS
    if n == 0:
        return []
    codes = (pd.Series(sub[:n]).map({"A": 0, "D": 1, "H": 2, "X": 3}).fillna(3)
             .to_numpy(dtype=int).reshape(-1, FPS))
    counts = np.stack([(codes == k).sum(axis=1) for k in range(4)], axis=1)
    majority = np.array(["A", "D", "H", "X"])[counts.argmax(axis=1)]
    out = []
    start = 0
    for i in range(1, len(majority) + 1):
        if i == len(majority) or majority[i] != majority[start]:
            out.append((lo + start * FPS, lo + i * FPS, majority[start]))
            start = i
    return out


def strip_figure(rows, lo, hi):
    """Horizontal strips: rows = [(name, [(start, stop, key)])], key in COLORS. No legend:
    the caller writes one above the plot."""
    fig = go.Figure()
    for name, segs in rows:
        by_key = {}
        for a, z, key in segs:
            a, z = max(a, lo), min(z, hi)
            if z > a:
                by_key.setdefault(key, []).append((a, z))
        for key, spans in by_key.items():
            fig.add_bar(orientation="h", y=[name] * len(spans), base=[a for a, _ in spans],
                        x=[z - a for a, z in spans], marker_color=COLORS.get(key, "#888"),
                        marker_line_width=0, showlegend=False,
                        hovertext=["{}: {}  {} → {} ({:.1f} s)".format(
                            name, LABEL_NAMES.get(key, key), L.clock(a, SOURCE),
                            L.clock(z, SOURCE), (z - a) / FPS) for a, z in spans],
                        hoverinfo="text")
    fig.update_layout(barmode="overlay", height=60 + 30 * len(rows),
                      margin=dict(l=10, r=10, t=10, b=10))
    fig.update_yaxes(categoryorder="array", categoryarray=[r[0] for r in rows],
                     autorange="reversed")
    fig.update_xaxes(range=[lo, hi])
    shade_halves(fig, lo, hi)
    mark_selected(fig)
    frame_ticks(fig, lo, hi)
    return fig


# -- pitch drawing -----------------------------------------------------------------------------

def pitch_shapes(xref="x", yref="y", filled=True):
    """Pitch markings in metres, origin at a corner, as plotly layout shapes."""
    line = dict(color=LINE, width=2)
    common = dict(xref=xref, yref=yref, line=line)
    shapes = [dict(type="rect", x0=0, y0=0, x1=LEN, y1=WID, layer="below",
                   fillcolor=GRASS if filled else None, **common),
              dict(type="line", x0=LEN / 2, y0=0, x1=LEN / 2, y1=WID, **common),
              dict(type="circle", x0=LEN / 2 - 9.15, y0=WID / 2 - 9.15, x1=LEN / 2 + 9.15,
                   y1=WID / 2 + 9.15, **common)]
    if not filled:
        shapes[0]["layer"] = "above"
    for depth, box in ((16.5, 40.3), (5.5, 18.3)):
        for x0, x1 in ((0, depth), (LEN - depth, LEN)):
            shapes.append(dict(type="rect", x0=x0, x1=x1, y0=(WID - box) / 2,
                               y1=(WID + box) / 2, **common))
    return shapes


def frame_points(tr, f):
    """{'H': (x, y, slots), 'A': ..., 'ball': (x, y)} in metres for one frame."""
    players = tr["players"][f]
    side = tr["slot_side"]
    out = {}
    for key, code in (("H", L.HOME), ("A", L.AWAY)):
        m = (side == code) & np.isfinite(players[:, 0])
        slots = np.flatnonzero(m)
        out[key] = ((players[m, 0] * LEN).astype(float).tolist(),
                    (players[m, 1] * WID).astype(float).tolist(), (slots + 1).tolist())
    b = tr["ball"][f]
    out["ball"] = ([float(b[0]) * LEN], [float(b[1]) * WID]) if np.isfinite(b[0]) else ([], [])
    return out


def player_traces(points, show_numbers, axes):
    """[H, A, ball] scatter traces on the given axes ({'xaxis': 'x2', 'yaxis': 'y2'})."""
    traces = []
    for key in ("H", "A"):
        x, y, slots = points[key]
        traces.append(go.Scatter(
            x=x, y=y, mode="markers+text" if show_numbers else "markers", showlegend=False,
            text=[str(s) for s in slots], textposition="top center",
            textfont=dict(color="white", size=10),
            marker=dict(size=13, color=COLORS[key], line=dict(color="black", width=1)),
            hovertemplate="slot %{text}<br>x %{x:.1f} m, y %{y:.1f} m<extra>" + key + "</extra>",
            **axes))
    bx, by = points["ball"]
    traces.append(go.Scatter(x=bx, y=by, mode="markers", showlegend=False,
                             marker=dict(size=10, color="white", line=dict(color="black", width=2)),
                             hovertemplate="ball<br>x %{x:.1f} m, y %{y:.1f} m<extra></extra>",
                             **axes))
    return traces


# -- video drawing -----------------------------------------------------------------------------

def pitch_polylines():
    """Pitch markings as polylines in normalised (0-1) pitch coordinates, densified so the
    projective warp of long straight lines is drawn faithfully."""
    def rect(x0, y0, x1, y1):
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    lines = [rect(0, 0, 1, 1), [(0.5, 0), (0.5, 1)]]
    t = np.linspace(0, 2 * np.pi, 60)
    lines.append(list(zip(0.5 + 9.15 / LEN * np.cos(t), 0.5 + 9.15 / WID * np.sin(t))))
    for depth, box in ((16.5, 40.3), (5.5, 18.3)):
        d, b = depth / LEN, box / WID
        lines.append(rect(0, 0.5 - b / 2, d, 0.5 + b / 2))
        lines.append(rect(1 - d, 0.5 - b / 2, 1, 0.5 + b / 2))
    dense = []
    for line in lines:
        pts = []
        for (x0, y0), (x1, y1) in zip(line[:-1], line[1:]):
            for s in np.linspace(0, 1, 20, endpoint=False):
                pts.append((x0 + s * (x1 - x0), y0 + s * (y1 - y0)))
        pts.append(line[-1])
        dense.append(np.array(pts))
    return dense


PITCH_POLYLINES = pitch_polylines()


def draw_overlay(image, f, opts, matrices, valid, tr):
    """Draw pitch lines, players and ball for frame f onto a BGR image (in place)."""
    cv2 = cv2_or_none()
    if not valid[f]:
        return image
    H = matrices[f]

    def to_px(x, y):
        y = 1 - np.asarray(y) if opts["swap_y"] else y
        px, py = L.project(H, x, y)
        return np.asarray(px), np.asarray(py)

    if opts["lines"]:
        for line in PITCH_POLYLINES:
            px, py = to_px(line[:, 0], line[:, 1])
            ok = np.isfinite(px) & np.isfinite(py)
            pts = np.stack([px[ok], py[ok]], axis=1).round().astype(np.int32)
            if len(pts) > 1:
                cv2.polylines(image, [pts], False, BGR["line"], 2, cv2.LINE_AA)
    if tr is None:
        return image
    players = tr["players"][f]
    if opts["players"]:
        for key, code in (("H", L.HOME), ("A", L.AWAY)):
            slots = np.flatnonzero((tr["slot_side"] == code) & np.isfinite(players[:, 0]))
            px, py = to_px(players[slots, 0], players[slots, 1])
            for s, x, y in zip(slots, np.atleast_1d(px), np.atleast_1d(py)):
                if np.isfinite(x) and np.isfinite(y):
                    centre = (int(round(x)), int(round(y)))
                    cv2.circle(image, centre, 10, BGR[key], 3, cv2.LINE_AA)
                    if opts["numbers"]:
                        cv2.putText(image, str(s + 1), (centre[0] + 9, centre[1] - 9),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, BGR[key], 2, cv2.LINE_AA)
    b = tr["ball"][f]
    if opts["ball"] and np.isfinite(b[0]):
        px, py = to_px(b[0], b[1])
        if np.isfinite(px) and np.isfinite(py):
            centre = (int(round(float(px))), int(round(float(py))))
            cv2.circle(image, centre, 8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.circle(image, centre, 8, BGR["ball"], 2, cv2.LINE_AA)
    return image


def to_data_uri(image, width):
    """BGR image -> resized JPEG data URI for a plotly Image trace."""
    cv2 = cv2_or_none()
    h, w = image.shape[:2]
    if w != width:
        image = cv2.resize(image, (width, int(round(h * width / float(w)))),
                           interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 72])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def read_video_frame(path, f):
    """(BGR image, reported position) for frame f, seeking directly."""
    cv2 = cv2_or_none()
    cap = get_capture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, image = cap.read()
    reported = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    return (image if ok else None), reported


def read_video_frame_sequential(path, f, lead=60):
    """Frame f reached by decoding forward from f - lead: the reference for checking seeks."""
    cv2 = cv2_or_none()
    cap = cv2.VideoCapture(path)
    try:
        start = max(0, f - lead)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for _ in range(f - start):
            cap.grab()
        ok, image = cap.read()
        return image if ok else None
    finally:
        cap.release()


def ffmpeg_or_none():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg
    except ImportError:
        return None


def pitch_canvas(height, flip_y, margin=4.0):
    """(empty pitch image BGR, metres -> pixel function) at a given height, drawn with OpenCV."""
    cv2 = cv2_or_none()
    scale = height / (WID + 2 * margin)
    width = int(round((LEN + 2 * margin) * scale))
    width += width % 2
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = PITCH_BG_BGR

    def to_px(x, y):
        px = (np.asarray(x, dtype=float) + margin) * scale
        py = (np.asarray(y, dtype=float) + margin) * scale if flip_y \
            else (WID - np.asarray(y, dtype=float) + margin) * scale
        return px, py

    def pt(x, y):
        px, py = to_px(x, y)
        return int(round(float(px))), int(round(float(py)))

    cv2.rectangle(image, pt(0, 0), pt(LEN, WID), GRASS_BGR, -1)
    white, thick = (255, 255, 255), 2
    cv2.rectangle(image, pt(0, 0), pt(LEN, WID), white, thick, cv2.LINE_AA)
    cv2.line(image, pt(LEN / 2, 0), pt(LEN / 2, WID), white, thick, cv2.LINE_AA)
    cv2.circle(image, pt(LEN / 2, WID / 2), int(round(9.15 * scale)), white, thick, cv2.LINE_AA)
    for depth, box in ((16.5, 40.3), (5.5, 18.3)):
        for x0, x1 in ((0, depth), (LEN - depth, LEN)):
            cv2.rectangle(image, pt(x0, (WID - box) / 2), pt(x1, (WID + box) / 2), white, thick,
                          cv2.LINE_AA)
    return image, to_px, scale


def draw_pitch_frame(image, to_px, scale, tr, f, numbers):
    """Players and ball of frame f onto a copy of the empty pitch canvas."""
    cv2 = cv2_or_none()
    radius = max(5, int(round(0.95 * scale)))
    points = frame_points(tr, f)
    for key in ("H", "A"):
        xs, ys, slots = points[key]
        px, py = to_px(xs, ys)
        for x, y, s in zip(np.atleast_1d(px), np.atleast_1d(py), slots):
            centre = (int(round(x)), int(round(y)))
            cv2.circle(image, centre, radius, BGR[key], -1, cv2.LINE_AA)
            cv2.circle(image, centre, radius, (0, 0, 0), 1, cv2.LINE_AA)
            if numbers:
                cv2.putText(image, str(s), (centre[0] + radius, centre[1] - radius),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    bx, by = points["ball"]
    if bx:
        px, py = to_px(bx[0], by[0])
        centre = (int(round(float(px))), int(round(float(py))))
        cv2.circle(image, centre, max(4, radius - 2), (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(image, centre, max(4, radius - 2), (0, 0, 0), 2, cv2.LINE_AA)
    return image


def burn_label(image, text):
    """White text on a dark band, top-left. ASCII only: OpenCV fonts have no other glyphs."""
    cv2 = cv2_or_none()
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(image, (0, 0), (w + 16, h + base + 12), (0, 0, 0), -1)
    cv2.putText(image, text, (8, h + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                cv2.LINE_AA)


def clip_label(f, tr):
    text = "{} | frame {}".format(L.clock(f, SOURCE), f)
    if tr is not None:
        p = frame_points(tr, f)
        text += " | H {} | A {} | {}".format(len(p["H"][0]), len(p["A"][0]),
                                             "ball" if len(p["ball"][0]) else "NO BALL")
    return text


def clip_file(params):
    digest = hashlib.sha1(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()
    return os.path.join(CLIP_DIR, "clip_{}.mp4".format(digest[:16]))


def prune_clips():
    try:
        clips = sorted(glob.glob(os.path.join(CLIP_DIR, "clip_*.mp4")), key=os.path.getmtime)
        for old in clips[:-KEEP_CLIPS]:
            os.remove(old)
    except OSError:
        pass


def render_clip(out_path, start, count, width, opts, flip_y, numbers, progress):
    """Decode `count` frames from `start`, draw the overlay and the pitch, encode H.264.

    One seek, then sequential decoding, so every frame of the clip is consecutive. Returns the
    frame the decoder reported after the seek, to catch an inexact seek.
    """
    cv2 = cv2_or_none()
    imageio_ffmpeg = ffmpeg_or_none()
    tr = get_tracking(SOURCE, KEYS["atd_raw"]) if HAS_TRACKING else None
    height = int(round(width * 1080 / 1920.0))
    height += height % 2
    canvas = pitch_canvas(height, flip_y) if tr is not None else None
    video_width = width if HAS_VIDEO else 0
    total_width = video_width + (canvas[0].shape[1] if canvas else 0)
    if HAS_VIDEO:
        video = L.available_assets(SOURCE)["video"][0]
        matrices, _, valid, _ = get_homography(SOURCE, KEYS["homography"])

    os.makedirs(CLIP_DIR, exist_ok=True)
    partial = out_path + ".part.mp4"
    writer = imageio_ffmpeg.write_frames(
        partial, (total_width, height), fps=FPS, codec="libx264", pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p", quality=None, macro_block_size=2,
        output_params=["-preset", "veryfast", "-crf", "23", "-movflags", "+faststart"])
    writer.send(None)
    cap = cv2.VideoCapture(video) if HAS_VIDEO else None
    reported = start
    try:
        if cap is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            reported = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        for i in range(count):
            f = start + i
            panels = []
            if cap is not None:
                ok, image = cap.read()
                if not ok:
                    break
                draw_overlay(image, f, opts, matrices, valid, tr)
                panels.append(cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA))
            if canvas:
                panels.append(draw_pitch_frame(canvas[0].copy(), canvas[1], canvas[2], tr, f,
                                               numbers))
            image = np.hstack(panels) if len(panels) > 1 else panels[0]
            burn_label(image, clip_label(f, tr))
            writer.send(np.ascontiguousarray(image[:, :, ::-1]).tobytes())  # BGR -> RGB
            progress((i + 1) / float(count))
    finally:
        if cap is not None:
            cap.release()
        writer.close()
    os.replace(partial, out_path)
    prune_clips()
    return reported


# -- sidebar -----------------------------------------------------------------------------------

st.title("Data inspection")
with st.sidebar:
    st.header("Source")
    sources = [s for s in L.SOURCES if L.has(s, "atd_meta") or L.has(s, "homography")]
    if not sources:
        st.error("No data found. Set MTAH_DATA_DIR (see DATA.md) and restart.")
        st.stop()
    SOURCE = st.radio("Recording", sources, index=sources.index("tactical")
                      if "tactical" in sources else 0,
                      help="tactical is the reference: row i of a submission is frame i of it.")
    N = L.n_frames(SOURCE)
    HAS_TRACKING = L.has(SOURCE, "atd_meta", "atd_raw")
    HAS_HOMOGRAPHY = L.has(SOURCE, "homography")
    HAS_TAGGING = SOURCE == "tactical" and L.has(SOURCE, "smart_tagging")
    HAS_VIDEO = L.has(SOURCE, "video") and HAS_HOMOGRAPHY and cv2_or_none() is not None

    st.header("Selected frame")
    st.caption("Shown on the Pitch + video tab, and as the yellow line on every timeline.")
    if "frame" not in st.session_state or st.session_state.get("source") != SOURCE:
        st.session_state["frame"] = 11185 if SOURCE == "tactical" else 0
        st.session_state["source"] = SOURCE
    st.number_input("Video frame (0-based = submission row)", 0, N - 1, key="frame", step=1)
    c = st.columns(4)
    c[0].button("−1 s", on_click=set_frame, args=(st.session_state["frame"] - FPS,))
    c[1].button("−1", on_click=set_frame, args=(st.session_state["frame"] - 1,))
    c[2].button("+1", on_click=set_frame, args=(st.session_state["frame"] + 1,))
    c[3].button("+1 s", on_click=set_frame, args=(st.session_state["frame"] + FPS,))
    if SOURCE == "tactical":
        def goto_clock():
            start = dict((h, a) for h, a, _ in L.half_video_blocks(L.GAME))[
                st.session_state["goto_half"]]
            set_frame(start + (st.session_state["goto_min"] * 60
                               + st.session_state["goto_s"]) * FPS)

        with st.form("goto"):
            st.caption("Go to a match clock (time since the half's first graded frame)")
            g = st.columns(3)
            g[0].selectbox("Half", [1, 2], key="goto_half")
            g[1].number_input("min", 0, 60, 0, key="goto_min")
            g[2].number_input("s", 0, 59, 0, key="goto_s")
            st.form_submit_button("Go", on_click=goto_clock)
    FRAME = int(st.session_state["frame"])
    st.markdown("**{}** · frame {}".format(L.clock(FRAME, SOURCE), FRAME))

NAMES = {"H": "H · home", "A": "A · away"}
if HAS_TRACKING:
    try:
        home_name, away_name = L.team_names(SOURCE)
        NAMES = {"H": "H · {}".format(home_name), "A": "A · {}".format(away_name)}
    except Exception:  # noqa: BLE001
        pass

KEYS = {a: file_key(SOURCE, a) for a in L.available_assets(SOURCE)}

tabs = st.tabs(["Overview", "Timeline", "Pitch + video", "Tracking quality", "Smart Tagging",
                "Guide"])


# -- Overview ----------------------------------------------------------------------------------

def overview_tab():
    st.subheader("Files")
    st.dataframe(pd.DataFrame([{
        "asset": name, "file": os.path.basename(path), "present": exists,
        "MB": round(size / 1e6, 1)} for name, (path, exists, size)
        in L.available_assets(SOURCE).items()]), hide_index=True, width="stretch")

    st.subheader("Where each signal comes from")
    st.markdown("""
| file | gives you | time base → video frame | used on these pages |
|---|---|---|---|
| `*_atd.xml` | which slot is which team, pitch size, fps, the raw line layout | static | team colours everywhere |
| `*_atd.txt` | per frame: up to 46 slots × (x, y, speed) + ball, 0–1 units | `frame_id − 1` | Timeline, Pitch + video, Quality |
| `*_homography.csv` | per frame: 3×3 pitch→pixel matrix, error, valid | `frame` | Timeline (error), Pitch + video |
| `*_smart_tagging.xml` | 638 overlapping phases with code + Team/Type/Half tags | `round(seconds × 25)` | Timeline, Smart Tagging |
| `*.mp4` | the pixels everything else was computed from | frame index | Pitch + video |
""")

    st.subheader("Checks")
    st.caption("The same report as `python inspection/inspect_data.py`, with the numbers DATA.md "
               "publishes. ⚠️ is a question, not necessarily a bug.")
    if not st.toggle("Run the checks", value=False, key="run_checks",
                     help="Parses everything once; cached afterwards."):
        return
    report = get_report(SOURCE, tuple(sorted(KEYS.items())))
    for section in report["sections"]:
        rows = section["rows"]
        n_diff = sum(r["ok"] is False for r in rows)
        with st.expander("{}{}".format(section["title"], "  —  {} ⚠️".format(n_diff)
                                       if n_diff else ""), expanded=n_diff > 0):
            if rows:
                st.dataframe(pd.DataFrame([{
                    "check": r["check"], "got": I._fmt(r["got"]),
                    "expected": "" if r["expected"] is None else I._fmt(r["expected"]),
                    "": "" if r["ok"] is None else ("✅" if r["ok"] else "⚠️")} for r in rows]),
                    hide_index=True, width="stretch")
            if section["notes"]:
                st.code("\n".join(section["notes"]), language=None)


# -- Timeline ----------------------------------------------------------------------------------

@st.cache_resource(show_spinner="Aggregating per second…")
def per_second_table(source, key_tr, key_h):
    cols = {}
    x = None
    if key_tr != "missing":
        d = get_signals(source, key_tr)
        x, cols["ball visible"] = L.per_second(d["ball_seen"].astype(float))
        _, cols["players H"] = L.per_second(d["n_home"])
        _, cols["players A"] = L.per_second(d["n_away"])
        _, cols["speed H"] = L.per_second(d["home_speed"])
        _, cols["speed A"] = L.per_second(d["away_speed"])
        _, cols["centroid x H"] = L.per_second(d["home_cx"])
        _, cols["centroid x A"] = L.per_second(d["away_cx"])
        ns = d["nearest_side"].astype(float)
        ns[ns < 0] = np.nan
        _, cols["nearest is H"] = L.per_second(1.0 - ns)
        _, cols["ball x"] = L.per_second(d["ball_x_m"])
    if key_h != "missing":
        _, error, valid, _ = get_homography(source, key_h)
        x, cols["reprojection error"] = L.per_second(np.where(valid, error, np.nan))
    table = pd.DataFrame(cols)
    table.insert(0, "frame", x)
    return table


def timeline_tab():
    if not (HAS_TRACKING or HAS_HOMOGRAPHY):
        st.info("Needs tracking or homography for this source.")
        return
    total_min = N / FPS / 60.0
    lo_min, hi_min = st.slider("Window (video minutes)", 0.0, round(total_min, 1),
                               (0.0, round(total_min, 1)), step=0.5)
    lo, hi = int(lo_min * 60 * FPS), min(N, int(hi_min * 60 * FPS))
    table = per_second_table(SOURCE, KEYS["atd_raw"] if HAS_TRACKING else "missing",
                             KEYS["homography"] if HAS_HOMOGRAPHY else "missing")
    t = table[(table["frame"] >= lo) & (table["frame"] < hi)]
    x = t["frame"].to_numpy()

    H_, A_ = COLORS["H"], COLORS["A"]
    panels = []  # (title with inline legend, [(column, color, dash)], y-axis title, log?)
    if HAS_TRACKING:
        panels += [
            ("<b>Ball tracked</b> — share of each second's frames &nbsp; "
             + colored("ball visible", "#222"), [("ball visible", "#222", None)], "share", False),
            ("<b>Tracked players per team</b> &nbsp; " + colored(NAMES["H"], H_) + " &nbsp; "
             + colored(NAMES["A"], A_), [("players H", H_, None), ("players A", A_, None)],
             "players", False),
            ("<b>Mean player speed</b> — dips = stoppages? &nbsp; " + colored(NAMES["H"], H_)
             + " &nbsp; " + colored(NAMES["A"], A_),
             [("speed H", H_, None), ("speed A", A_, None)], "m/s", False),
            ("<b>Mean x position</b> — teams swap ends at halftime &nbsp; "
             + colored(NAMES["H"], H_) + " &nbsp; " + colored(NAMES["A"], A_) + " &nbsp; "
             + colored("ball", "#222", "dash"),
             [("centroid x H", H_, None), ("centroid x A", A_, None), ("ball x", "#222", "dot")],
             "metres", False),
            ("<b>Nearest player to the ball is H</b> — 1 = all H, 0 = all A &nbsp; "
             + colored("share of ball frames", "#5B3FD1"), [("nearest is H", "#5B3FD1", None)],
             "share", False)]
    if HAS_HOMOGRAPHY:
        panels += [("<b>Homography reprojection error</b> — lower is better, log scale &nbsp; "
                    + colored("error", "#E67E22"), [("reprojection error", "#E67E22", None)],
                    "pixels", True)]

    plot_header(
        "Whole match, one point per second",
        "Each panel's own legend is in its title. Hover anywhere to read every panel at that "
        "second. Move the yellow line with the sidebar, or from the Smart Tagging tab.",
        time_legend_items())
    fig = make_subplots(rows=len(panels), cols=1, shared_xaxes=True, vertical_spacing=0.045,
                        subplot_titles=[p[0] for p in panels])
    for row, (_, series, y_title, log) in enumerate(panels, 1):
        for name, color, dash in series:
            fig.add_trace(go.Scattergl(x=x, y=t[name], name=name, mode="lines",
                                       showlegend=False,
                                       line=dict(color=color, width=1, dash=dash),
                                       hovertemplate="%{y:.2f}<extra>" + name + "</extra>"),
                          row=row, col=1)
        fig.update_yaxes(title_text=y_title, type="log" if log else None, row=row, col=1)
    fig.update_annotations(xanchor="left", x=0, font_size=13)  # titles: left-aligned, above
    fig.update_layout(height=190 * len(panels) + 40, margin=dict(l=10, r=10, t=30, b=10),
                      hovermode="x unified")
    shade_halves(fig, lo, hi, subplots=True)
    mark_selected(fig, subplots=True)
    frame_ticks(fig, lo, hi, row=len(panels), col=1)
    show(fig)

    rows = []
    label_files = (sorted(glob.glob(os.path.join(GT_DIR, "*.csv")))
                   + sorted(glob.glob(os.path.join(RESULTS_DIR, "*.csv"))))
    picked = []
    if SOURCE == "tactical" and label_files:
        picked = st.multiselect("Label strips from evaluation/ (ground truth, submissions)",
                                [os.path.relpath(p, ROOT) for p in label_files],
                                default=[os.path.relpath(p, ROOT) for p in label_files
                                         if "public" in p][:1])
        for rel in picked:
            path = os.path.join(ROOT, rel)
            letters = get_labels(path, os.path.getmtime(path))
            if len(letters) == N:
                rows.append((os.path.basename(rel).replace(".csv", ""),
                             label_runs(letters, lo, hi)))
    if HAS_TAGGING:
        table_tags, _ = get_tagging(KEYS["smart_tagging"])
        for code in table_tags["code"].value_counts().index:
            sub = table_tags[table_tags["code"] == code]
            rows.append((code.title(), list(zip(sub["start_frame"], sub["end_frame"],
                                                sub["possession_side"]))))
    if rows:
        keys = ["H", "A"] + (["D", "X"] if picked else []) + (["N/A"] if HAS_TAGGING else [])
        plot_header(
            "Labels{} and Smart Tagging".format(" (top rows)" if picked else ""),
            "Label rows show the per-second majority label. Smart Tagging rows are coloured by "
            "the side the tag says has the ball (defending codes flipped). Tags overlap, and they "
            "are phases, not possessions.",
            [(COLORS[k], LABEL_NAMES[k], "box") for k in keys] + time_legend_items())
        show(strip_figure(rows, lo, hi))


# -- Pitch + video -----------------------------------------------------------------------------

def combined_figure(with_video, with_pitch, flip_y, height):
    """One figure: video on the left (if any), pitch on the right. Returns (fig, pitch axes)."""
    if with_video and with_pitch:
        fig = make_subplots(rows=1, cols=2, column_widths=[0.57, 0.43], horizontal_spacing=0.02)
        video_axes, pitch_axes = ("x", "y"), ("x2", "y2")
    else:
        fig = make_subplots(rows=1, cols=1)
        video_axes = pitch_axes = ("x", "y")
    if with_video:
        fig.update_xaxes(visible=False, row=1, col=1)
        fig.update_yaxes(visible=False, autorange="reversed", scaleanchor=video_axes[0],
                         row=1, col=1)
    if with_pitch:
        col = 2 if with_video else 1
        fig.update_layout(shapes=pitch_shapes(*pitch_axes))
        fig.update_xaxes(range=[-4, LEN + 4], showgrid=False, zeroline=False, title="x (m)",
                         row=1, col=col)
        fig.update_yaxes(range=[WID + 4, -4] if flip_y else [-4, WID + 4], showgrid=False,
                         zeroline=False, scaleanchor=pitch_axes[0], scaleratio=1, title="y (m)",
                         row=1, col=col)
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10),
                      plot_bgcolor="#2d6b3a" if with_pitch and not with_video else None)
    return fig, {"xaxis": pitch_axes[0], "yaxis": pitch_axes[1]}


def title_for(f, tr):
    if tr is None:
        return "{} · frame {}".format(L.clock(f, SOURCE), f)
    p = frame_points(tr, f)
    return "{} · frame {} · H {} · A {} · {}".format(
        L.clock(f, SOURCE), f, len(p["H"][0]), len(p["A"][0]),
        "ball tracked" if len(p["ball"][0]) else "NO BALL")


def pitch_video_tab():
    if not (HAS_TRACKING or HAS_VIDEO):
        st.info("Needs the tracking files, or the video + homography (with OpenCV installed).")
        return
    tr = get_tracking(SOURCE, KEYS["atd_raw"]) if HAS_TRACKING else None
    if L.has(SOURCE, "video") and cv2_or_none() is None:
        st.info("Install OpenCV to see the video next to the pitch: "
                "`pip install -r requirements-inspection.txt`.")

    c = st.columns([1.3, 1, 1, 1])
    mode = c[0].radio("Show", ["selected frame", "animation"], horizontal=True, key="pv_mode")
    numbers = c[1].toggle("slot numbers", value=False, key="pv_numbers")
    flip_y = c[2].toggle("pitch: y = 0 at the top", value=True, key="pv_flip_y",
                         help="Metrica's convention puts the origin top-left. Compare with the "
                              "video: the pitch should not look mirrored against it.")
    width = c[3].select_slider("video size (px)", VIDEO_WIDTHS, value=960, key="pv_width")
    opts = {"lines": True, "players": HAS_TRACKING, "ball": HAS_TRACKING, "numbers": numbers,
            "swap_y": False}
    if HAS_VIDEO:
        v = st.columns(4)
        opts["lines"] = v[0].toggle("video: pitch lines", True, key="pv_lines")
        opts["players"] = v[1].toggle("video: players", HAS_TRACKING, key="pv_players")
        opts["ball"] = v[2].toggle("video: ball", HAS_TRACKING, key="pv_ball")
        opts["swap_y"] = v[3].toggle("video: use 1 − y", False, key="pv_swap_y",
                                     help="If the projected circles are mirrored top-to-bottom "
                                          "against the players in the picture, the y convention "
                                          "differs. Flip here to test it -- then fix it in the "
                                          "pipeline, not by eye.")

    items = side_items() + [("#222", "ball (white, black outline)", "ring")]
    if HAS_VIDEO:
        items += [("#999", "on the video: pitch lines = homography", "line")]
    if mode == "selected frame":
        pitch_video_frame(tr, opts, numbers, flip_y, width, items)
    else:
        pitch_video_animation(tr, opts, numbers, flip_y, width, items)


def pitch_video_frame(tr, opts, numbers, flip_y, width, items):
    trail_s = st.slider("pitch trail (s of past movement)", 0, 10, 0, key="pv_trail")
    reported = None
    plot_header(
        "Selected frame: video with the tracking projected onto it, and the same frame on a pitch"
        if HAS_VIDEO else "Selected frame on a pitch",
        "On the video, pitch lines should sit on the painted lines (checks the homography) and "
        "circles on the players' feet in the right kit colours (checks the frame alignment, "
        "slot → team and the y convention). Step with the sidebar ±1 buttons.",
        items)
    fig, axes = combined_figure(HAS_VIDEO, HAS_TRACKING, flip_y, 560)
    if HAS_VIDEO:
        matrices, error, valid, _ = get_homography(SOURCE, KEYS["homography"])
        path = L.available_assets(SOURCE)["video"][0]
        image, reported = read_video_frame(path, FRAME)
        if image is None:
            st.error("Could not decode frame {}.".format(FRAME))
            return
        if not valid[FRAME]:
            st.warning("No valid homography on this frame: nothing is projected onto the video.")
        fig.add_trace(go.Image(source=to_data_uri(
            draw_overlay(image.copy(), FRAME, opts, matrices, valid, tr), width),
            hoverinfo="skip"), row=1, col=1)
    if tr is not None:
        if trail_s:
            a = max(0, FRAME - trail_s * FPS)
            seg = tr["players"][a:FRAME + 1]
            for key, code in (("H", L.HOME), ("A", L.AWAY)):
                xs, ys = [], []
                for s in np.flatnonzero(tr["slot_side"] == code):
                    if np.isfinite(seg[:, s, 0]).any():
                        xs += (seg[:, s, 0] * LEN).astype(float).tolist() + [None]
                        ys += (seg[:, s, 1] * WID).astype(float).tolist() + [None]
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", hoverinfo="skip",
                                         line=dict(color=COLORS[key], width=1.5),
                                         showlegend=False, opacity=0.7, **axes))
            b = tr["ball"][a:FRAME + 1]
            fig.add_trace(go.Scatter(x=(b[:, 0] * LEN).astype(float).tolist(),
                                     y=(b[:, 1] * WID).astype(float).tolist(), mode="lines",
                                     line=dict(color="white", width=1.5, dash="dot"),
                                     hoverinfo="skip", showlegend=False, **axes))
        for trace in player_traces(frame_points(tr, FRAME), numbers, axes):
            fig.add_trace(trace)
    fig.update_layout(title=title_for(FRAME, tr))
    show(fig)

    if HAS_VIDEO:
        st.caption("Decoder reports frame {}{} · reprojection error {:.2f}".format(
            reported, "" if reported == FRAME else "  ⚠️ MISMATCH with the requested frame",
            float(get_homography(SOURCE, KEYS["homography"])[1][FRAME])))
        if st.button("Verify the seek at this frame", key="pv_verify"):
            path = L.available_assets(SOURCE)["video"][0]
            image, _ = read_video_frame(path, FRAME)
            reference = read_video_frame_sequential(path, FRAME)
            if reference is None or image is None:
                st.error("Decoding failed.")
            else:
                same = reference.shape == image.shape and np.array_equal(reference, image)
                diff = float(np.abs(reference.astype(np.int16) - image.astype(np.int16)).mean()) \
                    if reference.shape == image.shape else float("nan")
                (st.success if same else st.warning)(
                    "Direct seek vs decoding forward from 60 frames earlier: {} (mean pixel "
                    "difference {:.2f}).".format("identical" if same else "DIFFERENT", diff))


def pitch_video_animation(tr, opts, numbers, flip_y, width, items):
    length = st.slider("clip length (s), starting at the selected frame", 2, MAX_ANIMATION_S, 10,
                       key="pv_length")
    count = min(length * FPS, N - FRAME)
    if ffmpeg_or_none() is not None and cv2_or_none() is not None:
        pitch_video_clip(tr, opts, numbers, flip_y, width, items, count)
        return
    st.info("For a smooth video clip (video + pitch side by side), install OpenCV and the "
            "encoder: `pip install -r requirements-inspection.txt`, then restart. Falling back "
            "to a pitch-only animation.")
    if tr is None:
        return
    pitch_animation(tr, numbers, flip_y, items, count)


def pitch_video_clip(tr, opts, numbers, flip_y, width, items, count):
    """Render the range to a real H.264 clip -- video + overlay on the left, pitch on the
    right -- and let the browser's own player play it: no flicker, real-time, scrubbable."""
    params = {"source": SOURCE, "keys": KEYS, "start": FRAME, "count": count, "width": width,
              "opts": opts, "flip_y": flip_y, "numbers": numbers, "version": 1}
    path = clip_file(params)
    plot_header(
        "{:.0f} s clip from {}{}".format(
            count / float(FPS), L.clock(FRAME, SOURCE),
            ": video and pitch side by side, in sync" if HAS_VIDEO and tr is not None
            else (": pitch only" if tr is not None else ": video only")),
        "A real video clip: use the player's own controls (play, scrub, full screen; right-click "
        "for playback speed). Every frame of the range is in it, at 25 fps. The label in the "
        "top-left corner is burned into each frame: match clock | frame | players | ball.",
        items)
    if not os.path.isfile(path):
        bar = st.progress(0.0, text="Rendering the clip: decoding, drawing, encoding…")
        reported = render_clip(path, FRAME, count, width, opts, flip_y, numbers,
                               lambda done: bar.progress(min(done, 1.0)))
        bar.empty()
        if reported != FRAME:
            st.warning("The decoder put the clip start at frame {}, not {}: the clip may be "
                       "offset. Check with 'Verify the seek' on the selected-frame view."
                       .format(reported, FRAME))
    with open(path, "rb") as handle:
        st.video(handle.read(), format="video/mp4")
    st.caption("Rendered once and kept (the last {} clips) under the data directory, so "
               "revisiting the same range is instant.".format(KEEP_CLIPS))


def pitch_animation(tr, numbers, flip_y, items, count):
    """Pitch-only animation in plotly (no video, so nothing heavy to redraw: no flicker)."""
    step = st.select_slider("every n-th frame", [1, 2, 3, 5], value=2, key="pv_step")
    frames_idx = list(range(FRAME, FRAME + count, step))
    plot_header("{:.0f} s on the pitch from {}".format(count / float(FPS),
                                                      L.clock(FRAME, SOURCE)),
                "Press ▶ Play above the pitch; drag the slider under it to scrub.", items)
    fig, axes = combined_figure(False, True, flip_y, 600)
    for trace in player_traces(frame_points(tr, frames_idx[0]), numbers, axes):
        fig.add_trace(trace)
    fig.frames = [go.Frame(data=player_traces(frame_points(tr, f), numbers, axes), name=str(f),
                           layout=dict(title=title_for(f, tr)))
                  for f in frames_idx]
    delay = int(1000 * step / FPS)
    button_style = dict(bgcolor="#2F3640", bordercolor="#2F3640", font=dict(color="white",
                                                                              size=13))
    fig.update_layout(
        title=title_for(frames_idx[0], tr),
        updatemenus=[dict(type="buttons", x=0, y=1.12, xanchor="left", yanchor="bottom",
                          direction="left", showactive=False, pad=dict(r=6, t=0),
                          buttons=[dict(label="▶ Play", method="animate",
                                        args=[None, dict(frame=dict(duration=delay, redraw=False),
                                                         transition=dict(duration=0),
                                                         fromcurrent=True)]),
                                   dict(label="⏸ Pause", method="animate",
                                        args=[[None], dict(mode="immediate",
                                                           frame=dict(duration=0,
                                                                      redraw=False))])],
                          **button_style)],
        sliders=[dict(x=0, len=1.0, y=-0.02, yanchor="top", ticklen=0,
                      font=dict(size=1, color="rgba(0,0,0,0)"),
                      currentvalue=dict(prefix="frame ", visible=True, font=dict(size=12)),
                      steps=[dict(method="animate", label=str(f),
                                  args=[[str(f)], dict(mode="immediate",
                                                       frame=dict(duration=0, redraw=False))])
                             for f in frames_idx])],
        margin=dict(l=10, r=10, t=90, b=70))
    show(fig)


# -- Tracking quality --------------------------------------------------------------------------

@st.cache_resource(show_spinner="Computing quality plots…")
def quality_data(source, key):
    tr = get_tracking(source, key)
    d = get_signals(source, key)
    present = np.isfinite(tr["players"][:, :, 0])
    n = len(present) // FPS * FPS
    per_sec = present[:n].reshape(-1, FPS, present.shape[1]).any(axis=1)  # (T, S)
    occupancy = np.where(per_sec, (tr["slot_side"] == L.AWAY).astype(np.int8) + 1, 0).T
    graded = np.isin(d["phase"], L.GRADED_PHASES) if source == "tactical" \
        else np.ones(len(present), dtype=bool)
    gaps = np.array([z - a for a, z in L.runs(~d["ball_seen"] & graded)], dtype=float) / FPS
    sample = np.arange(0, len(present), 5)
    speed = tr["players"][sample, :, 2].ravel()
    speed = speed[np.isfinite(speed)]
    heat = {}
    edges_x, edges_y = np.linspace(0, LEN, 54), np.linspace(0, WID, 35)
    periods = ((L.FIRST, "1st half"), (L.SECOND, "2nd half")) if source == "tactical" \
        else ((None, "all"),)
    for code_phase, name in periods:
        rows = sample if code_phase is None else sample[d["phase"][sample] == code_phase]
        for key, code in (("H", L.HOME), ("A", L.AWAY)):
            pts = tr["players"][rows][:, tr["slot_side"] == code, :2].reshape(-1, 2)
            pts = pts[np.isfinite(pts[:, 0])]
            heat[(name, key)] = np.histogram2d(pts[:, 0] * LEN, pts[:, 1] * WID,
                                               [edges_x, edges_y])[0].T
        b = tr["ball"][rows, :2]
        b = b[np.isfinite(b[:, 0])]
        heat[(name, "ball")] = np.histogram2d(b[:, 0] * LEN, b[:, 1] * WID,
                                              [edges_x, edges_y])[0].T
    return {"occupancy": occupancy, "gaps": gaps, "speed": speed, "heat": heat,
            "edges": (edges_x, edges_y), "n_present": d["n_present"], "phase": d["phase"]}


def quality_tab():
    if not HAS_TRACKING:
        st.info("Needs the tracking files for this source.")
        return
    q = quality_data(SOURCE, KEYS["atd_raw"])

    left, right = st.columns(2)
    with left:
        plot_header("Ball gaps on graded frames: how long the ball stays lost",
                    "One bar per length bucket; log scale. Short gaps can be bridged by "
                    "interpolation, long ones need another signal.",
                    [("#5B6B7A", "number of gaps", "box")])
        fig = go.Figure(go.Histogram(x=q["gaps"], nbinsx=80, marker_color="#5B6B7A"))
        fig.update_layout(yaxis_type="log", height=300, margin=dict(t=10, b=10),
                          xaxis_title="gap length (s)", yaxis_title="gaps")
        show(fig)
    with right:
        phases = list(enumerate(L.PHASES)) if SOURCE == "tactical" else [(-1, "all")]
        plot_header("Tracked objects per frame, by match phase",
                    "Share of each phase's frames with that many objects. During play expect "
                    "~20–22; more than 22 (dashed line) means duplicates or non-players.",
                    [(PHASE_COLORS[name], name, "box") for _, name in phases]
                    + [("#888", "22 players", "dash")])
        fig = go.Figure()
        for p, name in phases:
            vals = q["n_present"][q["phase"] == p]
            if len(vals):
                counts = np.bincount(vals)
                fig.add_bar(x=np.arange(len(counts)), y=counts / counts.sum(), name=name,
                            marker_color=PHASE_COLORS[name], showlegend=False)
        fig.add_vline(x=22, line_dash="dash", line_color="#888")
        fig.update_layout(height=300, margin=dict(t=10, b=10), barmode="group",
                          xaxis_title="objects in the frame", yaxis_title="share of frames")
        show(fig)

    occ = q["occupancy"]
    plot_header("Slot occupancy: when each tracking slot is in use",
                "One row per slot, one column per second. A slot is a tracklet, not a person: "
                "look at how long slots live, how many are in use at once, and whether any is "
                "used across halftime.",
                [(COLORS["H"], "slot assigned to " + NAMES["H"], "box"),
                 (COLORS["A"], "slot assigned to " + NAMES["A"], "box"),
                 ("#E5E8EA", "slot empty", "box")] + time_legend_items())
    fig = go.Figure(go.Heatmap(
        z=occ, x=np.arange(occ.shape[1]) * FPS, y=["slot {}".format(i + 1)
                                                   for i in range(occ.shape[0])],
        colorscale=[[0, "#F4F6F7"], [0.33, "#F4F6F7"], [0.34, COLORS["H"]], [0.66, COLORS["H"]],
                    [0.67, COLORS["A"]], [1, COLORS["A"]]], zmin=0, zmax=2, showscale=False,
        hovertemplate="%{y}<br>frame %{x}<extra></extra>"))
    fig.update_layout(height=max(400, 13 * occ.shape[0]), margin=dict(t=10, b=10))
    fig.update_yaxes(autorange="reversed")
    shade_halves(fig, 0, N)
    mark_selected(fig)
    frame_ticks(fig, 0, N)
    show(fig)

    left, right = st.columns(2)
    with left:
        plot_header("Provided player speed",
                    "Every 5th frame, clipped at 15. Players rarely exceed ~10 m/s; a median "
                    "that looks like km/h would mean the units differ from DATA.md.",
                    [("#5B6B7A", "player-frames", "box")])
        fig = go.Figure(go.Histogram(x=np.clip(q["speed"], 0, 15), nbinsx=75,
                                     marker_color="#5B6B7A"))
        fig.update_layout(height=300, margin=dict(t=10, b=10), xaxis_title="speed (m/s)")
        show(fig)

    edges_x, edges_y = q["edges"]
    st.markdown("---")
    flip = st.toggle("heatmaps: y = 0 at the top", value=True, key="heat_flip_y")
    plot_header("Where each team and the ball spend time, per half",
                "Every 5th frame, 2 m cells; brighter = more time there. If the teams switch "
                "ends at halftime, the two rows mirror each other.")
    keys = sorted({k[0] for k in q["heat"]}, key=lambda s: (s != "1st half", s))
    for period in keys:
        cols = st.columns(3)
        for col, what in zip(cols, ("H", "A", "ball")):
            z = q["heat"][(period, what)]
            fig = go.Figure(go.Heatmap(z=z, x=(edges_x[:-1] + edges_x[1:]) / 2,
                                       y=(edges_y[:-1] + edges_y[1:]) / 2,
                                       colorscale="Viridis", showscale=False))
            fig.update_layout(shapes=pitch_shapes(filled=False), height=250,
                              margin=dict(t=10, b=10, l=10, r=10))
            fig.update_yaxes(autorange="reversed" if flip else True, scaleanchor="x")
            with col:
                st.markdown("{} · **{}**".format(period, NAMES.get(what, "ball")))
                show(fig)


# -- Smart Tagging -----------------------------------------------------------------------------

def tagging_tab():
    if not HAS_TAGGING:
        st.info("Smart Tagging exists for the tactical recording only.")
        return
    table, mapping = get_tagging(KEYS["smart_tagging"])
    st.caption("Team tag → side: {}. The DEFENDING… and DEFENSIVE TRANSITION codes name the "
               "team without the ball; `possession_side` flips them.".format(
                   ", ".join("{} → {}".format(k, v) for k, v in mapping.items())))
    c = st.columns(3)
    codes = c[0].multiselect("Codes", sorted(table["code"].unique()), key="tag_codes")
    sides = c[1].multiselect("Team tag side", sorted(table["side"].unique()), key="tag_sides")
    halves = c[2].multiselect("Half", sorted(table["Half"].dropna().unique())
                              if "Half" in table else [], key="tag_halves")
    view = table
    if codes:
        view = view[view["code"].isin(codes)]
    if sides:
        view = view[view["side"].isin(sides)]
    if halves:
        view = view[view["Half"].isin(halves)]

    shown = view.assign(start=view["start_frame"].map(lambda f: L.clock(int(f), SOURCE)),
                        duration_s=view["duration_s"].round(1))
    columns = [col for col in ["id", "code", "side", "possession_side", "Team", "Type", "Half",
                               "start", "duration_s", "start_frame", "end_frame"]
               if col in shown.columns]
    st.markdown("**Tagged intervals** — click a row, then use the button to jump to it")
    event = st.dataframe(shown[columns], hide_index=True, width="stretch", height=300,
                         on_select="rerun", selection_mode="single-row", key="tag_table")
    selected = event.selection.rows if event is not None else []
    if selected:
        row = shown.iloc[selected[0]]
        st.button("Go to frame {} ({}) — then open the Pitch + video tab".format(
            int(row["start_frame"]), row["start"]), on_click=set_frame,
            args=(int(row["start_frame"]),), key="tag_goto")

    left, right = st.columns(2)
    with left:
        plot_header("Instances per code, by the raw Team tag", None,
                    [(COLORS[s], LABEL_NAMES[s] if s not in NAMES else NAMES[s], "box")
                     for s in ("H", "A", "N/A", "?")])
        counts = view.groupby(["code", "side"]).size().unstack(fill_value=0)
        fig = go.Figure()
        for side in counts.columns:
            fig.add_bar(y=counts.index, x=counts[side], orientation="h", showlegend=False,
                        marker_color=COLORS.get(side, "#888"), name=side)
        fig.update_layout(barmode="stack", height=420, margin=dict(t=10, b=10),
                          xaxis_title="instances")
        show(fig)
    with right:
        plot_header("Duration per code", "Box = middle half of the intervals, line = median; "
                                         "log scale.", [("#5B6B7A", "interval length (s)", "box")])
        fig = go.Figure()
        for code in view["code"].unique():
            fig.add_box(x=view.loc[view["code"] == code, "duration_s"], name=code.title(),
                        marker_color="#5B6B7A", boxpoints=False)
        fig.update_layout(height=420, showlegend=False, margin=dict(t=10, b=10),
                          xaxis_type="log", xaxis_title="seconds")
        show(fig)

    if not HAS_TRACKING or not len(view):
        return
    d = get_signals(SOURCE, KEYS["atd_raw"])
    st.markdown("**Does the Team tag agree with the tracking?**")
    st.caption("Over the frames inside each code's intervals: how often the ball is tracked, and "
               "how often the tracked player nearest to it is on the side the tag says has the "
               "ball. High = a usable possession prior; ~50% = it says little. The nearest "
               "player is itself a noisy proxy, so compare codes rather than read absolute values.")
    st.dataframe(L.tag_agreement(view, d["nearest_side"]), hide_index=True, width="stretch")

    st.markdown("---")
    c = st.columns(3)
    all_codes = sorted(table["code"].unique())
    code = c[0].selectbox("Code", all_codes, key="profile_code",
                          index=all_codes.index("SET PIECES") if "SET PIECES" in all_codes else 0)
    anchor = c[1].radio("Align on", ["start", "end"], horizontal=True, key="profile_anchor")
    types = sorted(table.loc[table["code"] == code, "Type"].dropna().unique()) \
        if "Type" in table else []
    type_pick = c[2].multiselect("Type", types, key="profile_types")
    sel = table[table["code"] == code]
    if type_pick:
        sel = sel[sel["Type"].isin(type_pick)]
    graded = np.isin(d["phase"], L.GRADED_PHASES)
    frames = [int(f) for f in sel["start_frame" if anchor == "start" else "end_frame"]
              if 0 <= f < N and graded[int(f)]]
    offsets, speed = L.event_profile(frames, d["all_speed"])
    _, ball = L.event_profile(frames, d["ball_seen"].astype(float))
    plot_header(
        "What happens around a tag: {} {} intervals in graded play, aligned on their {}".format(
            len(frames), code.title(), anchor),
        "An average over all the selected intervals, with 0 at the tag's {}. For set pieces, the "
        "dip in speed is the stoppage before the restart; where it sits relative to 0 shows "
        "where the restart is inside the tagged interval. Split by Type: throw-ins, corners and "
        "kick-offs behave differently.".format(anchor),
        [("#5B3FD1", "mean player speed (m/s, left axis)", "line"),
         ("#E67E22", "ball visible (%, right axis)", "line"),
         ("#888", "the tag's {}".format(anchor), "dash")])
    if not len(offsets):
        st.info("No intervals of this kind in graded play.")
        return
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_scatter(x=offsets, y=speed, line_color="#5B3FD1", showlegend=False,
                    name="speed", secondary_y=False)
    fig.add_scatter(x=offsets, y=100 * ball, line_color="#E67E22", showlegend=False,
                    name="ball visible", secondary_y=True)
    fig.add_vline(x=0, line_dash="dash", line_color="#888")
    fig.update_layout(height=340, margin=dict(t=10, b=10),
                      xaxis_title="seconds from the tag's " + anchor)
    fig.update_yaxes(title_text="m/s", secondary_y=False)
    fig.update_yaxes(title_text="%", range=[0, 100], secondary_y=True)
    show(fig)


# -- Guide -------------------------------------------------------------------------------------

def guide_tab():
    path = os.path.join(HERE, "README.md")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            st.markdown(handle.read())


for tab, fn in zip(tabs, (overview_tab, timeline_tab, pitch_video_tab, quality_tab, tagging_tab,
                          guide_tab)):
    with tab:
        guarded(fn)
