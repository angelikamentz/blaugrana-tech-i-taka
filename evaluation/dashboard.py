"""The scoreboard, drawn: an interactive companion to run_evaluation.py.

    pip install -r requirements-dashboard.txt
    streamlit run evaluation/dashboard.py

Every number on these pages comes from possession_metrics.evaluate(), the same function the
official scorer calls, so they match `python evaluation/run_evaluation.py` exactly. This file only
draws them. run_evaluation.py stays the authority and is not modified.

It reads label CSVs only -- the ground truth in evaluation/gt/ and submissions in
evaluation/results/, plus any you upload -- and never the match data in data/.
"""
import glob
import hashlib
import io
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import possession_metrics as pm  # noqa: E402
import problems as pr  # noqa: E402
from games import GAME, half_video_blocks  # noqa: E402

GT_DIR = os.path.join(HERE, "gt")
RESULTS_DIR = os.path.join(HERE, "results")
FPS = pm.FPS
GT_ROW = "Ground truth"

COLORS = {"H": "#2F6BD8", "A": "#D64541", "D": "#A3A8AE", "X": "#E9ECEF", "wrong": "#222222"}
NAMES = {"H": "Home in possession", "A": "Away in possession", "D": "Dead ball",
         "X": "No label (X)", "wrong": "Frame label wrong"}
GOOD, BAD, EXTRA = "#1E9E5A", "#E0453A", "#F39C12"

# Full-match scores of the four baselines, from EVALUATION.md. They are scored on the private
# ground truth, which nobody but the organisers holds, so they are quoted rather than computed.
FULL_MATCH = {  # name: (window in frames, macroF1, spellF1)
    "baseline_raw": (0, 0.6532, 0.0486),
    "baseline_smooth25": (25, 0.6752, 0.1996),
    "baseline_smooth50": (50, 0.6724, 0.2433),
    "baseline_smooth125": (125, 0.6356, 0.2242),
}
# Below this many true possessions spellF1 is too noisy to rank on (run_evaluation.py uses 100).
FEW_POSSESSIONS = 100
SWEEP_WINDOWS = (0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150, 175, 200, 250)

HALVES = half_video_blocks(GAME)


# -- small helpers -----------------------------------------------------------------------------

def clock(frame):
    """'1H 03:12' -- time since the start of the graded half, which reads like a match clock."""
    for half, first, last in HALVES:
        if first <= frame <= last + 1:
            seconds = (frame - first) / float(FPS)
            return "{}H {:02d}:{:02d}".format(half, int(seconds // 60), int(seconds % 60))
    return "not graded"


def rle(codes, lo, hi):
    """[(start, stop, code)] constant runs of codes[lo:hi], half-open, in absolute frames."""
    window = codes[lo:hi]
    if not len(window):
        return []
    breaks = np.flatnonzero(window[1:] != window[:-1]) + 1
    starts = np.concatenate(([0], breaks))
    stops = np.concatenate((breaks, [len(window)]))
    return [(lo + int(a), lo + int(z), int(window[a])) for a, z in zip(starts, stops)]


def letters_to_codes(letters):
    """Vectorised label_codes for arrays this file builds itself (already clean H/A/D/X)."""
    codes = np.full(len(letters), -1, dtype=np.int8)
    for index, name in enumerate(pm.LABELS):
        codes[letters == name] = index
    return codes


def greedy_pairs(overlaps, threshold):
    """matched_gt_flags' greedy loop, also keeping which prediction each truth spell went to."""
    pairs, used_gt, used_pred = [], set(), set()
    for iou, gt_index, pred_index in overlaps:
        if iou < threshold:
            break
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index)
        used_pred.add(pred_index)
        pairs.append((gt_index, pred_index, iou))
    return pairs


def frame_axis(fig, lo, hi, n_ticks=8):
    """Frames on the x axis, labelled with the match clock."""
    ticks = np.linspace(lo, hi, n_ticks).astype(int).tolist()
    fig.update_xaxes(range=[lo, hi], tickvals=ticks, ticktext=[clock(t) for t in ticks],
                     title_text="match clock (hover for exact frames)")


def show(fig):
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


# -- loading and scoring (cached: label_codes is a Python loop over 199k rows) -----------------

@st.cache_data(show_spinner=False)
def load_gt(path, mtime):
    return pm.label_codes(pm.read_label_csv(path))


@st.cache_data(show_spinner="Scoring…")
def score_csv(data, gt_key, _gt_codes):
    """(codes, result, reason) -- exactly run_evaluation.score_submission, from bytes."""
    try:
        table = pm.read_label_csv(io.BytesIO(data))
    except Exception as error:
        return None, None, "could not read: {}".format(error)
    reason = pm.validate_submission(table, len(_gt_codes))
    if reason:
        return None, None, reason
    codes = pm.label_codes(table)
    return codes, pm.evaluate(_gt_codes, codes), None


@st.cache_data(show_spinner=False)
def problems_for(codes_digest, gt_key, _gt_codes, _codes):
    return pr.find_problems(_gt_codes, _codes)


def problem_color(kind):
    if kind.startswith(("Late", "Early")):
        return EXTRA
    if kind.startswith("Flicker"):
        return "#8E44AD"
    if kind.startswith("Missed whole"):
        return BAD
    return COLORS["X"] if kind.startswith("No answer") else "#5B6B7A"


@st.cache_data(show_spinner="Re-smoothing the raw baseline…")
def smoothing_sweep(raw_data, gt_key, _gt_codes, windows):
    """The baseline's own smooth() at each window, scored. Returns (table, raw team codes)."""
    baseline = import_baseline()
    letters = pd.read_csv(io.BytesIO(raw_data), dtype={"label": "object"})["label"].to_numpy()
    raw = np.where(letters == "H", baseline.HOME,
                   np.where(letters == "A", baseline.AWAY, baseline.NOBODY)).astype(np.int8)
    rows = []
    for window in windows:
        result = pm.evaluate(_gt_codes, letters_to_codes(baseline.smooth(raw, window)))
        rows.append({"window": window, "seconds": window / float(FPS),
                     "spellF1": result["spell_f1_mean"], "macroF1": result["macro_f1"],
                     "possessions": result["n_pred_spells"]})
    return pd.DataFrame(rows), raw


def import_baseline():
    sys.path.insert(0, os.path.join(ROOT, "starter"))
    import baseline
    return baseline


# -- figures -----------------------------------------------------------------------------------

def strip_figure(rows, lo, hi):
    """One horizontal strip per row. rows: [(name, [(start, stop, key)])], key in COLORS."""
    traces = {}
    for name, segs in rows:
        for start, stop, key in segs:
            a, z = max(start, lo), min(stop, hi)
            if z <= a:
                continue
            t = traces.setdefault(key, {"x": [], "base": [], "y": [], "text": []})
            t["x"].append(z - a)
            t["base"].append(a)
            t["y"].append(name)
            t["text"].append("<b>{}</b><br>{}<br>{} → {}  ({:.1f} s)<br>frames {}–{}".format(
                name, NAMES[key], clock(start), clock(stop), (stop - start) / float(FPS),
                start, stop - 1))
    fig = go.Figure()
    for key in ("H", "A", "D", "X", "wrong"):
        if key in traces:
            t = traces[key]
            fig.add_bar(orientation="h", name=NAMES[key], marker_color=COLORS[key],
                        x=t["x"], base=t["base"], y=t["y"], hovertext=t["text"],
                        hoverinfo="text", marker_line_width=0)
    fig.update_layout(barmode="overlay", bargap=0.25, height=90 + 46 * len(rows),
                      margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.08, x=0))
    fig.update_yaxes(categoryorder="array", categoryarray=[r[0] for r in rows],
                     autorange="reversed")
    frame_axis(fig, lo, hi)
    return fig


def label_rows(name, codes, lo, hi):
    return [(name, [(a, z, pm.LABELS[k]) for a, z, k in rle(codes, lo, hi)])]


def error_row(name, gt, codes, lo, hi):
    wrong = ((gt != pm.X) & (codes != gt)).astype(np.int8)
    return [(name, [(a, z, "wrong") for a, z, k in rle(wrong, lo, hi) if k])]


# -- page --------------------------------------------------------------------------------------

st.set_page_config(page_title="Possession scoreboard", page_icon="⚽", layout="wide")
st.title("Possession scoreboard")
st.caption("The numbers from `python evaluation/run_evaluation.py`, drawn. Same metric code, "
           "same results — this page only visualises them.")

with st.sidebar:
    st.header("Inputs")
    gt_files = sorted(glob.glob(os.path.join(GT_DIR, "*.csv")))
    if not gt_files:
        st.error("No ground truth found in evaluation/gt/.")
        st.stop()
    gt_path = st.selectbox("Ground truth", gt_files, format_func=os.path.basename)
    gt_codes = load_gt(gt_path, os.path.getmtime(gt_path))
    gt_key = "{}:{}".format(gt_path, os.path.getmtime(gt_path))
    runs = pm.valid_runs(gt_codes)

    result_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.csv")))
    names = [os.path.splitext(os.path.basename(p))[0] for p in result_files]
    chosen = st.multiselect("Submissions from evaluation/results/", names, default=names)
    uploads = st.file_uploader("…or upload more CSVs", type="csv", accept_multiple_files=True)

sources = [(n, open(p, "rb").read()) for n, p in zip(names, result_files) if n in chosen]
sources += [(os.path.splitext(u.name)[0], u.getvalue()) for u in uploads or []]

subs, skipped = [], []
for name, data in sources:
    codes, result, reason = score_csv(data, gt_key, gt_codes)
    if reason:
        skipped.append((name, reason))
    else:
        subs.append({"name": name, "codes": codes, "result": result,
                     "filled": pm.fill_abstentions(codes, runs)})
subs.sort(key=lambda s: -s["result"]["spell_f1_mean"])

for name, reason in skipped:
    st.sidebar.warning("**{}** skipped: {}".format(name, reason))
if not subs:
    st.info("Pick or upload at least one valid submission.")
    st.stop()

graded_lo, graded_hi = runs[0][0], runs[-1][1]
N_FRAMES = len(gt_codes)
DEFAULT_ZOOM = (graded_lo, min(graded_hi, graded_lo + 7500))
if not st.session_state.get("zoom") or st.session_state.zoom[1] > N_FRAMES:
    st.session_state.zoom = DEFAULT_ZOOM


def zoom_to(start, stop):
    """Point the zoom slider at [start, stop) with some context either side, on 1 s steps."""
    pad = max(5 * FPS, (stop - start) // 2)
    st.session_state.zoom = (max(0, (start - pad) // FPS * FPS),
                             min(N_FRAMES, -(-(stop + pad) // FPS) * FPS))


def zoom_to_selected_problem():
    """Callback: runs before the rerun, so it may still set the slider's state."""
    rows = st.session_state.problem_table.selection.rows
    view = st.session_state.get("problem_view")
    if rows and view is not None and rows[0] < len(view):
        row = view.iloc[rows[0]]
        zoom_to(int(row.start_frame), int(row.end_frame) + 1)


def reset_zoom():
    st.session_state.zoom = DEFAULT_ZOOM


with st.sidebar:
    st.header("Zoom")
    lo, hi = st.slider("Frames shown on timelines", 0, N_FRAMES, step=25, key="zoom",
                       help="Defaults to the graded part of the ground truth. 25 frames = 1 s. "
                            "Clicking a row on the Problems tab moves it.")
    st.button("Reset zoom", on_click=reset_zoom)
    st.caption("{} → {}  ({:.0f} s)".format(clock(lo), clock(hi), (hi - lo) / float(FPS)))

first = subs[0]["result"]
n_gt_poss, gt_dur = first["n_gt_spells"], first["gt_mean_spell_duration_s"]
by_name = {s["name"]: s for s in subs}

tabs = st.tabs(["Overview", "Timeline", "Problems", "Possessions (spellF1)", "Frames (macroF1)",
                "Baselines: the smoothing window"])

# -- Overview ----------------------------------------------------------------------------------
with tabs[0]:
    c = st.columns(4)
    c[0].metric("Graded frames", "{:,}".format(first["n_graded"]),
                help="Frames where the ground truth is not X. Everything else is ignored.")
    c[1].metric("True possessions", n_gt_poss)
    c[2].metric("Mean true possession", "{:.1f} s".format(gt_dur))
    c[3].metric("True home share", "{:.0%}".format(first["gt_home_share"]))

    if n_gt_poss < FEW_POSSESSIONS:
        st.warning(
            "This ground truth has only **{} possessions**, so one possession found or missed "
            "moves spellF1 by several points. Use it to check that your pipeline is right, "
            "**not to choose between versions** — the rules forbid tuning on it, and the last "
            "tab shows how it picks the wrong smoothing window.".format(n_gt_poss))

    spell_vals = [s["result"]["spell_f1_mean"] for s in subs]
    macro_vals = [s["result"]["macro_f1"] for s in subs]
    spell_rank = [1 + sum(v > x for v in spell_vals) for x in spell_vals]
    macro_rank = [1 + sum(v > x for v in macro_vals) for x in macro_vals]
    loose = max(pm.TAUS)
    board = pd.DataFrame([{
        "Rank": spell_rank[i],
        "Submission": s["name"],
        "spellF1 (ranked)": s["result"]["spell_f1_mean"],
        "macroF1": s["result"]["macro_f1"],
        "macroF1 rank": macro_rank[i],
        "Possessions": s["result"]["n_pred_spells"],
        "Mean possession (s)": round(s["result"]["mean_spell_duration_s"], 1),
        "Label-change F1 (±1 s)": s["result"]["transitions"][loose]["f1"],
        "Segments vs truth": round(s["result"]["fragmentation"], 2),
        "Home share error": round(s["result"]["share_err"], 3),
    } for i, s in enumerate(subs)])
    st.dataframe(board, hide_index=True, width="stretch", column_config={
        "spellF1 (ranked)": st.column_config.ProgressColumn(format="%.4f", min_value=0,
                                                            max_value=1),
        "macroF1": st.column_config.ProgressColumn(format="%.4f", min_value=0, max_value=1),
        "Label-change F1 (±1 s)": st.column_config.NumberColumn(format="%.4f"),
        "Possessions": st.column_config.NumberColumn(
            help="Predicted possessions. The truth has {}.".format(n_gt_poss)),
        "Mean possession (s)": st.column_config.NumberColumn(
            help="The truth's mean is {:.1f} s.".format(gt_dur)),
        "Segments vs truth": st.column_config.NumberColumn(
            help="Predicted segments (possessions + stoppages) / true segments. 1.0 is right; "
                 "above 1 means the prediction is chopped up."),
    })

    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_bar(name="spellF1 (ranked)", x=board["Submission"], y=board["spellF1 (ranked)"],
                    marker_color="#5B3FD1", text=board["spellF1 (ranked)"].round(3))
        fig.add_bar(name="macroF1", x=board["Submission"], y=board["macroF1"],
                    marker_color="#9AA6B2", text=board["macroF1"].round(3))
        fig.update_layout(title="The two scores disagree", barmode="group", height=360,
                          yaxis_range=[0, 1], margin=dict(t=50, b=10),
                          legend=dict(orientation="h", y=-0.15))
        show(fig)
    with right:
        fig = go.Figure()
        fig.add_bar(x=board["Submission"], y=board["Possessions"], marker_color="#5B3FD1",
                    text=board["Possessions"], name="predicted")
        fig.add_hline(y=n_gt_poss, line_dash="dash", line_color=GOOD,
                      annotation_text="truth: {}".format(n_gt_poss))
        fig.update_layout(title="How many possessions each one reports", height=360,
                          margin=dict(t=50, b=10), yaxis_type="log",
                          yaxis_title="possessions (log scale)")
        show(fig)

    with st.expander("What the columns mean", expanded=True):
        st.markdown("""
- **spellF1** — *did you find the real possessions?* Each possession (an unbroken run of H, or of A)
  is one object. A predicted possession counts as found if it is the same team and overlaps a true
  one enough (IoU ≥ threshold). Scored at 0.3–0.7 and averaged. **This decides the ranking.**
  See the *Possessions* tab.
- **macroF1** — *is each frame's label right?* F1 for H, A and D separately, averaged. Barely notices
  a possession being chopped into ten pieces, because most frames are still right. See *Frames*.
- **Possessions / Mean possession** — the plainest symptom. Too many short possessions = flicker.
- **Label-change F1 (±1 s)** — were the moments the label changes (handovers, stoppages,
  restarts) predicted within one second?
- **Segments vs truth** — how chopped up the prediction is compared with the truth (1.0 = same).
- **Home share error** — difference in % of live frames given to home. Easy to get right even
  while flickering, so it is never ranked on.
""")
    with st.expander("What are the four baselines?"):
        st.markdown("""
They are **one algorithm with one knob**:

1. On each frame, find the tracked player nearest the ball and take their team (`H`/`A`). If the
   ball isn't tracked on that frame, write `D`. So the baselines' **D means "ball not seen", not
   "dead ball"** — which is why their D scores are poor.
2. Majority vote over a window of ±0 (`raw`), ±25, ±50 or ±125 frames (±1, 2, 5 s). Output `D`
   only if the ball wasn't seen anywhere in the window.

A wider window → fewer, longer possessions → spellF1 goes up, until the window starts swallowing
real short possessions and handovers drift late. The last tab lets you slide the knob.
""")

# -- Timeline ----------------------------------------------------------------------------------
with tabs[1]:
    st.markdown("Each strip is one submission's labels over time, under the truth. Hover any "
                "block for its times. Zoom with the sidebar slider.")
    show_errors = st.toggle("Add a strip under each submission marking the frames it gets wrong",
                            value=True)
    rows = label_rows(GT_ROW, gt_codes, lo, hi)
    for s in subs:
        rows += label_rows(s["name"], s["codes"], lo, hi)
        if show_errors:
            rows += error_row("   ✗ " + s["name"], gt_codes, s["codes"], lo, hi)
    show(strip_figure(rows, lo, hi))
    st.caption("Look for: grey (D) where the truth is coloured — the baselines call 'ball not "
               "seen' dead; confetti of short blocks — flicker that spellF1 punishes; colour "
               "changes that sit to the right of the truth's — late handovers.")

# -- Problems ---------------------------------------------------------------------------------
with tabs[2]:
    st.markdown("Every stretch where the submission disagrees with the truth, named by the kind "
                "of mistake. **Click a row** to zoom every timeline to it, and type its video "
                "time into a player on the tactical video to watch what happened.")
    name = st.selectbox("Submission", list(by_name), key="problem_sub")
    s = by_name[name]
    found = problems_for(hashlib.md5(s["codes"].tobytes()).hexdigest(), gt_key, gt_codes,
                         s["codes"])
    summary = pr.summary(found)

    left, right = st.columns([3, 2])
    with left:
        fig = go.Figure(go.Bar(
            orientation="h", y=summary["problem"], x=summary["seconds"],
            marker_color=[problem_color(p) for p in summary["problem"]],
            text=["{:.1f} s · {}×".format(t, n) for t, n in zip(summary["seconds"],
                                                                 summary["count"])],
            textposition="auto", hovertext=summary["meaning"], hoverinfo="text+y"))
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(title="Where the {:.0f} wrong seconds go".format(
            found["duration_s"].sum()), xaxis_title="seconds wrong", height=110 + 28 * len(
                summary), margin=dict(t=50, b=10, l=10, r=10))
        show(fig)
    with right:
        st.markdown("**Filter the list**")
        kinds = st.multiselect("Kinds", list(summary["problem"]), default=list(summary["problem"]),
                               key="problem_kinds_" + name)
        min_s = st.slider("Hide problems shorter than (s)", 0.0, 3.0, 0.0, 0.1,
                          key="problem_min_s")
        order = st.radio("Order", ["In time order", "Longest first"], horizontal=True,
                         key="problem_order")
        with st.expander("What the kinds mean"):
            st.markdown("\n".join("- **{}** — {}".format(k, v)
                                  for k, v in pr.CATEGORIES.items()))

    view = found[found["problem"].isin(kinds) & (found["duration_s"] >= min_s)]
    if order == "Longest first":
        view = view.sort_values("duration_s", ascending=False)
    view = view.reset_index(drop=True)
    st.session_state.problem_view = view
    shown = pd.DataFrame({
        "Video time": view["video_time"],
        "Match clock": [clock(f) for f in view["start_frame"]],
        "Length (s)": view["duration_s"],
        "Problem": view["problem"],
        "Truth": view["truth"],
        "Said": view["said"],
        "Frames": ["{}–{}".format(a, z) for a, z in zip(view["start_frame"], view["end_frame"])],
    })
    st.caption("{} of {} problems shown".format(len(view), len(found)))
    picked = st.dataframe(shown, key="problem_table", on_select=zoom_to_selected_problem,
                          selection_mode="single-row", hide_index=True, width="stretch",
                          height=min(420, 38 + 35 * len(shown)))
    st.download_button("Download this list (CSV)", view.to_csv(index=False),
                       file_name="{}_problems.csv".format(name), mime="text/csv")

    rows = (label_rows(GT_ROW, gt_codes, lo, hi) + label_rows(name, s["codes"], lo, hi)
            + error_row("   ✗ wrong", gt_codes, s["codes"], lo, hi))
    fig = strip_figure(rows, lo, hi)
    selected = picked.selection.rows if picked is not None else []
    if selected and selected[0] < len(view):
        row = view.iloc[selected[0]]
        fig.add_vrect(x0=int(row.start_frame), x1=int(row.end_frame) + 1, fillcolor="#FFD43B",
                      opacity=0.35, line_width=0, annotation_text=row.problem,
                      annotation_position="top left")
    show(fig)

# -- Possessions -------------------------------------------------------------------------------
with tabs[3]:
    st.markdown("""
**How spellF1 works.** Take the possessions only (stoppages ignored). Pair each predicted
possession with a true one of the **same team**, one-to-one, best overlap first. Overlap is
**IoU** = shared time ÷ combined time — a prediction that covers the true possession exactly
scores 1.0, one that is half as long scores 0.5. A pair counts if IoU ≥ threshold. Then
precision = found ÷ predicted, recall = found ÷ true, F1 combines them. Repeat at thresholds
0.3, 0.4, 0.5, 0.6, 0.7 and average.
""")
    c1, c2 = st.columns([2, 1])
    name = c1.selectbox("Submission", list(by_name), key="spell_sub")
    threshold = c2.select_slider("IoU threshold", options=list(pm.SPELL_IOUS),
                                 value=pm.SPELL_IOU_HEADLINE)
    s = by_name[name]
    gt_spells = pm.possession_spells(gt_codes, runs)
    pred_spells = pm.possession_spells(s["filled"], runs)
    overlaps = pm.spell_overlaps(gt_spells, pred_spells)
    pairs = greedy_pairs(overlaps, threshold)
    per = s["result"]["spell_f1"][threshold]
    assert len(pairs) == per["n_matched"], "dashboard matching disagrees with the scorer"

    c = st.columns(5)
    c[0].metric("Found (matched)", per["n_matched"])
    c[1].metric("Missed true possessions", per["n_missed"])
    c[2].metric("Extra predicted possessions", per["n_spurious"])
    c[3].metric("F1 at {}".format(threshold), "{:.3f}".format(per["f1"]))
    c[4].metric("spellF1 (mean of 5)", "{:.4f}".format(s["result"]["spell_f1_mean"]))

    gt_match = {g: (p, iou) for g, p, iou in pairs}
    pred_match = {p: (g, iou) for g, p, iou in pairs}
    fig = go.Figure()
    for row, codes in ((GT_ROW, gt_codes), (name, s["filled"])):
        dead = [(a, z) for a, z, k in rle(codes, lo, hi) if k == pm.D]
        fig.add_bar(orientation="h", y=[row] * len(dead), base=[a for a, _ in dead],
                    x=[z - a for a, z in dead], marker_color=COLORS["D"], opacity=0.35,
                    name="Dead ball", showlegend=row == GT_ROW, hoverinfo="skip")
    for row, spells, matches, miss_word in ((GT_ROW, gt_spells, gt_match, "missed"),
                                           (name, pred_spells, pred_match, "extra")):
        for team in (pm.H, pm.A):
            for hit in (True, False):
                sel = [(i, sp) for i, sp in enumerate(spells)
                       if sp[2] == team and (i in matches) == hit and sp[1] > lo and sp[0] < hi]
                if not sel:
                    continue
                label = pm.LABELS[team]
                fig.add_bar(
                    orientation="h", y=[row] * len(sel), base=[sp[0] for _, sp in sel],
                    x=[sp[1] - sp[0] for _, sp in sel], marker_color=COLORS[label],
                    opacity=1.0 if hit else 0.45,
                    marker_pattern_shape="" if hit else "/",
                    marker_line=dict(color=GOOD if hit else BAD, width=2),
                    text=["IoU {:.2f}".format(matches[i][1]) if hit else miss_word
                          for i, _ in sel],
                    textposition="inside", insidetextanchor="middle",
                    name="{} — {}".format(NAMES[label], "matched" if hit else miss_word),
                    hovertext=["<b>{}</b> possession #{}<br>{} → {}  ({:.1f} s)<br>{}".format(
                        row, i + 1, clock(sp[0]), clock(sp[1]), (sp[1] - sp[0]) / float(FPS),
                        "matched, IoU {:.2f}".format(matches[i][1]) if hit else miss_word)
                        for i, sp in sel], hoverinfo="text")
    xs, ys = [], []
    for g, p, _ in pairs:
        (ga, gz, _), (pa, pz, _) = gt_spells[g], pred_spells[p]
        if gz > lo and ga < hi:
            xs += [(ga + gz) / 2.0, (pa + pz) / 2.0, None]
            ys += [GT_ROW, name, None]
    fig.add_scatter(x=xs, y=ys, mode="lines", line=dict(color=GOOD, width=1.5, dash="dot"),
                    name="matched pair", hoverinfo="skip")
    fig.update_layout(barmode="overlay", height=330, margin=dict(l=10, r=10, t=30, b=10),
                      uniformtext=dict(minsize=9, mode="hide"),
                      legend=dict(orientation="h", y=-0.35))
    fig.update_yaxes(categoryorder="array", categoryarray=[GT_ROW, name], autorange="reversed")
    frame_axis(fig, lo, hi)
    show(fig)
    st.caption("Solid with a green edge = matched (dotted line joins the pair). Hatched with a "
               "red edge = missed true possession or extra predicted one. Change the threshold "
               "to watch loose matches drop out.")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Every true possession**")
        found_at = {t: {g for g, _, _ in greedy_pairs(overlaps, t)} for t in pm.SPELL_IOUS}
        best = {}
        for iou, g, _ in overlaps:
            best[g] = max(best.get(g, 0.0), iou)
        st.dataframe(pd.DataFrame([{
            "#": i + 1,
            "Team": NAMES[pm.LABELS[k]].split()[0],
            "Starts": clock(a),
            "Length (s)": round((b - a) / float(FPS), 1),
            "Best overlap (IoU)": round(best.get(i, 0.0), 2),
            "Found at thresholds": " ".join("{:.1f}".format(t) for t in pm.SPELL_IOUS
                                            if i in found_at[t]) or "never",
        } for i, (a, b, k) in enumerate(gt_spells)]), hide_index=True, width="stretch")
    with right:
        fig = go.Figure()
        for t in subs:
            fig.add_scatter(x=list(pm.SPELL_IOUS),
                            y=[t["result"]["spell_f1"][v]["f1"] for v in pm.SPELL_IOUS],
                            mode="lines+markers", name="{} (mean {:.3f})".format(
                                t["name"], t["result"]["spell_f1_mean"]))
        fig.update_layout(title="F1 at each threshold — spellF1 is their average",
                          xaxis_title="IoU threshold", yaxis_title="F1", yaxis_range=[0, 1],
                          height=380, margin=dict(t=50, b=10), legend=dict(y=-0.3,
                                                                          orientation="h"))
        show(fig)

# -- Frames ------------------------------------------------------------------------------------
with tabs[4]:
    st.markdown("""
**Confusion matrix**: rows are what the truth says, columns what the submission said, over graded
frames only. Each row is coloured by its share, so the diagonal is the recall of that class. The
`X` column is frames the submission declined to label — macroF1 counts those as wrong.
""")
    picked = st.multiselect("Submissions", list(by_name), default=list(by_name)[:4],
                            key="cm_subs")
    cols = st.columns(2)
    for i, name in enumerate(picked):
        cm3 = pm.scored_confusion(np.array(by_name[name]["result"]["confusion"]))
        share = cm3 / np.maximum(cm3.sum(axis=1, keepdims=True), 1)
        text = [["{:,}<br>{:.0%}".format(cm3[r, c], share[r, c]) for c in range(4)]
                for r in range(3)]
        fig = go.Figure(go.Heatmap(
            z=share, x=["said " + l for l in pm.LABELS], y=["truth " + l for l in pm.SCORED],
            text=text, texttemplate="%{text}", colorscale="Blues", zmin=0, zmax=1,
            showscale=False, hovertemplate="%{y}, %{x}<br>%{text}<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(title="{} — macroF1 {:.4f}".format(
            name, by_name[name]["result"]["macro_f1"]), height=330,
            margin=dict(t=50, b=10, l=10, r=10))
        with cols[i % 2]:
            show(fig)

    st.info("The baselines' **D recall is low**: they say D only when the ball is not tracked "
            "at all, and smoothing fills those gaps with a team — so most true stoppages get a "
            "team label. Spotting real stoppages is one of the biggest openings.")

# -- Smoothing window --------------------------------------------------------------------------
with tabs[5]:
    raw_path = os.path.join(RESULTS_DIR, "baseline_raw.csv")
    if not os.path.isfile(raw_path):
        st.info("Needs evaluation/results/baseline_raw.csv (run `python starter/baseline.py`).")
    else:
        sweep, raw_team = smoothing_sweep(open(raw_path, "rb").read(), gt_key, gt_codes,
                                          SWEEP_WINDOWS)
        st.markdown("""
The four baselines are `baseline_raw` smoothed with a majority vote over ±window frames — this
recomputes that with `starter/baseline.py`'s own `smooth()` at many window sizes and scores each
against the ground truth selected in the sidebar. **This is a demonstration of why the public
window cannot be tuned on**, not a tuning tool: the diamonds are the same four baselines scored
on the full match (from EVALUATION.md).
""")
        best_public = sweep.loc[sweep["spellF1"].idxmax()]
        best_full = max(FULL_MATCH.items(), key=lambda kv: kv[1][2])
        fig = go.Figure()
        fig.add_scatter(x=sweep["seconds"], y=sweep["spellF1"], mode="lines+markers",
                        name="spellF1 on this ground truth", line_color="#5B3FD1")
        fig.add_scatter(x=sweep["seconds"], y=sweep["macroF1"], mode="lines+markers",
                        name="macroF1 on this ground truth", line_color="#9AA6B2")
        fig.add_scatter(x=[v[0] / float(FPS) for v in FULL_MATCH.values()],
                        y=[v[2] for v in FULL_MATCH.values()], mode="markers+text",
                        marker=dict(symbol="diamond", size=12, color="#5B3FD1",
                                    line=dict(color="white", width=1)),
                        text=["{:.3f}".format(v[2]) for v in FULL_MATCH.values()],
                        textposition="bottom center", name="spellF1 on the FULL match")
        fig.add_scatter(x=[v[0] / float(FPS) for v in FULL_MATCH.values()],
                        y=[v[1] for v in FULL_MATCH.values()], mode="markers",
                        marker=dict(symbol="diamond", size=12, color="#9AA6B2",
                                    line=dict(color="white", width=1)),
                        name="macroF1 on the FULL match")
        fig.update_layout(height=420, xaxis_title="majority-vote window (± seconds)",
                          yaxis_title="score", yaxis_range=[0, 1], margin=dict(t=30, b=10),
                          legend=dict(orientation="h", y=-0.2))
        show(fig)
        st.warning("On this ground truth the best window is **±{:.1f} s** (spellF1 {:.3f}). On "
                   "the full match, the best of the four published windows is **±{:.0f} s** "
                   "({}, spellF1 {:.3f}). Seventeen possessions cannot pick a winner.".format(
                       best_public["seconds"], best_public["spellF1"], best_full[1][0] / FPS,
                       best_full[0], best_full[1][2]))

        fig = go.Figure()
        fig.add_scatter(x=sweep["seconds"], y=sweep["possessions"], mode="lines+markers",
                        line_color="#5B3FD1", name="predicted")
        fig.add_hline(y=n_gt_poss, line_dash="dash", line_color=GOOD,
                      annotation_text="truth: {}".format(n_gt_poss))
        fig.update_layout(title="Possessions reported vs window", yaxis_type="log",
                          xaxis_title="± seconds", height=300, margin=dict(t=50, b=10))
        show(fig)

        window = st.select_slider("See one window on the timeline (± frames)",
                                  options=list(SWEEP_WINDOWS), value=50)
        baseline = import_baseline()
        smoothed = letters_to_codes(baseline.smooth(raw_team, window))
        rows = (label_rows(GT_ROW, gt_codes, lo, hi)
                + label_rows("raw: nearest player", letters_to_codes(baseline.smooth(raw_team, 0)),
                             lo, hi)
                + label_rows("smoothed ±{}".format(window), smoothed, lo, hi)
                + error_row("   ✗ smoothed ±{}".format(window), gt_codes, smoothed, lo, hi))
        show(strip_figure(rows, lo, hi))
