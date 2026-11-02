# Data inspection

Tools to answer three questions before building anything: **what is in each file, how do the
files line up, and which signals are worth using.** File-by-file facts live in
[`DATA.md`](../DATA.md) and the stage plan in [`__training_pipeline.md`](../__training_pipeline.md).
This page covers how to look at the data and what to look for.

```powershell
pip install -r requirements-inspection.txt
python inspection/inspect_data.py              # text report: aggregates only, safe to share
streamlit run inspection/dashboard.py          # the pictures: for your eyes only
```

Run both in your own terminal. Inside Claude Code, `games.asset_paths()` refuses to run.

| file | role |
|---|---|
| `loaders.py` | One parse of each file. Reuses the starter loaders. The raw tracking is cached as arrays in `<DATA_DIR>/_inspection_cache/`. |
| `inspect_data.py` | Every check, printed as counts, percentages and medians, with DATA.md's published value next to each. |
| `dashboard.py` | Timeline, pitch, video overlay, quality plots and Smart Tagging. |

The first run parses the 199,319-line tracking file, which takes a minute or two. Later runs read
the cache. If the raw file changes, the cache rebuilds itself.

## What comes from where

```
                         tactical video (.mp4)  ── frame i = submission row i
                                  │  Metrica, automatic, nobody corrected anything
        ┌───────────────┬─────────┴────────┬──────────────────────┐
   _atd.xml         _atd.txt          _homography.csv      _smart_tagging.xml
   slot → team      per frame:        per frame:           638 phases:
   pitch size       46 slots x,y,v    3×3 pitch→pixel      code + Team/Type/Half
   line layout      + ball x,y        error, valid         seconds → round(s×25)
   (static)         frame_id − 1      frame                (intervals, overlap)
```

| signal | comes from | usable for possession? |
|---|---|---|
| team of each dot | `_atd.xml` (slot → team) joined to `_atd.txt` | Essential. It's the only team information in the tracking. |
| ball position | `_atd.txt` | The strongest signal, but missing on 41% of graded frames, mostly at the moments that matter. |
| nearest player to the ball | `_atd.txt` (derived) | The baseline's whole method. Noisy frame by frame and flickers without smoothing. |
| player speeds | `_atd.txt` | A dead-ball proxy: everyone slows down or walks during stoppages. |
| team shape and position | `_atd.txt` (derived) | Context: which side attacks, pressing, restarts. Teams switch ends at halftime. |
| slot lifetimes | `_atd.txt` (derived) | No direct use. They show how unstable "players" are. |
| reprojection error | `_homography.csv` | A confidence signal. High means the camera isn't on the pitch (halftime, cutaways). |
| pitch ↔ pixel mapping | `_homography.csv` | Needed for anything that uses the video, and for the overlay check. |
| phases (`BUILD UP`, …) and Team tags | `_smart_tagging.xml` | A prior, not an answer: automatic, overlapping, phases rather than possessions. `N/A` on all throw-ins. |
| set pieces | `_smart_tagging.xml` (`SET PIECES` + `Type`) | Restarts, so a dead ball just before. A strong hint for `D`. |
| pixels | `.mp4` | Anything vision-based. Also the only way to verify the other files by eye. |

The training games won't have homography, Smart Tagging or this video. Only signals you can
compute for them too can go into a trained model; see "Other matches" in `__training_pipeline.md`.

## Run order

1. **`inspect_data.py`, then read the Summary at the bottom.** Each `DIFF` against a published
   number needs an explanation before you trust anything downstream. Look hardest at:
   - **format facts**: how a line splits, how many fields a player group and the ball group
     have, and how a missing group is spelled. If these differ from the documented
     `frame:x,y,v;…:ball`, the parser in `loaders.py` needs adjusting.
   - **spec order == `<Player>` order**: `baseline.py` assumes slot *i* in a line is the *i*-th
     `<Player>` in the XML. If False, every team colour is wrong.
   - **provided / computed speed ratio**: about 1 means m/s.
   - **teams switch ends**: tells you whether coordinates are fixed or follow the attacking
     direction.
2. **Dashboard → Pitch + video.** This is the one check that ties everything together. Pick a
   frame mid-play and confirm three things:
   - the projected pitch lines sit on the painted lines
   - the circles sit on the players' feet
   - blue circles are on one kit and red on the other

   Then press **Verify the seek**: a direct seek in an mp4 can land a few frames off, and that
   looks exactly like a tracking offset. Repeat in the 2nd half and near the end.
3. **Timeline.** Scroll the whole match:
   - Halftime should show no players and a high reprojection error.
   - Team mean x should swap at halftime.
   - Speed should dip at stoppages.
   - Compare the public ground-truth strip against the signals around it, as a sanity check of
     the pipeline only. It isn't for tuning (`RULES.md`).
4. **Tracking quality.**
   - ball-gap lengths: decide how far interpolation can go
   - objects per frame: duplicates show as >22
   - slot occupancy: how long slots live
   - heatmaps: they mirror between halves if ends switch
5. **Smart Tagging.** The agreement table shows which codes' Team tag matches who is near the
   ball. Click a row, jump to it, and watch it on the Pitch + video tab.

## Common ways to inspect tracking data

These are the standard views for this kind of data. Most are in the dashboard.

- **Frame on a pitch** (Pitch + video tab): dots coloured by team, ball outlined. It's the first plot
  everyone makes. Add **trails** (the last few seconds of movement) to see direction and speed.
- **Animation** (Pitch + video → animation): up to 30 s, the video and the pitch playing in
  sync. It's the quickest way to
  see a slot jump between players, the ball freeze, or teams walking during a stoppage.
- **Video overlay** (Pitch + video tab): the tracking projected onto the pixels through the homography.
  It's the only way to check the data against reality.
- **Timelines** (Timeline tab): per-second summaries across the whole match, with events and
  labels as coloured strips underneath. This is where possession structure becomes visible.
- **Quality distributions** (Tracking quality tab): gap lengths, objects per frame and speed
  histograms. These show the failure modes of automatic tracking.
- **Occupancy / lifetime charts** (Tracking quality tab): one row per slot over time. They show
  how much "identity" the tracker really has.
- **Positional heatmaps** (Tracking quality tab): where each team and the ball spend time, per
  half.
- **Not built here, often useful later:**
  - **Voronoi / pitch-control maps**: which team controls each area of the pitch.
  - **Pass/possession networks**: need identities, which these slots don't have.
  - **Per-frame feature plots**: whatever features stage 4 produces, drawn on the same timeline
    as the labels.

Libraries people commonly use for this: `kloppy` (loading; already in
`requirements-starter.txt`), `mplsoccer` (pitch drawing in matplotlib) and `floodlight` (tracking
analysis). None are needed here. The dashboard draws its own pitch in plotly, and `starter/utils.py`
has a matplotlib one.

## Data rules for these tools

- The **text report** prints only aggregates (counts, percentages, medians, category names), so
  it's meant to be pasted back to Claude. Keep it that way when editing `inspect_data.py`. Never
  print rows, coordinates or per-frame values. Parse errors report line numbers and field
  counts, never the text of the line.
- The **dashboard** shows the data itself. Don't paste its screenshots, tables or exported
  numbers into an LLM conversation.
- **Everything derived** (the parsed-tracking cache, the JSON report) is written under the data
  directory, never into the repository.
