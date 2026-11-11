# Our pipeline

What we built for Phase 1 so far, how to run it, and where it stands. Start here if you're
joining the work. The challenge itself is explained in [CHALLENGE.md](../CHALLENGE.md),
[DATA.md](../DATA.md) and [EVALUATION.md](../EVALUATION.md).

**Deadline: November 17, 2026** ([SUBMISSION.md](../SUBMISSION.md)).

---

## The idea in one paragraph

For every frame we answer two questions: **is play stopped?** and, if not, **which team has the
ball?** Each frame gets a probability for both, from simple clues in the tracking and the Smart
Tagging. A decoder then glues the frames into possessions: every change of state has a cost, so
one odd frame can't split a possession, and with no evidence (the ball is missing on 41% of
graded frames) the current owner carries on. The ranking metric scores whole possessions, not
frames, so this glue step is where most of the score comes from.

```
raw files ─► features.py ─► rules.py / dead_model.py ─► decode.py ─► evaluation/results/<name>.csv
             one row per     p(dead) and p(home)          Viterbi:
             frame, 30+      per frame                    H / A / D,
             clues                                        X outside the halves
```

---

## Rules we work under

These come from the competition rules and from [CLAUDE.md](../CLAUDE.md). They shape the design,
so read them before changing anything.

- **No hand labels on this match, and no tuning on the public 5 minutes.** The public ground
  truth is only for checking the pipeline works. Settings are fixed, explained values in
  [config.py](config.py), or are learned from **other matches** (IDSSE, below).
- **The match data never goes to a hosted LLM**, Claude included. Claude writes the scripts, a
  person runs them in their own terminal, and only aggregates get pasted back. Every script here
  prints aggregates only. Keep it that way.
- **Nothing derived from the data goes in the repo.** Caches, converted matches and models are
  written under `$MTAH_DATA_DIR`, never here.
- **The repo must regenerate the submission exactly.** Everything is seeded and deterministic.

---

## Setup

Point `MTAH_DATA_DIR` at the folder that **directly contains** `tactical/` (and `broadcast/`). Keep
that folder outside the repo. Put the line in `~/.zshrc`, and use your own path:

```bash
export MTAH_DATA_DIR="/path/to/mtah2027"
```

Then install the pipeline requirements (numpy, pandas, kloppy, scikit-learn, joblib; works on
Python 3.9+):

```bash
pip install -r requirements-pipeline.txt
```

The dashboards need streamlit 1.66, which needs **Python 3.11+**. Use a 3.12 virtual environment
if you want those (`requirements-inspection.txt`), and check that `which python` points into it.

---

## Running it

All of these read the match data: run them in your own terminal.

| step | command | what it does |
|---|---|---|
| inspect the data | `python inspection/inspect_data.py` | checks every file against the published numbers |
| feature table | `python pipeline/features.py` | builds and caches the per-frame clues, prints a summary |
| **v1** | `python pipeline/run.py` | rules + decoder, writes `evaluation/results/v1.csv` |
| IDSSE download | `python pipeline/idsse.py` | downloads and converts the 7 IDSSE matches |
| dead-ball model | `python pipeline/dead_model.py` | leave-one-match-out check, then trains the final model |
| **v2** | `python pipeline/run.py --dead model` | v1 with the learned dead-ball model, writes `v2.csv` |

Scoring only reads label CSVs, so anyone can run it:

```bash
python evaluation/run_evaluation.py
```

```bash
python evaluation/problems.py evaluation/results/v1.csv
```

---

## The files

| file | stage | what's in it |
|---|---|---|
| [config.py](config.py) | all | every setting, each with the reason for its value |
| [features.py](features.py) | 1 | the feature table: ball (seen, gaps, out, speed, distance to lines), players (distance to ball per team, crowd round the ball, speeds, share walking, spread, team centres), Smart Tagging (set-piece timing and type, team hints). Flips one half so home always attacks +x |
| [rules.py](rules.py) | 2 | v1 evidence: dead-ball rules and team rules, in log-odds |
| [decode.py](decode.py) | 3 | Viterbi over H / A / D with switch costs; each half separately |
| [run.py](run.py) | all | one command from raw files to the CSV; prints aggregates of its own output |
| [idsse.py](idsse.py) | v2 | downloads the IDSSE matches through kloppy, converts them to our arrays |
| [degrade.py](degrade.py) | v2 | makes IDSSE's clean tracking look like ours |
| [dead_features.py](dead_features.py) | v2 | the dead-ball model's inputs (31 columns) |
| [dead_model.py](dead_model.py) | v2 | trains the dead-ball model, compares v1 and v2 on held-out IDSSE matches |

`compute()` in `features.py` and everything in `rules.py`, `decode.py`, `degrade.py` and
`dead_features.py` take arrays and do no file I/O, so they can be tested on made-up data without
touching the match.

---

## What we learned from the data

From `inspection/inspect_data.py` (aggregates only):

- **Every file loads as documented:** 49 of 50 published checks match. The one that doesn't is a
  halftime figure, on frames that aren't graded.
- **The tracking has no alive/dead or owning-team field.** We build both from clues.
- **The ball doesn't flicker out, it vanishes for long stretches:** 150 gaps in graded play, median
  11 s. A frame without a ball typically sits in a ~29 s gap, so filling short gaps adds nothing.
  Ball-free stretches must be bridged by the decoder.
- **Teams switch ends at halftime** and the coordinates don't correct for it. `features.py` flips
  the second half.
- **Set-piece tags start during the stoppage:** player speed and ball visibility bottom out from
  ~5 s before to ~7 s after a `SET PIECES` tag starts. Those intervals cover 28% of graded frames,
  mostly the live play after the restart, so only their first seconds count as "dead".
- **Some Smart Tagging codes are useful team hints and some aren't.** Compared with the team of
  the nearest player: `BUILD UP` 76%, `PROGRESSION` 70%, transitions and `LONG BALL` ~62%,
  anything in the box or `SET PIECES` ~53-56%. Codes come in mirrored pairs
  (`PROGRESSION` = `DEFENDING IN MIDDLE THIRD`, other side).

---

## Versions and results

### v1: rules + decoder

Public 5 minutes (17 possessions). **A sanity check, not a measurement**: this window ranks the
published baselines in the wrong order.

| | macroF1 | spellF1 | possessions (truth 17) |
|---|---|---|---|
| **v1** | 0.698 | **0.447** | 17 |
| baseline_smooth50 (published floor) | 0.621 | 0.326 | 26 |

- No flicker. Live vs dead F1 0.92; team right ~84% of the time while play is live.
- **Weak spot: stoppages.** When play is really dead, v1 says so only 37% of the time. The biggest
  single error is a 22 s stoppage labelled as live play.

### v2: dead-ball model learned on other matches

**Status: written and tested on made-up matches; not yet run on real IDSSE data or on BAR-ATH.**

- **IDSSE** is 7 Bundesliga 2022/23 matches from the DFL, CC-BY 4.0 (Bassek et al., Scientific
  Data 2025, doi 10.1038/s41597-025-04505-y). The official tracking marks every frame alive/dead
  and the team in possession, at 25 fps, which is the same kind of label as our ground truth.
- **Degradation:** IDSSE's tracking is clean, ours isn't. Before training, the ball is hidden in
  random long gaps until 41% of frames have none (median 10.7 s, p90 33.8 s), and ~7% of player
  positions are removed. The targets are our own inspection aggregates. Gaps are placed at random,
  not at stoppages, so the model isn't taught a link we can't measure on our match.
- **Model:** gradient-boosted trees (scikit-learn), tracking-derived inputs only. Smart Tagging is
  added on top with v1's fixed weight, because IDSSE has no tags.
- **Choosing between versions:** `dead_model.py` trains on six IDSSE matches, scores the seventh
  with the official metrics, and repeats for every match, for v1's dead-ball rules and for the
  model. That table, not the public window, decides whether v2 replaces v1.

### Open questions and risks

- IDSSE's "dead" flag may not mean exactly what our ground truth's does (it comes from DFL's
  tracking provider, ours from Metrica's corrected tracking).
- kloppy's match ids and the dataset card disagree on two of the seven matches. The script skips
  any id that fails, and says which.
- Where kloppy keeps the raw downloads isn't verified. Converted matches go to
  `$MTAH_DATA_DIR/external/idsse/`, the model to `$MTAH_DATA_DIR/cache/models/`.

---

## Next steps

1. Run v2 end to end on IDSSE and BAR-ATH, and read the leave-one-match-out table.
2. Learn the **team** part from IDSSE the same way (it still uses v1's nearest-player rules).
3. Choose the decoder's switch costs on IDSSE instead of fixed values.
4. Pin exact package versions, freeze, write the technical document
   ([SUBMISSION.md](../SUBMISSION.md) lists what it must cover), validate the exact upload file.
