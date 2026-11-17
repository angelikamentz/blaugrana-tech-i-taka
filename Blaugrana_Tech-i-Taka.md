# Ball possession from automatic tracking: technical document

**Team:** Blaugrana Tech-i-Taka · **Members:** Ana Baltaretu, Angeliki Mentzelopoulou, Diego Becerra
· **Challenge:** More Than A Hack 2027, BAR-ATH possession (Phase 1)
· **Repository:** https://github.com/angelikamentz/blaugrana-tech-i-taka · **Submitted file:** `Blaugrana_Tech-i-Taka.csv`, built by
`python pipeline/run.py --dead model --team model` (version **v2team**)

---

## 1. Approach

For every frame we answer two questions: **is play stopped?** and, if not, **which team has the
ball?** Each frame gets a probability for each, `p_dead` and `p_home`, from clues in the tracking
and the Smart Tagging. A decoder then turns the frame-by-frame guesses into possessions.

```
raw files ─► features.py ─► dead_model.py (p_dead) + team_model.py (p_home) ─► decode.py ─► CSV
             one row per     two gradient-boosted tree models, both trained      Viterbi over
             frame, 30+      on 7 other matches (IDSSE); Smart Tagging added     H / A / D,
             clues           on top with fixed weights                           X outside the halves
```

**Why this shape.** The ranking metric, spellF1, scores whole possessions, not frames. A system
that is right on most frames but flickers loses badly, because every flicker splits a possession.
And the ball is missing on 41% of graded frames, in gaps with a median of 11 s, often exactly at
duels, crowded boxes and aerial balls. So the decisive step isn't the per-frame guess but the
step that glues frames together and carries the owner through the gaps.

### 1.1 Features (`pipeline/features.py`)

One row per frame, in metres, with one half flipped so the home team always attacks towards +x:

- **ball:** seen or not, short gaps (≤ 12 frames) interpolated, length of the current gap, time
  since and until the ball is seen, out of the pitch, distance to the lines, speed
- **players:** distance from the ball to the nearest home and nearest away player, players of each
  team within 5 m, crowd within 10 m, team speeds, share of players walking, spread, team centres
- **Smart Tagging:** time since the start of a `SET PIECES` tag and its type, and team "votes"
  from the phase codes

### 1.2 Dead ball: a model learned on other matches (`pipeline/dead_model.py`)

Gradient-boosted trees (scikit-learn `HistGradientBoostingClassifier`, `random_state=0`) on 31
tracking-only inputs (`pipeline/dead_features.py`): the base signals above, their averages over
2 s and 10 s windows centred on the frame, and team speed over the next 3 s minus the last 3 s
(slowing into a stoppage, speeding up out of a restart). Labelling is offline, so looking ahead
is allowed.

The model is trained on **IDSSE**, seven Bundesliga matches whose official tracking marks every
frame alive or dead (section 3). IDSSE's tracking is clean and ours isn't, so before training we
**degrade** it (`pipeline/degrade.py`, seeded) to match aggregates we measured on our own match:
the ball is hidden in random long gaps until 41% of frames have none (median 10.7 s, p90 33.8 s),
and about 7% of player positions are removed. Gaps are placed at random, not at stoppages, so the
model can't learn a link between missing ball and stoppage that we can't measure on our match.

Smart Tagging can't go into the model, because IDSSE has no tags. It is added on top of the
model's log-odds with a fixed weight: a window from 5 s before to 7 s after the start of each
`SET PIECES` tag counts as evidence of a stoppage.

The first graded frames of each half, before the first `Kick Off (Start)` tag, are forced to `D`.

### 1.3 Team in possession: a second model learned on IDSSE (`pipeline/team_model.py`)

`p_home`, the probability that the home side has the ball while play is live, comes from a second
gradient-boosted tree model, set up like the dead-ball model. It uses the same tree settings, the
same degraded IDSSE tracking and every 5th frame. It is trained on live frames only, labelled with
IDSSE's team-in-possession flag. Its inputs (`pipeline/team_features.py`) are tracking only:

- **who is at the ball:** the nearest home and away distances and their difference, and the
  players of each team near the ball
- **where the ball is and where it goes:** position, and velocity along the pitch, where +x is the
  home team's attacking direction
- **team shape:** each team's centre relative to the ball, and team speeds
- **ball gaps:** how long the current gap is, and who was nearer the ball at the last frame before
  the gap and the first frame after it. The ball is missing on 41% of our frames, so these carry
  much of the decision.
- **context:** the "who is nearer" signal averaged over 2 s and 10 s windows centred on the frame,
  and over the past and the future separately

**Home/away symmetry.** Every training frame is also used with the two teams exchanged and the
pitch turned half a circle. A prediction averages the model's answer on the frame and on its
swapped copy, so the model can't prefer "home" for its own sake.

**Calibration.** The decoder adds log-probabilities against fixed switch costs, so the
probabilities must be honest. One temperature, a single number dividing the model's log-odds, is
fitted on the pooled held-out predictions of the leave-one-match-out check (section 4).

**Smart Tagging** stays outside the model, because IDSSE has no tags. It is added on top as in our
rule-based version: `0.3 ×` the difference of team votes from the phase codes, in log-odds.

The rule-based team evidence it replaces, `1.5 · tanh((nearest away − nearest home) / 2 m)`,
remains in `pipeline/rules.py` as the fallback (`--team rules`).

### 1.4 Decoder (`pipeline/decode.py`)

A three-state Viterbi over `H`, `A` and `D`, run on each half separately. Each frame scores the
states by `log p_dead` for D and `log(1 − p_dead) + log p_home` (or `log(1 − p_home)`) for H and
A. Each change of state costs a fixed amount: **12** for H↔A, **10** for live↔dead. Per-frame
probabilities are clipped at 0.02, so no single frame outweighs a switch. In practice a change
needs roughly half a second to a second of consistent evidence, about the shortest real
possession or stoppage. Where there is no evidence, the current owner carries on through the gap.
Ungraded frames are `X`.

---

## 2. Assets: what we used and what we dropped

| asset | use |
|---|---|
| tracking (`_atd.txt`, `_atd.xml`) | **used**: every feature; the only source of team membership |
| Smart Tagging | **used** as fixed-weight hints on top: set-piece timing for `D`, phase codes for the team |
| homography | **not used** in the submission; only for checking the tracking against the video by eye |
| tactical and broadcast video | **not used** in the submission; watched to understand the data (allowed by `RULES.md`) |
| public 5-minute ground truth | **only** to check the pipeline end to end; never for training, tuning or choosing settings |

What we learned from the tracking (aggregates from `inspection/inspect_data.py`): the files load as
documented (49 of 50 published checks match). There is no alive/dead or owning-team field. The
ball vanishes for long stretches rather than flickering, so filling short gaps adds nothing. Teams
switch ends at halftime without the coordinates correcting for it. The Smart Tagging team tag
agrees with the nearest player 76% of the time on `BUILD UP`, 70% on `PROGRESSION` and only
53–56% on box and set-piece codes, and `Team = N/A` on every throw-in.

**Tried and not in the submission:**
- **v1, hand-written dead-ball rules:** replaced by the learned model (section 4).
- **v3, possession anchors:** stretches where one player of a team clearly controls the ball,
  with no opponent close, become hard constraints on the decoder.
- **v4, more realistic degradation:** wobble noise on the IDSSE ball, hiding it outside the pitch
  and while it lies still, calibrated to BAR-ATH aggregates only.
- **v2, rule-based team evidence:** replaced by the learned team model (section 4).

v3 and v4 were written on 16 November, the day before the deadline. Their IDSSE
leave-one-match-out comparison (`pipeline/idsse_eval.py`) wasn't finished in time, and that table
is the only legal way to choose between versions, so neither was adopted. Their code is on the
team's development branch.

---

## 3. Training data

| dataset | source | who labelled it, how | amount |
|---|---|---|---|
| **IDSSE** (Integrated Dataset for Sports Science and Engineering) | DFL / Bundesliga 2022/23, CC-BY 4.0. Bassek et al., *Scientific Data* 2025, doi:10.1038/s41597-025-04505-y. Downloaded from Hugging Face `pysport/idsse-data` | the official DFL tracking provider: per-frame ball alive/dead and team in possession, 25 fps | 7 full matches, about 1 million frames |

**No labels from the provided game were used**, in any amount: no hand labels, no public ground
truth for training or tuning, no event data. All learned settings come from IDSSE. The fixed
values in `config.py` are football sense or aggregates of our own match's tracking, which uses no
labels.

---

## 4. Possession rules we implemented

We follow the definition in `CHALLENGE.md`: while the ball is alive exactly one team has it, it
changes only when the other team gains *controlled* possession, and a dead ball ends a possession.

- **Loose balls and challenges:** the switch cost means a team change needs sustained evidence
  that the other team is now nearer the ball. Single touches, deflections and duels don't flip
  possession, and the last team in control keeps it.
- **Ball not visible:** no evidence either way, so possession carries on until the evidence turns.
- **Restarts:** play is dead while players are slow and the ball is still or out, around a
  set-piece tag, or outside the pitch. Dead ends when play speeds up again. After a restart the
  team evidence decides the new owner.
- **`H D H` is two possessions:** a stoppage always ends a spell, even when the same team keeps the
  ball.

**Results.** On IDSSE, leave-one-match-out (train on six, score the seventh with the official
metrics):

| mean of 7 held-out matches | spellF1 | macroF1 | alive/dead F1 | `D` F1 |
|---|---|---|---|---|
| v1, dead-ball rules | 0.223 | 0.582 | 0.806 | 0.587 |
| v2, dead-ball model | 0.335 | 0.785 | 0.964 | 0.957 |

v2 wins on all seven matches against v1.

**Learned team model vs plain v2.** Here both versions use the same dead-ball model, trained on the
other six matches, and the same decoder. Only `p_home` differs:

| mean of 7 held-out matches | spellF1 | macroF1 | possessions predicted / true |
|---|---|---|---|
| v2, team rules | `<v2 spellF1>` | `<v2 macroF1>` | `<...>` |
| **v2team, learned team model (submitted)** | **`<v2team spellF1>`** | **`<v2team macroF1>`** | `<...>` |

We agreed the decision rule before running the check: the team model replaces the rules only if
its mean spellF1 is higher **and** it wins on at least 5 of the 7 matches. It won on `<N>` of 7.
The temperature was fitted on the same pooled held-out predictions, so this score may be slightly
optimistic. That is a mild effect for a single number. On the public 5 minutes, as a pipeline check only: v2team `<spellF1 / macroF1,
possessions against 17 true>`. v2 gets spellF1 0.450 and macroF1 0.659, with 15 possessions against 17 true. v1 gets 0.447 / 0.698, and the published
`baseline_smooth50` gets 0.326 / 0.621. The window is too small to rank versions: its bootstrap
noise band is about ±0.37.

---

## 5. Limitations

- **The team part is still the bottleneck.** With the rule-based team evidence we found only
  about 250 possessions per IDSSE match against 370–511 true. `<Update with v2team's count.>` Short opponent control inside a long possession (under 2 s) is usually
  missed.
- **Restart team:** throw-ins have no team tag, so the team after a throw-in relies on proximity
  alone.
- **Hand-over timing:** switching on proximity tends to come a little early, at contact, while the
  ground truth switches at control.
- **Domain gap:** the IDSSE alive/dead flag comes from a different provider and may not mean exactly
  what our ground truth means. Our degradation matches aggregates but places ball gaps at random.
- **Fixed switch costs** were set from football sense, not learned.

**With more time:** choose the switch costs on
IDSSE leave-one-match-out; weight each Smart Tagging code by its reliability, estimated without
labels; and check that the degraded IDSSE tracking looks like ours, for example by training a
classifier to tell the two apart.

---

## 6. Reproduction

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements-pipeline.txt         # pinned exact versions
export MTAH_DATA_DIR=/path/to/data               # folder that directly contains tactical/
python pipeline/idsse.py                         # downloads and converts the 7 IDSSE matches (~2.6 GB)
python pipeline/dead_model.py                    # dead-ball model: leave-one-match-out table, final model
python pipeline/team_model.py                    # team model: v2 vs v2team table, verdict, final model
python pipeline/run.py --dead model --team model --name Blaugrana_Tech-i-Taka
python evaluation/validate_submission.py evaluation/results/Blaugrana_Tech-i-Taka.csv
```

Every random draw is seeded (degradation seed 0, model `random_state=0`), so the same data and the
same package versions give the same file. Caches, converted IDSSE matches and the trained models are
written under `$MTAH_DATA_DIR`, never into the repository. The match data is not in the repository.
