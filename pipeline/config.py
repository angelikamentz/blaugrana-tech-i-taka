"""Every fixed setting of the pipeline, in one place, each with the reason for its value.

Rule for this file: values are football-sensible defaults chosen *before* looking at any score.
None of them may be adjusted to raise the score on the public 5 minutes (RULES.md). When a value
changes, the reason changes with it, and the technical document repeats it.
"""

# -- version ------------------------------------------------------------------------------------
# Bump when the feature definitions change, so the cache rebuilds instead of serving stale columns.
FEATURE_VERSION = 2

# -- ball cleaning ------------------------------------------------------------------------------
# Gaps up to this many frames are filled by a straight line between the two ends. 12 frames is
# about half a second: short enough that a ball rarely changes owner inside it. Longer gaps stay
# missing -- the inspection showed most gaps last 10 s or more, far too long to draw a line across.
BALL_INTERP_MAX_FRAMES = 12

# The ball is "out" when it is more than this far beyond a touchline or goal line. 1 m absorbs
# tracking noise on the line itself; a ball genuinely out of play is usually well past it.
BALL_OUT_MARGIN_M = 1.0

# Ball speed is the distance moved per frame, averaged over this many frames, so a single noisy
# position doesn't look like a shot.
BALL_SPEED_SMOOTH_FRAMES = 5

# Above this the computed ball speed is treated as a tracking jump, not a real movement. The
# hardest shots in football are around 35-40 m/s.
BALL_SPEED_MAX_MS = 45.0

# -- players ------------------------------------------------------------------------------------
# Team speed is smoothed over one second, because a stoppage is a slow, sustained drop.
TEAM_SPEED_SMOOTH_FRAMES = 25

# Radius for "players close to the ball", per team. 5 m is about two strides: close enough to
# play the ball or press it.
NEAR_BALL_RADIUS_M = 5.0

# Radius for "a crowd round the ball", both teams together: about the 9.15 m a wall must keep at
# a free kick, so a wall or a box at a corner falls inside it.
CROWD_RADIUS_M = 10.0

# A player below this speed is walking or standing.
SLOW_PLAYER_MS = 1.0

# -- Smart Tagging ------------------------------------------------------------------------------
# Codes whose Team tag is used as a hint about which team has the ball. Picked from the
# inspection report, which compared each code's implied team with the team of the player nearest
# the ball -- an algorithmic signal, not the ground truth. Codes near 50% agreement say nothing
# and are left out: everything "in the box", SET PIECES, GOALS, CREATING CHANCES.
# Defending codes are listed too: loaders.load_tagging already flips them to the team WITH the
# ball (`possession_side`), and each one duplicates an attacking code's interval.
TAG_TEAM_CODES = (
    "BUILD UP", "DEFENDING IN ATTACKING THIRD",          # ~76% agreement
    "PROGRESSION", "DEFENDING IN MIDDLE THIRD",          # ~70%
    "ATTACKING TRANSITION", "DEFENSIVE TRANSITION",      # ~63%
    "LONG BALL",                                         # ~62%
    "BALL IN FINAL THIRD",                               # ~62%
)

# The set-piece code. Its intervals mark a restart; the inspection showed the stoppage sits in
# roughly the first 5-10 s of the interval.
TAG_SETPIECE_CODE = "SET PIECES"

# =============================================================================================
# Stage 2 -- rules. Each rule adds evidence, in log-odds, for "play is stopped" (dead) or for
# "home has the ball" (team). Positive = more likely. The numbers are round, plain-sense weights,
# not fitted values: a strong clue is worth 3-4, a hint 1-2.
# =============================================================================================

# -- dead ball ---------------------------------------------------------------------------------
DEAD_BIAS = -2.0                 # with no clue at all, play is far more often live than dead
DEAD_BALL_OUT = 4.0              # ball beyond a line: out of play, nearly always
# Around a SET PIECES tag start: players slowest and ball least visible from ~5 s before the tag
# starts to ~7 s after it (inspection report, 5-s bins). That window counts as a stoppage clue.
DEAD_SETPIECE_BEFORE_S = 5.0
DEAD_SETPIECE_AFTER_S = 7.0
DEAD_SETPIECE = 2.0
# Everyone walking. Live play has a median team speed of ~1.5 m/s.
DEAD_SLOW_MS = 1.0
DEAD_SLOW = 1.5
DEAD_VERY_SLOW_MS = 0.6
DEAD_VERY_SLOW = 1.5             # added on top of DEAD_SLOW
# Ball missing for a long time. Long gaps cluster around stoppages (ball out of view, held for
# a throw-in), but also happen in crowded play, so this is a hint, not a proof.
DEAD_LONG_GAP_S = 5.0
DEAD_LONG_GAP = 1.0
# Clear signs of live play.
LIVE_BALL_FAST_MS = 8.0          # the ball is travelling: a pass or a shot
LIVE_BALL_FAST = -2.0
LIVE_TEAM_FAST_MS = 2.2          # players running hard
LIVE_TEAM_FAST = -1.5

# -- which team ---------------------------------------------------------------------------------
# Nearest-player evidence: tanh of (distance of nearest away - distance of nearest home) over a
# 2 m scale, times a gain. Full weight only when someone is within playing distance of the ball;
# otherwise the ball is loose or in the air and the nearest player says much less.
TEAM_NEAR_GAIN = 1.5
TEAM_NEAR_SCALE_M = 2.0
TEAM_CONTROL_RADIUS_M = 2.0
TEAM_LOOSE_FACTOR = 0.3
TEAM_NO_PLAYER_M = 30.0          # distance used when a team has no tracked player at all
# Smart Tagging hint: per vote (H votes minus A votes). Small, because tags are phases, not
# possessions, and agree with the nearest player only ~67% of the time.
TEAM_TAG_GAIN = 0.3

# Per-frame probabilities are clipped, so one frame can never outweigh a switch cost on its own.
PROB_CLIP = 0.02

# =============================================================================================
# Stage 3 -- decoder (Viterbi over H / A / D). Costs in log-likelihood units: a switch has to be
# paid for by this much accumulated evidence. With the clipped per-frame evidence above, these
# work out to needing roughly half a second to a second of consistent clues before a change --
# about the shortest real possession or stoppage.
# =============================================================================================
SWITCH_COST_TEAM = 12.0          # H <-> A directly
SWITCH_COST_DEAD = 10.0          # live <-> dead

# =============================================================================================
# v2 -- dead-ball model learned on other matches (IDSSE, 7 Bundesliga games, CC-BY 4.0)
# =============================================================================================

# Degrading the clean IDSSE tracking so it looks like our automatic tracking. Targets are the
# BAR-ATH inspection report's own aggregates (no labels involved):
#   ball missing on 41% of graded frames, in gaps of median 10.7 s and p90 33.8 s
#   ~20.5 of 22 players tracked on average -> ~7% of player positions missing
DEGRADE_SEED = 0
DEGRADE_BALL_MISSING = 0.41
DEGRADE_BALL_GAP_MEDIAN_S = 10.7
DEGRADE_BALL_GAP_P90_S = 33.8
DEGRADE_PLAYER_MISSING = 0.07
DEGRADE_PLAYER_GAP_MEDIAN_S = 5.0

# Model: gradient-boosted trees (scikit-learn HistGradientBoostingClassifier), which handle the
# missing values natively. Modest, standard settings; trained on every DEAD_TRAIN_STEP-th frame
# because neighbouring frames are near-duplicates.
DEAD_MODEL_PARAMS = {"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 31,
                     "min_samples_leaf": 200, "l2_regularization": 1.0, "random_state": 0}
DEAD_TRAIN_STEP = 5

# Window lengths for context features, in seconds: a short look (a restart takes ~2 s) and a long
# one (a stoppage typically lasts 10-30 s). Offline, so the windows look both ways.
DEAD_WINDOWS_S = (2.0, 10.0)
DEAD_SHIFT_S = 3.0               # past-vs-future comparison: slowing down / speeding up
BALL_STILL_MS = 0.5              # the ball counts as still below this speed

# Smart Tagging is not available for IDSSE, so it cannot be in the model. It is added on top of
# the model's log-odds with the same fixed weight and window as in v1.
DEAD_TAG_PRIOR = DEAD_SETPIECE

# =============================================================================================
# Learned team model (team_model.py): p_home learned on IDSSE like the dead-ball model, in place
# of v1's nearest-player rules. Same modest tree settings, same degradation, every 5th live frame
# (neighbours are near-duplicates). Context windows centred on the frame, plus past-only and
# future-only means: 2 s is a touch or a short duel, 10 s a stretch of possession. Smart Tagging
# stays outside the model (IDSSE has none) and is added with TEAM_TAG_GAIN as in v1. Kept or
# dropped by the leave-one-match-out comparison on IDSSE only.
# =============================================================================================
TEAM_MODEL_PARAMS = dict(DEAD_MODEL_PARAMS)
TEAM_TRAIN_STEP = 5
TEAM_WINDOWS_S = (2.0, 10.0)
# v2team+: team-centre velocities over this many seconds around the frame. 2 s is long enough to
# average out tracking jitter and short enough to catch a team pushing up after winning the ball.
TEAM_MOVE_S = 2.0
# +bridge (bridge.py): unsure team evidence is filled from the confident evidence on both sides
# when both name the same team. "Confident" = the calibrated model gives one team at least 80%.
# Not further than 15 s on either side: about the length of a typical ball gap in our tracking
# (median 11 s, inspection aggregates), so a bridge spans a gap but not a long stretch of play.
# Football sense and tracking aggregates only, never the public labels.
BRIDGE_CONF_P = 0.8
BRIDGE_MAX_S = 15.0

# Switch costs chosen on IDSSE (team_model.py, cost_search). The costs above were set before any
# learned model existed; with stronger team evidence they may now be too high, which would explain
# finding ~295 possessions per IDSSE match against ~437 true. A small grid around them, decoded on
# the same held-out probabilities. The best pair replaces SWITCH_COST_TEAM / SWITCH_COST_DEAD for
# the team model only if it wins by the usual rule; it is saved with the model, not written here.
SWITCH_GRID_TEAM = (4.0, 6.0, 8.0, 10.0, 12.0)
SWITCH_GRID_DEAD = (6.0, 8.0, 10.0, 12.0)
