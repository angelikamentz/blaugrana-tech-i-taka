# Evaluation guide

## Run the dashboard

```bash
pip install -r requirements-dashboard.txt
streamlit run evaluation/dashboard.py
```

It shows the same numbers as `python evaluation/run_evaluation.py`, drawn as charts.

## Add your results

1. Save your labels to `evaluation/results/<name>.csv`. The file needs a `frame,label` header and
   199,319 rows (frames `0…199318`, labels `H`/`A`/`D`/`X`). The file name becomes the row name.
2. Validate it: `python evaluation/validate_submission.py evaluation/results/<name>.csv`
3. Refresh the dashboard. Or drag any CSV into the sidebar's upload box instead.

Result CSVs are gitignored, so they won't end up in your public repo.

## Problem lists per run

```bash
python evaluation/problems.py                                  # every CSV in evaluation/results/
python evaluation/problems.py evaluation/results/<name>.csv    # one file
```

This writes `evaluation/reports/<name>_problems.csv` (gitignored), one row per wrong stretch:
video time, frames, length, kind, what the truth said and what you said. It's the same list as
the dashboard's Problems tab.

## How to read it

| tab | what to look at | good sign |
|---|---|---|
| **Overview** | spellF1 (decides the ranking) and your possession count against the truth's 17 | possession count and mean length close to the truth's |
| **Timeline** | your strip under the ground truth, and the black "✗" strip of wrong frames | long solid blocks, colour changes lining up with the truth's |
| **Problems** | every wrong stretch with its video time, named by kind (late handover, flicker, missed stoppage…); click a row to zoom the timelines to it | the bar chart's biggest bar tells you what to fix next |
| **Possessions** | which true possessions you found (solid, green edge) and which you missed or invented (hatched, red edge) | few hatched blocks, and matches that survive the higher thresholds |
| **Frames** | the confusion matrix, where each row shows how that true label was predicted | a strong diagonal, especially the `D` row |

Warning signs:
- **Many short possessions:** flicker. spellF1 drops even if macroF1 looks fine.
- **Grey where the truth is coloured, or the reverse:** you're getting live ball vs dead ball wrong.
- **Mostly "Late …" problems:** your handovers come late. Being 1 s late roughly costs a third of
  a possession's score.

## Remember

The public ground truth is only 5 minutes and 17 possessions. Use it to check that your pipeline
works, not to choose between versions. The rules forbid tuning on it, and the last tab shows it
picking the wrong smoothing window for the baselines.
