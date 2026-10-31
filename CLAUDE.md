# CLAUDE.md

More Than A Hack 2027 entry: label every frame of one match `H` / `A` / `D` / `X`. Start with
`CHALLENGE.md`, `DATA.md` and `EVALUATION.md`.

## Never access the match data

The competition terms forbid making the match data (videos, tracking, tagging, homography, or
any extract of them) available to a hosted LLM. That includes Claude.

- **Never read, list, open, grep or load** match data files, wherever they live: `data/`,
  `data_DO_NOT_PUT_DATA_HERE/`, the folder `MTAH_DATA_DIR` points at (outside the repo), or any
  `BAR-ATH_*` file that isn't a label CSV.
- **Never run anything that loads the data**, e.g. `starter/baseline.py`, `starter/load_*.py`,
  the starter notebooks, or our own pipeline. Running it would bring data extracts into context.
- **When a task involves the data, write a script and give it to the user to run.** Base it on
  the formats documented in `DATA.md`, and find files with `games.asset_paths()` /
  `MTAH_DATA_DIR`. Never hard-code a data path.
  - Scripts should write results to files and print only aggregates: scores, counts, summary
    stats, categories. Never print raw rows, coordinates or frame-level tracking.
  - Then ask the user to paste back only what's needed. Don't ask for raw data.
- If you're unsure whether something counts as data, ask before opening it.

Fine to read:
- the docs and code
- the public ground truth in `evaluation/gt/`
- submission/label CSVs in `evaluation/results/`
- the reports in `evaluation/reports/`

## Other rules

- `evaluation/run_evaluation.py` and `evaluation/possession_metrics.py` are the official scorer.
  Don't modify them.
- No hand labels on this match, and no tuning on the public 5 minutes. They're for checking the
  pipeline only (`RULES.md`).
- Nothing from `data/` or any data folder may be committed. The repo will be public.

## Commands

```bash
python evaluation/run_evaluation.py                       # score everything in evaluation/results/
python evaluation/validate_submission.py <file>.csv       # before every submission
python evaluation/problems.py                             # per-run lists of wrong stretches
streamlit run evaluation/dashboard.py                     # visual scoreboard (requirements-dashboard.txt)
```
