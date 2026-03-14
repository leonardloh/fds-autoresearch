# autoresearch

This is an experiment to have the LLM do its own research on developing it's own fraud detection algorithm.

## Setup
To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `CONTEXT.md` — repository context.
   - `prepare.py` — the file you modify to load the data, create training and test set, perform feature engineering and find the best features.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop. Prefer explainable model like xgboost.
4. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
5. **Confirm and go**: Confirm setup looks good.
Once you get confirmation, kick off the experimentation.

## Experimentation
You launch it simply as: `uv run train.py`.

The goal is simple: get the highest F1 score. The training speed should be fast since it is run on CPU.
## Output format

Once the script finishes it prints a summary like this:

```
---
f1:          0.997900
precision:  0.887132
recall:     0.980123
training_seconds: 300.1
total_seconds:    325.9
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 6 columns:

```
commit	f1	precision   recall	status	description
```

1. git commit hash (short, 7 chars)
2. f1 achieved (e.g. 1.234567) — use 0.000000 for crashes
3. precision acheived (e.g. 1.234567) — use 0.000000 for crashes
4. recall acheived (e.g. 1.234567) — use 0.000000 for crashes
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:
```
commit	f1	precision	recall	status	description
c2d8f3a	0.000000	0.000000	0.000000	crash	OOM encoding high-cardinality merchant_id one-hot
e5a1b7c	0.851234	0.812345	0.894567	keep	target-encoded merchant_id + frequency features
f9c3d2e	0.839012	0.856789	0.821890	discard	oversampled fraud class with SMOTE - precision up but recall dropped
```
## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Tune `prepare.py` and/or `train.py` with an experimental idea by directly hacking the code.
3. git commit.
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context).
5. Read out the results: `grep "^f1:\|^precision:\|^recall:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in `results.tsv` (NOTE: do not commit results.tsv, leave it untracked by git).
8. If F1 improved (higher), you "advance" the branch, keeping the git commit.
9. If F1 is equal or worse, you `git reset` back to where you started.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should be fast since it runs on CPU. If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log `crash` as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read CONTEXT.md, prepare.py and train.py for new angles, try combining previous near-misses, try more radical feature engineering or model changes. The loop runs until the human interrupts you, period.