# autoresearch

This is an experiment to have the LLM do its own research on developing a fraud detection algorithm.

## Problem reframing

The original goal was binary classification (fraud / non-fraud). After extensive experimentation, we discovered:

1. **The label `CASE_NO > 0` means "flagged by the CC rule system"**, not "confirmed fraud."
2. **ALL CC rule features are data leakage** — CC rules fire as part of the case creation pipeline that produces `CASE_NO`. Using CC rule outputs to predict `CASE_NO > 0` is circular (the system predicting its own output). This includes both "notification" rules (NOT1, N001, TA*) AND "detection" rules (TSCC*, QACC*, CC*).
3. **RAD rule features are NOT leakage** (they fire pre-authorization, independently of case creation) — but they only cover 2.1% of fraud transactions, providing minimal signal.
4. The existing rule system itself has a massive false alarm problem: 4,415 transactions flagged, only 33 confirmed as fraud (CUH status 700), 17 confirmed as non-fraud (CUH status 750), ~4,365 unresolved.
5. Without rule features, the binary F1 caps at ~0.41 because the model inherits the rule system's false alarm rate.

**New objective: PRIORITIZATION MODEL.** Instead of binary fraud/non-fraud, build a model that ranks flagged cases by fraud likelihood so analysts review the most suspicious ones first.

## What constitutes data leakage

- **ALL CC rule outputs** (binary rule hits, rule counts, rule scores, individual rule IDs) — these are intermediate outputs of the case creation pipeline that produces the label.
- **Response codes / auth status** (DE039, TRANS_AUTH_STAT) — these encode post-decision information.
- **Settlement amounts** (DE005) — identical to DE004 in this dataset, adds no signal. DE006 (billing amount) differs only for currency conversions, already captured by `is_foreign_currency`.

## What is allowed

- **RAD rule features**: Pre-authorization, independent of case creation. Low coverage but legitimate.
- **Rule-INSPIRED features from raw data**: Understand what rules check, then compute the underlying signal from raw transaction data. Example: if a rule checks "> 3 transactions in 10 minutes", compute `card_txn_count_10min` from raw timestamps. This captures the continuous signal without using the rule system's output.
- **All raw transaction attributes**: Amount, MCC, POS entry mode, EMV/chip/contactless status, CVC2, ECI, currency, timestamps, card/merchant IDs.
- **Engineered features**: Velocity, volume, target encoding, behavioral patterns — all computed from raw data.

## Setup
To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `CONTEXT.md` — repository context.
   - `prepare.py` — feature engineering, label creation, data splitting. FOCUS YOUR TIME HERE! Use data centric ai approach!
   - `train.py` — model training + dual evaluation (binary F1 + prioritization AUC/Precision@k).
4. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
5. **Confirm and go**: Confirm setup looks good.
Once you get confirmation, kick off the experimentation.

## Experimentation
You launch it simply as: `uv run train.py`.

The script outputs two evaluations:

### 1. Binary classification (flagged vs not-flagged)
```
---
f1:          0.407761
precision:  0.260787
recall:     0.934337
training_seconds: 0.7
total_seconds:    6.0
---
```

### 2. Prioritization (ranking confirmed cases)
```
=== PRIORITIZATION (all confirmed cases) ===
Confirmed fraud cases: 30
Confirmed non-fraud cases: 16
AUC (case-level, max score):   0.6167
Avg Precision (max score):     0.7654
Precision@5:  0.800 (4/5 fraud)
Precision@10: 0.800 (8/10 fraud)
```

### Primary metric: AUC (case-level)
The primary goal is to maximize **AUC** on confirmed cases — how well the model separates confirmed fraud from confirmed non-fraud in the ranking. Higher AUC = analysts find real fraud faster.

Secondary metrics: Precision@5, Precision@10 (how many of the top-k ranked cases are real fraud).

Binary F1 is tracked but is a secondary concern (capped by label noise).

## Output format

The script prints both evaluations. Read them with:
```
grep "^f1:\|^precision:\|^recall:\|^AUC\|^Avg Prec\|^Precision@" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 8 columns:

```
commit	f1	auc	p_at_5	p_at_10	precision	recall	status	description
```

1. git commit hash (short, 7 chars)
2. f1 achieved — use 0.000000 for crashes
3. auc (case-level prioritization AUC) — use 0.000000 for crashes
4. p_at_5 (Precision@5) — use 0.000000 for crashes
5. p_at_10 (Precision@10) — use 0.000000 for crashes
6. precision (binary classification)
7. recall (binary classification)
8. status: `keep`, `discard`, or `crash`
9. short text description of what this experiment tried

Example:
```
commit	f1	auc	p_at_5	p_at_10	precision	recall	status	description
a2cab76	0.407761	0.6167	0.800	0.800	0.260787	0.934337	keep	Baseline: rule-inspired features from raw data
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar14b`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Tune `prepare.py` and/or `train.py` with an experimental idea by directly hacking the code.
3. git commit.
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context).
5. Read out the results: `grep "^f1:\|^precision:\|^recall:\|^AUC\|^Avg Prec\|^Precision@" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in `results.tsv` (NOTE: do not commit results.tsv, leave it untracked by git).
8. If AUC improved (higher), you "advance" the branch, keeping the git commit.
9. If AUC is equal or worse, you `git reset` back to where you started.

**Timeout**: Each experiment should be fast since it runs on CPU. If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log `crash` as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read CONTEXT.md, prepare.py and train.py for new angles, try combining previous near-misses, try more radical feature engineering or model changes. The loop runs until the human interrupts you, period.

## Ideas to explore

Feature engineering (from raw data, no rule outputs):
- More granular velocity windows (2min, 5min, 30min)
- Per-MCC velocity (e.g., count at MCC 5411 in 1h)
- Average transaction amount in window (AVERAGE rules)
- Same-amount repeat detection (TSCC2: same amount 5x in 60min)
- Terminal diversity (TSCC14: different terminals in 30min)
- Time-of-day interactions (midnight + contactless + high amount)
- Card age / first-seen features
- Merchant category risk tiers

Model approaches:
- Two-stage: first predict flagged/not, then rank within flagged
- Semi-supervised: use the 50 confirmed labels more directly
- Different objectives: LambdaRank or pairwise ranking loss
- Probability calibration for better risk scores
