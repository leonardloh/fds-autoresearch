# Weekend Project: Autonomous AI-Driven Fraud Detection Research

## What I Did

Inspired by Andrej Karpathy's [autoresearch](https://github.com/karpathy/autoresearch), I ran an experiment to see how far an AI agent (Claude) could go in autonomously developing a fraud detection algorithm with minimal human intervention. The setup:

- I gave Claude access to Finexus's transaction data (ISO 8583 records, case creation details, case update history, and rule tables)
- I wrote a `PROGRAM.md` file defining the research loop: engineer features, train a model, evaluate, log results, iterate
- I wrote a `REVIEW.md` file that acts as a separate "senior data scientist" reviewer to catch mistakes
- Then I let it run

**Data used:** the UAT data for technical interview

## What Happened

In a single afternoon (12:28pm to 11:15pm), the agent autonomously produced 46 experiments covering the full ML lifecycle:

1. **Data exploration** — understood the 5-table schema, identified that `CASE_NO > 0` means "flagged by rules" not "confirmed fraud"

2. **Discovered and fixed data leakage (3 rounds):**
   - Round 1: Caught that using CC rule outputs to predict case creation is circular (the system predicting its own decisions) — F1 dropped from 0.98 to 0.41, which was the *honest* number
   - Round 2: The review agent flagged target-encoded risk features (card_risk, mcc_risk) as label leakage — removed them, overfit gap dropped from 0.61 to 0.05
   - Round 3: I flagged remaining temporal leakage in batch aggregate features — agent converted them to expanding past-only windows

3. **Reframed the problem** — pivoted from binary classification (capped at F1 ~0.41 due to noisy labels) to case prioritization (rank flagged cases by fraud likelihood for analyst review)

4. **Ran 21 logged experiments** — systematically explored tree depth, learning rate, regularization, ensembles, CatBoost, feature engineering variants — tracking AUC on confirmed cases

5. **Built a production-ready pipeline** with:
   - Point-in-time feature engineering (no future data leakage)
   - Point-in-time label construction (training labels only use analyst decisions known at the split date)
   - Early stopping with validation split to prevent memorization
   - SHAP explainability for every scored case

6. **Produced a full analysis report** (PDF) with EDA, methodology, results, SHAP explanations, and business recommendations — written for non-technical stakeholders

## Key Finding

The model works, but **the real bottleneck is analyst resolutions, not the algorithm.** Out of 1,520 flagged cases, only 50 have been resolved by analysts (33 fraud, 17 non-fraud). Only 10 confirmed cases fall in the test set. With that few data points, we can't reliably measure whether the model outperforms random prioritization.

The #1 recommendation: resolve ~100 more cases from the existing backlog to unlock meaningful evaluation.

## What This Demonstrates

- An AI agent can go from raw data to a reviewed, leakage-free ML pipeline **in hours, not weeks**
- The autonomous research loop (experiment → review → fix → iterate) catches real issues that a human might miss under time pressure
- The "review agent" pattern (a separate AI persona acting as senior reviewer) is effective at catching data leakage and overfitting — it flagged 4 separate issues across the session
- Human oversight is still essential for domain judgment calls (e.g., whether to use test-only evaluation, how to handle label maturity)
