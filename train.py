import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
)
from prepare import prepare_prioritization


def train():
    t0 = time.time()

    result = prepare_prioritization()
    (X_train, X_test, y_train, y_test, feature_cols,
     test_case_nos, test_confirmed_fraud, test_confirmed_nf) = result

    # Split training into train (85%) + calibration (15%) for threshold tuning
    cal_size = int(len(X_train) * 0.15)
    cal_idx = len(X_train) - cal_size
    X_tr = X_train[:cal_idx]
    y_tr = y_train[:cal_idx]
    X_cal = X_train[cal_idx:]
    y_cal = y_train[cal_idx:]

    model = xgb.XGBClassifier(
        n_estimators=800,
        max_depth=3,
        learning_rate=0.1,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.5,
        scale_pos_weight=1.0,
        gamma=2.0,
        reg_alpha=0.5,
        reg_lambda=5.0,
        eval_metric="logloss",
        random_state=6,
        n_jobs=-1,
    )

    train_start = time.time()
    model.fit(X_tr, y_tr)
    training_seconds = time.time() - train_start

    # === Standard binary evaluation (flagged vs not-flagged) ===
    cal_proba = model.predict_proba(X_cal)[:, 1]
    best_thresh = 0.5
    best_cal_f1 = 0
    for t in np.arange(0.05, 0.95, 0.01):
        score = f1_score(y_cal, (cal_proba >= t).astype(int), zero_division=0)
        if score > best_cal_f1:
            best_cal_f1 = score
            best_thresh = t

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= best_thresh).astype(int)

    f1 = f1_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    total_seconds = time.time() - t0

    print("---")
    print(f"f1:          {f1:.6f}")
    print(f"precision:  {precision:.6f}")
    print(f"recall:     {recall:.6f}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"total_seconds:    {total_seconds:.1f}")
    print("---")
    print()

    # === Prioritization evaluation on confirmed cases ===
    # Among test-set flagged transactions, how well does the model
    # rank confirmed fraud above confirmed non-fraud?
    flagged_mask = y_test == 1  # transactions flagged by rule system
    if flagged_mask.sum() > 0:
        flagged_probas = y_proba[flagged_mask]
        flagged_case_nos = test_case_nos[flagged_mask]

        # Build case-level risk scores (max proba per case)
        case_scores = {}
        for cn, prob in zip(flagged_case_nos, flagged_probas):
            if cn not in case_scores or prob > case_scores[cn]:
                case_scores[cn] = prob

        # Evaluate on confirmed cases only
        confirmed_cases = test_confirmed_fraud | test_confirmed_nf
        eval_cases = {cn: score for cn, score in case_scores.items()
                      if cn in confirmed_cases}

        if len(eval_cases) > 0:
            eval_cn = list(eval_cases.keys())
            eval_scores = np.array([eval_cases[cn] for cn in eval_cn])
            eval_labels = np.array([1 if cn in test_confirmed_fraud else 0
                                    for cn in eval_cn])

            n_fraud = eval_labels.sum()
            n_nf = len(eval_labels) - n_fraud
            print(f"=== PRIORITIZATION (confirmed cases in test set) ===")
            print(f"Confirmed fraud cases: {n_fraud}")
            print(f"Confirmed non-fraud cases: {n_nf}")

            if n_fraud > 0 and n_nf > 0:
                auc = roc_auc_score(eval_labels, eval_scores)
                ap = average_precision_score(eval_labels, eval_scores)
                print(f"AUC (case-level):  {auc:.4f}")
                print(f"Avg Precision:     {ap:.4f}")

                # Rank and show top/bottom
                ranked = sorted(zip(eval_cn, eval_scores, eval_labels),
                                key=lambda x: -x[1])
                print(f"\nTop 10 ranked cases:")
                print(f"  {'Case':>8}  {'Score':>8}  {'Label':>8}")
                for cn, sc, lb in ranked[:10]:
                    label_str = "FRAUD" if lb == 1 else "clean"
                    print(f"  {cn:>8}  {sc:>8.4f}  {label_str:>8}")

                # Precision@k
                for k in [5, 10, 15]:
                    if k <= len(ranked):
                        top_k_labels = [lb for _, _, lb in ranked[:k]]
                        p_at_k = sum(top_k_labels) / k
                        print(f"Precision@{k}: {p_at_k:.3f}")
            else:
                print("Not enough confirmed cases of both types in test set")
        else:
            print("No confirmed cases found in test set flagged transactions")


if __name__ == "__main__":
    train()
