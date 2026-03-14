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
     test_case_nos, confirmed_fraud, confirmed_nf) = result

    # FIX #3: Use validation split for early stopping instead of no regularization
    # Split training into train (85%) + validation (15%) for early stopping
    val_size = int(len(X_train) * 0.15)
    val_idx = len(X_train) - val_size
    X_tr = X_train[:val_idx]
    y_tr = y_train[:val_idx]
    X_val = X_train[val_idx:]
    y_val = y_train[val_idx:]

    # FIX #3: Reasonable depth (4) with early stopping to prevent memorization
    model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=4,
        learning_rate=0.05,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.6,
        scale_pos_weight=1.0,
        gamma=1.0,
        reg_alpha=0.5,
        reg_lambda=5.0,
        eval_metric="logloss",
        random_state=6,
        n_jobs=-1,
        early_stopping_rounds=50,
    )

    train_start = time.time()
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    training_seconds = time.time() - train_start
    best_iter = model.best_iteration

    # === Binary evaluation on HELD-OUT test set only ===
    y_proba_test = model.predict_proba(X_test)[:, 1]

    # Threshold tuning on validation set (not test)
    val_proba = model.predict_proba(X_val)[:, 1]
    best_thresh = 0.5
    best_val_f1 = 0
    for t in np.arange(0.05, 0.95, 0.01):
        score = f1_score(y_val, (val_proba >= t).astype(int), zero_division=0)
        if score > best_val_f1:
            best_val_f1 = score
            best_thresh = t

    y_pred = (y_proba_test >= best_thresh).astype(int)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    # Train metrics for overfit check
    y_proba_train = model.predict_proba(X_tr)[:, 1]
    train_f1 = f1_score(y_tr, (y_proba_train >= best_thresh).astype(int), zero_division=0)

    total_seconds = time.time() - t0

    print("---")
    print(f"f1:          {f1:.6f}")
    print(f"precision:  {precision:.6f}")
    print(f"recall:     {recall:.6f}")
    print(f"train_f1:   {train_f1:.6f}")
    print(f"best_iter:  {best_iter}")
    print(f"threshold:  {best_thresh:.2f}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"total_seconds:    {total_seconds:.1f}")
    print("---")
    print()

    # === FIX #1: Prioritization evaluated on TEST SET ONLY ===
    # Only score test-set flagged transactions with truly unseen predictions
    flagged_mask = test_case_nos > 0
    flagged_probas = y_proba_test[flagged_mask]
    flagged_case_nos = test_case_nos[flagged_mask]

    case_scores = {}
    for cn, prob in zip(flagged_case_nos, flagged_probas):
        if cn not in case_scores or prob > case_scores[cn]:
            case_scores[cn] = prob

    # Evaluate on confirmed cases in test set only
    confirmed_all = confirmed_fraud | confirmed_nf
    eval_cases = {cn: case_scores[cn] for cn in case_scores if cn in confirmed_all}

    if len(eval_cases) > 0:
        eval_cn = list(eval_cases.keys())
        eval_scores = np.array([eval_cases[cn] for cn in eval_cn])
        eval_labels = np.array([1 if cn in confirmed_fraud else 0 for cn in eval_cn])

        n_fraud = int(eval_labels.sum())
        n_nf = len(eval_labels) - n_fraud
        print(f"=== PRIORITIZATION (TEST SET confirmed cases only) ===")
        print(f"Confirmed fraud cases in test: {n_fraud}")
        print(f"Confirmed non-fraud cases in test: {n_nf}")
        print(f"Total flagged cases in test: {len(case_scores)}")
        print()

        if n_fraud > 0 and n_nf > 0:
            auc = roc_auc_score(eval_labels, eval_scores)
            ap = average_precision_score(eval_labels, eval_scores)
            print(f"AUC (case-level, test only):   {auc:.4f}")
            print(f"Avg Precision (test only):     {ap:.4f}")
            print()

            ranked = sorted(zip(eval_cn, eval_scores, eval_labels),
                            key=lambda x: -x[1])

            for k in [3, 5, 8]:
                if k <= len(ranked):
                    top_k_labels = [lb for _, _, lb in ranked[:k]]
                    p_at_k = sum(top_k_labels) / k
                    print(f"Precision@{k:>2}: {p_at_k:.3f} ({sum(top_k_labels)}/{k} fraud)")

            print(f"\nRanked confirmed cases (test set):")
            print(f"  {'Rank':>4}  {'Case':>8}  {'Score':>8}  {'Label':>10}")
            for i, (cn, sc, lb) in enumerate(ranked):
                label_str = "FRAUD" if lb == 1 else "clean"
                print(f"  {i+1:>4}  {cn:>8}  {sc:>8.4f}  {label_str:>10}")
        else:
            print(f"Not enough confirmed cases of both types in test set")
            print(f"(need both fraud and non-fraud; have {n_fraud} fraud, {n_nf} non-fraud)")
    else:
        print("No confirmed cases found in test set")


if __name__ == "__main__":
    train()
