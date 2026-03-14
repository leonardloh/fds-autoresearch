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
     train_case_nos, test_case_nos,
     confirmed_fraud, confirmed_nf) = result

    model = xgb.XGBClassifier(
        n_estimators=2000,
        max_depth=8,
        learning_rate=0.015,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.5,
        scale_pos_weight=1.0,
        gamma=0.5,
        reg_alpha=0.5,
        reg_lambda=5.0,
        eval_metric="logloss",
        random_state=6,
        n_jobs=-1,
    )

    train_start = time.time()
    model.fit(X_train, y_train)
    training_seconds = time.time() - train_start

    # === Standard binary evaluation (fixed threshold) ===
    y_proba_test = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba_test >= 0.5).astype(int)

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

    # === Prioritization: score ALL originally-flagged transactions ===
    # Use case_no > 0 (original flag), NOT the cleaned label
    y_proba_train = model.predict_proba(X_train)[:, 1]

    all_probas = np.concatenate([y_proba_train, y_proba_test])
    all_case_nos = np.concatenate([train_case_nos, test_case_nos])

    # Flagged = originally had a case (case_no > 0), regardless of label cleanup
    flagged_mask = all_case_nos > 0
    flagged_probas = all_probas[flagged_mask]
    flagged_case_nos = all_case_nos[flagged_mask]

    case_scores = {}
    case_mean_scores = {}
    case_txn_counts = {}
    for cn, prob in zip(flagged_case_nos, flagged_probas):
        if cn not in case_scores:
            case_scores[cn] = prob
            case_mean_scores[cn] = [prob]
            case_txn_counts[cn] = 1
        else:
            case_scores[cn] = max(case_scores[cn], prob)
            case_mean_scores[cn].append(prob)
            case_txn_counts[cn] += 1

    for cn in case_mean_scores:
        case_mean_scores[cn] = np.mean(case_mean_scores[cn])

    # Evaluate on confirmed cases
    confirmed_all = confirmed_fraud | confirmed_nf
    eval_cases = {cn: case_scores[cn] for cn in case_scores if cn in confirmed_all}

    if len(eval_cases) > 0:
        eval_cn = list(eval_cases.keys())
        eval_scores = np.array([eval_cases[cn] for cn in eval_cn])
        eval_labels = np.array([1 if cn in confirmed_fraud else 0 for cn in eval_cn])

        n_fraud = int(eval_labels.sum())
        n_nf = len(eval_labels) - n_fraud
        print(f"=== PRIORITIZATION (all confirmed cases) ===")
        print(f"Confirmed fraud cases: {n_fraud}")
        print(f"Confirmed non-fraud cases: {n_nf}")
        print(f"Total flagged cases scored: {len(case_scores)}")
        print()

        if n_fraud > 0 and n_nf > 0:
            auc = roc_auc_score(eval_labels, eval_scores)
            ap = average_precision_score(eval_labels, eval_scores)
            print(f"AUC (case-level, max score):   {auc:.4f}")
            print(f"Avg Precision (max score):     {ap:.4f}")

            # Also evaluate with mean score
            eval_mean = np.array([case_mean_scores[cn] for cn in eval_cn])
            auc_mean = roc_auc_score(eval_labels, eval_mean)
            print(f"AUC (case-level, mean score):  {auc_mean:.4f}")
            print()

            # Ranked list
            ranked = sorted(zip(eval_cn, eval_scores, eval_labels),
                            key=lambda x: -x[1])
            print(f"Ranked confirmed cases (highest risk first):")
            print(f"  {'Rank':>4}  {'Case':>8}  {'Score':>8}  {'Label':>10}")
            for i, (cn, sc, lb) in enumerate(ranked):
                label_str = "FRAUD" if lb == 1 else "clean"
                print(f"  {i+1:>4}  {cn:>8}  {sc:>8.4f}  {label_str:>10}")

            print()
            # Precision@k
            for k in [5, 10, 15, 20, 25]:
                if k <= len(ranked):
                    top_k_labels = [lb for _, _, lb in ranked[:k]]
                    p_at_k = sum(top_k_labels) / k
                    print(f"Precision@{k:>2}: {p_at_k:.3f} ({sum(top_k_labels)}/{k} fraud)")

            # How many cases would analyst need to review to catch all fraud?
            fraud_found = 0
            for i, (cn, sc, lb) in enumerate(ranked):
                if lb == 1:
                    fraud_found += 1
                if fraud_found == n_fraud:
                    print(f"\nAll {n_fraud} fraud cases found after reviewing {i+1}/{len(ranked)} cases ({100*(i+1)/len(ranked):.1f}%)")
                    break


if __name__ == "__main__":
    train()
