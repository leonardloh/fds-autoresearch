import time
import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from prepare import prepare


def train():
    t0 = time.time()

    X_train, X_test, y_train, y_test, feature_cols = prepare()

    # Class weight for imbalanced data
    n_neg = (y_train == 0).sum()
    n_pos = max((y_train == 1).sum(), 1)
    scale_pos_weight = n_neg / n_pos

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    train_start = time.time()
    model.fit(X_train, y_train)
    training_seconds = time.time() - train_start

    # Threshold tuning via CV on training data (not test set)
    best_thresh = 0.5
    if y_train.sum() >= 5:
        kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        best_cv_f1 = 0
        for t in np.arange(0.05, 0.95, 0.01):
            cv_scores = []
            for tr_idx, val_idx in kf.split(X_train, y_train):
                m = xgb.XGBClassifier(
                    n_estimators=200, max_depth=6, learning_rate=0.1,
                    scale_pos_weight=scale_pos_weight, eval_metric="logloss",
                    random_state=42, n_jobs=-1,
                )
                m.fit(X_train[tr_idx], y_train[tr_idx])
                p = m.predict_proba(X_train[val_idx])[:, 1]
                cv_scores.append(f1_score(y_train[val_idx], (p >= t).astype(int), zero_division=0))
            mean_f1 = np.mean(cv_scores)
            if mean_f1 > best_cv_f1:
                best_cv_f1 = mean_f1
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


if __name__ == "__main__":
    train()
