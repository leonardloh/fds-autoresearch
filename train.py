import time
import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
from prepare import prepare


def train():
    t0 = time.time()

    X_train, X_test, y_train, y_test, feature_cols = prepare()

    # Class weight
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

    # Threshold tuning: train once per CV fold, sweep thresholds on predictions
    best_thresh = 0.5
    if y_train.sum() >= 5:
        tscv = TimeSeriesSplit(n_splits=3)
        fold_probas = []
        fold_labels = []
        for tr_idx, val_idx in tscv.split(X_train):
            if y_train[tr_idx].sum() == 0:
                continue
            fold_neg = (y_train[tr_idx] == 0).sum()
            fold_pos = max((y_train[tr_idx] == 1).sum(), 1)
            m = xgb.XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                scale_pos_weight=fold_neg / fold_pos,
                eval_metric="logloss", random_state=42, n_jobs=-1,
            )
            m.fit(X_train[tr_idx], y_train[tr_idx])
            fold_probas.append(m.predict_proba(X_train[val_idx])[:, 1])
            fold_labels.append(y_train[val_idx])

        if fold_probas:
            all_probas = np.concatenate(fold_probas)
            all_labels = np.concatenate(fold_labels)
            best_cv_f1 = 0
            for t in np.arange(0.05, 0.95, 0.01):
                score = f1_score(all_labels, (all_probas >= t).astype(int), zero_division=0)
                if score > best_cv_f1:
                    best_cv_f1 = score
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
