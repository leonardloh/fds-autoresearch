import time
import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
from prepare import prepare


def train():
    t0 = time.time()

    X_train, X_test, y_train, y_test, feature_cols = prepare()

    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=2,
        learning_rate=0.2,
        min_child_weight=5,
        subsample=1.0,
        colsample_bytree=0.5,
        scale_pos_weight=0.7,
        gamma=0.3,
        reg_alpha=0.01,
        reg_lambda=2.0,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    train_start = time.time()
    model.fit(X_train, y_train)
    training_seconds = time.time() - train_start

    # Threshold tuning via CV with minimum threshold of 0.08
    # to prevent over-predicting in low-fraud periods
    best_thresh = 0.5
    if y_train.sum() >= 5:
        tscv = TimeSeriesSplit(n_splits=3)
        fold_probas = []
        fold_labels = []
        for tr_idx, val_idx in tscv.split(X_train):
            if y_train[tr_idx].sum() == 0:
                continue
            m = xgb.XGBClassifier(
                n_estimators=400, max_depth=2, learning_rate=0.2,
                min_child_weight=5, subsample=1.0, colsample_bytree=0.5,
                scale_pos_weight=0.7, gamma=0.3, reg_alpha=0.01, reg_lambda=2.0,
                eval_metric="logloss", random_state=42, n_jobs=-1,
            )
            m.fit(X_train[tr_idx], y_train[tr_idx])
            fold_probas.append(m.predict_proba(X_train[val_idx])[:, 1])
            fold_labels.append(y_train[val_idx])

        if fold_probas:
            all_probas = np.concatenate(fold_probas)
            all_labels = np.concatenate(fold_labels)
            best_cv_f1 = 0
            for t in np.arange(0.08, 0.95, 0.01):
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
