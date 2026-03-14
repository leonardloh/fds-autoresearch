import time
import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score, precision_score, recall_score
from prepare import prepare


def train():
    t0 = time.time()

    X_train, X_test, y_train, y_test, feature_cols = prepare()

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

    # Threshold tuning on calibration set
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


if __name__ == "__main__":
    train()
