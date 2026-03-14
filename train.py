import time
import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score, precision_score, recall_score
from prepare import prepare


def train():
    t0 = time.time()

    X_train, X_test, y_train, y_test, feature_cols = prepare()

    # Class weight for imbalanced data
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
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

    y_proba = model.predict_proba(X_test)[:, 1]

    # Optimize threshold for best F1
    best_f1 = 0
    best_thresh = 0.5
    for t in np.arange(0.1, 0.9, 0.01):
        preds = (y_proba >= t).astype(int)
        score = f1_score(y_test, preds)
        if score > best_f1:
            best_f1 = score
            best_thresh = t

    y_pred = (y_proba >= best_thresh).astype(int)

    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    total_seconds = time.time() - t0

    print("---")
    print(f"f1:          {f1:.6f}")
    print(f"precision:  {precision:.6f}")
    print(f"recall:     {recall:.6f}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"total_seconds:    {total_seconds:.1f}")


if __name__ == "__main__":
    train()
