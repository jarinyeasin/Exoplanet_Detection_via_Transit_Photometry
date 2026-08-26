"""
Model Evaluation

Input:  X_train.csv, X_test.csv, y_train.csv, y_test.csv
Output: Printed analysis + evaluation_context.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    precision_recall_curve, confusion_matrix, classification_report,
    average_precision_score
)

# 1. Load data and retrain the flags-removed model
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").squeeze()
y_test = pd.read_csv("y_test.csv").squeeze()

FLAG_COLS = ["koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec"]
X_train_nf = X_train.drop(columns=FLAG_COLS)
X_test_nf = X_test.drop(columns=FLAG_COLS)

imputer = SimpleImputer(strategy="median")
X_train_i = pd.DataFrame(imputer.fit_transform(X_train_nf), columns=X_train_nf.columns)
X_test_i = pd.DataFrame(imputer.transform(X_test_nf), columns=X_test_nf.columns)

model = GradientBoostingClassifier(n_estimators=200, random_state=42)
model.fit(X_train_i, y_train)

y_proba = model.predict_proba(X_test_i)[:, 1]
y_pred_default = (y_proba >= 0.5).astype(int)  # standard 0.5 threshold

print("Model: Gradient Boosting, flags removed (physics-only features)")
print(f"Features used: {list(X_train_nf.columns)}")

# 2. Confusion matrix at the default 0.5 threshold, named explicitly
cm = confusion_matrix(y_test, y_pred_default)
tn, fp, fn, tp = cm.ravel()

print("\n=== Confusion Matrix at default threshold (0.5) ===")
print(f"True Negatives  (correctly identified FALSE POSITIVE): {tn}")
print(f"False Positives (wrongly called CONFIRMED):             {fp}")
print(f"False Negatives (missed a real CONFIRMED planet):       {fn}")
print(f"True Positives  (correctly identified CONFIRMED):       {tp}")

print(f"\n--> {fn} real exoplanets would be missed entirely by this model.")
print(f"--> {fp} non-planets would be wrongly sent for expensive follow-up observation.")

print("\nFull classification report:")
print(classification_report(y_test, y_pred_default, target_names=["FALSE POSITIVE", "CONFIRMED"]))

# 3. Precision-Recall curve and threshold analysis
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
avg_precision = average_precision_score(y_test, y_proba)

print(f"\nAverage Precision (area under PR curve): {avg_precision:.3f}")

# 4. Explore what happens at a lower threshold
LOWER_THRESHOLD = 0.3

y_pred_lower = (y_proba >= LOWER_THRESHOLD).astype(int)
cm_lower = confusion_matrix(y_test, y_pred_lower)
tn_l, fp_l, fn_l, tp_l = cm_lower.ravel()

print(f"\n=== Comparison: default threshold (0.5) vs lower threshold ({LOWER_THRESHOLD}) ===")
print(f"{'Metric':<20}{'Threshold 0.5':>15}{'Threshold ' + str(LOWER_THRESHOLD):>15}")
print("-" * 50)
print(f"{'Missed planets (FN)':<20}{fn:>15}{fn_l:>15}")
print(f"{'False alarms (FP)':<20}{fp:>15}{fp_l:>15}")
print(f"{'True planets found':<20}{tp:>15}{tp_l:>15}")

print(f"\nLowering the threshold from 0.5 to {LOWER_THRESHOLD}:")
print(f"  Missed planets changed by: {fn_l - fn:+d}")
print(f"  False alarms changed by:   {fp_l - fp:+d}")
print("\nThis is the real tradeoff a mission scientist navigates: how many")
print("false alarms is it worth accepting to avoid missing one more real planet?")
print("There is no single 'correct' threshold -- it depends on how expensive")
print("follow-up telescope time is versus how costly a missed discovery is.")

X_test_reset = X_test_nf.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)

results_df = X_test_reset.copy()
results_df["actual"] = y_test_reset.map({0: "FALSE POSITIVE", 1: "CONFIRMED"})
results_df["predicted_proba"] = y_proba
results_df["predicted"] = np.where(y_pred_default == 1, "CONFIRMED", "FALSE POSITIVE")

false_negatives = results_df[(y_test_reset == 1) & (y_pred_default == 0)]
false_positives = results_df[(y_test_reset == 0) & (y_pred_default == 1)]

print(f"\n=== Sample of {min(5, len(false_negatives))} missed real planets (false negatives) ===")
if len(false_negatives) > 0:
    print(false_negatives[["koi_period", "koi_prad", "koi_model_snr", "predicted_proba"]].head(5).to_string(index=False))
else:
    print("None in this test set.")

print(f"\n=== Sample of {min(5, len(false_positives))} false alarms (false positives) ===")
if len(false_positives) > 0:
    print(false_positives[["koi_period", "koi_prad", "koi_model_snr", "predicted_proba"]].head(5).to_string(index=False))
else:
    print("None in this test set.")

# 6. Visualize
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

axes[0].plot(recalls, precisions, color="steelblue", label=f"AP = {avg_precision:.3f}")
axes[0].scatter(
    [tp / (tp + fn)], [tp / (tp + fp)],
    color="darkorange", zorder=5, s=80, label="Threshold = 0.5"
)
axes[0].scatter(
    [tp_l / (tp_l + fn_l)], [tp_l / (tp_l + fp_l)],
    color="seagreen", zorder=5, s=80, label=f"Threshold = {LOWER_THRESHOLD}"
)
axes[0].set_xlabel("Recall (fraction of real planets found)")
axes[0].set_ylabel("Precision (fraction of predictions that are real)")
axes[0].set_title("Precision-Recall Curve")
axes[0].legend()

axes[1].plot(thresholds, precisions[:-1], label="Precision", color="steelblue")
axes[1].plot(thresholds, recalls[:-1], label="Recall", color="indianred")
axes[1].axvline(0.5, color="darkorange", linestyle="--", alpha=0.6, label="Threshold = 0.5")
axes[1].axvline(LOWER_THRESHOLD, color="seagreen", linestyle="--", alpha=0.6, label=f"Threshold = {LOWER_THRESHOLD}")
axes[1].set_xlabel("Decision Threshold")
axes[1].set_ylabel("Score")
axes[1].set_title("Precision & Recall vs Threshold")
axes[1].legend()

plt.tight_layout()
plt.savefig("evaluation_context.png", dpi=110)
print("\nSaved evaluation_context.png")
print("\nevaluation complete.")