"""
Random Forest & Gradient Boosting

Input:  X_train.csv, X_test.csv, y_train.csv, y_test.csv
Output: Printed comparison table + model_comparison.png

  1. Random Forest       - all 16 features (including pipeline flags)
  2. Gradient Boosting    - all 16 features (including pipeline flags)
  3. Random Forest       - flags REMOVED (raw physics only)
  4. Gradient Boosting    - flags REMOVED (raw physics only)

Why the flags-removed experiments matter:
    The baseline logistic regression scored ~99% accuracy, but almost
  entirely by leaning on koi_fpflag_* columns -- which are the PIPELINE'S
  OWN pre-computed verdict on whether a signal looks like a real transit.
  That's closer to memorizing an existing decision than learning the
  underlying transit physics. Removing those four flags and re-training
  tells us how well the model does using only genuine signal/star physics
  (period, depth, radius, temperature, SNR, etc.).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

# 1. Load train/test splits from Step 4
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").squeeze()
y_test = pd.read_csv("y_test.csv").squeeze()

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# 2. Impute any stray missing values

imputer = SimpleImputer(strategy="median")
X_train_full = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test_full = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# 3. Build the "flags removed" versions of the data
FLAG_COLS = ["koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec"]

X_train_noflags = X_train_full.drop(columns=FLAG_COLS)
X_test_noflags = X_test_full.drop(columns=FLAG_COLS)

print(f"\nFull feature set: {list(X_train_full.columns)}")
print(f"\nFlags-removed feature set: {list(X_train_noflags.columns)}")

# 4. Define a helper to train + evaluate + collect results
def evaluate_model(name, model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]

    results = {
        "name": name,
        "accuracy": accuracy_score(y_te, y_pred),
        "precision": precision_score(y_te, y_pred),
        "recall": recall_score(y_te, y_pred),
        "f1": f1_score(y_te, y_pred),
        "roc_auc": roc_auc_score(y_te, y_proba),
        "y_pred": y_pred,
        "y_proba": y_proba,
        "model": model,
        "feature_names": list(X_tr.columns),
    }
    return results


# 5. Run all four experiments
experiments = []

rf_full = RandomForestClassifier(
    n_estimators=200, random_state=42, class_weight="balanced"
)
experiments.append(evaluate_model(
    "Random Forest (all features)", rf_full,
    X_train_full, X_test_full, y_train, y_test
))

gb_full = GradientBoostingClassifier(n_estimators=200, random_state=42)
experiments.append(evaluate_model(
    "Gradient Boosting (all features)", gb_full,
    X_train_full, X_test_full, y_train, y_test
))

rf_noflags = RandomForestClassifier(
    n_estimators=200, random_state=42, class_weight="balanced"
)
experiments.append(evaluate_model(
    "Random Forest (flags removed)", rf_noflags,
    X_train_noflags, X_test_noflags, y_train, y_test
))

gb_noflags = GradientBoostingClassifier(n_estimators=200, random_state=42)
experiments.append(evaluate_model(
    "Gradient Boosting (flags removed)", gb_noflags,
    X_train_noflags, X_test_noflags, y_train, y_test
))

# 6. Print comparison table
print("\n" + "=" * 90)
print("MODEL COMPARISON")
print("=" * 90)
header = f"{'Model':<38}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>8}{'ROC-AUC':>10}"
print(header)
print("-" * 90)
for exp in experiments:
    print(f"{exp['name']:<38}{exp['accuracy']:>10.3f}{exp['precision']:>11.3f}"
          f"{exp['recall']:>9.3f}{exp['f1']:>8.3f}{exp['roc_auc']:>10.3f}")
print("=" * 90)

print("\nInterpretation:")
print("- 'All features' models include koi_fpflag_* (pipeline's own diagnostic flags)")
print("- 'Flags removed' models rely only on raw transit/star physics")
print("- The gap between these two groups shows how much of the ~99% baseline")
print("  accuracy came from leaning on a pre-existing pipeline judgment, versus")
print("  how well the model learns the underlying transit physics on its own.")

# 7. Feature importance for the best "flags removed" model
best_noflags = max(
    [e for e in experiments if "flags removed" in e["name"]],
    key=lambda e: e["f1"]
)
importances = pd.DataFrame({
    "feature": best_noflags["feature_names"],
    "importance": best_noflags["model"].feature_importances_
}).sort_values("importance", ascending=False)

print(f"\nFeature importances -- {best_noflags['name']}:")
print(importances.to_string(index=False))

# 8. Visualize: ROC curves for all 4 + feature importance bar chart
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

colors = ["steelblue", "indianred", "seagreen", "darkorange"]
for exp, color in zip(experiments, colors):
    fpr, tpr, _ = roc_curve(y_test, exp["y_proba"])
    axes[0].plot(fpr, tpr, label=f"{exp['name']} (AUC={exp['roc_auc']:.3f})", color=color)
axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curves: All Four Models")
axes[0].legend(fontsize=8)

axes[1].barh(importances["feature"], importances["importance"], color="steelblue")
axes[1].set_xlabel("Importance")
axes[1].set_title(f"Feature Importance\n({best_noflags['name']})")
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=110)
print("\nSaved model_comparison.png")
print("\nComplete.")