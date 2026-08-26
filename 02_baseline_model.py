"""
Step 5: Baseline Model
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

# 1. Load train/test splits
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").squeeze()
y_test = pd.read_csv("y_test.csv").squeeze()

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# 2. Handle any stray missing values
remaining_missing = X_train.isnull().sum()
if remaining_missing.sum() > 0:
    print("\nStray missing values found before modeling:")
    print(remaining_missing[remaining_missing > 0])

imputer = SimpleImputer(strategy="median")
X_train_imputed = pd.DataFrame(
    imputer.fit_transform(X_train), columns=X_train.columns
)
X_test_imputed = pd.DataFrame(
    imputer.transform(X_test), columns=X_test.columns
)

# 3. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# 4. Train baseline logistic regression
model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
model.fit(X_train_scaled, y_train)

# 5. Evaluate on the held-out test set
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]  # probability of CONFIRMED

print("\n=== Baseline Logistic Regression Results ===")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}  "
      "(of predicted CONFIRMED, how many were actually real?)")
print(f"Recall:    {recall_score(y_test, y_pred):.3f}  "
      "(of actual CONFIRMED planets, how many did we find?)")
print(f"F1 score:  {f1_score(y_test, y_pred):.3f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.3f}")

print("\nFull classification report:")
print(classification_report(y_test, y_pred, target_names=["FALSE POSITIVE", "CONFIRMED"]))

cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(f"                Predicted FALSE POS   Predicted CONFIRMED")
print(f"Actual FALSE POS       {cm[0][0]:5d}                {cm[0][1]:5d}")
print(f"Actual CONFIRMED       {cm[1][0]:5d}                {cm[1][1]:5d}")

# 6. Feature importance (logistic regression coefficients)
coef_df = pd.DataFrame({
    "feature": X_train.columns,
    "coefficient": model.coef_[0]
}).sort_values("coefficient", key=abs, ascending=False)

print("\nFeature coefficients (sorted by influence):")
print(coef_df.to_string(index=False))

# 7. Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Confusion matrix heatmap
im = axes[0].imshow(cm, cmap="Blues")
axes[0].set_xticks([0, 1])
axes[0].set_yticks([0, 1])
axes[0].set_xticklabels(["FALSE POSITIVE", "CONFIRMED"])
axes[0].set_yticklabels(["FALSE POSITIVE", "CONFIRMED"])
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")
axes[0].set_title("Confusion Matrix")
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, str(cm[i][j]), ha="center", va="center",
                     color="white" if cm[i][j] > cm.max()/2 else "black", fontsize=14)

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
axes[1].plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_score(y_test, y_proba):.3f}", color="steelblue")
axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curve")
axes[1].legend()

plt.tight_layout()
plt.savefig("baseline_results.png", dpi=110)
print("\nSaved baseline_results.png")
print("\nBaseline Model complete.")