"""
Shap Analysis

Input:  X_train.csv, X_test.csv, y_train.csv, y_test.csv
Output: Printed analysis + shap_summary.png, shap_individual_examples.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer

# 1. Load data
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

print("Model retrained: Gradient Boosting, flags removed (physics-only)")
print(f"Features: {list(X_train_nf.columns)}")

# 2. Compute SHAP values
print("\nComputing SHAP values (this may take a moment)...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_i)
base_value = explainer.expected_value
if isinstance(base_value, (list, np.ndarray)):
    base_value = base_value[0] if len(np.atleast_1d(base_value)) == 1 else base_value[-1]

print(f"SHAP values shape: {shap_values.shape}")
print(f"Base value (average model output before any features applied): {base_value:.3f}")
print("(This is roughly the log-odds of CONFIRMED averaged across the training set --")
print(" each feature's SHAP value shows how much it pushes an individual prediction")
print(" away from this baseline.)")

# 3. Global importance via mean absolute SHAP value
mean_abs_shap = pd.DataFrame({
    "feature": X_test_i.columns,
    "mean_abs_shap": np.abs(shap_values).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False)

print("\n=== Global feature importance (mean |SHAP value|) ===")
print(mean_abs_shap.to_string(index=False))

# 4. Directional analysis: does a HIGH value of each top feature push
#    toward CONFIRMED or FALSE POSITIVE?
print("\n=== Direction of effect for top 5 features ===")
top5_features = mean_abs_shap["feature"].head(5).tolist()

for feat in top5_features:
    feat_idx = list(X_test_i.columns).index(feat)
    feat_values = X_test_i[feat].values
    feat_shap = shap_values[:, feat_idx]

    # correlation between feature value and its SHAP value tells us direction
    correlation = np.corrcoef(feat_values, feat_shap)[0, 1]
    direction = "HIGHER values push toward CONFIRMED" if correlation > 0 else "HIGHER values push toward FALSE POSITIVE"
    print(f"  {feat:<15} correlation={correlation:+.3f}  -->  {direction}")

# 5. Explain individual predictions
y_proba = model.predict_proba(X_test_i)[:, 1]

X_test_reset = X_test_i.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)

confident_confirmed_idx = np.argmax(y_proba * (y_test_reset.values == 1))
confident_falsepos_idx = np.argmax((1 - y_proba) * (y_test_reset.values == 0))
uncertain_idx = np.argmin(np.abs(y_proba - 0.5))

examples = {
    "Confident CONFIRMED prediction": confident_confirmed_idx,
    "Confident FALSE POSITIVE prediction": confident_falsepos_idx,
    "Most uncertain prediction (near 0.5)": uncertain_idx,
}

print("\n" + "=" * 70)
print("INDIVIDUAL PREDICTION EXPLANATIONS")
print("=" * 70)

for label, idx in examples.items():
    print(f"\n--- {label} ---")
    print(f"Predicted probability of CONFIRMED: {y_proba[idx]:.3f}")
    print(f"Actual label: {'CONFIRMED' if y_test_reset.iloc[idx]==1 else 'FALSE POSITIVE'}")
    print("Top contributing features:")

    row_shap = shap_values[idx]
    row_features = X_test_reset.iloc[idx]
    contribution_df = pd.DataFrame({
        "feature": X_test_reset.columns,
        "value": row_features.values,
        "shap_value": row_shap
    }).sort_values("shap_value", key=abs, ascending=False)

    for _, r in contribution_df.head(4).iterrows():
        push = "toward CONFIRMED" if r["shap_value"] > 0 else "toward FALSE POSITIVE"
        print(f"    {r['feature']:<15} = {r['value']:>10.3f}   "
              f"(pushed {push}, magnitude {abs(r['shap_value']):.3f})")

# 6. Visualize
fig, ax = plt.subplots(figsize=(9, 6))
shap.summary_plot(shap_values, X_test_i, show=False, plot_size=None)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=110, bbox_inches="tight")
plt.close()
print("\nSaved shap_summary.png")
print("  (Each dot = one candidate. Red = high feature value, blue = low.")
print("   Position on x-axis = how much that feature pushed this specific")
print("   prediction toward CONFIRMED (right) or FALSE POSITIVE (left).)")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, (label, idx) in zip(axes, examples.items()):
    row_shap = shap_values[idx]
    row_features = X_test_reset.iloc[idx]
    contribution_df = pd.DataFrame({
        "feature": X_test_reset.columns,
        "shap_value": row_shap
    }).sort_values("shap_value", key=abs, ascending=True).tail(8)

    colors = ["steelblue" if v > 0 else "indianred" for v in contribution_df["shap_value"]]
    ax.barh(contribution_df["feature"], contribution_df["shap_value"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"{label}\n(P(CONFIRMED)={y_proba[idx]:.2f})", fontsize=9)
    ax.set_xlabel("SHAP value (- toward FALSE POS | + toward CONFIRMED)")

plt.tight_layout()
plt.savefig("shap_individual_examples.png", dpi=110)
print("Saved shap_individual_examples.png")

print("\nComplete.")
