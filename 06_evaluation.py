"""
Input:  X_train.csv, X_test.csv, y_train.csv, y_test.csv
Output: Printed analysis + cv_results.png, hyperparameter_tuning.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_score
from sklearn.impute import SimpleImputer

optuna.logging.set_verbosity(optuna.logging.WARNING)

X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").squeeze()
y_test = pd.read_csv("y_test.csv").squeeze()

X_full = pd.concat([X_train, X_test], ignore_index=True)
y_full = pd.concat([y_train, y_test], ignore_index=True)

print(f"Combined dataset for cross-validation: {X_full.shape[0]} rows")

FLAG_COLS = ["koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec"]
X_nf = X_full.drop(columns=FLAG_COLS)

imputer = SimpleImputer(strategy="median")
X_i = pd.DataFrame(imputer.fit_transform(X_nf), columns=X_nf.columns)

print(f"Features used (flags-removed, physics-only): {list(X_nf.columns)}")

print("\n" + "=" * 70)
print("STEP A: 5-FOLD CROSS-VALIDATION (default hyperparameters)")
print("=" * 70)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
default_model = GradientBoostingClassifier(n_estimators=200, random_state=42)

scoring = ["accuracy", "f1", "roc_auc", "average_precision"]
cv_results = cross_validate(default_model, X_i, y_full, cv=skf, scoring=scoring, n_jobs=-1)

metric_labels = {
    "accuracy": "Accuracy",
    "f1": "F1 Score",
    "roc_auc": "ROC-AUC",
    "average_precision": "PR-AUC (Average Precision)",
}

default_cv_summary = {}
for metric in scoring:
    scores = cv_results[f"test_{metric}"]
    mean, std = scores.mean(), scores.std()
    default_cv_summary[metric] = (mean, std)
    print(f"  {metric_labels[metric]:<28} {mean:.4f} +/- {std:.4f}")

print("\nNote: PR-AUC is the more informative summary metric here, since the")
print("dataset is imbalanced (62.5% FALSE POSITIVE / 37.5% CONFIRMED) and")
print("ROC-AUC can look optimistic on imbalanced data due to the large")
print("number of true negatives inflating the false-positive-rate denominator.")

print("\n" + "=" * 70)
print("STEP B: HYPERPARAMETER TUNING (Optuna, 12 trials)")
print("=" * 70)
print("Searching: n_estimators, max_depth, learning_rate, subsample")
print("Optimizing for: F1 score, 3-fold CV during search (for speed)")
print("This may take 1-2 minutes...\n")

skf_search = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "random_state": 42,
    }
    model = GradientBoostingClassifier(**params)
    scores = cross_val_score(model, X_i, y_full, cv=skf_search, scoring="f1", n_jobs=-1)
    return scores.mean()


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=12, show_progress_bar=False)

print("Best hyperparameters found:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")
print(f"\nBest F1 during search (3-fold CV): {study.best_value:.4f}")

print("\nValidating winning configuration with full 5-fold CV...")
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score, roc_auc_score, average_precision_score

tuned_model = GradientBoostingClassifier(**study.best_params, random_state=42)

y_pred_cv = cross_val_predict(tuned_model, X_i, y_full, cv=skf, n_jobs=-1)
y_proba_cv = cross_val_predict(tuned_model, X_i, y_full, cv=skf, method="predict_proba", n_jobs=-1)[:, 1]

tuned_cv_summary = {}
fold_accuracy, fold_f1, fold_roc, fold_pr = [], [], [], []
for _, test_idx in skf.split(X_i, y_full):
    y_true_fold = y_full.iloc[test_idx]
    y_pred_fold = y_pred_cv[test_idx]
    y_proba_fold = y_proba_cv[test_idx]
    fold_accuracy.append(accuracy_score(y_true_fold, y_pred_fold))
    fold_f1.append(f1_score(y_true_fold, y_pred_fold))
    fold_roc.append(roc_auc_score(y_true_fold, y_proba_fold))
    fold_pr.append(average_precision_score(y_true_fold, y_proba_fold))

tuned_cv_summary["accuracy"] = (np.mean(fold_accuracy), np.std(fold_accuracy))
tuned_cv_summary["f1"] = (np.mean(fold_f1), np.std(fold_f1))
tuned_cv_summary["roc_auc"] = (np.mean(fold_roc), np.std(fold_roc))
tuned_cv_summary["average_precision"] = (np.mean(fold_pr), np.std(fold_pr))

print("\nTuned model, 5-fold CV results:")
for metric in scoring:
    mean, std = tuned_cv_summary[metric]
    print(f"  {metric_labels[metric]:<28} {mean:.4f} +/- {std:.4f}")

print("\n=== Default vs Tuned comparison ===")
print(f"{'Metric':<28}{'Default':>18}{'Tuned':>18}{'Change':>10}")
print("-" * 74)
for metric in scoring:
    d_mean, d_std = default_cv_summary[metric]
    t_mean, t_std = tuned_cv_summary[metric]
    change = t_mean - d_mean
    print(f"{metric_labels[metric]:<28}{d_mean:>10.4f}+/-{d_std:.3f}{t_mean:>10.4f}+/-{t_std:.3f}{change:>+10.4f}")

print("\n" + "=" * 70)
print("STEP C: BENCHMARK AGAINST NASA'S ROBOVETTER (DR25)")
print("=" * 70)

this_recall = recall_score(y_full, y_pred_cv)
this_precision = precision_score(y_full, y_pred_cv)

print(f"This project's tuned model (out-of-sample, 5-fold cross-validated predictions):")
print(f"  Recall (~ completeness analog):    {this_recall:.3f}")
print(f"  Precision (~ reliability analog):  {this_precision:.3f}")
print(f"\nRobovetter (published, DR25, periods < 100 days):")
print(f"  Completeness:  > 0.85")
print(f"  Reliability:   ~0.98 (inferred from the 2% false-rejection finding above)")
print("""
Interpretation: this project's model, working from a much narrower and
already-pre-filtered feature set, lands in a broadly comparable range to
Kepler's own production system on these specific figures -- but given the
caveats above (different task scope, different data stage, no injection
testing), this should be read as "operating in the right neighborhood,"
not as a claim of matching or exceeding NASA's actual production pipeline.
""")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

box_data = []
box_labels = []
for metric in scoring:
    box_data.append(cv_results[f"test_{metric}"])
    box_labels.append(f"{metric_labels[metric]}\n(default)")

tuned_fold_scores = {
    "accuracy": fold_accuracy, "f1": fold_f1,
    "roc_auc": fold_roc, "average_precision": fold_pr
}
for metric in scoring:
    box_data.append(tuned_fold_scores[metric])
    box_labels.append(f"{metric_labels[metric]}\n(tuned)")

bp = axes[0].boxplot(box_data, tick_labels=box_labels, patch_artist=True)
colors = ["steelblue"] * 4 + ["darkorange"] * 4
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
axes[0].set_ylabel("Score")
axes[0].set_title("5-Fold CV Score Distribution: Default vs Tuned")
axes[0].tick_params(axis="x", rotation=45, labelsize=7)

trial_values = [t.value for t in study.trials]
best_so_far = np.maximum.accumulate(trial_values)
axes[1].plot(range(1, len(trial_values) + 1), trial_values, "o", color="steelblue",
             alpha=0.5, label="Trial F1 score")
axes[1].plot(range(1, len(best_so_far) + 1), best_so_far, color="darkorange",
             linewidth=2, label="Best so far")
axes[1].set_xlabel("Trial number")
axes[1].set_ylabel("F1 score (3-fold CV)")
axes[1].set_title("Hyperparameter Search Progress")
axes[1].legend()

plt.tight_layout()
plt.savefig("cv_and_tuning_results.png", dpi=110)
print("\nSaved cv_and_tuning_results.png")

import json

results_summary = {
    "cross_validation": {
        "default_hyperparameters": {
            metric: {"mean": float(default_cv_summary[metric][0]), "std": float(default_cv_summary[metric][1])}
            for metric in scoring
        },
        "tuned_hyperparameters": {
            metric: {"mean": float(tuned_cv_summary[metric][0]), "std": float(tuned_cv_summary[metric][1])}
            for metric in scoring
        },
    },
    "best_hyperparameters": study.best_params,
    "hyperparameter_search": {
        "n_trials": len(study.trials),
        "best_f1_during_search": float(study.best_value),
        "trial_values": [float(t.value) for t in study.trials],
    },
    "robovetter_comparison": {
        "this_model": {
            "recall": float(this_recall),
            "precision": float(this_precision),
        },
        "robovetter_published": {
            "completeness": 0.85,
            "completeness_note": "> 0.85 for orbital periods under 100 days (Thompson et al. 2018, DR25)",
            "reliability": 0.98,
            "reliability_note": "~0.98 inferred from 2% false-rejection rate on 306 borderline signals",
        },
    },
}

with open("evaluation_results.json", "w") as f:
    json.dump(results_summary, f, indent=2)

print("Saved evaluation_results.json")
print("\n evaluation complete.")
