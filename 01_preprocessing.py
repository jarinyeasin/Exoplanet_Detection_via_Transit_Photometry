"""
Data Cleaning and Preprocessing
Exoplanet Detection Project - Kepler Objects of Interest (KOI)

Input:  koi_cumulative.csv  (raw download from NASA Exoplanet Archive)
Output: koi_clean.csv, X_train.csv, X_test.csv, y_train.csv, y_test.csv
"""

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_FILE = "koi_cumulative.csv"

df = pd.read_csv(RAW_FILE, comment="#")
print(f"Loaded {RAW_FILE}: {df.shape[0]} rows, {df.shape[1]} columns")

df_binary = df[df["koi_disposition"] != "CANDIDATE"].copy()
print(f"\nAfter dropping CANDIDATE rows: {len(df_binary)} rows")
print(df_binary["koi_disposition"].value_counts())

key_cols_for_missing_check = [
    "koi_depth", "koi_prad", "koi_teq", "koi_model_snr",
    "koi_impact", "koi_steff", "koi_slogg", "koi_srad", "koi_insol"
]

missing_mask = df_binary[key_cols_for_missing_check].isnull().any(axis=1)
print(f"\nRows with missing key values: {missing_mask.sum()}")
print("Disposition breakdown of missing rows:")
print(df_binary[missing_mask]["koi_disposition"].value_counts())

df_clean = df_binary.dropna(subset=key_cols_for_missing_check).copy()
print(f"\nAfter dropping missing rows: {len(df_clean)} rows")
print("New class balance:")
print(df_clean["koi_disposition"].value_counts(normalize=True).round(3) * 100)

df_clean.to_csv("koi_clean.csv", index=False)
print("\nSaved koi_clean.csv")

feature_cols = [
    "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec",
    "koi_period", "koi_impact", "koi_duration", "koi_depth",
    "koi_prad", "koi_teq", "koi_insol", "koi_model_snr",
    "koi_steff", "koi_slogg", "koi_srad", "koi_kepmag",
]

X = df_clean[feature_cols]
y = (df_clean["koi_disposition"] == "CONFIRMED").astype(int)  # 1=CONFIRMED, 0=FALSE POSITIVE

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target distribution: {y.value_counts().to_dict()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain shape: {X_train.shape}")
print(f"Test shape:  {X_test.shape}")
print(f"Train class balance: {y_train.value_counts(normalize=True).round(3).to_dict()}")
print(f"Test class balance:  {y_test.value_counts(normalize=True).round(3).to_dict()}")

X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("\nSaved: X_train.csv, X_test.csv, y_train.csv, y_test.csv")
print("\nPreprocessing complete.")