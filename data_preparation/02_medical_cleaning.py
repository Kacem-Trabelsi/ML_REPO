"""
CRISP-DM: Data Preparation â€” Script 2
Dataset : Medicaldataset.csv  (Cardiac / Heart-Attack Prediction)
Output  : cleaned/medical_cleaned.csv
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os

# â”€â”€ Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE  = os.path.join(BASE_DIR, "Medicaldataset.csv")
OUTPUT_DIR  = os.path.join(BASE_DIR, "cleaned")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "medical_cleaned.csv")
SCALER_FILE = os.path.join(OUTPUT_DIR, "medical_scaler.pkl")

# â”€â”€ 1. Load â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
df = pd.read_csv(INPUT_FILE)
print(f"[1] Loaded  â†’  {df.shape[0]} rows Ã— {df.shape[1]} cols")
print(f"    Columns: {df.columns.tolist()}")

# â”€â”€ 2. Verify no missing values / duplicates â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print(f"\n[2] Missing values:\n{df.isnull().sum().to_string()}")
print(f"    Duplicate rows: {df.duplicated().sum()}")

# â”€â”€ 3. Encode target column â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
df["Result"] = df["Result"].str.strip().str.lower().map({"positive": 1, "negative": 0})
print(f"\n[3] Encoded 'Result'  â†’  positive=1 / negative=0")
print(f"    Class distribution:\n{df['Result'].value_counts().to_string()}")
pos_pct = df["Result"].mean() * 100
print(f"    Class imbalance  â†’  {pos_pct:.1f}% positive  (flag for modeling phase)")

# â”€â”€ 4. Verify Gender encoding â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print(f"\n[4] Gender unique values: {sorted(df['Gender'].unique())}")
print("    Gender already binary-encoded (0=Female / 1=Male) â€” no action needed")

# â”€â”€ 5. Outlier detection and capping (IQR Winsorization) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
numeric_cols = ["Age", "Heart rate", "Systolic blood pressure",
                "Diastolic blood pressure", "Blood sugar", "CK-MB", "Troponin"]

print("\n[5] Outlier capping (IQR-based, 1stâ€“99th percentile):")
outlier_report = []
for col in numeric_cols:
    lo  = df[col].quantile(0.01)
    hi  = df[col].quantile(0.99)
    n   = ((df[col] < lo) | (df[col] > hi)).sum()
    df[col] = df[col].clip(lower=lo, upper=hi)
    outlier_report.append({"column": col, "lower_cap": round(lo, 4),
                            "upper_cap": round(hi, 4), "values_capped": n})
    print(f"    {col:35s}  capped {n:3d} values  â†’  [{lo:.4f}, {hi:.4f}]")

# â”€â”€ 6. Feature scaling â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#   Scale only continuous numeric columns (not Gender or the target Result).
scale_cols = numeric_cols  # Age + 6 vitals/labs

scaler    = StandardScaler()
scaled_df = df.copy()
scaled_df[scale_cols] = scaler.fit_transform(df[scale_cols])

print(f"\n[6] StandardScaler applied to: {scale_cols}")
print("    Scaler saved for inverse-transform at modeling stage.")

# Persist the scaler
with open(SCALER_FILE, "wb") as f:
    pickle.dump(scaler, f)
print(f"    Scaler written  â†’  {SCALER_FILE}")

# â”€â”€ 7. Summary statistics post-cleaning â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print(f"\n[7] Post-cleaning shape: {scaled_df.shape[0]} rows Ã— {scaled_df.shape[1]} cols")
print("    Describe (scaled numeric cols):")
print(scaled_df[scale_cols].describe().round(4).to_string())

# â”€â”€ 8. Save â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
scaled_df.to_csv(OUTPUT_FILE, index=False)
print(f"\n[8] Saved  â†’  {OUTPUT_FILE}")

