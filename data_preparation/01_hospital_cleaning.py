"""
CRISP-DM: Data Preparation â€” Script 1
Dataset : Hopsital Dataset.csv
Output  : cleaned/hospital_cleaned.csv
"""

import pandas as pd
import numpy as np
import os

# â”€â”€ Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, "Hopsital Dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "cleaned")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "hospital_cleaned.csv")

# â”€â”€ 1. Load â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
df = pd.read_csv(INPUT_FILE)
print(f"[1] Loaded  â†’  {df.shape[0]} rows Ã— {df.shape[1]} cols")

# â”€â”€ 2. Remove embedded header rows â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#   The raw file contains duplicate header rows mixed into the data.
#   Detected by sentinel values: Gender='Sex', Route='Route', Frequency='Freq'/'Frequency'
header_mask = (
    df["Gender"].isin(["Sex", "Gender"]) |
    df["Route"].isin(["Route"]) |
    df["Frequency"].isin(["Freq", "Frequency"])
)
removed_headers = header_mask.sum()
df = df[~header_mask].reset_index(drop=True)
print(f"[2] Removed {removed_headers} embedded header rows  â†’  {df.shape[0]} rows")

# â”€â”€ 3. Drop duplicate rows â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
dupes = df.duplicated().sum()
df = df.drop_duplicates().reset_index(drop=True)
print(f"[3] Dropped {dupes} duplicate rows  â†’  {df.shape[0]} rows")

# â”€â”€ 4. Fix data types â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
df["Age"]            = pd.to_numeric(df["Age"], errors="coerce")
df["Dosage (gram)"]  = pd.to_numeric(df["Dosage (gram)"], errors="coerce")
df["Duration (days)"] = pd.to_numeric(df["Duration (days)"], errors="coerce")
df["Date of Data Entry"] = pd.to_datetime(df["Date of Data Entry"],
                                           dayfirst=True, errors="coerce")
print(f"[4] Fixed dtypes  â€”  Age, Dosage, Duration â†’ numeric; Date â†’ datetime")

# â”€â”€ 5. Extract datetime features â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
df["entry_year"]        = df["Date of Data Entry"].dt.year
df["entry_month"]       = df["Date of Data Entry"].dt.month
df["entry_day_of_week"] = df["Date of Data Entry"].dt.dayofweek   # 0=Monday
df["entry_hour"]        = df["Date of Data Entry"].dt.hour
df = df.drop(columns=["Date of Data Entry"])
print("[5] Extracted year / month / day_of_week / hour from Date of Data Entry")

# â”€â”€ 6. Standardise text columns â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
text_cols = ["Gender", "Diagnosis", "Name of Drug", "Route", "Frequency", "Indication"]
for col in text_cols:
    df[col] = df[col].astype(str).str.strip().str.lower()
print(f"[6] Standardised text (strip + lower): {text_cols}")

# â”€â”€ 7. Handle 1 missing Indication â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
missing_ind = df["Indication"].isna().sum() + (df["Indication"] == "nan").sum()
df["Indication"] = df["Indication"].replace("nan", np.nan)
df["Indication"] = df["Indication"].fillna("unknown")
print(f"[7] Filled {missing_ind} missing Indication value(s) with 'unknown'")

# â”€â”€ 8. Encode Gender â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
df["Gender"] = df["Gender"].map({"male": 1, "female": 0})
unmapped_gender = df["Gender"].isna().sum()
if unmapped_gender:
    print(f"    WARNING: {unmapped_gender} Gender rows could not be mapped â†’ set to NaN")
print("[8] Encoded Gender  â†’  male=1 / female=0")

# â”€â”€ 9. Outlier capping (Winsorization) on numeric clinical columns â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def winsorise(series, lower_pct=0.01, upper_pct=0.99):
    lo = series.quantile(lower_pct)
    hi = series.quantile(upper_pct)
    capped = series.clip(lower=lo, upper=hi)
    n_capped = ((series < lo) | (series > hi)).sum()
    return capped, lo, hi, n_capped

for col in ["Age", "Dosage (gram)", "Duration (days)"]:
    df[col], lo, hi, n = winsorise(df[col])
    print(f"[9] Winsorised '{col}'  â†’  clipped {n} values to [{lo:.2f}, {hi:.2f}]")

# â”€â”€ 10. Final state â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print(f"\n[10] Final shape: {df.shape[0]} rows Ã— {df.shape[1]} cols")
print("     Missing values remaining:")
print(df.isnull().sum()[df.isnull().sum() > 0].to_string() or "     None")
print("\n     Dtypes:")
print(df.dtypes.to_string())

# â”€â”€ 11. Save â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
df.to_csv(OUTPUT_FILE, index=False)
print(f"\n[11] Saved  â†’  {OUTPUT_FILE}")

