# Medical ML Project — CRISP-DM Pipeline

## Overview
This repository follows the **CRISP-DM** methodology for three medical datasets:

| Dataset | Rows | Cols | Task |
|---|---|---|---|
| Hospital Dataset | 833 | 10 | Medication / drug prescription analysis |
| Medical Dataset (Cardiac) | 1,319 | 9 | Heart attack prediction |
| Kaggle Sirio-Libanes ICU | 1,925 | 231 | ICU admission prediction |

---

## Project Structure

```
ML/
├── data_preparation/
│   ├── 01_hospital_cleaning.py     # Hospital dataset cleaning
│   ├── 02_medical_cleaning.py      # Cardiac dataset cleaning + scaling
│   ├── 03_icu_cleaning.py          # ICU dataset imputation + scaling
│   └── 04_visualization.py         # Post-cleaning visualisations (18 plots)
└── cleaned/
    └── plots/                      # All generated charts (PNG)
```

---

## Data Preparation Steps (per dataset)

### Hospital Dataset
- Removed embedded header rows and 7 duplicates
- Fixed dtypes (Age, Dosage, Duration, Date)
- Extracted datetime features (year, month, day-of-week, hour)
- Standardised text columns (lowercase + strip)
- Filled 1 missing Indication with `'unknown'`
- Encoded Gender (male=1 / female=0)
- Winsorised outliers at 1st–99th percentile

### Medical (Cardiac) Dataset
- Encoded target: `Result` → positive=1 / negative=0
- IQR-based outlier capping on all vitals and lab values
- StandardScaler applied (scaler saved as `.pkl`)
- Flagged 61.4% class imbalance for modeling phase

### ICU Dataset (Sirio-Libanes)
- Ordinal-encoded `AGE_PERCENTIL` (10th=1 … Above 90th=10)
- Reduced missing values from **223,818 → 0** via:
  1. Within-patient forward-fill + backward-fill
  2. Global median imputation
- Winsorised 216 continuous columns
- Saved two versions: full (5 windows) + first-window-only (0-2h)

---

## Visualisations

### Hospital (7 plots)
- Age distribution, Gender balance, Top 10 drugs
- Route of administration, Dosage by route, Duration distribution, Monthly entries

### Medical Cardiac (5 plots)
- Class balance, Correlation heatmap, Vitals by result
- CK-MB / Troponin distributions, Gender vs result

### ICU (6 plots)
- ICU rate by window, Class balance, Age percentile vs ICU
- Disease groupings vs ICU, Demographic correlation heatmap, Window distribution

---

## Requirements

```bash
pip install pandas numpy scikit-learn openpyxl seaborn matplotlib
```

## How to Run

```bash
python data_preparation/01_hospital_cleaning.py
python data_preparation/02_medical_cleaning.py
python data_preparation/03_icu_cleaning.py
python data_preparation/04_visualization.py
```

> Place the three raw dataset files in the project root before running.
