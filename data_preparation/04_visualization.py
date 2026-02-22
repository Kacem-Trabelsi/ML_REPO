"""
CRISP-DM: Data Preparation â€” Script 4
Post-cleaning visualisation for all three datasets.
Reads from: cleaned/hospital_cleaned.csv
            cleaned/medical_cleaned.csv
            cleaned/icu_cleaned_full.csv
Saves plots to: cleaned/plots/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

# â”€â”€ Style â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_DIR  = os.path.join(BASE_DIR, "cleaned")
PLOTS_DIR  = os.path.join(CLEAN_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# HELPER
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def save(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved  â†’  {path}")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 1. HOSPITAL DATASET
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
print("\nâ”€â”€ Hospital Dataset visualisations â”€â”€")
h = pd.read_csv(os.path.join(CLEAN_DIR, "hospital_cleaned.csv"))

# 1-A  Age distribution
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(h["Age"].dropna(), bins=20, color="#4C72B0", edgecolor="white")
ax.set_title("Hospital â€” Age Distribution (post-cleaning)")
ax.set_xlabel("Age (years)")
ax.set_ylabel("Count")
save(fig, "H1_age_distribution.png")

# 1-B  Gender balance
fig, ax = plt.subplots(figsize=(5, 4))
gender_counts = h["Gender"].value_counts().rename({1: "Male", 0: "Female"})
bars = ax.bar(gender_counts.index, gender_counts.values,
              color=["#4C72B0", "#DD8452"], edgecolor="white")
ax.bar_label(bars, fmt="%d")
ax.set_title("Hospital â€” Gender Balance")
ax.set_ylabel("Count")
save(fig, "H2_gender_balance.png")

# 1-C  Top 10 drugs
fig, ax = plt.subplots(figsize=(9, 5))
top_drugs = h["Name of Drug"].value_counts().head(10)
sns.barplot(x=top_drugs.values, y=top_drugs.index, ax=ax,
            hue=top_drugs.index, palette="Blues_r", legend=False)
ax.set_title("Hospital â€” Top 10 Prescribed Drugs")
ax.set_xlabel("Frequency")
save(fig, "H3_top_drugs.png")

# 1-D  Route of Administration
fig, ax = plt.subplots(figsize=(6, 4))
route_counts = h["Route"].value_counts()
ax.pie(route_counts.values, labels=route_counts.index, autopct="%1.1f%%",
       colors=sns.color_palette("pastel"), startangle=90)
ax.set_title("Hospital â€” Route of Administration")
save(fig, "H4_route_pie.png")

# 1-E  Dosage boxplot by Route
fig, ax = plt.subplots(figsize=(7, 4))
route_labels = {0: "IV", 1: "Oral", 2: "IM"}   # may differ after encoding
data_by_route = h.groupby("Route")["Dosage (gram)"].apply(list)
ax.boxplot(data_by_route.values, tick_labels=data_by_route.index, patch_artist=True)
ax.set_title("Hospital â€” Dosage (gram) by Route")
ax.set_ylabel("Dosage (gram)")
save(fig, "H5_dosage_by_route.png")

# 1-F  Duration distribution
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(h["Duration (days)"].dropna(), bins=15, color="#55A868", edgecolor="white")
ax.set_title("Hospital â€” Treatment Duration Distribution")
ax.set_xlabel("Duration (days)")
ax.set_ylabel("Count")
save(fig, "H6_duration_distribution.png")

# 1-G  Entries by month
fig, ax = plt.subplots(figsize=(9, 4))
monthly = h["entry_month"].value_counts().sort_index()
ax.bar(monthly.index, monthly.values, color="#C44E52", edgecolor="white")
ax.set_xticks(range(1, 13))
ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"])
ax.set_title("Hospital â€” Data Entries by Month")
ax.set_ylabel("Count")
save(fig, "H7_entries_by_month.png")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 2. MEDICAL (CARDIAC) DATASET
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
print("\nâ”€â”€ Medical Dataset visualisations â”€â”€")
m = pd.read_csv(os.path.join(CLEAN_DIR, "medical_cleaned.csv"))
result_label = m["Result"].map({1: "Positive", 0: "Negative"})

# 2-A  Class balance
fig, ax = plt.subplots(figsize=(5, 4))
counts = m["Result"].value_counts().rename({1: "Positive", 0: "Negative"})
bars = ax.bar(counts.index, counts.values, color=["#C44E52", "#4C72B0"], edgecolor="white")
ax.bar_label(bars, fmt="%d")
ax.set_title("Medical â€” Class Balance (Heart Attack Result)")
ax.set_ylabel("Count")
save(fig, "M1_class_balance.png")

# 2-B  Correlation heatmap
fig, ax = plt.subplots(figsize=(9, 7))
corr = m.corr(numeric_only=True)
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=ax, linewidths=0.5)
ax.set_title("Medical â€” Feature Correlation Heatmap")
save(fig, "M2_correlation_heatmap.png")

# 2-C  Vitals boxplots by Result
vitals = ["Age", "Heart rate", "Systolic blood pressure",
          "Diastolic blood pressure", "Blood sugar"]
fig, axes = plt.subplots(1, len(vitals), figsize=(16, 5))
for ax, col in zip(axes, vitals):
    data = [m.loc[m["Result"] == v, col].values for v in [0, 1]]
    bp = ax.boxplot(data, patch_artist=True, tick_labels=["Negative", "Positive"])
    bp["boxes"][0].set_facecolor("#4C72B0")
    bp["boxes"][1].set_facecolor("#C44E52")
    ax.set_title(col, fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
fig.suptitle("Medical â€” Vitals by Result (scaled)", y=1.02)
fig.tight_layout()
save(fig, "M3_vitals_by_result.png")

# 2-D  CK-MB and Troponin distributions by Result
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, col in zip(axes, ["CK-MB", "Troponin"]):
    for val, label, color in [(0, "Negative", "#4C72B0"), (1, "Positive", "#C44E52")]:
        subset = m.loc[m["Result"] == val, col]
        ax.hist(subset, bins=30, alpha=0.6, label=label, color=color, edgecolor="white")
    ax.set_title(f"Medical â€” {col} Distribution by Result (scaled)")
    ax.set_xlabel(col)
    ax.legend()
fig.tight_layout()
save(fig, "M4_cardiac_markers.png")

# 2-E  Gender vs Result
fig, ax = plt.subplots(figsize=(6, 4))
gender_result = m.groupby(["Gender", "Result"]).size().unstack(fill_value=0)
gender_result.index = ["Female (0)", "Male (1)"]
gender_result.columns = ["Negative", "Positive"]
gender_result.plot(kind="bar", ax=ax, color=["#4C72B0", "#C44E52"],
                   edgecolor="white", rot=0)
ax.set_title("Medical â€” Result by Gender")
ax.set_ylabel("Count")
ax.legend(title="Result")
save(fig, "M5_gender_vs_result.png")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# 3. ICU DATASET
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
print("\nâ”€â”€ ICU Dataset visualisations â”€â”€")
icu = pd.read_csv(os.path.join(CLEAN_DIR, "icu_cleaned_full.csv"))

# 3-A  ICU admission rate by time window
fig, ax = plt.subplots(figsize=(8, 4))
window_order = ["0-2", "2-4", "4-6", "6-12", "ABOVE_12"]
icu_rate = (icu.groupby("WINDOW")["ICU"]
              .mean()
              .reindex(window_order) * 100)
bars = ax.bar(icu_rate.index, icu_rate.values, color="#4C72B0", edgecolor="white")
ax.bar_label(bars, fmt="%.1f%%", fontsize=9)
ax.set_title("ICU â€” Admission Rate by Time Window")
ax.set_xlabel("Time Window (hours)")
ax.set_ylabel("ICU Admission Rate (%)")
save(fig, "I1_icu_rate_by_window.png")

# 3-B  Overall ICU class balance
fig, ax = plt.subplots(figsize=(5, 4))
icu_counts = icu["ICU"].value_counts().rename({0: "Not ICU", 1: "ICU"})
bars = ax.bar(icu_counts.index, icu_counts.values,
              color=["#4C72B0", "#C44E52"], edgecolor="white")
ax.bar_label(bars, fmt="%d")
ax.set_title("ICU â€” Class Balance")
ax.set_ylabel("Count")
save(fig, "I2_icu_class_balance.png")

# 3-C  Age percentile vs ICU admission rate
fig, ax = plt.subplots(figsize=(8, 4))
age_icu = icu.groupby("AGE_PERCENTIL")["ICU"].mean() * 100
age_icu.plot(kind="bar", ax=ax, color="#55A868", edgecolor="white", rot=0)
ax.set_title("ICU â€” Admission Rate by Age Percentile")
ax.set_xlabel("Age Percentile (ordinal encoded)")
ax.set_ylabel("ICU Admission Rate (%)")
save(fig, "I3_age_percentile_vs_icu.png")

# 3-D  Disease groupings vs ICU rate
disease_cols = [c for c in icu.columns if "DISEASE GROUPING" in c]
if disease_cols:
    fig, ax = plt.subplots(figsize=(9, 4))
    rates = {col: icu.loc[icu[col] == 1, "ICU"].mean() * 100 for col in disease_cols}
    ax.bar(list(rates.keys()), list(rates.values()), color="#DD8452", edgecolor="white")
    ax.set_title("ICU â€” Admission Rate by Disease Grouping")
    ax.set_ylabel("ICU Admission Rate (%)")
    ax.tick_params(axis="x", rotation=20)
    save(fig, "I4_disease_grouping_vs_icu.png")

# 3-E  Correlation heatmap for static / demographic features
static_cols = ["AGE_ABOVE65", "AGE_PERCENTIL", "GENDER",
               "DISEASE GROUPING 1", "DISEASE GROUPING 2", "DISEASE GROUPING 3",
               "DISEASE GROUPING 4", "DISEASE GROUPING 5", "DISEASE GROUPING 6",
               "HTN", "IMMUNOCOMPROMISED", "OTHER", "ICU"]
fig, ax = plt.subplots(figsize=(10, 8))
corr_icu = icu[static_cols].corr()
mask = np.triu(np.ones_like(corr_icu, dtype=bool))
sns.heatmap(corr_icu, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=ax, linewidths=0.5)
ax.set_title("ICU â€” Demographic Feature Correlation Heatmap")
save(fig, "I5_demo_correlation_heatmap.png")

# 3-F  WINDOW distribution (sanity check â€” should be uniform)
fig, ax = plt.subplots(figsize=(7, 4))
win_counts = icu["WINDOW"].value_counts().reindex(window_order)
ax.bar(win_counts.index, win_counts.values, color="#8172B2", edgecolor="white")
ax.set_title("ICU â€” Row Count per Time Window")
ax.set_ylabel("Count")
save(fig, "I6_window_distribution.png")

print(f"\nAll plots saved to: {PLOTS_DIR}")
print("Done.")

