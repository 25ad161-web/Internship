#!/usr/bin/env python
# coding: utf-8

# =============================================================================
# DATA CLEANING & VISUALIZATION PROJECT
# =============================================================================
# Compatible with: Jupyter Notebook / Google Colab
# Libraries: Pandas, NumPy, Matplotlib, Seaborn
# Style: PEP 8 compliant, modular, well-commented
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 ─ LIBRARY IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# Suppress non-critical warnings for cleaner notebook output
warnings.filterwarnings("ignore")

# Global plot aesthetics
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

print("✅ Libraries imported successfully.")


# ─────────────────────────────────────────────────────────────────────────────
# HELPER ─ SAMPLE DATASET GENERATOR
# Creates sample.csv if it doesn't already exist so the notebook runs
# end-to-end even without a pre-supplied file.
# Swap in your own sample.csv to analyse real data — the rest of the code
# adapts automatically via column-type detection.
# ─────────────────────────────────────────────────────────────────────────────

def create_sample_csv(filepath: str = "sample.csv", n_rows: int = 300) -> None:
    """Generate a realistic retail-sales sample dataset and save to CSV."""
    np.random.seed(42)

    regions = ["North", "South", "East", "West"]
    categories = ["Electronics", "Clothing", "Furniture", "Groceries", "Sports"]
    payment_methods = ["Credit Card", "Cash", "UPI", "Net Banking"]

    data = {
        "Order_ID": [f"ORD-{1000 + i}" for i in range(n_rows)],
        "Date": pd.date_range("2023-01-01", periods=n_rows, freq="D").strftime("%Y-%m-%d"),
        "Region": np.random.choice(regions, n_rows),
        "Category": np.random.choice(categories, n_rows),
        "Payment_Method": np.random.choice(payment_methods, n_rows),
        "Units_Sold": np.random.randint(1, 50, n_rows).astype(float),
        "Unit_Price": np.round(np.random.uniform(10, 500, n_rows), 2),
        "Discount_Pct": np.round(np.random.uniform(0, 30, n_rows), 1),
        "Customer_Age": np.random.randint(18, 70, n_rows).astype(float),
        "Rating": np.round(np.random.uniform(1, 5, n_rows), 1),
    }

    df = pd.DataFrame(data)

    # Derived revenue column
    df["Revenue"] = np.round(
        df["Units_Sold"] * df["Unit_Price"] * (1 - df["Discount_Pct"] / 100), 2
    )

    # ── Inject realistic data quality issues ──────────────────────────────────
    # Missing values (~5–8 % per sensitive column)
    for col in ["Units_Sold", "Unit_Price", "Customer_Age", "Rating", "Region"]:
        mask = np.random.choice([True, False], n_rows, p=[0.06, 0.94])
        df.loc[mask, col] = np.nan

    # Duplicate rows (10 exact duplicates)
    duplicate_rows = df.sample(10, random_state=7)
    df = pd.concat([df, duplicate_rows], ignore_index=True)

    # Outliers (a handful of extreme revenue values)
    outlier_indices = np.random.choice(df.index, 8, replace=False)
    df.loc[outlier_indices, "Revenue"] = np.random.choice([50000, 75000, 100000], 8)

    # Wrong data type: corrupt a few dates to strings
    df.loc[np.random.choice(df.index, 5, replace=False), "Date"] = "not-a-date"

    df.to_csv(filepath, index=False)
    print(f"✅ '{filepath}' created with {len(df)} rows (including injected issues).")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 ─ DATA LOADING AND INSPECTION
# ─────────────────────────────────────────────────────────────────────────────

CSV_FILE = "sample.csv"

# Create the sample file only if it doesn't exist (won't overwrite yours)
if not os.path.exists(CSV_FILE):
    print(f"ℹ️  '{CSV_FILE}' not found — generating sample data …")
    create_sample_csv(CSV_FILE)

# Load the dataset
df_raw = pd.read_csv(CSV_FILE)

print("\n" + "=" * 60)
print("SECTION 1 — DATA LOADING AND INSPECTION")
print("=" * 60)

# 1.1 First 5 rows
print("\n📋 First 5 rows:")
print(df_raw.head())

# 1.2 Last 5 rows
print("\n📋 Last 5 rows:")
print(df_raw.tail())

# 1.3 Shape
print(f"\n📐 Dataset shape: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")

# 1.4 Column names
print("\n🏷️  Column names:")
print(list(df_raw.columns))

# 1.5 Data types
print("\n🔢 Data types:")
print(df_raw.dtypes)

# 1.6 Summary statistics (numerical)
print("\n📊 Summary statistics:")
print(df_raw.describe(include="all").T)

# 1.7 Missing values
print("\n❓ Missing values per column:")
missing = df_raw.isnull().sum()
missing_pct = (missing / len(df_raw) * 100).round(2)
missing_report = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
print(missing_report[missing_report["Missing Count"] > 0])

# 1.8 Duplicate rows
n_duplicates_raw = df_raw.duplicated().sum()
print(f"\n🔁 Duplicate rows detected: {n_duplicates_raw}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 ─ DATA CLEANING
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 2 — DATA CLEANING")
print("=" * 60)

# Work on a copy so raw data is preserved for comparison
df = df_raw.copy()

records_before = len(df)

# ── 2.1 Detect column types automatically ────────────────────────────────────
# We never hard-code column names; instead we inspect dtype and cardinality.

# Identify numeric columns
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Identify categorical / object columns (excluding IDs and dates heuristically)
cat_cols = [
    col for col in df.select_dtypes(include=["object"]).columns
    if df[col].nunique(dropna=True) <= 20          # low cardinality = categorical
    and not col.lower().endswith("_id")             # skip ID columns
    and "date" not in col.lower()                   # skip date-like columns
]

# Identify date-like columns (by name convention or later coercion)
date_cols = [col for col in df.columns if "date" in col.lower()]

# Identify high-cardinality string columns (IDs, free text, etc.) — kept but not analyzed
id_cols = [col for col in df.columns if col.lower().endswith("_id")]

print(f"\n🔍 Detected numeric columns   : {num_cols}")
print(f"🔍 Detected categorical columns: {cat_cols}")
print(f"🔍 Detected date columns       : {date_cols}")
print(f"🔍 Detected ID / key columns   : {id_cols}")

# ── 2.2 Fix data types — parse dates ─────────────────────────────────────────
print("\n🔧 Step 1: Convert date columns to datetime (coerce bad values to NaT).")
for col in date_cols:
    before_nulls = df[col].isnull().sum()
    df[col] = pd.to_datetime(df[col], errors="coerce")
    after_nulls = df[col].isnull().sum()
    newly_null = after_nulls - before_nulls
    if newly_null > 0:
        print(f"   ↳ '{col}': {newly_null} unparseable value(s) coerced to NaT.")

# ── 2.3 Remove duplicate rows ─────────────────────────────────────────────────
print(f"\n🔧 Step 2: Remove duplicate rows.")
n_before_dedup = len(df)
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
n_duplicates_removed = n_before_dedup - len(df)
print(f"   ↳ Removed {n_duplicates_removed} duplicate row(s). Rows remaining: {len(df)}")

# ── 2.4 Handle missing values ────────────────────────────────────────────────
print("\n🔧 Step 3: Handle missing values.")

total_missing_handled = 0

for col in num_cols:
    n_missing = df[col].isnull().sum()
    if n_missing == 0:
        continue
    # Use median (robust to outliers) for numerical columns
    fill_value = df[col].median()
    df[col].fillna(fill_value, inplace=True)
    print(f"   ↳ Numerical '{col}': filled {n_missing} NaN(s) with median = {fill_value:.2f}")
    total_missing_handled += n_missing

for col in cat_cols:
    n_missing = df[col].isnull().sum()
    if n_missing == 0:
        continue
    # Use mode (most frequent value) for categorical columns
    fill_value = df[col].mode()[0]
    df[col].fillna(fill_value, inplace=True)
    print(f"   ↳ Categorical '{col}': filled {n_missing} NaN(s) with mode = '{fill_value}'")
    total_missing_handled += n_missing

for col in date_cols:
    n_missing = df[col].isnull().sum()
    if n_missing == 0:
        continue
    # Drop rows with unparseable dates (small fraction, can't meaningfully impute)
    df.dropna(subset=[col], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"   ↳ Date '{col}': dropped {n_missing} row(s) with unparseable date.")
    total_missing_handled += n_missing

print(f"   ✅ Total missing values handled: {total_missing_handled}")

# ── 2.5 Outlier detection & treatment using the IQR method ───────────────────
print("\n🔧 Step 4: Detect and cap outliers using the IQR method.")

total_outliers_treated = 0

for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    n_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()

    if n_outliers > 0:
        # Cap (Winsorise) outliers at the fence values rather than dropping rows
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        print(
            f"   ↳ '{col}': {n_outliers} outlier(s) capped "
            f"to [{lower_bound:.2f}, {upper_bound:.2f}]"
        )
        total_outliers_treated += n_outliers

print(f"   ✅ Total outliers treated: {total_outliers_treated}")

# ── 2.6 Drop columns that add no analytical value ────────────────────────────
print("\n🔧 Step 5: Evaluate and remove low-value columns.")

# Drop high-cardinality unique-ID columns (every value is unique → no signal)
cols_to_drop = [
    col for col in id_cols
    if df[col].nunique() > 0.95 * len(df)  # >95 % unique values
]

if cols_to_drop:
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"   ↳ Dropped high-cardinality ID column(s): {cols_to_drop}")
else:
    print("   ↳ No low-value columns to drop.")

records_after = len(df)

print(f"\n✅ Cleaning complete. Rows: {records_before} → {records_after}")
print("\n📋 Cleaned dataset preview:")
print(df.head())
print("\n🔢 Final dtypes:")
print(df.dtypes)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 ─ EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 3 — EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# Re-resolve column lists after cleaning (some columns may have been dropped)
num_cols_clean = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols_clean = [
    col for col in df.select_dtypes(include=["object"]).columns
    if df[col].nunique() <= 20
]
date_cols_clean = [col for col in df.select_dtypes(include=["datetime64"]).columns]

# 3.1 Distribution of numerical features
print("\n📊 3.1 Numerical feature distributions:")
print(df[num_cols_clean].describe().T.round(2))

# 3.2 Frequency of categorical features
print("\n📊 3.2 Categorical feature frequencies:")
for col in cat_cols_clean:
    print(f"\n  {col}:")
    print(df[col].value_counts().to_string())

# 3.3 Correlation analysis
print("\n📊 3.3 Correlation matrix (numerical features):")
corr_matrix = df[num_cols_clean].corr()
print(corr_matrix.round(2))

# 3.4 Identify top correlations (excluding self-correlations)
print("\n📊 3.4 Top variable pairs by absolute correlation:")
corr_pairs = (
    corr_matrix.abs()
    .unstack()
    .sort_values(ascending=False)
    .drop_duplicates()
)
# Remove self-pairs
corr_pairs = corr_pairs[corr_pairs < 1.0]
print(corr_pairs.head(10).round(3))


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — SAVE FIGURE
# ─────────────────────────────────────────────────────────────────────────────

FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

def save_fig(filename: str) -> None:
    """Save the current matplotlib figure to the figures directory."""
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"   💾 Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 ─ DATA VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 4 — DATA VISUALIZATIONS")
print("=" * 60)

# Pick the "primary" numeric columns for targeted plots
# (use first 4 to keep output manageable regardless of dataset size)
primary_num = num_cols_clean[:4] if len(num_cols_clean) >= 4 else num_cols_clean
primary_cat = cat_cols_clean[0] if cat_cols_clean else None
secondary_cat = cat_cols_clean[1] if len(cat_cols_clean) > 1 else primary_cat

# Heuristic: choose a revenue/price-like column for monetary analysis
revenue_col = next(
    (c for c in num_cols_clean if any(k in c.lower() for k in ["revenue", "sales", "price", "amount"])),
    num_cols_clean[0],
)
units_col = next(
    (c for c in num_cols_clean if any(k in c.lower() for k in ["unit", "qty", "quantity", "sold", "count"])),
    num_cols_clean[1] if len(num_cols_clean) > 1 else revenue_col,
)


# ── 4.1 HISTOGRAMS — Distribution of numerical features ──────────────────────
print("\n📈 4.1 Histograms")

n_num = len(primary_num)
fig, axes = plt.subplots(1, n_num, figsize=(5 * n_num, 4))
if n_num == 1:
    axes = [axes]

for ax, col in zip(axes, primary_num):
    ax.hist(df[col].dropna(), bins=25, color=sns.color_palette("muted")[0],
            edgecolor="white", linewidth=0.6)
    ax.set_title(f"Distribution of {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")

fig.suptitle("Histograms — Numerical Feature Distributions", fontsize=15, y=1.02)
plt.tight_layout()
save_fig("01_histograms.png")
print("   💡 Insight: Histograms reveal whether each feature is normally distributed,")
print("      skewed, or multimodal, guiding appropriate statistical treatment.")


# ── 4.2 BOX PLOTS — Spread and outliers ──────────────────────────────────────
print("\n📈 4.2 Box Plots")

fig, axes = plt.subplots(1, n_num, figsize=(5 * n_num, 4))
if n_num == 1:
    axes = [axes]

for ax, col in zip(axes, primary_num):
    sns.boxplot(y=df[col], ax=ax, color=sns.color_palette("pastel")[1])
    ax.set_title(f"Box Plot: {col}")
    ax.set_ylabel(col)

fig.suptitle("Box Plots — Spread & Residual Outliers After Capping", fontsize=15, y=1.02)
plt.tight_layout()
save_fig("02_boxplots.png")
print("   💡 Insight: After IQR capping, boxes should be compact; any remaining")
print("      whisker extension signals naturally wide distributions.")


# ── 4.3 BAR CHART — Average revenue by primary category ──────────────────────
print("\n📈 4.3 Bar Chart")

if primary_cat:
    agg = df.groupby(primary_cat)[revenue_col].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(agg.index, agg.values,
                  color=sns.color_palette("Set2", len(agg)))
    ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=9)
    ax.set_title(f"Average {revenue_col} by {primary_cat}", fontsize=14)
    ax.set_xlabel(primary_cat)
    ax.set_ylabel(f"Average {revenue_col}")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x:,.0f}"
    ))
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    save_fig("03_bar_chart.png")
    print(f"   💡 Insight: Reveals which {primary_cat} generates the highest")
    print(f"      average {revenue_col} — useful for resource allocation decisions.")


# ── 4.4 COUNT PLOT — Frequency of a categorical variable ─────────────────────
print("\n📈 4.4 Count Plot")

if primary_cat:
    fig, ax = plt.subplots(figsize=(8, 5))
    order = df[primary_cat].value_counts().index
    sns.countplot(data=df, x=primary_cat, order=order, palette="Set3", ax=ax)
    ax.set_title(f"Count Plot — Frequency of {primary_cat}", fontsize=14)
    ax.set_xlabel(primary_cat)
    ax.set_ylabel("Count")
    for patch in ax.patches:
        ax.annotate(f"{int(patch.get_height())}",
                    (patch.get_x() + patch.get_width() / 2, patch.get_height()),
                    ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    save_fig("04_count_plot.png")
    print(f"   💡 Insight: Highlights class imbalance in '{primary_cat}'.")
    print("      Imbalanced categories may require stratified sampling in ML tasks.")


# ── 4.5 SCATTER PLOT — Revenue vs Units Sold ─────────────────────────────────
print("\n📈 4.5 Scatter Plot")

hue_var = primary_cat if primary_cat else None

fig, ax = plt.subplots(figsize=(9, 6))
scatter_data = df[[revenue_col, units_col]].dropna()
if hue_var and hue_var in df.columns:
    for label, grp in df.groupby(hue_var):
        ax.scatter(grp[units_col], grp[revenue_col], label=label, alpha=0.6, s=40)
    ax.legend(title=hue_var, bbox_to_anchor=(1.01, 1), loc="upper left")
else:
    ax.scatter(df[units_col], df[revenue_col], alpha=0.5, s=40,
               color=sns.color_palette("muted")[2])

# Trend line
z = np.polyfit(df[units_col].fillna(0), df[revenue_col].fillna(0), 1)
p = np.poly1d(z)
xs = np.linspace(df[units_col].min(), df[units_col].max(), 200)
ax.plot(xs, p(xs), "r--", linewidth=1.5, label="Trend")

ax.set_title(f"Scatter: {units_col} vs {revenue_col}", fontsize=14)
ax.set_xlabel(units_col)
ax.set_ylabel(revenue_col)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
plt.tight_layout()
save_fig("05_scatter_plot.png")
print(f"   💡 Insight: The trend line direction tells us whether more units sold")
print(f"      consistently translates to higher revenue.")


# ── 4.6 LINE CHART — Revenue over time (if date column exists) ────────────────
print("\n📈 4.6 Line Chart")

if date_cols_clean:
    date_col = date_cols_clean[0]
    ts = (
        df.set_index(date_col)[revenue_col]
        .resample("W")          # weekly aggregation
        .sum()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ts[date_col], ts[revenue_col],
            color=sns.color_palette("muted")[3], linewidth=2, marker="o", markersize=3)
    ax.fill_between(ts[date_col], ts[revenue_col], alpha=0.15,
                    color=sns.color_palette("muted")[3])
    ax.set_title(f"Weekly {revenue_col} Over Time", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Total {revenue_col}")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    save_fig("06_line_chart.png")
    print("   💡 Insight: The time-series line reveals seasonality, growth trends,")
    print("      and any sudden dips/spikes that warrant investigation.")
else:
    print("   ⚠️  No datetime column found — skipping line chart.")


# ── 4.7 CORRELATION HEATMAP ──────────────────────────────────────────────────
print("\n📈 4.7 Correlation Heatmap")

if len(num_cols_clean) >= 2:
    corr = df[num_cols_clean].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))   # upper triangle mask

    fig, ax = plt.subplots(figsize=(max(6, len(num_cols_clean)), max(5, len(num_cols_clean) - 1)))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, linewidths=0.5,
        square=True, ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Correlation Heatmap — Numerical Features", fontsize=14)
    plt.tight_layout()
    save_fig("07_correlation_heatmap.png")
    print("   💡 Insight: Strong positive correlations (dark red) indicate redundant")
    print("      features; strong negative correlations reveal opposing trends.")


# ── 4.8 PAIR PLOT ─────────────────────────────────────────────────────────────
print("\n📈 4.8 Pair Plot")

pair_cols = primary_num[:4]   # cap at 4 to keep it readable
if len(pair_cols) >= 2:
    hue_col = primary_cat if primary_cat and primary_cat in df.columns else None
    g = sns.pairplot(
        df[pair_cols + ([hue_col] if hue_col else [])].dropna(),
        hue=hue_col,
        diag_kind="kde",
        plot_kws={"alpha": 0.5, "s": 20},
        palette="Set2",
    )
    g.fig.suptitle("Pair Plot — Multi-variable Relationships", y=1.02, fontsize=14)
    save_fig("08_pair_plot.png")
    print("   💡 Insight: The pair plot matrix gives an at-a-glance view of all")
    print("      pairwise relationships and per-category distributions.")


# ── 4.9 PIE CHART — Category share of total revenue ─────────────────────────
print("\n📈 4.9 Pie Chart")

if primary_cat and revenue_col in df.columns:
    pie_data = df.groupby(primary_cat)[revenue_col].sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 7))
    wedge_props = {"edgecolor": "white", "linewidth": 1.5}
    ax.pie(
        pie_data.values,
        labels=pie_data.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=sns.color_palette("Set2", len(pie_data)),
        wedgeprops=wedge_props,
        pctdistance=0.82,
    )
    ax.set_title(f"Revenue Share by {primary_cat}", fontsize=14)
    plt.tight_layout()
    save_fig("09_pie_chart.png")
    print(f"   💡 Insight: The pie chart shows which {primary_cat} segment dominates")
    print(f"      total {revenue_col} — useful for portfolio prioritisation.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 ─ DASHBOARD / VISUAL REPORT
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 5 — DASHBOARD / VISUAL REPORT")
print("=" * 60)

fig = plt.figure(figsize=(20, 22))
fig.suptitle("📊  Data Analysis Dashboard", fontsize=20, fontweight="bold", y=1.00)

# ── Row 1: Histogram + Box Plot ───────────────────────────────────────────────
ax1 = fig.add_subplot(4, 3, 1)
ax1.hist(df[revenue_col].dropna(), bins=25,
         color=sns.color_palette("muted")[0], edgecolor="white")
ax1.set_title(f"Distribution of {revenue_col}")
ax1.set_xlabel(revenue_col)
ax1.set_ylabel("Frequency")

ax2 = fig.add_subplot(4, 3, 2)
sns.boxplot(y=df[revenue_col], ax=ax2, color=sns.color_palette("pastel")[1])
ax2.set_title(f"Box Plot: {revenue_col}")

ax3 = fig.add_subplot(4, 3, 3)
if len(num_cols_clean) > 1:
    ax3.hist(df[units_col].dropna(), bins=20,
             color=sns.color_palette("muted")[2], edgecolor="white")
    ax3.set_title(f"Distribution of {units_col}")
    ax3.set_xlabel(units_col)
    ax3.set_ylabel("Frequency")

# ── Row 2: Bar Chart + Count Plot + Pie ───────────────────────────────────────
ax4 = fig.add_subplot(4, 3, 4)
if primary_cat:
    agg_dash = df.groupby(primary_cat)[revenue_col].mean().sort_values(ascending=False)
    ax4.bar(agg_dash.index, agg_dash.values,
            color=sns.color_palette("Set2", len(agg_dash)))
    ax4.set_title(f"Avg {revenue_col} by {primary_cat}")
    ax4.set_xlabel(primary_cat)
    ax4.set_ylabel(f"Avg {revenue_col}")
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=30, ha="right")

ax5 = fig.add_subplot(4, 3, 5)
if primary_cat:
    order_dash = df[primary_cat].value_counts().index
    sns.countplot(data=df, x=primary_cat, order=order_dash, palette="Set3", ax=ax5)
    ax5.set_title(f"Count by {primary_cat}")
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=30, ha="right")

ax6 = fig.add_subplot(4, 3, 6)
if primary_cat:
    pie_d = df.groupby(primary_cat)[revenue_col].sum()
    ax6.pie(pie_d.values, labels=pie_d.index, autopct="%1.0f%%",
            colors=sns.color_palette("Set2", len(pie_d)),
            wedgeprops={"edgecolor": "white"})
    ax6.set_title(f"{revenue_col} Share")

# ── Row 3: Scatter + Line Chart ───────────────────────────────────────────────
ax7 = fig.add_subplot(4, 3, 7)
ax7.scatter(df[units_col], df[revenue_col], alpha=0.4, s=15,
            color=sns.color_palette("muted")[4])
xs = np.linspace(df[units_col].min(), df[units_col].max(), 200)
ax7.plot(xs, p(xs), "r--", linewidth=1.2)
ax7.set_title(f"{units_col} vs {revenue_col}")
ax7.set_xlabel(units_col)
ax7.set_ylabel(revenue_col)

ax8 = fig.add_subplot(4, 3, (8, 9))
if date_cols_clean:
    ax8.plot(ts[date_col], ts[revenue_col],
             color=sns.color_palette("muted")[3], linewidth=1.5)
    ax8.fill_between(ts[date_col], ts[revenue_col], alpha=0.15,
                     color=sns.color_palette("muted")[3])
    ax8.set_title(f"Weekly {revenue_col} Trend")
    ax8.set_xlabel("Date")
    plt.setp(ax8.xaxis.get_majorticklabels(), rotation=30, ha="right")
else:
    ax8.axis("off")

# ── Row 4: Correlation Heatmap ────────────────────────────────────────────────
ax9 = fig.add_subplot(4, 3, (10, 12))
if len(num_cols_clean) >= 2:
    corr_d = df[num_cols_clean].corr()
    mask_d = np.triu(np.ones_like(corr_d, dtype=bool))
    sns.heatmap(corr_d, mask=mask_d, annot=True, fmt=".2f",
                cmap="coolwarm", center=0, linewidths=0.4,
                ax=ax9, cbar_kws={"shrink": 0.6})
    ax9.set_title("Correlation Heatmap")

plt.tight_layout()
save_fig("10_dashboard.png")
print("   ✅ Dashboard saved.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 ─ STORYTELLING WITH DATA
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 6 — STORYTELLING WITH DATA")
print("=" * 60)

# Compute top-performing category
if primary_cat and revenue_col in df.columns:
    top_cat = df.groupby(primary_cat)[revenue_col].mean().idxmax()
    top_cat_val = df.groupby(primary_cat)[revenue_col].mean().max()

    print(f"""
📖 KEY NARRATIVE
─────────────────────────────────────────────────────────

1. 📌 MAJOR INSIGHTS
   • The dataset spans {records_after} clean records across {len(df.columns)} features.
   • '{top_cat}' is the highest-performing {primary_cat} with an average
     {revenue_col} of {top_cat_val:,.1f}.
   • Numerical features show {'positive' if corr_pairs.iloc[0] > 0 else 'negative'}
     correlation between the top two paired variables
     ({corr_pairs.index[0][0]} & {corr_pairs.index[0][1]}: ρ = {corr_pairs.iloc[0]:.2f}).

2. 📈 TRENDS DISCOVERED
   • Time-series analysis (weekly aggregation) {'revealed' if date_cols_clean else 'was not possible —'} 
     {'a discernible trend in ' + revenue_col + ' over the period.' if date_cols_clean else 'no datetime column exists.'}
   • Count plots confirmed {'uneven' if df[primary_cat].value_counts().std() > 5 else 'relatively balanced'}
     class distribution across '{primary_cat}' — important for any downstream modelling.

3. ⚠️  OUTLIERS FOUND & HANDLED
   • {total_outliers_treated} outlier(s) were detected using the 1.5×IQR fence rule.
   • Outliers were capped (Winsorised) at the fence values rather than removed
     to preserve the overall record count while eliminating distortion.

4. 🔗 RELATIONSHIPS BETWEEN VARIABLES
   • The scatter plot of {units_col} vs {revenue_col} shows a
     {'positive' if z[0] > 0 else 'negative'} linear trend (slope ≈ {z[0]:.2f}).
   • The correlation heatmap highlights that the strongest numerical relationship
     is between {corr_pairs.index[0][0]} and {corr_pairs.index[0][1]}
     (ρ = {corr_pairs.iloc[0]:.2f}).

5. 🌍 BUSINESS / REAL-WORLD INTERPRETATION
   • High-revenue categories should receive priority inventory and marketing spend.
   • The positive relationship between units sold and revenue validates volume-
     driven pricing strategies.
   • Discount percentages warrant scrutiny: if high discounts aren't matched by
     proportional volume increases, margin erosion may be occurring.
   • Missing value patterns (random vs. systematic) should be investigated
     operationally — systematic gaps may indicate data pipeline failures.
""")
else:
    print("   ⚠️  Narrative skipped — primary categorical column not identified.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 ─ FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 7 — FINAL SUMMARY")
print("=" * 60)

summary = {
    "Total records BEFORE cleaning": records_before,
    "Total records AFTER cleaning": records_after,
    "Duplicate rows removed": n_duplicates_removed,
    "Missing values handled (imputed/dropped)": total_missing_handled,
    "Outliers detected and treated (capped)": total_outliers_treated,
    "Numerical columns analysed": len(num_cols_clean),
    "Categorical columns analysed": len(cat_cols_clean),
}

print()
for key, value in summary.items():
    print(f"  {key:<45}: {value}")

print(f"""
MAIN CONCLUSIONS
────────────────
  ✔  The raw dataset contained missing values, duplicate records, and revenue
     outliers that would have skewed all downstream analysis.
  ✔  After cleaning, the dataset is fully typed, free of duplicates, and
     outlier-capped — ready for modelling or further reporting.
  ✔  '{revenue_col}' is the primary KPI; its distribution and relationships with
     volume and categorical dimensions are well-characterised across 9 plots.
  ✔  All visualisations are saved under the '{FIGURES_DIR}/' directory and
     combined into a single dashboard image (10_dashboard.png).

  ▶  Next steps: feature engineering (e.g., revenue-per-unit), time-series
     forecasting, customer segmentation, or classification modelling.
""")

print("=" * 60)
print("✅ PROJECT COMPLETE — all figures saved to:", os.path.abspath(FIGURES_DIR))
print("=" * 60)
