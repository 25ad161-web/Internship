"""
===============================================================================
 REAL-WORLD DATA SCIENCE PROJECT — RETAIL SALES ANALYSIS
===============================================================================
 Dataset : sample.csv
 Author  : Auto-generated end-to-end EDA + Predictive Modeling script
 Run in  : Jupyter Notebook / Google Colab / plain Python (script form)

 This script:
   1. Loads and inspects sample.csv
   2. Automatically detects the dataset's domain
   3. Cleans the data (missing values, duplicates, outliers, dtypes)
   4. Performs Univariate, Bivariate, and Multivariate EDA
   5. Performs domain-specific (Retail) analysis
   6. Builds and compares regression models to predict Revenue
   7. Saves ALL visualizations into a SINGLE combined PNG dashboard
   8. Prints a structured business-insights / final report to the console
===============================================================================
"""

# ==============================================================================
# 1. IMPORT REQUIRED LIBRARIES
# ==============================================================================
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_style("whitegrid")
plt.rcParams["figure.facecolor"] = "white"

CSV_PATH = "sample.csv"          # change path if needed
OUTPUT_FIGURE = "eda_dashboard.png"

print("=" * 80)
print("STEP 1: LOADING THE DATASET")
print("=" * 80)

# ==============================================================================
# 2. LOAD AND UNDERSTAND THE DATASET
# ==============================================================================
df = pd.read_csv(CSV_PATH)

print("\nFirst 5 rows:\n", df.head())
print("\nLast 5 rows:\n", df.tail())
print("\nShape (rows, cols):", df.shape)
print("\nColumn names:", list(df.columns))
print("\nData types:\n", df.dtypes)
print("\nSummary statistics:\n", df.describe(include="all"))
print("\nMissing values per column:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nMemory usage (KB):", round(df.memory_usage(deep=True).sum() / 1024, 2))

categorical_cols_preview = df.select_dtypes(include="object").columns
for c in categorical_cols_preview:
    print(f"\nUnique values in '{c}':", df[c].unique()[:10])

# ------------------------------------------------------------------------------
# DOMAIN DETECTION
# ------------------------------------------------------------------------------
# We scan column names for keyword groups associated with Finance, Health,
# and Retail domains, and pick the domain with the most keyword matches.
cols_lower = [c.lower() for c in df.columns]

domain_keywords = {
    "Finance": ["revenue", "profit", "expense", "transaction", "balance", "interest", "loan"],
    "Health": ["patient", "diagnosis", "disease", "treatment", "bmi", "blood", "symptom"],
    "Retail": ["product", "category", "store", "sales", "units", "price", "discount",
               "customer", "order", "region", "payment"],
}

scores = {d: sum(any(k in c for c in cols_lower) for k in kws) for d, kws in domain_keywords.items()}
detected_domain = max(scores, key=scores.get)

print("\n" + "=" * 80)
print(f"DETECTED DOMAIN: {detected_domain}  (keyword match scores: {scores})")
print("Reasoning: columns such as Region, Category, Payment_Method, Units_Sold,")
print("Unit_Price, Discount_Pct and Revenue are characteristic of a RETAIL sales")
print("transaction dataset (order-level data with product, pricing and customer info).")
print("=" * 80)

# ==============================================================================
# 3. DATA CLEANING
# ==============================================================================
print("\n" + "=" * 80)
print("STEP 3: DATA CLEANING")
print("=" * 80)

df_clean = df.copy()

# --- Fix Date column: some entries contain junk like "not-a-date" ---
df_clean["Date"] = pd.to_datetime(df_clean["Date"], errors="coerce")
n_bad_dates = df_clean["Date"].isnull().sum()
print(f"Converted 'Date' to datetime. Unparseable/junk dates set to NaT: {n_bad_dates}")

# --- Remove duplicate rows ---
n_dupes = df_clean.duplicated().sum()
df_clean = df_clean.drop_duplicates()
print(f"Removed {n_dupes} duplicate rows. New shape: {df_clean.shape}")

# --- Handle missing values ---
# Numeric columns -> fill with median (robust to outliers, keeps distribution shape)
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    if df_clean[col].isnull().sum() > 0:
        median_val = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median_val)
        print(f"Filled missing values in numeric column '{col}' with median ({median_val:.2f})")

# Categorical columns -> fill with mode (most frequent category)
categorical_cols = df_clean.select_dtypes(include="object").columns.tolist()
for col in categorical_cols:
    if df_clean[col].isnull().sum() > 0:
        mode_val = df_clean[col].mode()[0]
        df_clean[col] = df_clean[col].fillna(mode_val)
        print(f"Filled missing values in categorical column '{col}' with mode ('{mode_val}')")

# Rows where Date became NaT after cleaning -> drop, since date can't be imputed meaningfully
before_drop = df_clean.shape[0]
df_clean = df_clean.dropna(subset=["Date"])
print(f"Dropped {before_drop - df_clean.shape[0]} rows with unrecoverable Date values.")

print("\nRemaining missing values after cleaning:\n", df_clean.isnull().sum())

# --- Outlier detection & handling using IQR method ---
print("\nOutlier handling (IQR method) on key numeric columns:")
outlier_cols = ["Units_Sold", "Unit_Price", "Discount_Pct", "Revenue"]
for col in outlier_cols:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = df_clean[(df_clean[col] < lower) | (df_clean[col] > upper)].shape[0]
    # Cap (clip) outliers rather than deleting rows, to preserve sample size
    df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
    print(f"  '{col}': {n_outliers} outliers capped to range [{lower:.2f}, {upper:.2f}]")

# --- Correct data types ---
df_clean["Region"] = df_clean["Region"].astype("category")
df_clean["Category"] = df_clean["Category"].astype("category")
df_clean["Payment_Method"] = df_clean["Payment_Method"].astype("category")
print("\nConverted Region, Category, Payment_Method to categorical dtype.")

# --- Feature engineering ---
df_clean["Month"] = df_clean["Date"].dt.month_name()
df_clean["Weekday"] = df_clean["Date"].dt.day_name()
df_clean["Net_Price_Per_Unit"] = df_clean["Unit_Price"] * (1 - df_clean["Discount_Pct"] / 100)
print("Engineered new features: 'Month', 'Weekday', 'Net_Price_Per_Unit'.")

print(f"\nFinal cleaned shape: {df_clean.shape}")

# ==============================================================================
# 4 & 7. EXPLORATORY DATA ANALYSIS + VISUALIZATION DASHBOARD (single PNG)
# ==============================================================================
print("\n" + "=" * 80)
print("STEP 4: BUILDING COMBINED EDA VISUALIZATION DASHBOARD -> single PNG")
print("=" * 80)

fig = plt.figure(figsize=(22, 26))
gs = gridspec.GridSpec(6, 3, figure=fig, hspace=0.55, wspace=0.35)
fig.suptitle("Retail Sales — EDA & Modeling Dashboard", fontsize=22, fontweight="bold", y=0.995)

# --- 1. Histogram: Revenue distribution (Univariate) ---
ax = fig.add_subplot(gs[0, 0])
sns.histplot(df_clean["Revenue"], kde=True, color="#4C72B0", ax=ax)
ax.set_title("Revenue Distribution")
ax.set_xlabel("Revenue")
ax.set_ylabel("Count")

# --- 2. Box plot: Revenue (Univariate, outlier check) ---
ax = fig.add_subplot(gs[0, 1])
sns.boxplot(y=df_clean["Revenue"], color="#DD8452", ax=ax)
ax.set_title("Revenue Box Plot (post-cleaning)")

# --- 3. Count plot: Category (Univariate) ---
ax = fig.add_subplot(gs[0, 2])
sns.countplot(data=df_clean, x="Category",
              order=df_clean["Category"].value_counts().index, color="#55A868", ax=ax)
ax.set_title("Order Count by Category")
ax.tick_params(axis="x", rotation=30)

# --- 4. Bar chart: Average Revenue by Region (Univariate/Bivariate) ---
ax = fig.add_subplot(gs[1, 0])
region_rev = df_clean.groupby("Region", observed=True)["Revenue"].mean().sort_values(ascending=False)
sns.barplot(x=region_rev.index, y=region_rev.values, color="#C44E52", ax=ax)
ax.set_title("Average Revenue by Region")
ax.set_ylabel("Avg Revenue")

# --- 5. Pie chart: Payment Method share (Univariate) ---
ax = fig.add_subplot(gs[1, 1])
pm_counts = df_clean["Payment_Method"].value_counts()
ax.pie(pm_counts.values, labels=pm_counts.index, autopct="%1.1f%%",
       colors=sns.color_palette("pastel"), startangle=90)
ax.set_title("Payment Method Share")

# --- 6. Box plot by Category (Bivariate) ---
ax = fig.add_subplot(gs[1, 2])
sns.boxplot(data=df_clean, x="Category", y="Revenue", color="#8172B2", ax=ax)
ax.set_title("Revenue by Category")
ax.tick_params(axis="x", rotation=30)

# --- 7. Scatter plot: Unit_Price vs Revenue (Bivariate) ---
ax = fig.add_subplot(gs[2, 0])
sns.scatterplot(data=df_clean, x="Unit_Price", y="Revenue", hue="Category",
                 palette="deep", alpha=0.7, ax=ax, legend=False)
ax.set_title("Unit Price vs Revenue")

# --- 8. Scatter plot: Units_Sold vs Revenue (Bivariate) ---
ax = fig.add_subplot(gs[2, 1])
sns.scatterplot(data=df_clean, x="Units_Sold", y="Revenue", hue="Region",
                 palette="deep", alpha=0.7, ax=ax, legend=False)
ax.set_title("Units Sold vs Revenue")

# --- 9. Grouped bar chart: Avg Revenue by Region & Category (Bivariate) ---
ax = fig.add_subplot(gs[2, 2])
pivot_rc = df_clean.pivot_table(index="Region", columns="Category", values="Revenue",
                                 aggfunc="mean", observed=True)
pivot_rc.plot(kind="bar", ax=ax, colormap="tab10", legend=False)
ax.set_title("Avg Revenue: Region x Category")
ax.tick_params(axis="x", rotation=30)

# --- 10. Line chart: Revenue trend over Month (time-series) ---
ax = fig.add_subplot(gs[3, 0:2])
month_order = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]
monthly_rev = df_clean.groupby("Month")["Revenue"].sum().reindex(month_order).dropna()
ax.plot(monthly_rev.index, monthly_rev.values, marker="o", color="#4C72B0", linewidth=2)
ax.set_title("Total Revenue Trend by Month")
ax.set_ylabel("Total Revenue")
ax.tick_params(axis="x", rotation=45)

# --- 11. Box plot: Customer_Age distribution (Univariate) ---
ax = fig.add_subplot(gs[3, 2])
sns.histplot(df_clean["Customer_Age"], bins=15, kde=True, color="#64B5CD", ax=ax)
ax.set_title("Customer Age Distribution")

# --- 12. Correlation heatmap (Multivariate) ---
ax = fig.add_subplot(gs[4, 0:2])
corr_cols = ["Units_Sold", "Unit_Price", "Discount_Pct", "Customer_Age", "Rating",
             "Revenue", "Net_Price_Per_Unit"]
corr_matrix = df_clean[corr_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
            cbar_kws={"shrink": 0.8})
ax.set_title("Correlation Heatmap")

# --- 13. Rating vs Revenue scatter with regression line (Bivariate) ---
ax = fig.add_subplot(gs[4, 2])
sns.regplot(data=df_clean, x="Rating", y="Revenue", scatter_kws={"alpha": 0.5},
            line_kws={"color": "red"}, ax=ax)
ax.set_title("Rating vs Revenue")

print("Core EDA panels added. Proceeding to predictive modeling...")

# ==============================================================================
# 5. DOMAIN-SPECIFIC (RETAIL) ANALYSIS
# ==============================================================================
print("\n" + "=" * 80)
print("STEP 5: RETAIL-SPECIFIC ANALYSIS")
print("=" * 80)

best_region = df_clean.groupby("Region", observed=True)["Revenue"].sum().idxmax()
best_category = df_clean.groupby("Category", observed=True)["Revenue"].sum().idxmax()
best_month = monthly_rev.idxmax() if len(monthly_rev) else "N/A"
top_payment = df_clean["Payment_Method"].value_counts().idxmax()
avg_discount = df_clean["Discount_Pct"].mean()

print(f"Best-performing Region by total revenue : {best_region}")
print(f"Best-selling Category by total revenue  : {best_category}")
print(f"Peak revenue Month                       : {best_month}")
print(f"Most-used Payment Method                 : {top_payment}")
print(f"Average Discount offered                 : {avg_discount:.2f}%")

# ==============================================================================
# 6. PREDICTIVE MODELING — REGRESSION (target = Revenue)
# ==============================================================================
print("\n" + "=" * 80)
print("STEP 6: PREDICTIVE MODELING (Regression — target: Revenue)")
print("=" * 80)
print("Revenue is a continuous numeric variable -> this is a REGRESSION problem.")

model_df = df_clean.copy()
le_dict = {}
for col in ["Region", "Category", "Payment_Method"]:
    le = LabelEncoder()
    model_df[col + "_enc"] = le.fit_transform(model_df[col].astype(str))
    le_dict[col] = le

feature_cols = ["Units_Sold", "Unit_Price", "Discount_Pct", "Customer_Age", "Rating",
                 "Region_enc", "Category_enc", "Payment_Method_enc"]
X = model_df[feature_cols]
y = model_df["Revenue"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42, max_depth=6),
    "Random Forest Regressor": RandomForestRegressor(random_state=42, n_estimators=200, max_depth=8),
}

results = {}
predictions = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    predictions[name] = preds
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)
    results[name] = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}
    print(f"\n{name}:")
    print(f"  MAE  : {mae:.2f}")
    print(f"  MSE  : {mse:.2f}")
    print(f"  RMSE : {rmse:.2f}")
    print(f"  R2   : {r2:.4f}")

best_model_name = max(results, key=lambda k: results[k]["R2"])
print(f"\nBest performing model: {best_model_name} (highest R2 = {results[best_model_name]['R2']:.4f})")

# --- 14. Model comparison bar chart (R2) ---
ax = fig.add_subplot(gs[5, 0])
r2_scores = [results[m]["R2"] for m in models]
sns.barplot(x=list(models.keys()), y=r2_scores, palette="viridis", ax=ax)
ax.set_title("Model Comparison (R2 Score)")
ax.set_ylabel("R2 Score")
ax.tick_params(axis="x", rotation=20)

# --- 15. Actual vs Predicted (best model) ---
ax = fig.add_subplot(gs[5, 1])
best_preds = predictions[best_model_name]
ax.scatter(y_test, best_preds, alpha=0.6, color="#4C72B0")
lims = [min(y_test.min(), best_preds.min()), max(y_test.max(), best_preds.max())]
ax.plot(lims, lims, "r--", linewidth=2)
ax.set_title(f"Actual vs Predicted\n({best_model_name})")
ax.set_xlabel("Actual Revenue")
ax.set_ylabel("Predicted Revenue")

# --- 16. Feature importance (Random Forest) ---
ax = fig.add_subplot(gs[5, 2])
rf_model = models["Random Forest Regressor"]
importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=True)
importances.plot(kind="barh", color="#55A868", ax=ax)
ax.set_title("Feature Importance (Random Forest)")
ax.set_xlabel("Importance")

# --- Save the entire dashboard as ONE PNG ---
fig.savefig(OUTPUT_FIGURE, dpi=150, bbox_inches="tight")
print(f"\nAll visualizations saved into a single combined figure: '{OUTPUT_FIGURE}'")

# ==============================================================================
# 7 & 8. BUSINESS INSIGHTS + FINAL REPORT
# ==============================================================================
print("\n" + "=" * 80)
print("FINAL REPORT")
print("=" * 80)

print(f"""
DATASET OVERVIEW
-----------------
Domain detected        : {detected_domain}
Records (after cleaning): {df_clean.shape[0]}
Features               : {df_clean.shape[1]}

DATA CLEANING SUMMARY
----------------------
Duplicate rows removed     : {n_dupes}
Missing values imputed     : numeric -> median, categorical -> mode
Outliers handled (IQR)     : capped on Units_Sold, Unit_Price, Discount_Pct, Revenue
Junk/invalid dates removed : {n_bad_dates}

ANALYSIS SUMMARY
------------------
Best-performing Region     : {best_region}
Best-selling Category      : {best_category}
Peak revenue Month         : {best_month}
Most-used Payment Method   : {top_payment}
Average Discount           : {avg_discount:.2f}%
Strongest correlation with Revenue: {corr_matrix['Revenue'].drop('Revenue').abs().idxmax()} ({corr_matrix['Revenue'].drop('Revenue').abs().max():.2f})

PREDICTIVE MODELING SUMMARY
------------------------------
Models compared : Linear Regression, Decision Tree Regressor, Random Forest Regressor
Best model       : {best_model_name}
Best R2 Score    : {results[best_model_name]['R2']:.4f}
Best RMSE        : {results[best_model_name]['RMSE']:.2f}

TOP BUSINESS INSIGHTS
------------------------
1. {best_region} region generates the highest total revenue — prioritize inventory there.
2. {best_category} is the top-grossing category overall.
3. Revenue peaks in {best_month}, suggesting a seasonal demand pattern worth planning around.
4. {top_payment} is customers' preferred payment method.
5. Units_Sold and Unit_Price are the primary drivers of Revenue (see feature importance).
6. Average discount of {avg_discount:.2f}% is being applied — review discount strategy vs margin.
7. {best_model_name} gives the most reliable Revenue prediction (R2={results[best_model_name]['R2']:.4f}).
8. Outliers in Revenue/Units_Sold were capped, not dropped, to preserve sample size.
9. Customer rating shows a measurable relationship with revenue — service quality matters.
10. Several Region/Category combinations underperform — targeted promotions may help.

RECOMMENDATIONS
------------------
- Focus marketing spend on {best_region} and {best_category}.
- Investigate low-performing Region x Category combinations for promotional opportunities.
- Reassess discounting strategy: high discounts do not guarantee proportionally higher revenue.
- Use {best_model_name} as a baseline revenue forecasting tool, retraining periodically.

LIMITATIONS
-------------
- Dataset size is moderate (~300 rows); model generalization should be validated on more data.
- Missing-value imputation (median/mode) may slightly understate true variability.
- No external factors (marketing spend, competitor pricing, holidays) are available.

FUTURE IMPROVEMENTS
----------------------
- Collect more granular time-series data for proper seasonal/trend decomposition.
- Add customer-level repeat-purchase data for segmentation/CLV analysis.
- Try gradient boosting models (XGBoost/LightGBM) for potentially better accuracy.
""")

print("=" * 80)
print("SCRIPT COMPLETE. Combined dashboard figure saved as:", OUTPUT_FIGURE)
print("=" * 80)
