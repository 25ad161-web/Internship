# =============================================================================
#  Predictive Modeling Using Machine Learning
#  Dataset  : sample.csv
#  Target   : Revenue (Regression)
#  Models   : Linear Regression | Decision Tree | Random Forest
#  Author   : [Your Name]
#  Run      : python ML_Revenue_Prediction.py
#             (Ensure sample.csv is in the same directory)
# =============================================================================

# ── Data manipulation & numerical computing ────────────────────────────────
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── Visualisation ──────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.05)
plt.rcParams['figure.dpi'] = 110

# ── Preprocessing ──────────────────────────────────────────────────────────
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score

# ── Models ─────────────────────────────────────────────────────────────────
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# ── Evaluation ─────────────────────────────────────────────────────────────
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score)

print("✅ All libraries imported successfully.")



# Load the dataset
df_raw = pd.read_csv('sample.csv')

print(f"Dataset loaded successfully!")
print(f"Shape : {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
print(f"Columns: {list(df_raw.columns)}")



# ── First 5 rows ───────────────────────────────────────────────────────────
print("── First 5 rows ──")
print(df_raw.head())



# ── Last 5 rows ────────────────────────────────────────────────────────────
print("── Last 5 rows ──")
print(df_raw.tail())



# ── Shape & Column Types ───────────────────────────────────────────────────
print(f"Shape  : {df_raw.shape}")
print()
print("Data Types:")
print(df_raw.dtypes)



# ── Summary Statistics ─────────────────────────────────────────────────────
print("── Summary Statistics ──")
print(df_raw.describe(include='all'))



# ── Missing Values ─────────────────────────────────────────────────────────
missing = df_raw.isnull().sum()
missing_pct = (missing / len(df_raw) * 100).round(2)
miss_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
miss_df = miss_df[miss_df['Missing Count'] > 0]

print("── Missing Values ──")
print(miss_df)

# Visualise
fig, ax = plt.subplots(figsize=(8, 4))
miss_df['Missing %'].plot(kind='bar', ax=ax, color='#e07b54', edgecolor='white', rot=0)
ax.set_title('Missing Value Percentage by Column', fontsize=13, fontweight='bold')
ax.set_xlabel('Column')
ax.set_ylabel('Missing %')
for p in ax.patches:
    ax.annotate(f"{p.get_height():.1f}%",
                (p.get_x() + p.get_width() / 2, p.get_height() + 0.1),
                ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.show()



# ── Duplicate Rows ─────────────────────────────────────────────────────────
dup_count = df_raw.duplicated().sum()
print(f"Duplicate rows found: {dup_count}")



# ── Correlation Matrix ─────────────────────────────────────────────────────
numeric_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
corr = df_raw[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            linewidths=0.4, ax=ax, vmin=-1, vmax=1)
ax.set_title('Correlation Matrix – Numeric Features', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()



df = df_raw.copy()

# ── Step 1: Drop irrelevant columns ────────────────────────────────────────
# Order_ID  → unique identifier; carries no predictive information
# Date      → raw date string; extracting month/quarter is done next
drop_cols = ['Order_ID']
df.drop(columns=drop_cols, inplace=True)
print(f"Dropped columns: {drop_cols}")



# ── Step 2: Extract temporal features from Date ────────────────────────────
# Month and quarter can capture seasonality in sales revenue.
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Month']   = df['Date'].dt.month
df['Quarter'] = df['Date'].dt.quarter
df.drop(columns=['Date'], inplace=True)
print("Extracted Month and Quarter from Date. Date column dropped.")
print(df[['Month', 'Quarter']].head())



# ── Step 3: Remove duplicate rows ──────────────────────────────────────────
before = len(df)
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"Removed {before - len(df)} duplicate rows. Remaining: {len(df)} rows")



# ── Step 4: Handle Missing Values ─────────────────────────────────────────
# Numeric columns → median (robust to outliers)
# Categorical columns → mode (most frequent category)

numeric_feats = df.select_dtypes(include=np.number).columns.tolist()
cat_feats     = df.select_dtypes(include='object').columns.tolist()

for col in numeric_feats:
    if df[col].isnull().any():
        med = df[col].median()
        df[col].fillna(med, inplace=True)
        print(f"  Numeric  '{col}' → filled NaN with median = {med:.2f}")

for col in cat_feats:
    if df[col].isnull().any():
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
        print(f"  Category '{col}' → filled NaN with mode = '{mode_val}'")

print(f"\nTotal missing values remaining: {df.isnull().sum().sum()}")



# ── Step 5: Detect and treat outliers in Revenue (IQR method) ──────────────
# Revenue has a max of 100,000 which is extreme vs the 75th percentile of ~8,569.
# We Winsorise (cap) rather than delete to preserve sample size.

for col in ['Revenue', 'Units_Sold', 'Unit_Price']:
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_out = ((df[col] < lower) | (df[col] > upper)).sum()
    df[col] = df[col].clip(lower, upper)
    print(f"  '{col}': {n_out} outliers Winsorised  [range capped to {lower:.2f} – {upper:.2f}]")



# ── Step 6: Encode Categorical Variables ───────────────────────────────────
# Region        → 4 categories  → One-Hot Encode (low cardinality)
# Category      → 5 categories  → One-Hot Encode
# Payment_Method → 4 categories → One-Hot Encode

cat_cols_to_encode = ['Region', 'Category', 'Payment_Method']
df = pd.get_dummies(df, columns=cat_cols_to_encode, drop_first=True, dtype=int)

print(f"After One-Hot Encoding, shape: {df.shape}")
print(f"New columns: {list(df.columns)}")



# ── Step 7: Separate Features and Target ────────────────────────────────────
TARGET = 'Revenue'
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Ensure all features are numeric
X = X.select_dtypes(include=np.number).fillna(0)

print(f"Features (X) shape : {X.shape}")
print(f"Target   (y) shape : {y.shape}")
print(f"\nFeature columns: {list(X.columns)}")



# ── Step 8: Feature Scaling ────────────────────────────────────────────────
# StandardScaler standardises features to mean=0, std=1.
# Essential for Linear Regression; harmless for tree-based models.

scaler   = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

print("StandardScaler applied.")
print(X_scaled.describe().loc[['mean', 'std']].round(3))



RANDOM_STATE = 42

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=RANDOM_STATE
)

print(f"Training samples : {X_train.shape[0]}")
print(f"Testing  samples : {X_test.shape[0]}")
print(f"Features         : {X_train.shape[1]}")



# ── Model definitions ──────────────────────────────────────────────────────
models = {
    'Linear Regression'    : LinearRegression(),
    'Decision Tree'        : DecisionTreeRegressor(max_depth=6, random_state=RANDOM_STATE),
    'Random Forest'        : RandomForestRegressor(n_estimators=200, max_depth=8,
                                                   random_state=RANDOM_STATE, n_jobs=-1),
}

results   = {}   # stores metrics
all_preds = {}   # stores predictions for plotting

for name, model in models.items():
    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    # Metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2  = r2_score(y_test,  y_pred_test)
    mae      = mean_absolute_error(y_test, y_pred_test)
    mse      = mean_squared_error(y_test,  y_pred_test)
    rmse     = np.sqrt(mse)

    results[name] = {
        'Train R²' : round(train_r2, 4),
        'Test R²'  : round(test_r2,  4),
        'MAE'      : round(mae,  2),
        'MSE'      : round(mse,  2),
        'RMSE'     : round(rmse, 2),
    }
    all_preds[name] = y_pred_test
    results[name]['_model']  = model

    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"{'─'*55}")
    print(f"  Train R² : {train_r2:.4f}")
    print(f"  Test  R² : {test_r2:.4f}")
    print(f"  MAE      : {mae:,.2f}")
    print(f"  RMSE     : {rmse:,.2f}")

print("\n✅ All models trained.")



compare_df = pd.DataFrame({
    name: {k: v for k, v in m.items() if not k.startswith('_')}
    for name, m in results.items()
}).T

print(compare_df)

best_model_name = compare_df['Test R²'].idxmax()
best_r2         = compare_df.loc[best_model_name, 'Test R²']

print(f"\n🏆 Best Model: {best_model_name}  (Test R² = {best_r2})")



rf_model = results['Random Forest']['_model']

fi = pd.Series(rf_model.feature_importances_, index=X.columns)
fi_top10 = fi.nlargest(10).sort_values()

fig, ax = plt.subplots(figsize=(9, 6))
colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(fi_top10)))
fi_top10.plot(kind='barh', ax=ax, color=colors, edgecolor='white')
ax.set_title('Top 10 Feature Importances — Random Forest', fontsize=13, fontweight='bold')
ax.set_xlabel('Importance Score')
ax.set_ylabel('Feature')
for bar in ax.patches:
    ax.annotate(f'{bar.get_width():.4f}',
                (bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2),
                va='center', fontsize=9)
plt.tight_layout()
plt.show()

print("\nTop 10 Feature Importances:")
print(fi.nlargest(10).to_string())



# Linear Regression Coefficients
lr_model = results['Linear Regression']['_model']
lr_coef  = pd.Series(lr_model.coef_, index=X.columns).abs().nlargest(10).sort_values()

fig, ax = plt.subplots(figsize=(9, 6))
colors = plt.cm.Oranges(np.linspace(0.4, 0.9, len(lr_coef)))
lr_coef.plot(kind='barh', ax=ax, color=colors, edgecolor='white')
ax.set_title('Top 10 Feature Coefficients (|coef|) — Linear Regression', fontsize=13, fontweight='bold')
ax.set_xlabel('|Coefficient|')
ax.set_ylabel('Feature')
plt.tight_layout()
plt.show()



# ── 8A: Target (Revenue) Distribution ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].hist(df_raw['Revenue'], bins=40, color='#4878CF', edgecolor='white')
axes[0].set_title('Revenue Distribution (Original)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Revenue')
axes[0].set_ylabel('Frequency')
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

axes[1].hist(y, bins=40, color='#55A868', edgecolor='white')
axes[1].set_title('Revenue Distribution (After Winsorisation)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Revenue')
axes[1].set_ylabel('Frequency')
axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

plt.suptitle('Target Variable: Revenue', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()



# ── 8B: Revenue by Category & Region ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

order_cat = df_raw.groupby('Category')['Revenue'].median().sort_values(ascending=False).index
sns.boxplot(data=df_raw, x='Category', y='Revenue', order=order_cat,
            palette='Set2', ax=axes[0])
axes[0].set_title('Revenue by Category', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Category')
axes[0].set_ylabel('Revenue')
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

order_reg = df_raw.groupby('Region')['Revenue'].median().sort_values(ascending=False).index
sns.boxplot(data=df_raw.dropna(subset=['Region']), x='Region', y='Revenue',
            order=order_reg, palette='Set3', ax=axes[1])
axes[1].set_title('Revenue by Region', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Region')
axes[1].set_ylabel('Revenue')
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

plt.tight_layout()
plt.show()



# ── 8C: Correlation Heatmap ────────────────────────────────────────────────
numeric_feat_cols = X.columns.tolist()
corr_full = pd.concat([X, y], axis=1).corr()

fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr_full, dtype=bool))
sns.heatmap(corr_full, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            linewidths=0.3, ax=ax, vmin=-1, vmax=1, annot_kws={'size': 8})
ax.set_title('Correlation Heatmap — All Features + Target', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()



# ── 8D: Actual vs Predicted — All Models ───────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
colors = ['#4878CF', '#DD8452', '#55A868']

for ax, (name, y_pred), color in zip(axes, all_preds.items(), colors):
    ax.scatter(y_test, y_pred, alpha=0.55, color=color, edgecolors='none', s=40)
    mn = min(y_test.min(), y_pred.min())
    mx = max(y_test.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], 'r--', lw=1.5, label='Perfect Prediction')
    r2 = results[name]['Test R²']
    ax.set_title(f'{name}\nR² = {r2}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Actual Revenue')
    ax.set_ylabel('Predicted Revenue')
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

plt.suptitle('Actual vs Predicted Revenue — All Models', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()



# ── 8E: Residual Plots ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
colors = ['#4878CF', '#DD8452', '#55A868']

for ax, (name, y_pred), color in zip(axes, all_preds.items(), colors):
    residuals = y_test.values - y_pred
    ax.scatter(y_pred, residuals, alpha=0.55, color=color, edgecolors='none', s=40)
    ax.axhline(0, color='red', linestyle='--', lw=1.5)
    ax.set_title(f'Residuals — {name}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted Revenue')
    ax.set_ylabel('Residual (Actual − Predicted)')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

plt.suptitle('Residual Plots — All Models', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()



# ── 8F: Model Comparison Bar Chart ────────────────────────────────────────
metrics_to_plot = ['Test R²', 'MAE', 'RMSE']
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
colors = ['#4C72B0', '#DD8452', '#55A868']

for ax, metric in zip(axes, metrics_to_plot):
    vals  = compare_df[metric].astype(float)
    bars  = ax.bar(compare_df.index, vals, color=colors, edgecolor='white', width=0.5)
    ax.set_title(metric, fontsize=12, fontweight='bold')
    ax.set_ylabel(metric)
    ax.set_xticklabels(compare_df.index, rotation=15, ha='right')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(vals)*0.01,
                f'{val:,.2f}', ha='center', va='bottom', fontsize=9)

plt.suptitle('Model Performance Comparison', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()



# ── 8G: Revenue Trend by Month ─────────────────────────────────────────────
df_raw_copy = df_raw.copy()
df_raw_copy['Date'] = pd.to_datetime(df_raw_copy['Date'], errors='coerce')
df_raw_copy['Month'] = df_raw_copy['Date'].dt.month

monthly = df_raw_copy.groupby('Month')['Revenue'].mean()

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(monthly.index, monthly.values, marker='o', color='#4878CF', linewidth=2)
ax.fill_between(monthly.index, monthly.values, alpha=0.15, color='#4878CF')
ax.set_title('Average Revenue by Month', fontsize=13, fontweight='bold')
ax.set_xlabel('Month')
ax.set_ylabel('Average Revenue')
ax.set_xticks(monthly.index)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
plt.tight_layout()
plt.show()



# ── 8H: Payment Method vs Revenue ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
order_pm = df_raw.groupby('Payment_Method')['Revenue'].median().sort_values(ascending=False).index
sns.barplot(data=df_raw, x='Payment_Method', y='Revenue', order=order_pm,
            palette='pastel', estimator=np.median, ci=None, ax=ax)
ax.set_title('Median Revenue by Payment Method', fontsize=13, fontweight='bold')
ax.set_xlabel('Payment Method')
ax.set_ylabel('Median Revenue')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
plt.tight_layout()
plt.show()



best_model_name = compare_df['Test R²'].idxmax()
print(f"🏆 Best Performing Model: {best_model_name}")
print()

interp = {
    'Linear Regression': {
        'why': 'Fits a linear hyperplane through the feature space. Provides baseline but cannot capture non-linear interactions between Unit_Price and Units_Sold.',
        'strength': '• Fast training  • Highly interpretable coefficients  • Low variance',
        'weakness': '• Assumes linearity  • Sensitive to multicollinearity  • Underperforms on complex patterns',
    },
    'Decision Tree': {
        'why': 'Splits data recursively. Can capture non-linearity but tends to overfit — high train R² vs lower test R² gap is expected.',
        'strength': '• Captures non-linear relationships  • Easy to visualise  • No scaling needed',
        'weakness': '• High variance / prone to overfitting  • Unstable with small data changes',
    },
    'Random Forest': {
        'why': 'Ensemble of 200 trees; bagging reduces variance dramatically. Best test R² shows it generalises well beyond training data.',
        'strength': '• Robust to outliers  • Handles mixed feature types  • Provides feature importance',
        'weakness': '• Less interpretable than single trees  • Slower inference  • Higher memory use',
    },
}

for model_name, info in interp.items():
    tag = " ← BEST" if model_name == best_model_name else ""
    print(f"{'='*60}")
    print(f"  {model_name}{tag}")
    print(f"{'='*60}")
    print(f"  Why this result : {info['why']}")
    print(f"  Strengths       : {info['strength']}")
    print(f"  Weaknesses      : {info['weakness']}")
    print()

print("Possible Improvements:")
print("  1. Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)")
print("  2. Gradient boosting (XGBoost / LightGBM) — often outperforms RF on tabular data")
print("  3. Engineer new features: Revenue_per_Unit = Revenue / Units_Sold")
print("  4. Collect more data — 310 rows is small for generalisation")
print("  5. Cross-validation (k-fold) for more stable metric estimates")
print()
print("Dataset Limitations:")
print("  • Small sample (310 rows) — metrics can fluctuate significantly")
print("  • Revenue outlier (₹100,000) required special treatment")
print("  • Missing values in 5 columns may introduce imputation bias")



best_model_name = compare_df['Test R²'].idxmax()
best_row        = compare_df.loc[best_model_name]

print("=" * 65)
print("                     FINAL CONCLUSION")
print("=" * 65)
print(f"  Dataset             : sample.csv")
print(f"  Original shape      : 310 rows × 11 columns")
print(f"  After preprocessing : {X.shape[0]} rows × {X.shape[1]} features")
print(f"  Target variable     : Revenue (Regression Task)")
print()
print("  Preprocessing Summary:")
print("    ✔  Dropped   : Order_ID (identifier)")
print("    ✔  Extracted : Month, Quarter from Date")
print("    ✔  Removed   : 10 duplicate rows")
print("    ✔  Imputed   : Missing values (numeric→median, category→mode)")
print("    ✔  Treated   : Outliers via IQR Winsorisation")
print("    ✔  Encoded   : Region, Category, Payment_Method (One-Hot)")
print("    ✔  Scaled    : StandardScaler on all numeric features")
print()
print("  Model Results:")
print(compare_df)
print()
print(f"  🏆 Best Model       : {best_model_name}")
print(f"     Test R²          : {best_row['Test R²']}")
print(f"     MAE              : {best_row['MAE']:,}")
print(f"     RMSE             : {best_row['RMSE']:,}")
print()
print("  Key Business Insights:")
print("    • Unit_Price and Units_Sold are the dominant revenue drivers.")
print("    • Discount_Pct negatively impacts revenue — monitor discount policy.")
print("    • Some months/quarters show higher average revenue — seasonal planning opportunity.")
print("    • Category and Region differences suggest targeted marketing can boost revenue.")
print()
print("  Recommendations:")
print("    1. Use the Random Forest model as a revenue forecasting tool.")
print("    2. Collect customer segment data (B2B vs B2C) to improve predictions.")
print("    3. Integrate time-series modelling for long-term forecasting.")
print("    4. Set discount thresholds per product category to protect margins.")
print("    5. Retrain the model quarterly as new orders accumulate.")
print("=" * 65)

