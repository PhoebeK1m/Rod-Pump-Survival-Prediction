import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer

# ---------------------------------- Load Data -----------------------------------------------------
file_path = "/Users/phoebekim/Downloads/nsc data/training_data/prev/correct_types_rodpump.csv"
df = pd.read_csv(file_path)

target_col = "FAILED"

# Identify numeric and categorical columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=[np.number, "datetime"]).columns.tolist()

# ---------------------------------- Basic Statistical Imputation ----------------------------------
df_basic = df.copy()

# Fill numeric columns with median values
for col in numeric_cols:
    df_basic[col] = df_basic[col].fillna(df_basic[col].median())

# Fill categorical columns with mode or fallback token
for col in categorical_cols:
    mode_vals = df_basic[col].mode()
    fill_value = mode_vals.iloc[0] if not mode_vals.empty else "unknown"
    df_basic[col] = df_basic[col].fillna(fill_value)

print("Basic imputation complete.")
print(df_basic.isna().sum().sum(), "missing values remaining.")

# ---------------------------------- KNN Imputation for Numeric Columns ----------------------------
knn_imputer = KNNImputer(n_neighbors=5)
df_knn = df.copy()

# Apply KNN imputation to numeric features
df_knn[numeric_cols] = knn_imputer.fit_transform(df_knn[numeric_cols])

# Re-identify categorical columns after transformation
categorical_cols = df_knn.select_dtypes(exclude=[np.number, "datetime"]).columns.tolist()

# Fill categorical columns with mode after KNN
# Do NOT fill columns that will be handled by the RandomForest model
rf_target_cols = ["bha_configuration","packer_vs_tac","wellbore_category","rod_apigrade"]

for col in categorical_cols:
    if col in rf_target_cols:
        continue  # skip RF-imputed columns
    mode_vals = df_knn[col].mode()
    fill_value = mode_vals.iloc[0] if not mode_vals.empty else "unknown"
    df_knn[col] = df_knn[col].fillna(fill_value)

print("KNN imputation complete.")
print(df_knn.isna().sum().sum(), "missing values remaining.")

# Save basic
df_basic.to_csv("control.csv", index=False)  # median + mode
print("\nFiles saved:")
print(" - control.csv (basic imputation)")

# ---------------------------------- Encode Categorical Columns ----------------------------------
encoder = OrdinalEncoder()
X_full = df_knn[categorical_cols + numeric_cols].copy()

# Ensure categorical data is string type before encoding
X_full[categorical_cols] = encoder.fit_transform(X_full[categorical_cols].astype(str))

# Columns selected for predictive imputation
columns_to_impute = [
    "bha_configuration",
    "packer_vs_tac",
    "wellbore_category",
    "rod_apigrade"
]

# ---------------------------------- Predictive Imputation ---------------------------------------
for col in columns_to_impute:
    missing_mask = df_knn[col].isna()

    # Skip if no missing values
    if missing_mask.sum() == 0:
        print(f"'{col}' has no missing values — skipping.")
        continue

    print(f"Predicting missing values for '{col}'...")

    # Prepare training and prediction datasets
    train_df = X_full.loc[~missing_mask]
    test_df = X_full.loc[missing_mask]
    y_train = df_knn.loc[~missing_mask, col]

    # Train classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(train_df.fillna(0), y_train)

    # Predict missing values
    preds = rf.predict(test_df.fillna(0))
    df_knn.loc[missing_mask, col] = preds
    print(f"Filled {len(preds)} missing values in '{col}'.")

# ---------------------------------- Report Remaining Missing Values ------------------------------
missing_after = df_knn.isna().sum()
print("\nMissing values after imputation:")
print(missing_after[missing_after > 0].sort_values(ascending=False))

# ---------------------------------- Save Outputs --------------------------------------------------
df_knn.to_csv("knn_rf.csv", index=False)
print("\nFiles saved:")
print(" - knn_rf.csv (KNN + RF imputation)")
