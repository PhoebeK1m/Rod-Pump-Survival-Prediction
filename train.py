import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from lifelines import CoxPHFitter
from sklearn.preprocessing import StandardScaler
from lifelines import KaplanMeierFitter

# ---------------------------------- Load Data -----------------------------------------------------
# df = pd.read_csv("/Users/phoebekim/Downloads/nsc data/control.csv")

# ---------------------------------- Drop Unused Columns -------------------------------------------
drop_cols = ["FAILURETYPE", "UWI", "tbguid", "lifetime_start"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# ---------------------------------- Encode Categoricals -------------------------------------------
selected_cats = ["bha_configuration", "rod_sinker_type", "rod_apigrade", "ROUTE"]
for col in selected_cats:
    df[col] = df[col].astype(str)

df = pd.get_dummies(df, columns=selected_cats, drop_first=True)

# ---------------------------------- Clean Column Names --------------------------------------------
def clean_col(s):
    s = re.sub(r"[^0-9a-zA-Z_]", "_", s)
    s = re.sub(r"_+", "_", s)
    return s

df.columns = [clean_col(c) for c in df.columns]

# ---------------------------------- Fill Missing Numeric Values -----------------------------------
numeric_cols_all = df.select_dtypes(include=[np.number]).columns
df[numeric_cols_all] = df[numeric_cols_all].fillna(df[numeric_cols_all].median())

# ---------------------------------- Drop Zero-Variance Columns -------------------------------------
zero_var = [c for c in numeric_cols_all if df[c].var() == 0]
if zero_var:
    df = df.drop(columns=zero_var)

numeric_cols_all = df.select_dtypes(include=[np.number]).columns

# ---------------------------------- Scale Numeric Columns ------------------------------------------
exclude = ["FAILED", "sample_weight", "lifetime_duration_days"]
numeric_cols = [c for c in numeric_cols_all if c not in exclude]

scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# ---------------------------------- Log Transform Skewed Features ----------------------------------
def is_skewed(x):
    return abs(pd.Series(x).skew()) > 1

for col in numeric_cols:
    if is_skewed(df[col]):
        df[col] = np.sign(df[col]) * np.log1p(np.abs(df[col]))

# ---------------------------------- Select Spline Candidates ----------------------------------------
continuous_cols = [c for c in numeric_cols if df[c].nunique() >= 10]
variances = df[continuous_cols].var().sort_values(ascending=False)
top_spline_vars = list(variances.head(23).index)

# ---------------------------------- Build Spline Formula -------------------------------------------
spline_terms = " + ".join([
    f"bs({col}, df=4, include_intercept=False)" for col in top_spline_vars
])

# ---------------------------------- Build Full Formula ---------------------------------------------
base_vars = [c for c in df.columns if c not in exclude]
base_formula = " + ".join(base_vars)

full_formula = spline_terms + " + " + base_formula

# ---------------------------------- LASSO Cox for Feature Reduction --------------------------------
cox_lasso = CoxPHFitter(penalizer=1.0, l1_ratio=1.0)

# uncomment if data included sample weight
cox_lasso.fit(
    df,
    duration_col="lifetime_duration_days",
    event_col="FAILED",
    weights_col="sample_weight",
    formula=full_formula,
    robust=True
)
# cox_lasso.fit(
#     df,
#     duration_col="lifetime_duration_days",
#     event_col="FAILED",
#     # weights_col="sample_weight",
#     formula=full_formula,
#     robust=True
# )

lasso_summary = cox_lasso.summary
lasso_summary["abs_z"] = lasso_summary["z"].abs()

top_selected = lasso_summary.sort_values("abs_z", ascending=False).head(30).index.tolist()
spline_basis_terms = [s for s in lasso_summary.index if s.startswith("bs(")]
top_selected += spline_basis_terms

# ---------------------------------- Clean Selected Names --------------------------------------------
def to_df_col(name):
    if name.startswith("bs("):
        return name
    name = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    name = re.sub(r"_+", "_", name)
    return name

selected_clean = [to_df_col(c) for c in top_selected]
valid_df_cols = set(df.columns)
final_terms = []

for term in selected_clean:
    if term.startswith("bs("):
        final_terms.append(term)
    elif term in valid_df_cols:
        final_terms.append(term)

final_formula = " + ".join(final_terms)

# ---------------------------------- Fit Final Cox Model (Ridge) ------------------------------------
final_mod = CoxPHFitter(penalizer=0.5, l1_ratio=0.0)

final_mod.fit(
    df,
    duration_col="lifetime_duration_days",
    event_col="FAILED",
    weights_col="sample_weight",
    formula=final_formula,
    robust=True
)

# ---------------------------------- Compute Risk Scores ---------------------------------------------
df["risk_score"] = final_mod.predict_partial_hazard(df)

df_sorted = df.sort_values("risk_score", ascending=False)
top10 = df_sorted.head(10)
bottom10 = df_sorted.tail(10)

# ---------------------------------- Plot Survival Curves (High Risk) --------------------------------
plt.figure(figsize=(10, 8))
for i in top10.index:
    surv = final_mod.predict_survival_function(df.loc[[i]])
    plt.plot(surv.index, surv.values, alpha=0.7)

plt.title("Top 10 Highest-Risk Wells – Predicted Survival Curves")
plt.xlabel("Days")
plt.ylabel("Survival Probability")
plt.grid(True)
plt.tight_layout()
plt.savefig("top10_high_risk_survival.png", dpi=300)
plt.show()

# ---------------------------------- Plot Survival Curves (Low Risk) ---------------------------------
plt.figure(figsize=(10, 8))
for i in bottom10.index:
    surv = final_mod.predict_survival_function(df.loc[[i]])
    plt.plot(surv.index, surv.values, alpha=0.7)

plt.title("Bottom 10 Lowest-Risk Wells – Predicted Survival Curves")
plt.xlabel("Days")
plt.ylabel("Survival Probability")
plt.grid(True)
plt.tight_layout()
plt.savefig("bottom10_low_risk_survival.png", dpi=300)
plt.show()

# ---------------------------------- Coefficient Cleaning --------------------------------------------
def clean_spline_name(term):
    m = re.match(r"bs\((.*?),.*?\)\[(\d+)\]", term)
    if m:
        feature = m.group(1)
        idx = m.group(2)
        return f"{feature}_spline_{idx}"
    return term

coef_series = final_mod.summary["coef"]
coef_series.index = [clean_spline_name(i) for i in coef_series.index]
coef_series = coef_series.sort_values(ascending=False)

top10_coef = coef_series.head(10)
bottom10_coef = coef_series.tail(10)

plt.figure(figsize=(7, 8))
top10_coef.plot(kind="barh")
plt.gca().invert_yaxis()
plt.title("Top 10 Highest-Risk Features")
plt.xlabel("Coefficient (log hazard ratio)")
plt.tight_layout()
plt.savefig("top10_highest_coefficients.png", dpi=300)
plt.show()

plt.figure(figsize=(7, 8))
bottom10_coef.plot(kind="barh", color='gray')
plt.gca().invert_yaxis()
plt.title("Top 10 Most Protective Features")
plt.xlabel("Coefficient (log hazard ratio)")
plt.tight_layout()
plt.savefig("top10_lowest_coefficients.png", dpi=300)
plt.show()

# ---------------------------------- Highest and Lowest Risk Wells -----------------------------------
highest_risk_idx = df_sorted.index[0]
lowest_risk_idx = df_sorted.index[-1]

print("\n=== Highest-risk well info ===")
print("Index:", highest_risk_idx)
print("Hazard Ratio:", df.loc[highest_risk_idx, "risk_score"])

print("\n=== Lowest-risk well info ===")
print("Index:", lowest_risk_idx)
print("Hazard Ratio:", df.loc[lowest_risk_idx, "risk_score"])

# ---------------------------------- Save Risk Scores ------------------------------------------------
df.to_csv("cox_failure_risk_scores.csv", index=False)

# ---------------------------------- Kaplan-Meier Curves ---------------------------------------------
df["risk_group"] = pd.qcut(df["risk_score"], 2, labels=["Low Risk", "High Risk"])

kmf = KaplanMeierFitter()
plt.figure(figsize=(10, 6))

for grp in ["Low Risk", "High Risk"]:
    mask = df["risk_group"] == grp
    kmf.fit(
        df.loc[mask, "lifetime_duration_days"],
        df.loc[mask, "FAILED"],
        label=grp
    )
    kmf.plot(ci_show=False)

plt.title("Kaplan–Meier Curves: High vs Low Risk Wells")
plt.xlabel("Days")
plt.ylabel("Survival Probability")
plt.grid(True)
plt.tight_layout()
plt.savefig("Kaplan–Meier_Curves_High_vs_Low_Risk.png", dpi=300)
plt.show()