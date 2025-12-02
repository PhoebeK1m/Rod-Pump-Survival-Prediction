import os
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import statsmodels.api as sm

# ---------------------------------- LOWESS Linearity Test -----------------------------------------
def test_linearity(martingale, x, frac=0.3):
    """Compute deviation of LOWESS smooth from zero to assess linearity."""
    df = pd.DataFrame({"m": martingale, "x": x}).dropna()
    if df["x"].nunique() < 3:
        return np.nan, None

    lowess = sm.nonparametric.lowess(df["m"], df["x"], frac=frac)
    lowess = lowess[np.argsort(lowess[:, 0])]  # ensure monotonic x ordering

    deviation = np.mean(np.abs(lowess[:, 1]))
    return deviation, lowess


# ---------------------------------- Martingale Residual Diagnostics -------------------------------
def plot_martingale_residuals(cph, train_df, duration_col, event_col, out_prefix):
    """Generate martingale residual plots for assessing linearity of numeric predictors."""

    raw_resid = cph.compute_residuals(train_df, kind="martingale")
    martingale = raw_resid["martingale"]

    numeric_transformed_cols = [c for c in train_df.columns if c.startswith("num__")]
    results = []

    for col in numeric_transformed_cols:
        x = train_df[col].values
        score, lowess = test_linearity(martingale.values, x)

        if lowess is not None:
            plt.figure(figsize=(6, 4))
            plt.scatter(x, martingale, s=10, alpha=0.4)
            plt.plot(lowess[:, 0], lowess[:, 1], color="red", linewidth=2)
            plt.axhline(0, color="black", linestyle="--")
            plt.xlabel(col.replace("num__", ""))
            plt.ylabel("Martingale Residual")
            plt.title(f"Martingale Residual Plot\n{col.replace('num__','')}")
            plt.tight_layout()
            plt.savefig(f"{out_prefix}_martingale_{col.replace('num__','')}.png")
            plt.close()

        results.append({
            "variable": col.replace("num__", ""),
            "linearity_score": score
        })

    return pd.DataFrame(results)


# ---------------------------------- Main Workflow --------------------------------------------------
data_dir = "/training data"
csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

os.makedirs("martingale", exist_ok=True)

for file in csv_files:
    print(f"\nDiagnostics for {file}")

    df = pd.read_csv(os.path.join(data_dir, file))

    if "FAILED" not in df.columns or "lifetime_duration_days" not in df.columns:
        print(f"Skipping {file} (required columns missing)")
        continue

    df = df.dropna(subset=["lifetime_duration_days", "FAILED"]).copy()

    drop_cols = [
        "lifetime_start", "lifetime_end", "roduid", "UWI", "NODEID",
        "IDWELL", "tbguid", "IDRECJOBPULL", "REPORTTO/FAILURETYPE"
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    duration_col = "lifetime_duration_days"
    event_col = "FAILED"

    numeric_cols = df.select_dtypes(include=["int64", "float64"])
    numeric_cols = numeric_cols.drop(columns=[duration_col, event_col], errors="ignore")
    categorical_cols = df.select_dtypes(exclude=["int64", "float64"])

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols.columns),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols.columns)
    ])

    X = df[numeric_cols.columns.union(categorical_cols.columns)]
    X_prep = preprocessor.fit_transform(X)
    X_df = pd.DataFrame(X_prep, columns=preprocessor.get_feature_names_out())

    train_df = X_df.copy()
    train_df[duration_col] = df[duration_col].values
    train_df[event_col] = df[event_col].values

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(train_df, duration_col=duration_col, event_col=event_col)

    out_dir = f"martingale/{file.replace('.csv','')}"
    os.makedirs(out_dir, exist_ok=True)

    print("Testing linearity (Martingale residuals)...")

    linearity_df = plot_martingale_residuals(
        cph, train_df, duration_col, event_col,
        out_prefix=f"{out_dir}/diagnostics"
    )