"""
Beginner-friendly feature selection analysis.

Loads the full engineered dataset, drops rows with missing
`treatment_outcome` for the supervised analysis, computes numeric
Pearson correlations and categorical associations (Cramer's V),
saves heatmaps and writes a summary report.

Do not modify the source dataset.
"""
from pathlib import Path
import warnings
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency


OUTPUT_DIR = Path("outputs")
ENGINEERED_FP = OUTPUT_DIR / "data_engineered.parquet"
REPORT_FP = OUTPUT_DIR / "feature_selection_analysis_report.txt"
NUM_HEATMAP = OUTPUT_DIR / "numerical_correlation_heatmap.png"
CAT_HEATMAP = OUTPUT_DIR / "categorical_association_heatmap.png"


def cramers_v(x, y):
    """Compute Cramer's V statistic for categorical-categorical association.

    Uses bias correction via chi2_contingency and returns 0..1.
    """
    confusion = pd.crosstab(x, y)
    if confusion.size == 0:
        return 0.0
    chi2, p, dof, ex = chi2_contingency(confusion, correction=False)
    n = confusion.sum().sum()
    if n == 0:
        return 0.0
    phi2 = chi2 / n
    r, k = confusion.shape
    # apply bias correction
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        phi2corr = max(0.0, phi2 - ((k - 1)*(r - 1)) / (n - 1)) if n > 1 else phi2
    rcorr = r - ((r - 1)**2) / (n - 1) if n > 1 else r
    kcorr = k - ((k - 1)**2) / (n - 1) if n > 1 else k
    denom = min((kcorr - 1), (rcorr - 1))
    if denom <= 0:
        return 0.0
    return math.sqrt(phi2corr / denom)


def main():
    print("Progress: Loading engineered dataset (full load). This may use substantial memory.")
    if not ENGINEERED_FP.exists():
        raise FileNotFoundError(f"{ENGINEERED_FP} not found")

    df = pd.read_parquet(ENGINEERED_FP)

    # 11. Validate source dataset shape remains 1,050,000 rows and 38 columns
    expected_shape = (1050000, 38)
    actual_shape = df.shape
    # If the source dataset does not match exactly, stop with a clear error.
    if actual_shape != expected_shape:
        raise ValueError(f"Source dataset shape {actual_shape} does not match expected {expected_shape}. Aborting.")

    print(f"Progress: dataset loaded with shape {actual_shape}")

    # 2. For supervised analysis, remove rows where treatment_outcome is missing
    print("Progress: Dropping rows with missing treatment_outcome for analysis only...")
    if "treatment_outcome" not in df.columns:
        raise KeyError("treatment_outcome column not found in dataset")
    df_labelled = df[df["treatment_outcome"].notna()].copy()

    # Number of labelled rows
    labelled_rows = df_labelled.shape[0]

    # 3. Exclude specified columns from feature analysis (but do not remove from df)
    excluded_columns = {
        "patient_id": "identifier",
        "admission_date": "raw date",
        "adverse_event": "post-treatment outcome (leakage)",
        "readmission_30d": "post-treatment outcome (leakage)",
    }

    # Prepare columns lists
    analysis_cols = [c for c in df_labelled.columns if c not in excluded_columns]

    # Identify numeric and categorical predictors (exclude label column from predictors)
    predictors = [c for c in analysis_cols if c != "treatment_outcome"]
    numeric_cols = df_labelled[predictors].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in predictors if c not in numeric_cols]

    # Ensure treatment_outcome included appropriately
    # For numeric heatmap include treatment_outcome if numeric
    numeric_heatmap_cols = numeric_cols.copy()
    if pd.api.types.is_numeric_dtype(df_labelled["treatment_outcome"]):
        numeric_heatmap_cols.append("treatment_outcome")
    else:
        print("Note: treatment_outcome is not numeric; it will be included only in categorical association heatmap.")

    # 4. Numerical correlation heatmap (Pearson)
    print("Progress: Computing numerical Pearson correlations...")
    NUM_HEATMAP.parent.mkdir(parents=True, exist_ok=True)
    if len(numeric_heatmap_cols) >= 2:
        num_df = df_labelled[numeric_heatmap_cols].copy()
        corr = num_df.corr(method="pearson")
        plt.figure(figsize=(max(8, len(corr)*0.4), max(6, len(corr)*0.4)))
        sns.heatmap(corr, annot=False, cmap="vlag", center=0)
        plt.title("Numerical Pearson Correlation")
        plt.tight_layout()
        plt.savefig(NUM_HEATMAP, dpi=150)
        plt.close()
    else:
        print("Warning: Not enough numeric columns to compute a numerical heatmap.")

    # 5. Categorical association heatmap (Cramer's V)
    print("Progress: Computing categorical associations (Cramer's V)...")
    CAT_HEATMAP.parent.mkdir(parents=True, exist_ok=True)
    # Always include treatment_outcome as a categorical variable for Cramer's V analysis.
    # Create a temporary categorical DataFrame containing all categorical predictors,
    # then add treatment_outcome converted to integer->string.
    cat_cols_with_label = categorical_cols + ["treatment_outcome"]
    cats = cat_cols_with_label
    cat_data = df_labelled[categorical_cols].astype(str).copy()
    # Convert treatment_outcome to int then to str to create discrete categories
    cat_data["treatment_outcome"] = df_labelled["treatment_outcome"].astype(int).astype(str)

    # prepare an empty square DataFrame for symmetric results
    mat = pd.DataFrame(index=cats, columns=cats, dtype=float)
    if len(cats) >= 2:
        # compute only upper triangle and mirror to lower triangle to save work
        for i in range(len(cats)):
            for j in range(i, len(cats)):
                a = cats[i]
                b = cats[j]
                try:
                    v = cramers_v(cat_data[a], cat_data[b])
                except Exception:
                    v = 0.0
                mat.iat[i, j] = v
                mat.iat[j, i] = v
        # set diagonal explicitly to 1.0
        for c in cats:
            mat.at[c, c] = 1.0

        plt.figure(figsize=(max(8, len(cats)*0.4), max(6, len(cats)*0.4)))
        sns.heatmap(mat.astype(float), annot=False, cmap="YlGnBu", vmin=0, vmax=1)
        plt.title("Categorical associations (Cramer's V)")
        plt.tight_layout()
        plt.savefig(CAT_HEATMAP, dpi=150)
        plt.close()
    else:
        print("Warning: Not enough categorical columns to compute categorical association heatmap.")

    # 6. Assemble report
    print("Progress: Generating report...")
    lines = []
    lines.append("FEATURE SELECTION ANALYSIS REPORT")
    lines.append("================================")
    lines.append(f"Source dataset path: {ENGINEERED_FP}")
    lines.append(f"Source dataset shape: {actual_shape}")
    lines.append(f"Expected source shape: {expected_shape}")
    lines.append("")
    lines.append(f"Labelled rows used (treatment_outcome not null): {labelled_rows}")
    lines.append("")
    lines.append("Columns excluded from predictor analysis and reasons:")
    for c, reason in excluded_columns.items():
        lines.append(f"- {c}: {reason}")

    lines.append("")
    # Numerical features ranked by absolute correlation with treatment_outcome
    lines.append("Numerical features ranked by absolute Pearson correlation with treatment_outcome:")
    num_rank = []
    if "treatment_outcome" in numeric_heatmap_cols:
        for col in numeric_cols:
            if col == "treatment_outcome":
                continue
            try:
                r = corr.at[col, "treatment_outcome"]
            except Exception:
                r = np.nan
            num_rank.append((col, r, abs(r) if r is not None and not np.isnan(r) else np.nan))
        num_rank.sort(key=lambda x: (np.nan_to_num(x[2], nan=-1)), reverse=True)
        for col, r, ar in num_rank:
            lines.append(f"- {col}: r={r:.4f} | |r|={ar:.4f}")
    else:
        lines.append("- treatment_outcome is not numeric; numerical ranking skipped.")

    lines.append("")
    # Categorical features ranked by Cramer's V with treatment_outcome
    lines.append("Categorical features ranked by Cramer's V association with treatment_outcome:")
    cat_rank = []
    # Rank every categorical predictor (exclude the label itself) by its Cramer's V with treatment_outcome
    if "treatment_outcome" in cats:
        for col in categorical_cols:
            # safe fetch from matrix
            try:
                v = float(mat.at[col, "treatment_outcome"])
            except Exception:
                v = 0.0
            cat_rank.append((col, v))
        cat_rank.sort(key=lambda x: x[1], reverse=True)
        for col, v in cat_rank:
            lines.append(f"- {col}: Cramer's V = {v:.4f}")
    else:
        lines.append("- treatment_outcome not present in categorical set; categorical ranking skipped.")

    lines.append("")
    # Any numerical feature pairs with abs(corr) > 0.90
    lines.append("Highly correlated numerical feature pairs (|r| > 0.90):")
    high_pairs = []
    if len(numeric_heatmap_cols) >= 2:
        for i, a in enumerate(numeric_heatmap_cols):
            for b in numeric_heatmap_cols[i+1:]:
                rv = corr.at[a, b]
                if abs(rv) > 0.90:
                    high_pairs.append((a, b, rv))
        if high_pairs:
            for a, b, rv in high_pairs:
                lines.append(f"- {a} <-> {b}: r={rv:.4f}")
        else:
            lines.append("- None found")
    else:
        lines.append("- Not enough numeric columns to evaluate high correlations.")

    lines.append("")
    # Constant or near-constant columns
    lines.append("Constant or near-constant columns (low variance or single unique value):")
    const_cols = []
    for c in df.columns:
        nunique = df[c].nunique(dropna=False)
        if nunique <= 1:
            const_cols.append((c, nunique))
    if const_cols:
        for c, n in const_cols:
            lines.append(f"- {c}: unique values={n}")
    else:
        lines.append("- None found")

    lines.append("")
    # Recommendations (basic, beginner friendly)
    lines.append("Recommendations:")
    if len(num_rank) > 0:
        top_num = [c for c, _, _ in num_rank[:5]]
        lines.append(f"- Useful numeric predictors (top ~5 by |r|): {top_num}")
    else:
        lines.append("- No numeric ranking available.")
    if len(cat_rank) > 0:
        top_cat = [c for c, _ in cat_rank[:5]]
        lines.append(f"- Useful categorical predictors (top ~5 by Cramer's V): {top_cat}")
    else:
        lines.append("- No categorical ranking available.")
    if high_pairs:
        lines.append("- Potentially redundant numeric pairs detected; consider removing or combining one of each pair.")
    else:
        lines.append("- No strongly redundant numeric pairs detected.")

    lines.append("")
    lines.append("Notes:")
    lines.append("- Columns excluded from analysis were not removed from the saved dataset; exclusion was only for feature analysis.")
    lines.append("- Cramer's V is used for categorical associations; it is symmetric and ranges 0..1.")

    # Save report
    REPORT_FP.write_text("\n".join(lines), encoding="utf-8")
    print(f"Progress: Report saved to {REPORT_FP}")
    print(f"Progress: Numerical heatmap saved to {NUM_HEATMAP}")
    print(f"Progress: Categorical heatmap saved to {CAT_HEATMAP}")


if __name__ == "__main__":
    main()
