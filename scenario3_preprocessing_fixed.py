"""Reusable Scenario 3 preprocessing transformers.

The transformer learns all data-dependent values from the training split only
and reuses those stored statistics for validation/test transforms.
It operates on the 26 base predictor columns used by Scenario 3.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


BASE_NUMERIC_COLUMNS = [
    "age",
    "weight_kg",
    "height_cm",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "heart_rate",
    "temperature_f",
    "hemoglobin",
    "wbc_count",
    "alt_enzyme",
    "ast_enzyme",
    "creatinine",
    "egfr",
    "hba1c",
    "total_cholesterol",
    "dosage_mg",
    "duration_days",
    "concurrent_drugs",
]

BASE_CATEGORICAL_COLUMNS = [
    "gender",
    "ethnicity",
    "drug_name",
    "route",
    "diagnosis",
    "smoking_status",
    "alcohol_use",
]

EXPECTED_BASE_PREDICTOR_COLUMNS = BASE_NUMERIC_COLUMNS + BASE_CATEGORICAL_COLUMNS

STEP02_IQR_COLUMNS = [
    "age",
    "bmi",
    "dosage_mg",
    "hemoglobin",
    "creatinine",
    "alt_enzyme",
    "ast_enzyme",
]

ENGINEERED_CATEGORICAL_COLUMNS = ["kidney_stage", "bmi_category", "age_group"]
ENGINEERED_NUMERIC_COLUMNS = [
    "liver_risk",
    "polypharmacy",
    "elderly_high_dose",
    "de_ritis_ratio",
]

FINAL_NUMERIC_COLUMNS = BASE_NUMERIC_COLUMNS + ENGINEERED_NUMERIC_COLUMNS
FINAL_CATEGORICAL_COLUMNS = BASE_CATEGORICAL_COLUMNS + ENGINEERED_CATEGORICAL_COLUMNS
FINAL_FEATURE_COLUMNS = FINAL_NUMERIC_COLUMNS + FINAL_CATEGORICAL_COLUMNS


def _make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def _as_dataframe(X):
    if not isinstance(X, pd.DataFrame):
        raise TypeError("Scenario3ValidationPreprocessor expects a pandas DataFrame.")
    return X


def _to_na_like(value):
    if pd.isna(value):
        return pd.NA
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "" or text in {"-", "--", "na", "n/a", "unknown", "<missing>"}:
            return pd.NA
    return value


class Scenario3ValidationPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.base_columns_ = None
        self.predictor_columns_ = None
        self.numeric_predictor_columns_ = None
        self.categorical_predictor_columns_ = None
        self.numeric_medians_ = None
        self.categorical_modes_ = None
        self.iqr_limits_ = None
        self.dosage_median_ = None
        self.ratio_median_ = None
        self.ratio_q1_ = None
        self.ratio_q3_ = None
        self.ratio_iqr_ = None
        self.ratio_lower_limit_ = None
        self.ratio_upper_limit_ = None
        self.feature_columns_ = None
        self.feature_numeric_columns_ = None
        self.feature_categorical_columns_ = None
        self.ohe_ = None
        self.encoded_feature_names_ = None
        self.encoded_feature_count_ = None

    def _validate_and_order_base_predictors(self, X):
        X = _as_dataframe(X).copy()
        missing = [column for column in EXPECTED_BASE_PREDICTOR_COLUMNS if column not in X.columns]
        extra = [column for column in X.columns if column not in EXPECTED_BASE_PREDICTOR_COLUMNS]
        if missing:
            raise SystemExit(f"Scenario 3 input is missing expected predictors: {missing}")
        if extra:
            raise SystemExit(f"Scenario 3 input contains unexpected predictors: {extra}")
        return X[EXPECTED_BASE_PREDICTOR_COLUMNS].copy()

    def _normalize_base_inputs(self, X):
        numeric_df = pd.DataFrame(index=X.index)
        categorical_df = pd.DataFrame(index=X.index)

        for column in BASE_NUMERIC_COLUMNS:
            series = pd.to_numeric(X[column], errors="coerce").astype("float64")
            series = series.replace([np.inf, -np.inf], np.nan)
            numeric_df[column] = series

        for column in BASE_CATEGORICAL_COLUMNS:
            categorical_df[column] = X[column].copy().apply(_to_na_like).astype("object")

        return numeric_df, categorical_df

    def _impute_numeric_from_stats(self, numeric_df):
        numeric = numeric_df.copy().apply(pd.to_numeric, errors="coerce").astype("float64")
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        for column in BASE_NUMERIC_COLUMNS:
            numeric[column] = numeric[column].fillna(self.numeric_medians_[column]).astype("float64")
        return numeric

    def _impute_categorical_from_stats(self, categorical_df):
        categorical = categorical_df.copy()
        for column in BASE_CATEGORICAL_COLUMNS:
            categorical[column] = categorical[column].apply(_to_na_like)
            categorical[column] = categorical[column].fillna(self.categorical_modes_[column])
        return categorical

    def _apply_iqr_clipping(self, numeric_df):
        clipped = numeric_df.copy()
        for column, limits in self.iqr_limits_.items():
            clipped[column] = clipped[column].clip(lower=limits["lower"], upper=limits["upper"])
        return clipped

    def _learn_base_statistics(self, numeric_df, categorical_df):
        self.numeric_medians_ = numeric_df.median(skipna=True).astype("float64")
        if self.numeric_medians_.isna().any():
            missing = self.numeric_medians_[self.numeric_medians_.isna()].index.tolist()
            raise SystemExit(f"Unable to learn training numeric medians for: {missing}")

        self.categorical_modes_ = {}
        for column in BASE_CATEGORICAL_COLUMNS:
            mode_series = categorical_df[column].mode(dropna=True)
            if mode_series.empty:
                raise SystemExit(f"Unable to learn a training mode for categorical column: {column}")
            self.categorical_modes_[column] = mode_series.iloc[0]

        imputed_numeric = self._impute_numeric_from_stats(numeric_df)
        self.iqr_limits_ = {}
        for column in STEP02_IQR_COLUMNS:
            series = pd.to_numeric(imputed_numeric[column], errors="coerce").astype("float64")
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            self.iqr_limits_[column] = {
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower": q1 - (1.5 * iqr),
                "upper": q3 + (1.5 * iqr),
            }

        clipped_numeric = self._apply_iqr_clipping(imputed_numeric)
        self.dosage_median_ = float(clipped_numeric["dosage_mg"].median(skipna=True))
        if pd.isna(self.dosage_median_):
            raise SystemExit("Unable to learn a training dosage_mg median.")

        alt = pd.to_numeric(clipped_numeric["alt_enzyme"], errors="coerce").astype("float64")
        ast = pd.to_numeric(clipped_numeric["ast_enzyme"], errors="coerce").astype("float64")
        ratio_series = pd.Series(np.nan, index=clipped_numeric.index, dtype="float64")
        valid_alt = alt > 0
        ratio_series.loc[valid_alt] = ast.loc[valid_alt] / alt.loc[valid_alt]
        valid_ratios = ratio_series.dropna()
        if valid_ratios.empty:
            raise SystemExit("No valid De Ritis ratios were available in training data.")
        self.ratio_median_ = float(valid_ratios.median())
        filled_ratio = ratio_series.fillna(self.ratio_median_)
        self.ratio_q1_ = float(filled_ratio.quantile(0.25))
        self.ratio_q3_ = float(filled_ratio.quantile(0.75))
        self.ratio_iqr_ = self.ratio_q3_ - self.ratio_q1_
        self.ratio_lower_limit_ = max(0.0, self.ratio_q1_ - (1.5 * self.ratio_iqr_))
        self.ratio_upper_limit_ = self.ratio_q3_ + (1.5 * self.ratio_iqr_)

    def _engineer_features(self, numeric_df, categorical_df):
        features = pd.DataFrame(index=numeric_df.index)

        for column in numeric_df.columns:
            features[column] = pd.to_numeric(numeric_df[column], errors="coerce").astype("float64")
        for column in categorical_df.columns:
            features[column] = categorical_df[column]

        egfr = pd.to_numeric(features["egfr"], errors="coerce").astype("float64")
        bmi = pd.to_numeric(features["bmi"], errors="coerce").astype("float64")
        age = pd.to_numeric(features["age"], errors="coerce").astype("float64")
        alt = pd.to_numeric(features["alt_enzyme"], errors="coerce").astype("float64")
        ast = pd.to_numeric(features["ast_enzyme"], errors="coerce").astype("float64")
        concurrent_drugs = pd.to_numeric(features["concurrent_drugs"], errors="coerce").astype("float64")
        dosage = pd.to_numeric(features["dosage_mg"], errors="coerce").astype("float64")

        kidney_stage = pd.Series("Unknown", index=features.index, dtype="object")
        kidney_stage.loc[egfr >= 90] = "Normal"
        kidney_stage.loc[(egfr >= 60) & (egfr < 90)] = "Mild"
        kidney_stage.loc[(egfr >= 30) & (egfr < 60)] = "Moderate"
        kidney_stage.loc[egfr < 30] = "Severe"
        features["kidney_stage"] = kidney_stage

        bmi_category = pd.Series("Unknown", index=features.index, dtype="object")
        bmi_category.loc[bmi < 18.5] = "Underweight"
        bmi_category.loc[(bmi >= 18.5) & (bmi < 25)] = "Normal"
        bmi_category.loc[(bmi >= 25) & (bmi < 30)] = "Overweight"
        bmi_category.loc[bmi >= 30] = "Obese"
        features["bmi_category"] = bmi_category

        age_group = pd.Series("Unknown", index=features.index, dtype="object")
        age_group.loc[age < 18] = "Pediatric"
        age_group.loc[(age >= 18) & (age < 35)] = "Young Adult"
        age_group.loc[(age >= 35) & (age < 65)] = "Middle Aged"
        age_group.loc[(age >= 65) & (age < 80)] = "Senior"
        age_group.loc[age >= 80] = "Elderly"
        features["age_group"] = age_group

        features["liver_risk"] = ((alt > 40) | (ast > 40)).astype("int8")
        features["polypharmacy"] = (concurrent_drugs >= 5).astype("int8")
        features["elderly_high_dose"] = ((age >= 65) & (dosage > self.dosage_median_)).astype("int8")

        de_ritis_ratio = pd.Series(np.nan, index=features.index, dtype="float64")
        valid_alt = alt > 0
        de_ritis_ratio.loc[valid_alt] = ast.loc[valid_alt] / alt.loc[valid_alt]
        if de_ritis_ratio.dropna().empty:
            de_ritis_ratio = pd.Series(self.ratio_median_, index=features.index, dtype="float64")
        else:
            de_ritis_ratio = de_ritis_ratio.fillna(self.ratio_median_)
        de_ritis_ratio = de_ritis_ratio.clip(lower=self.ratio_lower_limit_, upper=self.ratio_upper_limit_)
        features["de_ritis_ratio"] = de_ritis_ratio.astype("float64")

        return features[FINAL_FEATURE_COLUMNS]

    def fit(self, X, y=None):
        X = self._validate_and_order_base_predictors(X)
        self.base_columns_ = list(EXPECTED_BASE_PREDICTOR_COLUMNS)
        self.predictor_columns_ = list(EXPECTED_BASE_PREDICTOR_COLUMNS)
        self.numeric_predictor_columns_ = list(BASE_NUMERIC_COLUMNS)
        self.categorical_predictor_columns_ = list(BASE_CATEGORICAL_COLUMNS)

        normalized_numeric, normalized_categorical = self._normalize_base_inputs(X)
        self._learn_base_statistics(normalized_numeric, normalized_categorical)

        training_features = self.transform_features(X)
        if training_features.shape[1] != 33:
            raise SystemExit(f"Scenario 3 expected 33 engineered predictors, found {training_features.shape[1]}.")

        self.feature_columns_ = list(FINAL_FEATURE_COLUMNS)
        self.feature_numeric_columns_ = list(FINAL_NUMERIC_COLUMNS)
        self.feature_categorical_columns_ = list(FINAL_CATEGORICAL_COLUMNS)

        self.ohe_ = _make_one_hot_encoder()
        self.ohe_.fit(training_features[self.feature_categorical_columns_])
        self.encoded_feature_names_ = list(self.feature_numeric_columns_) + list(
            self.ohe_.get_feature_names_out(self.feature_categorical_columns_)
        )
        self.encoded_feature_count_ = len(self.encoded_feature_names_)
        return self

    def transform_features(self, X):
        if self.numeric_medians_ is None or self.categorical_modes_ is None or self.iqr_limits_ is None:
            raise SystemExit("Scenario3ValidationPreprocessor must be fitted before transform_features().")
        X = self._validate_and_order_base_predictors(X)
        normalized_numeric, normalized_categorical = self._normalize_base_inputs(X)
        imputed_numeric = self._impute_numeric_from_stats(normalized_numeric)
        imputed_categorical = self._impute_categorical_from_stats(normalized_categorical)
        clipped_numeric = self._apply_iqr_clipping(imputed_numeric)
        return self._engineer_features(clipped_numeric, imputed_categorical)

    def transform(self, X):
        if self.ohe_ is None:
            raise SystemExit("Scenario3ValidationPreprocessor must be fitted before transform().")
        features = self.transform_features(X)
        numeric_block = features[self.feature_numeric_columns_].astype("float64")
        categorical_block = features[self.feature_categorical_columns_]

        numeric_sparse = sp.csr_matrix(numeric_block.to_numpy(dtype="float64"))
        categorical_sparse = self.ohe_.transform(categorical_block)
        return sp.hstack([numeric_sparse, categorical_sparse], format="csr")

    def get_feature_names_out(self, input_features=None):
        if self.encoded_feature_names_ is None:
            raise SystemExit("Scenario3ValidationPreprocessor must be fitted before get_feature_names_out().")
        return np.asarray(self.encoded_feature_names_, dtype=object)


def build_scenario3_preprocessor():
    return Scenario3ValidationPreprocessor()
