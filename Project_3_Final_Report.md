# Project 3 Final Report

## 1) Objective
Build an educational end-to-end system to predict treatment outcome (`treatment_outcome`) from prepared clinical data, then operationalize it with SQL analytics and a local API.

## 2) Work completed
Completed components:
- EDA, cleaning, and feature engineering.
- Model training/tuning and Scenario 3 comparison.
- XGBoost explainability using SHAP and LIME.
- MySQL connection, full data load, and SQL analytics.
- Flask API for local prediction requests and result logging.

This project follows PDF guidance, adapted to this repository's Scenario 3 assets and MySQL environment.

## 3) Dataset and Scenario 3 setup
- Prepared patient records: **1,050,000**.
- Labelled records for supervised modeling: **985,876**.
- Scenario 3 preprocessing is learned from training split only.
- Split design: **60/20/20** (train/validation/test).
- Base predictors used for modeling: **26**.
- Predictors after feature engineering: **33**.
- Encoded model inputs: **121**.

Earlier tuning explored multiple settings and cutoffs. Those tuning outputs are retained as separate experiments and are not merged into the final fixed Scenario 3 comparison.

## 4) Final model comparison (saved report)
Model selection was based on validation ROC-AUC among Decision Tree, Random Forest, XGBoost, and Neural Network. LSTM is reported as an educational demo.

**Selected model for API use:** Neural Network (validation ROC-AUC selection rule).

**Important metric highlights:**
- Highest **test accuracy** among the five models: **XGBoost = 72.39%**.
- Neural Network **test ROC-AUC**: **0.6496**.

### Final test-results table
| Model | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Decision Tree | 57.95% | 0.2997 | 0.6603 | 0.4123 | 0.6459 |
| Random Forest | 58.02% | 0.2990 | 0.6541 | 0.4104 | 0.6449 |
| XGBoost | **72.39%** | 0.3637 | 0.3154 | 0.3379 | 0.6488 |
| Neural Network | 59.68% | 0.3048 | 0.6285 | 0.4105 | **0.6496** |
| LSTM Demo | 59.27% | 0.3037 | 0.6373 | 0.4114 | 0.6500 |

Note: LSTM remains educational because each patient contributes one timestep only.

## 5) Explainability summary (XGBoost)
- SHAP and LIME were implemented for the saved Scenario 3 XGBoost pipeline.
- SHAP provided stable global and local contribution views.
- LIME local fit quality was weak for the sampled case, so LIME outputs are interpreted cautiously.
- These explainability outputs describe model behavior; they do not prove causality.

## 6) MySQL loading and SQL analytics
- All **1,050,000** prepared patients were loaded into `gsk_medicine_db.patients`.
- Source-to-database verification matched row counts, null counts, and binary outcome distributions.
- Six SQL analytics queries were run from one authoritative SQL source in Python and Workbench.

Selected findings from saved SQL outputs:
1. Outcome distribution: **72.92% Ineffective**, **20.97% Effective**, **6.11% Unknown outcome**.
2. Observed effective rate decreases across age groups in this dataset: about **37.51% (Under 30)** vs **13.67% (Over 65)**.
3. Dosage-category effective rates are close (about **22.29% to 22.43%**) across low/medium/high groups.

These are observational SQL rates, not causal effects.

## 7) Flask API integration
- API reads patient predictors directly from MySQL.
- Saved Scenario 3 Neural Network artifacts are reused without retraining.
- Each successful prediction is written to `prediction_results` and verified in Workbench joins.

Verified local example:
- `patient_id`: **26319**
- Effective score (`probability_effective`): **46.93%**
- Cutoff: **50%**
- Predicted class/outcome: **0 / Ineffective**
- Saved `prediction_id`: **1**

## 8) Clarifications and limitations
- **Model accuracy** is a dataset-level test metric.
- An individual **prediction score** (for one patient) is not model accuracy.
- SQL **observed outcome rates** are descriptive aggregates and not equivalent to model performance.

Limitations:
- The 50% precision target was not met in final comparison.
- LSTM uses one timestep and is educational only.
- The same held-out test split was used in earlier scenarios.
- Deployment is local and educational, not clinical-ready.

## 9) Conclusion
Project 3 now provides a complete educational workflow: prepared data, compared models, explained XGBoost behavior, verified MySQL analytics, and exposed a local Flask prediction API that stores prediction events. The implementation demonstrates reproducible engineering and reporting while avoiding clinical-readiness claims.
