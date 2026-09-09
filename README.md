# Project 3: GSK Medicine Treatment-Outcome Prediction

## Purpose
This repository delivers an educational end-to-end workflow to predict treatment outcome (0 or 1) from prepared clinical patient data.

It includes EDA, cleaning, feature engineering, model training and tuning, model comparison, explainability (SHAP and LIME), MySQL analytics, and a Flask prediction API.

## Workflow overview
1. Convert raw CSV to parquet.
2. Run EDA and data cleaning.
3. Engineer clinical features.
4. Train and tune models.
5. Run Scenario 3 comparison and explainability.
6. Load data to MySQL and run SQL analytics.
7. Serve predictions through a local Flask API.

## Dependencies and setup
- Python 3.9+
- Install packages from [requirements.txt](requirements.txt)

### Original dataset location (excluded from GitHub)
The original CSV expected by this project should be placed at:
- outputs/clinical_data_raw.csv

This source file is excluded from GitHub and must be provided locally before running the workflow.

### Configure .env
1. Copy [.env.example](.env.example) to `.env`.
2. Fill local values.
3. Use `MYSQL_DATABASE=gsk_medicine_db` for Steps 17+.

Required keys:
- `MYSQL_HOST`
- `MYSQL_PORT`
- `MYSQL_USER`
- `MYSQL_PASSWORD`
- `MYSQL_DATABASE`

## Script execution order
1. [00_convert_to_parquet.py](00_convert_to_parquet.py)
2. [01_eda.py](01_eda.py)
3. [02_data_cleaning.py](02_data_cleaning.py)
4. [03_feature_engineering.py](03_feature_engineering.py)
5. [04_decision_tree.py](04_decision_tree.py)
6. [05_random_forest.py](05_random_forest.py)
7. [06_xgboost.py](06_xgboost.py)
8. [06b_gradient_boosting.py](06b_gradient_boosting.py)
9. [07_data_audit.py](07_data_audit.py)
10. [08_feature_selection_analysis.py](08_feature_selection_analysis.py)
11. [09_train_validation_test_baseline.py](09_train_validation_test_baseline.py)
12. [10_xgboost_validation_tuning.py](10_xgboost_validation_tuning.py)
13. [11_prepare_validation_input.py](11_prepare_validation_input.py)
14. [12_scenario3_model_baseline.py](12_scenario3_model_baseline.py)
15. [13_keras_neural_network.py](13_keras_neural_network.py)
16. [14_lstm_demonstration.py](14_lstm_demonstration.py)
17. [15_model_evaluation_comparison.py](15_model_evaluation_comparison.py)
18. [16_xgboost_explainability.py](16_xgboost_explainability.py)
19. [17_mysql_connection_check.py](17_mysql_connection_check.py)
20. [18_load_patients_mysql.py](18_load_patients_mysql.py)
21. [19_clinical_sql_analytics.py](19_clinical_sql_analytics.py)
22. [app.py](app.py)

### Exact runnable commands
Run from project root:

- `./gsk_env/bin/python 00_convert_to_parquet.py`
- `./gsk_env/bin/python 01_eda.py`
- `./gsk_env/bin/python 02_data_cleaning.py`
- `./gsk_env/bin/python 03_feature_engineering.py`
- `./gsk_env/bin/python 04_decision_tree.py`
- `./gsk_env/bin/python 05_random_forest.py`
- `./gsk_env/bin/python 06_xgboost.py`
- `./gsk_env/bin/python 06b_gradient_boosting.py`
- `./gsk_env/bin/python 07_data_audit.py`
- `./gsk_env/bin/python 08_feature_selection_analysis.py`
- `./gsk_env/bin/python 09_train_validation_test_baseline.py`
- `./gsk_env/bin/python 10_xgboost_validation_tuning.py`
- `./gsk_env/bin/python 11_prepare_validation_input.py`
- `./gsk_env/bin/python 12_scenario3_model_baseline.py`
- `./gsk_env/bin/python 13_keras_neural_network.py --train`
- `./gsk_env/bin/python 14_lstm_demonstration.py --train`
- `./gsk_env/bin/python 15_model_evaluation_comparison.py --train`
- `./gsk_env/bin/python 16_xgboost_explainability.py --run`
- `./gsk_env/bin/python 17_mysql_connection_check.py`
- `./gsk_env/bin/python 18_load_patients_mysql.py --load`
- `./gsk_env/bin/python 19_clinical_sql_analytics.py --run`
- `./gsk_env/bin/python app.py --setup-table`
- `./gsk_env/bin/python app.py`

## MySQL and SQL analytics in Workbench
1. Check connection with [17_mysql_connection_check.py](17_mysql_connection_check.py).
2. Load all records with [18_load_patients_mysql.py](18_load_patients_mysql.py) using `--load`.
3. Confirm parity in [outputs/mysql_patient_load_report.txt](outputs/mysql_patient_load_report.txt).
4. Run six analytics queries in Workbench from [19_clinical_sql_queries.sql](19_clinical_sql_queries.sql).
5. Run the same SQL source in Python with [19_clinical_sql_analytics.py](19_clinical_sql_analytics.py).
6. Review outputs in [outputs/clinical_sql](outputs/clinical_sql).

## Flask API on localhost:5001
Educational treatment-outcome prediction demo using saved Scenario 3 artifacts.

Start:
- Run [app.py](app.py). The app runs on `127.0.0.1:5001` with `debug=False`.

Endpoints:
- `GET /health`
- `POST /predict` with JSON, for example `{"patient_id": 26319}`

Endpoint commands:
- `curl -s http://127.0.0.1:5001/health`
- `curl -s -X POST http://127.0.0.1:5001/predict -H "Content-Type: application/json" -d '{"patient_id":26319}'`

prediction_results table and Workbench checks:
- Setup/query SQL: [20_prediction_results_setup.sql](20_prediction_results_setup.sql)
- Explicit setup command: `python app.py --setup-table`
- Use included SELECT statements to view predictions and joins with `patients` by `patient_id`.

## Regenerate excluded datasets and model files
Large datasets and model binaries are intentionally excluded from GitHub.

To regenerate locally, run scripts [00_convert_to_parquet.py](00_convert_to_parquet.py) through [16_xgboost_explainability.py](16_xgboost_explainability.py), then run [17_mysql_connection_check.py](17_mysql_connection_check.py), [18_load_patients_mysql.py](18_load_patients_mysql.py), [19_clinical_sql_analytics.py](19_clinical_sql_analytics.py), and [20_prediction_results_setup.sql](20_prediction_results_setup.sql).

## Key outputs
- [outputs/scenario3_final_model_comparison_report.txt](outputs/scenario3_final_model_comparison_report.txt)
- [outputs/scenario3_final_model_comparison.csv](outputs/scenario3_final_model_comparison.csv)
- [outputs/scenario3_xgboost_explainability_report.txt](outputs/scenario3_xgboost_explainability_report.txt)
- [outputs/mysql_patient_load_report.txt](outputs/mysql_patient_load_report.txt)
- [outputs/clinical_sql/clinical_sql_report.txt](outputs/clinical_sql/clinical_sql_report.txt)
- [Project_3_Final_Report.md](Project_3_Final_Report.md)
- [Project_3_Final_Report.pdf](Project_3_Final_Report.pdf)

## Important note
This repository is for educational analytics and engineering demonstration, not clinical deployment.
