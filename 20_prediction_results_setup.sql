-- Step 12 prediction-results table setup and checks
-- Run manually in MySQL Workbench OR via: python app.py --setup-table

USE gsk_medicine_db;

CREATE TABLE IF NOT EXISTS prediction_results (
    prediction_id BIGINT NOT NULL AUTO_INCREMENT,
    patient_id BIGINT NOT NULL,
    model_name VARCHAR(128) NOT NULL,
    model_version VARCHAR(64) NOT NULL,
    probability_effective DOUBLE NOT NULL,
    cutoff DOUBLE NOT NULL,
    predicted_class TINYINT NOT NULL,
    predicted_outcome VARCHAR(32) NOT NULL,
    predicted_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (prediction_id),
    KEY idx_prediction_results_patient_id (patient_id),
    KEY idx_prediction_results_predicted_at (predicted_at),
    CONSTRAINT fk_prediction_results_patient
        FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Basic setup checks
SELECT DATABASE() AS active_database;
SHOW TABLES LIKE 'prediction_results';
SHOW COLUMNS FROM prediction_results;
SELECT COUNT(*) AS prediction_count FROM prediction_results;

-- Workbench query 1: latest predictions
SELECT
    prediction_id,
    patient_id,
    model_name,
    model_version,
    probability_effective,
    cutoff,
    predicted_class,
    predicted_outcome,
    predicted_at
FROM prediction_results
ORDER BY predicted_at DESC, prediction_id DESC
LIMIT 100;

-- Workbench query 2: predictions joined with patient context
SELECT
    pr.prediction_id,
    pr.predicted_at,
    pr.patient_id,
    p.age,
    p.gender,
    p.drug_name,
    p.dosage_mg,
    p.concurrent_drugs,
    p.creatinine,
    p.treatment_outcome AS observed_treatment_outcome,
    p.adverse_event AS observed_adverse_event,
    pr.model_name,
    pr.model_version,
    pr.probability_effective,
    pr.cutoff,
    pr.predicted_class,
    pr.predicted_outcome
FROM prediction_results AS pr
INNER JOIN patients AS p
    ON p.patient_id = pr.patient_id
ORDER BY pr.predicted_at DESC, pr.prediction_id DESC
LIMIT 200;

-- Workbench query 3: patient-specific prediction history (set patient_id as needed)
SELECT
    pr.prediction_id,
    pr.predicted_at,
    pr.patient_id,
    pr.model_name,
    pr.model_version,
    pr.probability_effective,
    pr.cutoff,
    pr.predicted_class,
    pr.predicted_outcome,
    p.treatment_outcome AS observed_treatment_outcome,
    p.adverse_event AS observed_adverse_event,
    p.readmission_30d AS observed_readmission_30d
FROM prediction_results AS pr
INNER JOIN patients AS p
    ON p.patient_id = pr.patient_id
WHERE pr.patient_id = 26319
ORDER BY pr.predicted_at DESC, pr.prediction_id DESC;
