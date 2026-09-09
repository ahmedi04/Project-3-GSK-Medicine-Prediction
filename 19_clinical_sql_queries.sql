-- Step 11 Clinical SQL Analytics (Project 3)
-- Authoritative query source for both MySQL Workbench and Python runner.

USE gsk_medicine_db;

-- QUERY_ID: Q1
-- QUERY_TITLE: Treatment outcome distribution
SELECT
    CASE
        WHEN treatment_outcome = 0 THEN 'Ineffective (0)'
        WHEN treatment_outcome = 1 THEN 'Effective (1)'
        ELSE 'Unknown'
    END AS treatment_outcome_group,
    COUNT(*) AS total_records,
    SUM(CASE WHEN treatment_outcome IS NOT NULL THEN 1 ELSE 0 END) AS known_outcome_count,
    SUM(CASE WHEN treatment_outcome IS NULL THEN 1 ELSE 0 END) AS missing_outcome_count,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM patients), 2) AS pct_of_all_records
FROM patients
GROUP BY treatment_outcome_group
ORDER BY
    CASE treatment_outcome_group
        WHEN 'Ineffective (0)' THEN 1
        WHEN 'Effective (1)' THEN 2
        ELSE 3
    END,
    treatment_outcome_group;

-- QUERY_ID: Q2
-- QUERY_TITLE: Treatment effectiveness by age group
SELECT
    CASE
        WHEN age IS NULL THEN 'Unknown'
        WHEN age < 30 THEN 'Under 30'
        WHEN age <= 50 THEN '30-50'
        WHEN age <= 65 THEN '>50-65'
        ELSE 'Over 65'
    END AS age_group,
    COUNT(*) AS total_records,
    SUM(CASE WHEN treatment_outcome IS NOT NULL THEN 1 ELSE 0 END) AS known_outcome_count,
    SUM(CASE WHEN treatment_outcome IS NULL THEN 1 ELSE 0 END) AS missing_outcome_count,
    CASE
        WHEN SUM(CASE WHEN treatment_outcome IS NOT NULL THEN 1 ELSE 0 END) = 0 THEN NULL
        ELSE ROUND(
            100.0 * SUM(CASE WHEN treatment_outcome = 1 THEN 1 ELSE 0 END)
            / SUM(CASE WHEN treatment_outcome IS NOT NULL THEN 1 ELSE 0 END),
            2
        )
    END AS observed_effective_rate_pct
FROM patients
GROUP BY age_group
ORDER BY
    CASE age_group
        WHEN 'Under 30' THEN 1
        WHEN '30-50' THEN 2
        WHEN '>50-65' THEN 3
        WHEN 'Over 65' THEN 4
        ELSE 5
    END,
    age_group;

-- QUERY_ID: Q3
-- QUERY_TITLE: Top 10 drugs by observed treatment effectiveness (min 100 records)
SELECT
    drug_group,
    total_records,
    known_outcome_count,
    missing_outcome_count,
    observed_effective_count,
    observed_effective_rate_pct
FROM (
    SELECT
        COALESCE(NULLIF(TRIM(drug_name), ''), 'Unknown Drug') AS drug_group,
        COUNT(*) AS total_records,
        SUM(CASE WHEN treatment_outcome IS NOT NULL THEN 1 ELSE 0 END) AS known_outcome_count,
        SUM(CASE WHEN treatment_outcome IS NULL THEN 1 ELSE 0 END) AS missing_outcome_count,
        SUM(CASE WHEN treatment_outcome = 1 THEN 1 ELSE 0 END) AS observed_effective_count,
        CASE
            WHEN SUM(CASE WHEN treatment_outcome IS NOT NULL THEN 1 ELSE 0 END) = 0 THEN NULL
            ELSE ROUND(
                100.0 * SUM(CASE WHEN treatment_outcome = 1 THEN 1 ELSE 0 END)
                / SUM(CASE WHEN treatment_outcome IS NOT NULL THEN 1 ELSE 0 END),
                2
            )
        END AS observed_effective_rate_pct
    FROM patients
    GROUP BY COALESCE(NULLIF(TRIM(drug_name), ''), 'Unknown Drug')
) ranked
WHERE total_records >= 100
  AND drug_group <> 'Unknown Drug'
ORDER BY
    observed_effective_rate_pct DESC,
    total_records DESC,
    drug_group ASC
LIMIT 10;

-- QUERY_ID: Q4
-- QUERY_TITLE: Adverse-event rates by drug and age category (top 15)
SELECT
    COALESCE(NULLIF(TRIM(drug_name), ''), 'Unknown Drug') AS drug_group,
    CASE
        WHEN age IS NULL THEN 'Unknown'
        WHEN age > 65 THEN 'Elderly'
        ELSE 'Non-Elderly'
    END AS age_category,
    COUNT(*) AS total_records,
    SUM(CASE WHEN adverse_event IS NOT NULL THEN 1 ELSE 0 END) AS known_adverse_count,
    SUM(CASE WHEN adverse_event IS NULL THEN 1 ELSE 0 END) AS missing_adverse_count,
    SUM(CASE WHEN adverse_event = 1 THEN 1 ELSE 0 END) AS observed_adverse_count,
    CASE
        WHEN SUM(CASE WHEN adverse_event IS NOT NULL THEN 1 ELSE 0 END) = 0 THEN NULL
        ELSE ROUND(
            100.0 * SUM(CASE WHEN adverse_event = 1 THEN 1 ELSE 0 END)
            / SUM(CASE WHEN adverse_event IS NOT NULL THEN 1 ELSE 0 END),
            2
        )
    END AS observed_adverse_rate_pct
FROM patients
GROUP BY
    COALESCE(NULLIF(TRIM(drug_name), ''), 'Unknown Drug'),
    CASE
        WHEN age IS NULL THEN 'Unknown'
        WHEN age > 65 THEN 'Elderly'
        ELSE 'Non-Elderly'
    END
ORDER BY
    observed_adverse_rate_pct DESC,
    total_records DESC,
    drug_group ASC,
    age_category ASC
LIMIT 15;

-- QUERY_ID: Q5
-- QUERY_TITLE: High-risk patients (age > 65, creatinine > 1.5, concurrent_drugs >= 5)
SELECT
    patient_id,
    age,
    bmi,
    creatinine,
    concurrent_drugs,
    treatment_outcome,
    adverse_event
FROM patients
WHERE age > 65
  AND creatinine > 1.5
  AND concurrent_drugs >= 5
ORDER BY
    creatinine DESC,
    patient_id ASC
LIMIT 20;

-- QUERY_ID: Q6
-- QUERY_TITLE: Treatment and adverse-event rates by dosage category
SELECT
    CASE
        WHEN dosage_mg IS NULL THEN 'Unknown'
        WHEN dosage_mg < 100 THEN 'Low (<100mg)'
        WHEN dosage_mg <= 500 THEN 'Medium (100-500mg)'
        ELSE 'High (>500mg)'
    END AS dosage_category,
    COUNT(*) AS total_records,
    SUM(CASE WHEN treatment_outcome IS NOT NULL THEN 1 ELSE 0 END) AS known_outcome_count,
    SUM(CASE WHEN treatment_outcome IS NULL THEN 1 ELSE 0 END) AS missing_outcome_count,
    SUM(CASE WHEN adverse_event IS NOT NULL THEN 1 ELSE 0 END) AS known_adverse_count,
    SUM(CASE WHEN adverse_event IS NULL THEN 1 ELSE 0 END) AS missing_adverse_count,
    CASE
        WHEN SUM(CASE WHEN treatment_outcome IS NOT NULL THEN 1 ELSE 0 END) = 0 THEN NULL
        ELSE ROUND(
            100.0 * SUM(CASE WHEN treatment_outcome = 1 THEN 1 ELSE 0 END)
            / SUM(CASE WHEN treatment_outcome IS NOT NULL THEN 1 ELSE 0 END),
            2
        )
    END AS observed_effective_rate_pct,
    CASE
        WHEN SUM(CASE WHEN adverse_event IS NOT NULL THEN 1 ELSE 0 END) = 0 THEN NULL
        ELSE ROUND(
            100.0 * SUM(CASE WHEN adverse_event = 1 THEN 1 ELSE 0 END)
            / SUM(CASE WHEN adverse_event IS NOT NULL THEN 1 ELSE 0 END),
            2
        )
    END AS observed_adverse_rate_pct
FROM patients
GROUP BY dosage_category
ORDER BY
    CASE dosage_category
        WHEN 'Low (<100mg)' THEN 1
        WHEN 'Medium (100-500mg)' THEN 2
        WHEN 'High (>500mg)' THEN 3
        ELSE 4
    END,
    dosage_category;
