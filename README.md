# GSK Medicine Outcome Prediction System

## Project Overview

This project prepares a large clinical medicine dataset for a machine-learning system that will predict treatment outcomes.

The original dataset contains:

- 1,155,000 rows
- 41 columns

The first three project stages completed are:

1. Exploratory Data Analysis (EDA)
2. Data Cleaning and Preprocessing
3. Clinical Feature Engineering

Parquet format was used because it is faster and more memory-efficient than CSV for this large dataset.

## Project Workflow

```text
Raw CSV
   ↓
Parquet Conversion
   ↓
Exploratory Data Analysis
   ↓
Data Cleaning
   ↓
Feature Engineering