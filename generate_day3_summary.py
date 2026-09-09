#!/usr/bin/env python3
"""Generate a one-page Day 3 model summary PDF.

Sources:
- outputs/scenario3_final_model_comparison.csv
- outputs/scenario3_final_model_comparison_report.txt

Output:
- Day_3_Model_Summary.pdf

This script does not retrain models or modify existing results.
"""
from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


CSV_FILE = Path("outputs/scenario3_final_model_comparison.csv")
REPORT_FILE = Path("outputs/scenario3_final_model_comparison_report.txt")
OUTPUT_PDF = Path("Day_3_Model_Summary.pdf")


def read_inputs() -> tuple[pd.DataFrame, str]:
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"Missing required file: {CSV_FILE}")
    if not REPORT_FILE.exists():
        raise FileNotFoundError(f"Missing required file: {REPORT_FILE}")

    df = pd.read_csv(CSV_FILE)
    report_text = REPORT_FILE.read_text(encoding="utf-8")

    required_cols = [
        "model",
        "val_roc_auc",
        "val_f1",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_roc_auc",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    expected_models = ["DecisionTree", "RandomForest", "XGBoost", "NeuralNetwork", "LSTM"]
    model_set = set(df["model"].astype(str))
    missing_models = [m for m in expected_models if m not in model_set]
    if missing_models:
        raise ValueError(f"Missing expected models in CSV: {missing_models}")

    return df, report_text


def parse_report_values(report_text: str) -> dict[str, str]:
    def extract(pattern: str, label: str) -> str:
        m = re.search(pattern, report_text, flags=re.MULTILINE)
        if not m:
            raise ValueError(f"Could not parse {label} from report")
        return m.group(1).strip()

    selected_model = extract(r"^Selected Model:\s*([A-Za-z0-9_]+)", "selected model")
    baseline_accuracy = extract(r"AlwaysZero reference:\s*Accuracy=([0-9.]+)", "always-zero accuracy")
    split_line = extract(r"^- Test:\s*(.+)$", "test split summary")

    return {
        "selected_model": selected_model,
        "baseline_accuracy": baseline_accuracy,
        "test_split_line": split_line,
    }


def compute_summary_stats(df: pd.DataFrame) -> dict[str, object]:
    order = ["DecisionTree", "RandomForest", "XGBoost", "NeuralNetwork", "LSTM"]
    df = df.set_index("model").loc[order].reset_index()

    eligible = df[df["model"].isin(["DecisionTree", "RandomForest", "XGBoost", "NeuralNetwork"])].copy()
    eligible_sorted = eligible.sort_values(["val_roc_auc", "val_f1"], ascending=False).reset_index(drop=True)
    eligible_best = eligible_sorted.iloc[0]
    eligible_second = eligible_sorted.iloc[1]
    eligible_auc_gap = float(eligible_best["val_roc_auc"] - eligible_second["val_roc_auc"])

    xgb_row = df[df["model"] == "XGBoost"].iloc[0]
    test_best_acc_model = df.sort_values("test_accuracy", ascending=False).iloc[0]["model"]
    test_best_prec_model = df.sort_values("test_precision", ascending=False).iloc[0]["model"]

    max_precision = float(df["test_precision"].max())
    precision_target_met = max_precision >= 0.50

    return {
        "ordered_df": df,
        "eligible_best_model": str(eligible_best["model"]),
        "eligible_best_auc": float(eligible_best["val_roc_auc"]),
        "eligible_auc_gap": eligible_auc_gap,
        "xgb_test_accuracy": float(xgb_row["test_accuracy"]),
        "xgb_test_precision": float(xgb_row["test_precision"]),
        "test_best_acc_model": str(test_best_acc_model),
        "test_best_prec_model": str(test_best_prec_model),
        "max_test_precision": max_precision,
        "precision_target_met": precision_target_met,
    }


def make_styles():
    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=21,
            leading=24,
            textColor=colors.HexColor("#17365D"),
            alignment=TA_CENTER,
            spaceAfter=6,
        )
    )

    styles.add(
        ParagraphStyle(
            name="ReportSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=12,
            textColor=colors.HexColor("#3E4A59"),
            alignment=TA_CENTER,
            spaceAfter=10,
        )
    )

    styles.add(
        ParagraphStyle(
            name="SectionHeading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=14,
            textColor=colors.HexColor("#17365D"),
            spaceBefore=5,
            spaceAfter=4,
        )
    )

    styles.add(
        ParagraphStyle(
            name="BulletText",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=11,
            textColor=colors.HexColor("#263238"),
            leftIndent=10,
            firstLineIndent=-7,
            spaceAfter=2,
        )
    )

    styles.add(
        ParagraphStyle(
            name="SmallNote",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8.2,
            leading=10,
            textColor=colors.HexColor("#556270"),
            alignment=TA_CENTER,
            spaceBefore=6,
        )
    )

    return styles


def format_pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def format_dec(v: float) -> str:
    return f"{v:.4f}"


def display_model_name(raw_name: str) -> str:
    mapping = {
        "DecisionTree": "Decision Tree",
        "RandomForest": "Random Forest",
        "NeuralNetwork": "Neural Network",
        "LSTM": "LSTM Demo",
    }
    return mapping.get(raw_name, raw_name)


def build_pdf(df: pd.DataFrame, parsed: dict[str, str], stats: dict[str, object]) -> None:
    styles = make_styles()

    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=letter,
        rightMargin=0.60 * inch,
        leftMargin=0.60 * inch,
        topMargin=0.52 * inch,
        bottomMargin=0.52 * inch,
        title="Day 3 Model Summary",
        author="Project 3 Team",
    )

    story = []
    story.append(Paragraph("Day 3: Scenario 3 Model Comparison", styles["ReportTitle"]))
    story.append(Paragraph("GSK Medicine Prediction System | One-Page Summary", styles["ReportSubtitle"]))

    story.append(Paragraph("What was completed", styles["SectionHeading"]))
    bullets = [
        "Readmission conversion fix was retained, and preprocessing remained training-only to avoid leakage.",
        "A fixed 60/20/20 split (train/validation/test) was used for comparison across five saved models.",
        "No retraining, no retuning, and no split changes were performed in this summary step.",
        "LSTM is included as a one-timestep educational demonstration (one row per patient).",
    ]
    for line in bullets:
        story.append(Paragraph(f"- {line}", styles["BulletText"]))

    story.append(Spacer(1, 4))
    story.append(Paragraph("Test results: Treatment outcome prediction", styles["SectionHeading"]))
    story.append(Paragraph("Precision, recall and F1 refer to Effective (1).", styles["BulletText"]))

    table_rows = [["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]]
    for _, row in df.iterrows():
        table_rows.append([
            display_model_name(str(row["model"])),
            format_pct(float(row["test_accuracy"])),
            format_pct(float(row["test_precision"])),
            format_pct(float(row["test_recall"])),
            format_dec(float(row["test_f1"])),
            format_dec(float(row["test_roc_auc"])),
        ])

    table = Table(table_rows, colWidths=[110, 73, 73, 73, 62, 62])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF3F7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#17365D")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.8),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#B5C3D1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FAFCFE")]),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(table)

    story.append(Spacer(1, 5))
    story.append(Paragraph("Key findings", styles["SectionHeading"]))

    selected_from_report = parsed["selected_model"]
    eligible_best = stats["eligible_best_model"]
    gap = stats["eligible_auc_gap"]

    findings = [
        f"Neural Network was selected using validation ROC-AUC among the four eligible models; the top eligible ROC-AUC gap is small ({gap:.4f}).",
        f"XGBoost has the highest test accuracy ({format_pct(stats['xgb_test_accuracy'])}) and precision ({format_pct(stats['xgb_test_precision'])}).",
        f"The 50% precision target remains unmet (best test precision: {format_pct(stats['max_test_precision'])}).",
        f"Always-zero baseline accuracy is {format_pct(float(parsed['baseline_accuracy']))} with zero recall.",
        "The test split was also evaluated in earlier scenarios.",
    ]

    if selected_from_report != eligible_best:
        findings.insert(
            0,
            f"Saved report selection is {selected_from_report}; CSV tie-break computation indicates {eligible_best}. Review only if this mismatch was unexpected.",
        )

    for line in findings:
        story.append(Paragraph(f"- {line}", styles["BulletText"]))

    story.append(
        Paragraph(
            "This page is generated from outputs/scenario3_final_model_comparison.csv and outputs/scenario3_final_model_comparison_report.txt.",
            styles["SmallNote"],
        )
    )

    doc.build(story)


def verify_one_page(pdf_path: Path) -> int:
    try:
        from PyPDF2 import PdfReader  # type: ignore

        pages = len(PdfReader(str(pdf_path)).pages)
        return pages
    except Exception:
        try:
            from pypdf import PdfReader  # type: ignore

            pages = len(PdfReader(str(pdf_path)).pages)
            return pages
        except Exception as exc:
            raise RuntimeError("Could not verify page count. Install PyPDF2 or pypdf.") from exc


def main() -> None:
    print("Progress: Reading Day 3 comparison inputs...")
    df, report_text = read_inputs()
    parsed = parse_report_values(report_text)
    stats = compute_summary_stats(df)

    print("Progress: Building Day_3_Model_Summary.pdf...")
    build_pdf(stats["ordered_df"], parsed, stats)

    if not OUTPUT_PDF.exists():
        raise RuntimeError(f"Failed to create {OUTPUT_PDF}")

    page_count = verify_one_page(OUTPUT_PDF)
    if page_count != 1:
        raise RuntimeError(f"{OUTPUT_PDF} must be exactly 1 page; found {page_count} pages.")

    print(f"Done: {OUTPUT_PDF} created with exactly {page_count} page.")


if __name__ == "__main__":
    main()
