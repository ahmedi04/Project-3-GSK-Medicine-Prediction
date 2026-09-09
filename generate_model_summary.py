#!/usr/bin/env python3
"""
Generate a compact model-summary page and comparison charts.

This script reads testing metrics and confusion matrices from existing
report TXT files (Steps 4-6) and produces two PNG charts and a one-page
PDF summary. It performs validations and prints progress messages.

Do not modify existing reports or models. This script is safe to run
multiple times; it overwrites its own outputs.
"""
from pathlib import Path
import re
import sys
import json
from typing import Dict, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle


# Beginner-friendly helper functions
def read_report(path: Path) -> str:
    """Return the text of a report file."""
    return path.read_text(encoding='utf-8')


def parse_testing_metrics(report_text: str) -> Dict[str, float]:
    """Parse the 'Testing metrics' block and return a dict of metrics.

    Expected lines like:
      - accuracy: 0.5768
      - precision: 0.2989
    """
    metrics = {}
    # find 'Testing metrics:' and then gather subsequent lines until a blank line
    m = re.search(r"Testing metrics:\s*\n(.*?)\n\s*\n", report_text + "\n\n", re.S)
    if not m:
        # fallback: find 'Testing metrics:' and read following lines
        m2 = re.search(r"Testing metrics:\s*\n(.*)", report_text, re.S)
        block = m2.group(1) if m2 else ''
    else:
        block = m.group(1)

    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        mm = re.match(r"^-?\s*([a-zA-Z_]+):\s*([0-9.+eE-]+)", line)
        if mm:
            key = mm.group(1)
            try:
                val = float(mm.group(2))
            except Exception:
                continue
            metrics[key] = val
    return metrics


def parse_confusion_matrix(report_text: str) -> np.ndarray:
    """Find a 2x2 confusion matrix printed like [[a b]\n [c d]] and return as numpy array."""
    txt = report_text.replace(',', ' ')
    # look for [[...][...]] pattern
    m = re.search(r"\[\s*\[([^\]]+)\]\s*\[([^\]]+)\]\s*\]", txt, re.S)
    if m:
        part1 = m.group(1)
        part2 = m.group(2)
        nums = re.findall(r"(-?\d+)", part1 + ' ' + part2)
    else:
        # fallback: find the first [[...]]
        m2 = re.search(r"\[\[.*?\]\]", txt, re.S)
        if not m2:
            raise ValueError('Confusion matrix not found in report')
        nums = re.findall(r"(-?\d+)", m2.group(0))

    if len(nums) < 4:
        raise ValueError('Could not parse 4 integers from confusion matrix')
    a, b, c, d = map(int, nums[:4])
    return np.array([[a, b], [c, d]])


def validate_reports_exist(paths):
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        print('Missing report files:', missing)
        raise SystemExit('Required report files are missing')


def make_accuracy_bar_chart(results: Dict[str, float], out_path: Path):
    """Create a vertical bar chart of accuracies (percent)."""
    labels = list(results.keys())
    vals = [results[k] * 100.0 for k in labels]
    colors_list = ['#4C72B0', '#55A868', '#C44E52', '#DD8452']

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, vals, color=colors_list[:len(labels)])
    plt.ylim(0, 100)
    plt.ylabel('Test Accuracy (%)')
    plt.title('Model Accuracy Comparison - Steps 4 to 6')

    for bar, v in zip(bars, vals):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 1.0, f"{v:.2f}%", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def make_xgboost_pie(matrix: np.ndarray, out_path: Path):
    """Create pie chart for XGBoost correct vs incorrect predictions."""
    tn, fp = int(matrix[0, 0]), int(matrix[0, 1])
    fn, tp = int(matrix[1, 0]), int(matrix[1, 1])
    correct = tn + tp
    incorrect = fp + fn
    sizes = [correct, incorrect]
    labels = ['Correct', 'Incorrect']
    colors = ['#2ca02c', '#d62728']

    plt.figure(figsize=(4, 4))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%.2f%%', startangle=90)
    plt.title('XGBoost Test Prediction Results')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def build_pdf(summary: Dict[str, Dict[str, float]], xg_matrix: np.ndarray, acc_chart: Path, pie_chart: Path, out_pdf: Path):
    """Create a one-page PDF summary using ReportLab."""
    doc = SimpleDocTemplate(str(out_pdf), pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    style_h = ParagraphStyle('Heading', parent=styles['Heading1'], alignment=1, textColor=colors.HexColor('#0B3D91'))
    style_sub = ParagraphStyle('Subtitle', parent=styles['Normal'], alignment=1, textColor=colors.HexColor('#333333'))
    normal = styles['Normal']

    elements = []
    elements.append(Paragraph('Steps 4-6: Model Training & Evaluation', style_h))
    elements.append(Paragraph('GSK Medicine Prediction System | Day 2 Summary', style_sub))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph('<b>Work Completed</b>', styles['Heading3']))
    work_lines = [
        '985,876 labelled patient records were used',
        '64,124 missing-target records were excluded only for supervised model training',
        'The same stratified 80% training and 20% testing split with random_state=42 was used for all models',
        'Decision Tree, Random Forest, XGBoost and Gradient Boosting were trained',
        'Numeric missing values were filled using training medians',
        'Categorical missing values were filled using training modes and then one-hot encoded',
        'Preprocessing was fitted only on training data',
        'patient_id, treatment_outcome, admission_date, adverse_event and readmission_30d were excluded from predictors'
    ]
    for ln in work_lines:
        elements.append(Paragraph(f' - {ln}', normal))
    elements.append(Spacer(1, 8))

    elements.append(Paragraph('<b>Model Results (Test)</b>', styles['Heading3']))
    data = [['Model', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC']]
    for model in ['Decision Tree', 'Random Forest', 'Gradient Boosting', 'XGBoost']:
        metrics = summary.get(model, {})
        data.append([
            model,
            f"{metrics.get('accuracy', 0.0)*100:.2f}%",
            f"{metrics.get('precision', 0.0)*100:.2f}%",
            f"{metrics.get('recall', 0.0)*100:.2f}%",
            f"{metrics.get('f1_score', 0.0)*100:.2f}%",
            f"{metrics.get('roc_auc', 0.0)*100:.2f}%",
        ])

    tbl = Table(data, colWidths=[100, 70, 70, 70, 70, 70])
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F2F2F2')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
    ]))
    elements.append(tbl)
    elements.append(Spacer(1, 8))

    elements.append(Paragraph('<b>Charts</b>', styles['Heading3']))
    img1 = Image(str(acc_chart), width=260, height=180)
    img2 = Image(str(pie_chart), width=180, height=180)
    t = Table([[img1, img2]], colWidths=[280, 200])
    t.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))
    elements.append(t)
    elements.append(Spacer(1, 8))

    elements.append(Paragraph('<b>Key Findings</b>', styles['Heading3']))
    kf = [
        'XGBoost achieved the highest accuracy and precision',
        'Decision Tree achieved the highest recall',
        'Gradient Boosting achieved the highest ROC-AUC',
        'Random Forest did not significantly improve over Decision Tree with the supplied settings',
        'Training and testing scores were close, indicating no serious overfitting',
        'Accuracy should not be considered alone because the target is imbalanced'
    ]
    for ln in kf:
        elements.append(Paragraph(f' - {ln}', normal))
    elements.append(Spacer(1, 8))

    elements.append(Paragraph('<b>Conclusion</b>', styles['Heading3']))
    conclusion = 'Steps 4, 5 and 6 were completed successfully. XGBoost is the accuracy-leading model, while Decision Tree is strongest for identifying a larger percentage of effective-treatment cases.'
    elements.append(Paragraph(conclusion, normal))
    elements.append(Spacer(1, 12))

    footer = Paragraph('GSK Medicine Prediction System - Steps 4 to 6', ParagraphStyle('footer', alignment=1, fontSize=8, textColor=colors.grey))
    elements.append(Spacer(1, 12))
    elements.append(footer)

    doc.build(elements)


def main():
    r4 = Path('outputs/step4_decision_tree_report.txt')
    r5 = Path('outputs/step5_random_forest_report.txt')
    r6 = Path('outputs/step6_xgboost_report.txt')
    r6b = Path('outputs/step6b_gradient_boosting_report.txt')

    validate_reports_exist([r4, r5, r6, r6b])

    print('Reading reports...')
    t4 = read_report(r4)
    t5 = read_report(r5)
    t6 = read_report(r6)
    t6b = read_report(r6b)

    m4 = parse_testing_metrics(t4)
    m5 = parse_testing_metrics(t5)
    m6 = parse_testing_metrics(t6)
    m6b = parse_testing_metrics(t6b)

    summary = {
        'Decision Tree': m4,
        'Random Forest': m5,
        'Gradient Boosting': m6b,
        'XGBoost': m6,
    }

    acc_chart = Path('outputs/steps4_to_6_accuracy_bar_chart.png')
    pie_chart = Path('outputs/steps4_to_6_xgboost_results_pie_chart.png')

    xg_cm = parse_confusion_matrix(t6)
    make_accuracy_bar_chart({k: summary[k].get('accuracy', 0.0) for k in ['Decision Tree', 'Random Forest', 'Gradient Boosting', 'XGBoost']}, acc_chart)
    make_xgboost_pie(xg_cm, pie_chart)

    pdf_path = Path('Day_2_Model_Summary.pdf')
    build_pdf(summary, xg_cm, acc_chart, pie_chart, pdf_path)

    print('Verifying outputs...')
    ok_images = acc_chart.exists() and pie_chart.exists()
    page_count = None
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(str(pdf_path))
        page_count = len(reader.pages)
    except Exception:
        page_count = None

    if not ok_images:
        raise SystemExit('One or more chart PNGs were not created')
    if page_count is None:
        print('Warning: PyPDF2 not available to verify PDF page count; generated PDF at', pdf_path)
    elif page_count != 1:
        raise SystemExit(f'PDF page count is {page_count}, expected 1')

    print('Generated:', acc_chart, pie_chart, pdf_path)
    print('PDF page count verified as 1' if page_count == 1 else 'PDF generated (page count not verified)')


if __name__ == '__main__':
    main()
