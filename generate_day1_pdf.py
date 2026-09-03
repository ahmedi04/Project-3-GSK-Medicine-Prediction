#!/usr/bin/env python3
"""
Generate the Day 1 project summary PDF.

Creates:
    Day_1_Data.pdf
"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
)


# ---------------------------------------------------------
# File location
# ---------------------------------------------------------
OUTPUT_FILE = Path("Day_1_Data.pdf")


# ---------------------------------------------------------
# Colours used in the report
# ---------------------------------------------------------
NAVY = colors.HexColor("#17365D")
TEAL = colors.HexColor("#167D8D")
GREEN = colors.HexColor("#2E7D4F")
GOLD = colors.HexColor("#C69214")
LIGHT_BLUE = colors.HexColor("#EAF3F7")
LIGHT_GREEN = colors.HexColor("#EAF5EE")
LIGHT_GOLD = colors.HexColor("#FFF6DD")
LIGHT_GREY = colors.HexColor("#F3F5F7")
DARK_GREY = colors.HexColor("#3E4A59")


def add_page_number(canvas, document):
    """Add a footer and page number to every page."""
    canvas.saveState()

    canvas.setStrokeColor(colors.HexColor("#D5DCE3"))
    canvas.line(
        document.leftMargin,
        0.55 * inch,
        letter[0] - document.rightMargin,
        0.55 * inch,
    )

    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#68737D"))

    canvas.drawString(
        document.leftMargin,
        0.35 * inch,
        "GSK Medicine Prediction System - Day 1",
    )

    canvas.drawRightString(
        letter[0] - document.rightMargin,
        0.35 * inch,
        f"Page {document.page}",
    )

    canvas.restoreState()


def make_styles():
    """Create all text styles used in the PDF."""
    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=29,
            textColor=NAVY,
            alignment=TA_CENTER,
            spaceAfter=10,
        )
    )

    styles.add(
        ParagraphStyle(
            name="ReportSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=11,
            leading=15,
            textColor=DARK_GREY,
            alignment=TA_CENTER,
            spaceAfter=16,
        )
    )

    styles.add(
        ParagraphStyle(
            name="SectionHeading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=19,
            textColor=NAVY,
            spaceBefore=8,
            spaceAfter=8,
        )
    )

    styles.add(
        ParagraphStyle(
            name="WhiteHeading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=colors.white,
            spaceAfter=0,
        )
    )

    styles.add(
        ParagraphStyle(
            name="BodyTextCustom",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            textColor=colors.HexColor("#263238"),
            spaceAfter=7,
        )
    )

    styles.add(
        ParagraphStyle(
            name="BulletCustom",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=13,
            leftIndent=15,
            firstLineIndent=-8,
            textColor=colors.HexColor("#263238"),
            spaceAfter=5,
        )
    )

    styles.add(
        ParagraphStyle(
            name="SmallText",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=12,
            textColor=DARK_GREY,
        )
    )

    styles.add(
        ParagraphStyle(
            name="HighlightText",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=14,
            textColor=NAVY,
        )
    )

    return styles


def section_banner(title, background_color, styles):
    """Create a coloured section heading."""
    banner = Table(
        [[Paragraph(title, styles["WhiteHeading"])]],
        colWidths=[7.0 * inch],
    )

    banner.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background_color),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )

    return banner


def bullet(text, styles):
    """Create one formatted bullet point."""
    return Paragraph(f"- {text}", styles["BulletCustom"])


def build_pdf():
    """Build the complete Day 1 summary PDF."""
    styles = make_styles()

    document = SimpleDocTemplate(
        str(OUTPUT_FILE),
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.60 * inch,
        bottomMargin=0.75 * inch,
        title="Day 1 Data Preparation and Feature Engineering",
        author="Project 3 Team",
    )

    story = []

    # =====================================================
    # PAGE 1 - OVERVIEW
    # =====================================================
    story.append(
        Paragraph(
            "Day 1 Data Preparation &amp;<br/>Feature Engineering",
            styles["ReportTitle"],
        )
    )

    story.append(
        Paragraph(
            "GSK Medicine Prediction System | Project 3",
            styles["ReportSubtitle"],
        )
    )

    story.append(Spacer(1, 4))

    story.append(section_banner("Executive Summary", TEAL, styles))
    story.append(Spacer(1, 8))

    story.append(
        Paragraph(
            "Day 1 focused on understanding, cleaning and preparing the "
            "clinical dataset before machine-learning models are trained. "
            "The raw data was examined for missing values, duplicate patient "
            "records, inconsistent formats and medically implausible values. "
            "The cleaned data was then enriched with seven clinically "
            "meaningful features.",
            styles["BodyTextCustom"],
        )
    )

    story.append(Spacer(1, 5))

    # Dataset journey table
    journey_data = [
        [
            Paragraph("<b>Dataset stage</b>", styles["SmallText"]),
            Paragraph("<b>Rows</b>", styles["SmallText"]),
            Paragraph("<b>Columns</b>", styles["SmallText"]),
            Paragraph("<b>Purpose</b>", styles["SmallText"]),
        ],
        [
            "Raw data",
            "1,155,000",
            "41",
            "Original clinical records",
        ],
        [
            "Cleaned data",
            "1,050,000",
            "31",
            "One record per patient",
        ],
        [
            "Engineered data",
            "1,050,000",
            "38",
            "Seven new clinical features",
        ],
    ]

    journey_table = Table(
        journey_data,
        colWidths=[1.30 * inch, 1.10 * inch, 1.00 * inch, 3.60 * inch],
        repeatRows=1,
    )

    journey_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 1), (-1, -1), LIGHT_BLUE),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#C4D2DB")),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTNAME", (1, 1), (2, -1), "Helvetica-Bold"),
                ("TEXTCOLOR", (1, 1), (2, -1), NAVY),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 1), (2, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )

    story.append(journey_table)
    story.append(Spacer(1, 14))

    story.append(section_banner("Step 1 - Exploratory Data Analysis", GOLD, styles))
    story.append(Spacer(1, 8))

    story.append(
        Paragraph(
            "The first step examined the complete raw Parquet dataset without "
            "changing any records. This established the starting condition of "
            "the data and identified issues requiring correction.",
            styles["BodyTextCustom"],
        )
    )

    story.append(bullet("<b>1,155,000 rows and 41 columns</b> were analysed.", styles))
    story.append(bullet("<b>70,017 exact duplicate rows</b> were identified.", styles))
    story.append(
        bullet(
            "The dataset contained <b>1,050,000 unique Patient IDs</b> and "
            "105,000 additional repeated patient records.",
            styles,
        )
    )
    story.append(
        bullet(
            "Missing-value counts, column types, treatment-outcome distribution "
            "and numeric statistics were reviewed.",
            styles,
        )
    )
    story.append(
        bullet(
            "Clinical distribution charts and a correlation heatmap were created.",
            styles,
        )
    )

    story.append(Spacer(1, 7))

    eda_result = Table(
        [
            [
                Paragraph(
                    "<b>Result:</b> The EDA confirmed that cleaning was necessary "
                    "before the dataset could be used safely for modelling.",
                    styles["BodyTextCustom"],
                )
            ]
        ],
        colWidths=[7.0 * inch],
    )

    eda_result.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT_GOLD),
                ("BOX", (0, 0), (-1, -1), 1, GOLD),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )

    story.append(eda_result)

    # =====================================================
    # PAGE 2 - CLEANING
    # =====================================================
    story.append(PageBreak())

    story.append(section_banner("Step 2 - Data Cleaning", TEAL, styles))
    story.append(Spacer(1, 9))

    story.append(
        Paragraph(
            "The cleaning stage converted the raw dataset into one consistent "
            "record per patient while retaining useful clinical information.",
            styles["BodyTextCustom"],
        )
    )

    cleaning_sections = [
        (
            "Duplicate columns combined",
            "Duplicate versions of patient ID, age, gender, drug name and weight "
            "were consolidated. Weight recorded in pounds was converted to "
            "kilograms when required.",
        ),
        (
            "Repeated patients handled",
            "Only the first record for each Patient_ID was retained. This removed "
            "105,000 repeated patient rows and produced 1,050,000 unique patients.",
        ),
        (
            "Column names standardized",
            "Column names were converted into consistent lowercase names using "
            "underscores, making them easier to reference in Python.",
        ),
        (
            "Categorical values standardized",
            "Gender, ethnicity, medicine names, route, diagnosis, smoking status "
            "and alcohol-use values were corrected and standardized.",
        ),
        (
            "Numeric and clinical values corrected",
            "Numeric columns were converted to suitable data types. Invalid or "
            "implausible values were corrected, replaced or capped using "
            "reasonable data-cleaning rules.",
        ),
        (
            "Missing values handled",
            "Numeric predictor gaps were filled using median values and "
            "categorical predictor gaps were filled using the most common value.",
        ),
        (
            "Target values preserved",
            "The 64,124 records with missing treatment_outcome were retained in "
            "the cleaned dataset. They will be excluded only when supervised "
            "model training begins.",
        ),
    ]

    for heading, explanation in cleaning_sections:
        block = Table(
            [
                [
                    Paragraph(f"<b>{heading}</b>", styles["HighlightText"]),
                    Paragraph(explanation, styles["BodyTextCustom"]),
                ]
            ],
            colWidths=[1.75 * inch, 5.25 * inch],
        )

        block.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, 0), LIGHT_BLUE),
                    ("BACKGROUND", (1, 0), (1, 0), colors.white),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CED8DE")),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CED8DE")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )

        story.append(KeepTogether([block, Spacer(1, 6)]))

    story.append(Spacer(1, 5))

    cleaning_result = Table(
        [
            [
                Paragraph(
                    "<b>Cleaning result</b><br/>"
                    "1,050,000 rows | 31 columns | Patient IDs unique: True",
                    styles["BodyTextCustom"],
                )
            ]
        ],
        colWidths=[7.0 * inch],
    )

    cleaning_result.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT_GREEN),
                ("BOX", (0, 0), (-1, -1), 1, GREEN),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )

    story.append(cleaning_result)

    # =====================================================
    # PAGE 3 - FEATURE ENGINEERING
    # =====================================================
    story.append(PageBreak())

    story.append(section_banner("Step 3 - Feature Engineering", GREEN, styles))
    story.append(Spacer(1, 9))

    story.append(
        Paragraph(
            "Seven new features were created from the cleaned clinical variables. "
            "These features summarize useful health information and may help the "
            "machine-learning models identify patterns connected with treatment "
            "effectiveness.",
            styles["BodyTextCustom"],
        )
    )

    feature_data = [
        [
            Paragraph("<b>Feature</b>", styles["SmallText"]),
            Paragraph("<b>How it was created</b>", styles["SmallText"]),
            Paragraph("<b>Why it is useful</b>", styles["SmallText"]),
        ],
        [
            "kidney_stage",
            "Grouped from eGFR values",
            "Summarizes kidney function severity",
        ],
        [
            "bmi_category",
            "Underweight, Normal, Overweight or Obese",
            "Makes body-weight risk easier to interpret",
        ],
        [
            "age_group",
            "Grouped into five age categories",
            "Captures age-related treatment patterns",
        ],
        [
            "liver_risk",
            "ALT above 40 or AST above 40",
            "Flags possible liver-function risk",
        ],
        [
            "polypharmacy",
            "Five or more concurrent medicines",
            "Identifies patients using many medicines",
        ],
        [
            "elderly_high_dose",
            "Age 65+ and dosage above the median",
            "Highlights higher-dose risk in older patients",
        ],
        [
            "de_ritis_ratio",
            "AST divided by ALT",
            "Provides an additional liver-health indicator",
        ],
    ]

    feature_table = Table(
        feature_data,
        colWidths=[1.45 * inch, 2.65 * inch, 2.90 * inch],
        repeatRows=1,
    )

    feature_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("BACKGROUND", (0, 2), (-1, 2), LIGHT_GREY),
                ("BACKGROUND", (0, 4), (-1, 4), LIGHT_GREY),
                ("BACKGROUND", (0, 6), (-1, 6), LIGHT_GREY),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#C8D1D8")),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                ("TEXTCOLOR", (0, 1), (0, -1), NAVY),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("LEADING", (0, 0), (-1, -1), 11),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )

    story.append(feature_table)
    story.append(Spacer(1, 14))

    story.append(section_banner("Final Validation", GOLD, styles))
    story.append(Spacer(1, 8))

    story.append(bullet("Row count remained unchanged at <b>1,050,000</b>.", styles))
    story.append(bullet("Column count increased from <b>31 to 38</b>.", styles))
    story.append(bullet("<b>patient_id remained unique</b> after feature engineering.", styles))
    story.append(
        bullet(
            "<b>64,124 missing treatment outcomes</b> remained preserved.",
            styles,
        )
    )
    story.append(
        bullet(
            "The cleaned and engineered datasets were saved in efficient "
            "<b>Parquet format</b>.",
            styles,
        )
    )

    story.append(Spacer(1, 10))

    conclusion = Table(
        [
            [
                Paragraph(
                    "<b>Day 1 conclusion</b><br/>"
                    "The project now has a structured, consistent and clinically "
                    "enriched dataset. The next stage is to remove unlabeled rows "
                    "for supervised learning, split the labelled data into training "
                    "and testing sets, and evaluate the provided classification models.",
                    styles["BodyTextCustom"],
                )
            ]
        ],
        colWidths=[7.0 * inch],
    )

    conclusion.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT_GOLD),
                ("BOX", (0, 0), (-1, -1), 1, GOLD),
                ("LEFTPADDING", (0, 0), (-1, -1), 11),
                ("RIGHTPADDING", (0, 0), (-1, -1), 11),
                ("TOPPADDING", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )

    story.append(conclusion)

    # Generate PDF
    document.build(
        story,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number,
    )

    print("Day 1 PDF generated successfully.")
    print(f"Saved to: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    build_pdf()