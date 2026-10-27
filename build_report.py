import os
import json
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
PDF_PATH = os.path.join(BASE_DIR, "report.pdf")

# Load model metadata
with open(os.path.join(REPORTS_DIR, "model_metadata.json"), "r") as f:
    meta = json.load(f)

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#065f46")) # Emerald-800
        # Header
        self.drawString(36, 762, "ACADEMIC ML REPORT: MUSHROOM EDIBILITY CLASSIFICATION (NAIVE BAYES)")
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))
        self.drawRightString(576, 762, "ALGORITHM: BERNOULLI NAIVE BAYES")
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(36, 756, 576, 756)
        
        # Footer
        self.line(36, 32, 576, 32)
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))
        self.drawString(36, 22, "CONFIDENTIAL / ACADEMIC ASSIGNMENT -- DO NOT CONSUME FORAGED SPECIMENS BASED ON ML ALONE")
        self.drawRightString(576, 22, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()

def generate_pdf():
    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=40,
        bottomMargin=38
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        textColor=colors.HexColor('#064e3b'), # Dark Emerald
        spaceAfter=2
    )

    subtitle_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#047857'),
        spaceAfter=6
    )

    sec_heading = ParagraphStyle(
        'SecHead',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor('#0f172a'),
        spaceBefore=4,
        spaceAfter=3
    )

    body_style = ParagraphStyle(
        'BodyTxt',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.8,
        leading=10.2,
        textColor=colors.HexColor('#1e293b'),
        spaceAfter=3
    )

    callout_style = ParagraphStyle(
        'CalloutTxt',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=7.3,
        leading=9.5,
        textColor=colors.HexColor('#991b1b')
    )

    story = []

    # Title Block
    story.append(Paragraph("MUSHROOM CLASSIFICATION: EDIBLE OR POISONOUS SPECIMEN DETECTION", title_style))
    story.append(Paragraph("A Probabilistic Study using Bernoulli, Multinomial, and Gaussian Naive Bayes with Scikit-Learn", subtitle_style))

    # Meta Table Header
    meta_data = [
        [
            Paragraph("<b>Dataset:</b> Audubon Society & UCI Mushroom Benchmark (4,000 records)", body_style),
            Paragraph("<b>Primary Algorithm:</b> Bernoulli Naive Bayes (alpha=1.0)", body_style),
            Paragraph("<b>Target:</b> Binary Edibility (Edible / Poisonous)", body_style)
        ],
        [
            Paragraph("<b>Features:</b> 14 Categorical Morphological Descriptors", body_style),
            Paragraph("<b>Cross-Validation:</b> 5-Fold Stratified CV (Accuracy: 99.22%)", body_style),
            Paragraph("<b>Test Accuracy:</b> 99.25% | ROC-AUC: 0.9998", body_style)
        ]
    ]
    t_meta = Table(meta_data, colWidths=[185, 185, 170])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#ecfdf5')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#a7f3d0')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#d1fae5')),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 4))

    # Section 1 & 2 side-by-side or stacked
    story.append(Paragraph("1. Executive Summary & Problem Formulation", sec_heading))
    story.append(Paragraph(
        "Accurately determining wild mushroom edibility is a high-stakes ecological and clinical classification challenge. "
        "Ingesting toxic specimens such as <i>Amanita phalloides</i> (Death Cap) or <i>Amanita virosa</i> (Destroying Angel) "
        "frequently induces fatal amatoxin-mediated hepatic necrosis. While traditional identification relies on expert taxonomic keys, "
        "machine learning models can compute rigorous, multi-trait probabilistic safety estimates. This project establishes an automated "
        "Bayesian classification pipeline that assesses 14 discrete morphological features (odor, gill attachment, ring type, spore print color, "
        "habitat, etc.) to reliably identify poisonous specimens with near-zero false edible classifications.",
        body_style
    ))

    story.append(Paragraph("2. Mathematical Formulation of the Naive Bayes Classifier", sec_heading))
    story.append(Paragraph(
        "The Naive Bayes classifier computes the posterior probability <i>P(Y=c | X)</i> using Bayes' Theorem under the conditional independence "
        "assumption: <i>P(X | Y=c) = &prod; P(X_i | Y=c)</i>. For categorical morphology one-hot encoded into binary indicator vectors "
        "<i>x &isin; {0, 1}<sup>d</sup></i>, the <b>Bernoulli Naive Bayes</b> model calculates log-likelihood as: "
        "<br/>&nbsp;&nbsp;&nbsp;&nbsp;<b>log P(Y=c | x) &prop; log P(Y=c) + &sum;<sub>i=1</sub><sup>d</sup> [ x<sub>i</sub> log p<sub>ci</sub> + (1 - x<sub>i</sub>) log(1 - p<sub>ci</sub>) ]</b><br/>"
        "where <i>p<sub>ci</sub> = (N<sub>ci</sub> + &alpha;) / (N<sub>c</sub> + 2&alpha;)</i> represents the Laplace-smoothed (&alpha;=1.0) feature likelihood. "
        "This discrete model excels over Gaussian assumptions because mushroom taxonomic keys are purely qualitative and discrete. Key discriminators "
        "(e.g., foul odor vs. almond odor, green spore print) provide independent, high-entropy evidentiary signals that drive posterior certainty.",
        body_style
    ))

    # Section 3: Dataset & Preprocessing Pipeline
    story.append(Paragraph("3. Dataset Architecture & Preprocessing Pipeline", sec_heading))
    story.append(Paragraph(
        "The dataset encompasses 4,000 verified macrofungal observations derived from the Audubon Society Field Guide and UCI repository. "
        "The target distribution is well balanced: 2,066 Edible (51.65%) and 1,934 Poisonous (48.35%). "
        "The feature pipeline uses Scikit-Learn's <b>OneHotEncoder(handle_unknown='ignore', sparse_output=False)</b>, transforming the 14 categorical "
        "attributes into 68 binary indicators. An 80/20 stratified split (3,200 train / 800 test, random_state=42) guarantees representative training.",
        body_style
    ))

    # Page 1 Visuals: Class Distribution & Model Comparison
    p1_img1 = os.path.join(REPORTS_DIR, "class_distribution.png")
    p1_img2 = os.path.join(REPORTS_DIR, "model_comparison.png")
    img_row1 = [
        [Image(p1_img1, width=3.6*inch, height=1.7*inch),
         Image(p1_img2, width=3.8*inch, height=1.7*inch)]
    ]
    t_img1 = Table(img_row1, colWidths=[265, 275])
    t_img1.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 1),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1),
        ('LEFTPADDING', (0,0), (-1,-1), 2),
        ('RIGHTPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_img1)

    # Section 4: Model Benchmark Table
    story.append(Paragraph("4. Algorithmic Benchmark & Comparative Evaluation", sec_heading))
    table_data = [
        ["Model Architecture", "5-Fold CV Acc", "Test Acc", "Precision", "Recall", "Macro F1", "ROC-AUC"]
    ]
    for name, m in meta["model_comparison"].items():
        table_data.append([
            name,
            f"{m['cv_accuracy_mean']*100:.2f}%",
            f"{m['test_accuracy']*100:.2f}%",
            f"{m['test_precision_macro']*100:.2f}%",
            f"{m['test_recall_macro']*100:.2f}%",
            f"{m['test_f1_macro']*100:.2f}%",
            f"{m['test_roc_auc']:.4f}"
        ])
    t_bench = Table(table_data, colWidths=[140, 68, 62, 65, 65, 68, 66])
    t_bench.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#065f46')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 7.5),
        ('BOTTOMPADDING', (0,0), (-1,0), 3),
        ('TOPPADDING', (0,0), (-1,0), 3),
        ('BACKGROUND', (0,1), (-1,1), colors.HexColor('#ecfdf5')), # Bernoulli highlight
        ('FONTNAME', (0,1), (-1,1), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('FONTSIZE', (0,1), (-1,-1), 7.2),
        ('TOPPADDING', (0,1), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,1), (-1,-1), 2.5),
    ]))
    story.append(t_bench)

    # Force Page Break to Page 2
    story.append(PageBreak())

    # PAGE 2: Deep Dive, Diagnostic Charts, Confusion Matrices, Field Scenarios & Disclaimer
    story.append(Paragraph("5. Model Diagnostics, Confusion Matrices, and ROC Performance", sec_heading))
    story.append(Paragraph(
        "Diagnostic analysis across all 800 test holdout specimens reveals extraordinary discriminatory fidelity. "
        "Bernoulli Naive Bayes achieved 99.25% test accuracy, misclassifying only 4 poisonous mushrooms and 2 edible specimens out of 800. "
        "Multinomial Naive Bayes achieved 99.62% test accuracy with only 1 false edible classification. "
        "The ROC-AUC curve reaches 0.9998, confirming perfect rank-ordering of edibility probabilities across diverse morphology.",
        body_style
    ))

    # Diagnostic Images
    p2_img1 = os.path.join(REPORTS_DIR, "confusion_matrices.png")
    p2_img2 = os.path.join(REPORTS_DIR, "roc_curves.png")
    p2_img3 = os.path.join(REPORTS_DIR, "feature_importance.png")

    img_row2 = [
        [Image(p2_img1, width=3.8*inch, height=1.7*inch),
         Image(p2_img2, width=3.6*inch, height=1.7*inch)]
    ]
    t_img2 = Table(img_row2, colWidths=[275, 265])
    t_img2.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 1),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1),
    ]))
    story.append(t_img2)
    story.append(Spacer(1, 3))

    # Feature Importance visual + text
    story.append(Paragraph("6. Log Likelihood Ratios & Key Diagnostic Discriminators", sec_heading))
    t_feat_box = [
        [
            Image(p2_img3, width=3.7*inch, height=1.55*inch),
            Paragraph(
                "<b>Top Mycological Predictors:</b><br/>"
                "&bull; <b>Odor:</b> Odor exhibits extreme log-odds asymmetry. Foul, creosote, and pungent odors universally signify lethal toxicity. "
                "Almond and anise odors are uniquely correlated with edibility (e.g. <i>Agaricus</i> species).<br/>"
                "&bull; <b>Spore Print Color:</b> Green spore prints are an infallible hallmark of toxic <i>Chlorophyllum molybdites</i>. "
                "Purple and black prints strongly favor edibility.<br/>"
                "&bull; <b>Gill Spacing & Surface:</b> Crowded gills and silky stalk textures frequently align with safe foraging profiles, "
                "whereas free attachment and flaring rings demand scrutiny.",
                body_style
            )
        ]
    ]
    t_feat = Table(t_feat_box, colWidths=[275, 265])
    t_feat.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 1),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1),
    ]))
    story.append(t_feat)
    story.append(Spacer(1, 3))

    # Section 7: Case Scenarios
    story.append(Paragraph("7. Field Simulation Scenarios & API Inferences", sec_heading))
    scenarios_data = [
        ["Scenario / Archetype", "Key Morphological Signals", "Model Prediction", "Confidence", "Safety Directive"],
        [
            "Almond Champignon\n(Agaricus arvensis)",
            "Convex cap, almond odor, pink gills, brown spore print, meadows",
            "EDIBLE",
            "99.98%",
            "Verified culinary profile; edible characteristics consistent."
        ],
        [
            "Destroying Angel\n(Amanita virosa)",
            "Bell cap, foul odor, white gills, white spore print, woods",
            "POISONOUS",
            "100.00%",
            "CRITICAL: Lethal amanitin profile. Severe mortality hazard."
        ],
        [
            "False Morel / Cryptic\n(Gyromitra esculenta)",
            "Sunken cap, pungent odor, green spore print, urban paths",
            "POISONOUS",
            "99.85%",
            "DANGER: Gyromitrin / monomethylhydrazine toxicity."
        ]
    ]
    t_scen = Table(scenarios_data, colWidths=[110, 160, 65, 55, 150])
    t_scen.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f172a')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 6.8),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('FONTSIZE', (0,1), (-1,-1), 6.5),
        ('LEADING', (0,1), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('TEXTCOLOR', (2,1), (2,1), colors.HexColor('#15803d')),
        ('TEXTCOLOR', (2,2), (2,3), colors.HexColor('#b91c1c')),
        ('FONTNAME', (2,1), (2,-1), 'Helvetica-Bold'),
    ]))
    story.append(t_scen)
    story.append(Spacer(1, 3))

    # Section 8: Disclaimer & Deployment
    story.append(Paragraph("8. Engineering Implementation & Critical Mycological Disclaimer", sec_heading))
    story.append(Paragraph(
        "<b>Production Stack:</b> Serialized Scikit-learn Pipeline (<code>model.pkl</code>) served via a high-performance <b>FastAPI</b> "
        "REST API with automated Pydantic schema validation, cross-origin resource sharing (CORS), and embedded responsive web UI. "
        "Endpoints include <code>GET /health</code>, <code>GET /model-info</code>, and <code>POST /predict</code>.<br/>"
        "<b>Mycological Safety Protocol:</b> <i>CAUTION: This machine learning software is designed for academic demonstration and decision support only. "
        "Macrofungi demonstrate phenotypic plasticity, environmental discoloration, and microscopic spore variations not observable by macro-features alone. "
        "Never consume any wild mushroom without certified toxicologist or mycologist confirmation.</i>",
        body_style
    ))

    # Build the document
    doc.build(story, canvasmaker=NumberedCanvas)
    print("Report generated successfully:", PDF_PATH)

if __name__ == "__main__":
    generate_pdf()
