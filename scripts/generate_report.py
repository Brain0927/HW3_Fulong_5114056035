#!/usr/bin/env python3
"""
Generate comprehensive PDF report for Spam Email Classification Project
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
import json
from datetime import datetime
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_trainer import load_model, load_vectorizer
from data_loader import get_data

# Colors
PURPLE = colors.HexColor('#667eea')
LIGHT_PURPLE = colors.HexColor('#f5f3ff')
DARK_GRAY = colors.HexColor('#31333b')
LIGHT_GRAY = colors.HexColor('#f0f2f6')

def create_report():
    """Generate comprehensive PDF report"""
    
    # Create PDF
    pdf_file = "outputs/Spam_Classification_Report.pdf"
    os.makedirs("outputs", exist_ok=True)
    
    # Use A4 page size for better text display
    doc = SimpleDocTemplate(pdf_file, pagesize=A4, topMargin=0.7*inch, bottomMargin=0.7*inch, 
                           leftMargin=0.6*inch, rightMargin=0.6*inch)
    story = []
    
    # Define styles with better text rendering
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=26,
        textColor=PURPLE,
        spaceAfter=8,
        spaceBefore=4,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold',
        leading=32
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=PURPLE,
        spaceAfter=10,
        spaceBefore=10,
        fontName='Helvetica-Bold',
        leading=18
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=11,
        textColor=DARK_GRAY,
        spaceAfter=6,
        fontName='Helvetica-Bold',
        leading=14
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=9.5,
        alignment=TA_LEFT,
        spaceAfter=10,
        leading=13
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=9,
        leading=12,
        spaceAfter=8
    )
    
    # Title Page
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("Spam Email Classification", title_style))
    story.append(Paragraph("Advanced ML Pipeline with OpenSpec Workflow", heading_style))
    story.append(Spacer(1, 0.25*inch))
    
    # Date
    date_str = datetime.now().strftime("%Y年%m月%d日")
    story.append(Paragraph(f"Report Generated: {date_str}", normal_style))
    story.append(Spacer(1, 0.4*inch))
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))
    story.append(Paragraph(
        "This report documents a complete end-to-end machine learning project for spam email classification. "
        "The project implements a logistic regression model trained on 5,574 SMS messages with 96.95% test accuracy. "
        "The implementation includes data preprocessing, model training, evaluation, and an interactive Streamlit web application for real-time classification.",
        body_style
    ))
    story.append(Spacer(1, 0.15*inch))
    
    # Project Overview
    story.append(Paragraph("Project Overview", heading_style))
    
    overview_data = [
        ["Aspect", "Details"],
        ["Dataset Size", "5,574 SMS messages"],
        ["Spam Ratio", "13.4% (747 spam, 4,827 ham)"],
        ["Model Type", "Logistic Regression"],
        ["Test Accuracy", "96.95%"],
        ["Precision (Spam)", "100%"],
        ["Recall (Spam)", "77.18%"],
        ["F1 Score", "0.871"],
        ["Vectorization", "TF-IDF (max 5,000 features)"],
        ["N-grams", "Unigrams and Bigrams (1-2)"]
    ]
    
    overview_table = Table(overview_data, colWidths=[2*inch, 3.5*inch])
    overview_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PURPLE),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('TOPPADDING', (0, 0), (-1, 0), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (0, 1), (-1, -1), LIGHT_GRAY),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
    ]))
    story.append(overview_table)
    story.append(Spacer(1, 0.2*inch))
    
    # Load metrics
    try:
        with open("models/metrics_logistic_regression.json", 'r') as f:
            metrics = json.load(f)
    except:
        metrics = {}
    
    try:
        with open("models/threshold_sweep.json", 'r') as f:
            threshold_sweep = json.load(f)
    except:
        threshold_sweep = []
    
    # Threshold Sweep Analysis
    story.append(Paragraph("Threshold Sweep Analysis", heading_style))
    story.append(Paragraph(
        "The table below shows model performance metrics across different decision thresholds, "
        "enabling optimization for specific use cases.",
        body_style
    ))
    story.append(Spacer(1, 0.08*inch))
    
    # Create threshold table
    sweep_data = [["Threshold", "Precision", "Recall", "F1 Score"]]
    for item in threshold_sweep[:9]:  # Show 9 thresholds
        sweep_data.append([
            str(item.get('threshold', '')),
            f"{item.get('precision', 0):.4f}",
            f"{item.get('recall', 0):.4f}",
            f"{item.get('f1', 0):.4f}"
        ])
    
    sweep_table = Table(sweep_data, colWidths=[1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch])
    sweep_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PURPLE),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTSIZE', (0, 1), (-1, -1), 8.5),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('BACKGROUND', (0, 1), (-1, -1), LIGHT_GRAY),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
    ]))
    story.append(sweep_table)
    story.append(Spacer(1, 0.2*inch))
    
    # Page break
    story.append(PageBreak())
    
    # Data Preprocessing
    story.append(Paragraph("Data Preprocessing Pipeline", heading_style))
    story.append(Paragraph(
        "The project implements a comprehensive 7-stage text preprocessing pipeline:",
        body_style
    ))
    story.append(Spacer(1, 0.08*inch))
    
    preproc_data = [
        ["Stage", "Operation", "Purpose"],
        ["1. Raw", "Original text", "Baseline reference"],
        ["2. Lowercase", "Convert to lowercase", "Normalization"],
        ["3. Contact Masking", "Mask emails/phones", "Remove PII"],
        ["4. Number Replacement", "Replace digits with <NUM>", "Generalization"],
        ["5. Punctuation Removal", "Remove special characters", "Simplification"],
        ["6. Whitespace Normalization", "Normalize spaces", "Formatting"],
        ["7. Stopword Removal", "Remove common words", "Feature reduction"]
    ]
    
    preproc_table = Table(preproc_data, colWidths=[0.8*inch, 1.8*inch, 2*inch])
    preproc_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PURPLE),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('ROWHEIGHT', (0, 1), (-1, -1), 20),
        ('BACKGROUND', (0, 1), (-1, -1), LIGHT_GRAY),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
    ]))
    story.append(preproc_table)
    story.append(Spacer(1, 0.2*inch))
    
    # Key Features
    story.append(Paragraph("Key Features", heading_style))
    features_text = """
    <b>1. Multi-Format CSV Support:</b> Supports simple 2-column and 9-column preprocessing pipeline formats.<br/>
    <br/>
    <b>2. Interactive Dashboard:</b> Streamlit-based web application with real-time classification and token analysis.<br/>
    <br/>
    <b>3. Advanced Analytics:</b> Threshold sweep, ROC curves, confusion matrices, and precision-recall curves.<br/>
    <br/>
    <b>4. CLI Tools:</b> Command-line utilities for batch prediction and visualization generation.<br/>
    <br/>
    <b>5. Professional Documentation:</b> README, quick-start guides, and technical delivery summaries.
    """
    story.append(Paragraph(features_text, normal_style))
    story.append(Spacer(1, 0.15*inch))
    
    # Page break
    story.append(PageBreak())
    
    # Technology Stack
    story.append(Paragraph("Technology Stack", heading_style))
    
    tech_data = [
        ["Component", "Technology", "Purpose"],
        ["Language", "Python 3.12+", "Core implementation"],
        ["ML Framework", "Scikit-learn", "Model training & evaluation"],
        ["Data Processing", "Pandas, NumPy", "Data manipulation"],
        ["Visualization", "Plotly, Matplotlib, Seaborn", "Interactive & publication charts"],
        ["Web Framework", "Streamlit", "Interactive dashboard"],
        ["Serialization", "joblib", "Model & vectorizer storage"],
        ["Deployment", "Streamlit Cloud", "Public web application"],
        ["Version Control", "Git, GitHub", "Code management"],
        ["Workflow", "OpenSpec", "Specification-driven development"]
    ]
    
    tech_table = Table(tech_data, colWidths=[1.3*inch, 1.7*inch, 2*inch])
    tech_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PURPLE),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('BACKGROUND', (0, 1), (-1, -1), LIGHT_GRAY),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
    ]))
    story.append(tech_table)
    story.append(Spacer(1, 0.2*inch))
    
    # Project Structure
    story.append(Paragraph("Project Structure", heading_style))
    structure_text = """
    <font face="Courier" size="7">
    . (root)<br/>
    ├── app.py - Streamlit web application<br/>
    ├── train.py - Model training script<br/>
    ├── requirements.txt - Python dependencies<br/>
    ├── src/<br/>
    │   ├── data_loader.py - Data loading<br/>
    │   └── model_trainer.py - Model training<br/>
    ├── scripts/<br/>
    │   ├── predict_spam.py - CLI prediction<br/>
    │   ├── visualize_spam.py - Visualizations<br/>
    │   └── generate_report.py - PDF report<br/>
    ├── data/<br/>
    │   ├── sms_spam_clean.csv - 2-column format<br/>
    │   ├── sms_spam_preprocessing.csv - 9-column pipeline<br/>
    │   └── sms_spam_no_header.csv - Original format<br/>
    ├── models/<br/>
    │   ├── logistic_regression.pkl - Trained model<br/>
    │   ├── vectorizer.pkl - TF-IDF vectorizer<br/>
    │   ├── metrics_logistic_regression.json - Metrics<br/>
    │   ├── threshold_sweep.json - Threshold analysis<br/>
    │   └── test_predictions.json - Test predictions<br/>
    └── docs/ - Documentation files
    </font>
    """
    story.append(Paragraph(structure_text, normal_style))
    story.append(Spacer(1, 0.15*inch))
    
    # Page break
    story.append(PageBreak())
    
    # Results & Performance
    story.append(Paragraph("Model Performance Results", heading_style))
    story.append(Paragraph(
        "The Logistic Regression model achieved excellent performance on the spam classification task:",
        body_style
    ))
    story.append(Spacer(1, 0.1*inch))
    
    perf_data = [
        ["Metric", "Value", "Description"],
        ["Test Accuracy", "96.95%", "Overall correctness of predictions"],
        ["Precision (Spam)", "100%", "All spam predictions were correct"],
        ["Recall (Spam)", "77.18%", "77% of actual spam was detected"],
        ["F1 Score", "0.871", "Harmonic mean of precision & recall"],
        ["ROC-AUC", "~0.98", "Excellent discriminative ability"],
        ["Specificity", "100%", "No false positive rate"],
        ["True Negative Rate", "100%", "All legitimate emails correctly classified"]
    ]
    
    perf_table = Table(perf_data, colWidths=[1.5*inch, 1*inch, 2.5*inch])
    perf_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PURPLE),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('BACKGROUND', (0, 1), (-1, -1), LIGHT_GRAY),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
    ]))
    story.append(perf_table)
    story.append(Spacer(1, 0.2*inch))
    
    # Usage Instructions
    story.append(Paragraph("How to Use", heading_style))
    
    usage_text = """
    <b>1. Running the Web Application:</b><br/>
    <font face="Courier" size="8">streamlit run app.py</font><br/>
    <br/>
    
    <b>2. Making Predictions (CLI):</b><br/>
    <font face="Courier" size="8">python scripts/predict_spam.py --text "message"</font><br/>
    <br/>
    
    <b>3. Batch Predictions:</b><br/>
    <font face="Courier" size="8">python scripts/predict_spam.py --input data.csv</font><br/>
    <br/>
    
    <b>4. Training Model:</b><br/>
    <font face="Courier" size="8">python train.py</font>
    """
    story.append(Paragraph(usage_text, normal_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Conclusions
    story.append(Paragraph("Conclusions & Future Work", heading_style))
    
    conclusion_text = """
    <b>Achievements:</b><br/>
    • Built high-accuracy spam classification model (96.95% accuracy)<br/>
    • Implemented comprehensive 7-stage preprocessing pipeline<br/>
    • Created professional interactive dashboard with Streamlit<br/>
    • Developed CLI tools for batch processing<br/>
    • Demonstrated OpenSpec specification-driven workflow<br/>
    <br/>
    
    <b>Future Enhancements:</b><br/>
    • Support for multiple languages<br/>
    • Ensemble models combining multiple algorithms<br/>
    • Active learning with user feedback<br/>
    • Advanced NLP techniques (BERT, transformers)<br/>
    • Cloud platform deployment
    """
    story.append(Paragraph(conclusion_text, normal_style))
    story.append(Spacer(1, 0.4*inch))
    
    # Footer
    story.append(Paragraph(
        f"Generated: {date_str} | Project: Spam Classification | GitHub: Brain0927/HW3_Fulong_5114056035",
        ParagraphStyle('Footer', parent=styles['Normal'], fontSize=7, alignment=TA_CENTER, textColor=colors.grey)
    ))
    
    # Build PDF
    doc.build(story)
    print(f"✅ PDF報告已生成: {pdf_file}")
    print(f"📄 文件大小: {os.path.getsize(pdf_file) / 1024:.1f} KB")

if __name__ == "__main__":
    create_report()
