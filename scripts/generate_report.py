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
    
    doc = SimpleDocTemplate(pdf_file, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=28,
        textColor=PURPLE,
        spaceAfter=6,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=PURPLE,
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=DARK_GRAY,
        spaceAfter=6,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=10,
        alignment=TA_JUSTIFY,
        spaceAfter=10
    )
    
    # Title Page
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("📧 Spam Email Classification", title_style))
    story.append(Paragraph("Advanced ML Pipeline with OpenSpec Workflow", styles['Heading3']))
    story.append(Spacer(1, 0.3*inch))
    
    # Date
    date_str = datetime.now().strftime("%Y年%m月%d日")
    story.append(Paragraph(f"<font size=10>Report Generated: {date_str}</font>", styles['Normal']))
    story.append(Spacer(1, 0.5*inch))
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))
    story.append(Paragraph(
        """This report documents a complete end-to-end machine learning project for spam email classification. 
        The project implements a logistic regression model trained on 5,574 SMS messages with 96.95% test accuracy. 
        The implementation includes data preprocessing, model training, evaluation, and an interactive Streamlit web application 
        for real-time classification and analysis.""",
        body_style
    ))
    story.append(Spacer(1, 0.2*inch))
    
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
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), LIGHT_GRAY),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
    ]))
    story.append(overview_table)
    story.append(Spacer(1, 0.3*inch))
    
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
        "The following table shows model performance metrics across different decision thresholds, "
        "allowing optimization for specific use cases (prioritize precision or recall):",
        body_style
    ))
    story.append(Spacer(1, 0.1*inch))
    
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
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), LIGHT_GRAY),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
    ]))
    story.append(sweep_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Page break
    story.append(PageBreak())
    
    # Data Preprocessing
    story.append(Paragraph("Data Preprocessing Pipeline", heading_style))
    story.append(Paragraph(
        "The project implements a comprehensive 7-stage text preprocessing pipeline:",
        body_style
    ))
    story.append(Spacer(1, 0.1*inch))
    
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
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BACKGROUND', (0, 1), (-1, -1), LIGHT_GRAY),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
    ]))
    story.append(preproc_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Key Features
    story.append(Paragraph("Key Features", heading_style))
    features_text = """
    <b>1. Multi-Format CSV Support:</b> The application supports both simple 2-column CSV format and 
    advanced 9-column preprocessing pipeline format for detailed text transformation analysis.<br/>
    <br/>
    <b>2. Interactive Dashboard:</b> Streamlit-based web application with real-time classification, 
    token analysis, and model performance visualization.<br/>
    <br/>
    <b>3. Advanced Analytics:</b> Threshold sweep analysis, ROC curves, confusion matrices, and 
    precision-recall curves for comprehensive model evaluation.<br/>
    <br/>
    <b>4. CLI Tools:</b> Command-line utilities for batch prediction, visualization generation, 
    and model training with custom parameters.<br/>
    <br/>
    <b>5. Professional Documentation:</b> Comprehensive README, quick-start guides, and delivery summaries 
    with usage examples and technical details.
    """
    story.append(Paragraph(features_text, body_style))
    story.append(Spacer(1, 0.2*inch))
    
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
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), LIGHT_GRAY),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
    ]))
    story.append(tech_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Project Structure
    story.append(Paragraph("Project Structure", heading_style))
    structure_text = """
    <font face="Courier" size="8">
    .<br/>
    ├── app.py                         # Streamlit web application<br/>
    ├── train.py                       # Model training script<br/>
    ├── requirements.txt               # Python dependencies<br/>
    ├── README.md                      # Project documentation<br/>
    ├── src/<br/>
    │   ├── data_loader.py            # Data loading & preprocessing<br/>
    │   └── model_trainer.py          # Model training & evaluation<br/>
    ├── scripts/<br/>
    │   ├── predict_spam.py           # CLI prediction tool<br/>
    │   ├── visualize_spam.py         # Visualization toolkit<br/>
    │   └── generate_report.py        # PDF report generation<br/>
    ├── data/<br/>
    │   ├── sms_spam_clean.csv        # Clean 2-column format<br/>
    │   ├── sms_spam_preprocessing.csv# 9-column preprocessing pipeline<br/>
    │   └── sms_spam_no_header.csv    # Original format<br/>
    ├── models/<br/>
    │   ├── logistic_regression.pkl   # Trained model (joblib)<br/>
    │   ├── vectorizer.pkl            # TF-IDF vectorizer<br/>
    │   ├── label_mapping.json        # Label mappings<br/>
    │   ├── metrics_logistic_regression.json # Performance metrics<br/>
    │   ├── threshold_sweep.json      # Threshold analysis<br/>
    │   └── test_predictions.json     # Test predictions for ROC<br/>
    └── docs/<br/>
        └── PREPROCESSING.md          # Preprocessing documentation<br/>
    </font>
    """
    story.append(Paragraph(structure_text, body_style))
    story.append(Spacer(1, 0.2*inch))
    
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
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), LIGHT_GRAY),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
    ]))
    story.append(perf_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Usage Instructions
    story.append(Paragraph("How to Use", heading_style))
    
    usage_text = """
    <b>1. Running the Web Application:</b><br/>
    <font face="Courier">streamlit run app.py</font><br/>
    Open your browser to http://localhost:8501<br/>
    <br/>
    
    <b>2. Making Predictions (CLI):</b><br/>
    <font face="Courier">python scripts/predict_spam.py --text "Your message here"</font><br/>
    <br/>
    
    <b>3. Batch Predictions:</b><br/>
    <font face="Courier">python scripts/predict_spam.py --input data.csv --output predictions.csv</font><br/>
    <br/>
    
    <b>4. Generating Visualizations:</b><br/>
    <font face="Courier">python scripts/visualize_spam.py --input data.csv --dist --tokens</font><br/>
    <br/>
    
    <b>5. Training Model:</b><br/>
    <font face="Courier">python train.py --model logistic_regression</font><br/>
    """
    story.append(Paragraph(usage_text, body_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Conclusions
    story.append(Paragraph("Conclusions & Future Work", heading_style))
    
    conclusion_text = """
    <b>Achievements:</b><br/>
    • Successfully built a high-accuracy spam classification model (96.95% accuracy)<br/>
    • Implemented comprehensive data preprocessing pipeline with 7 stages<br/>
    • Created professional interactive dashboard with real-time classification<br/>
    • Developed CLI tools for batch processing and automation<br/>
    • Demonstrated OpenSpec specification-driven development workflow<br/>
    <br/>
    
    <b>Future Enhancements:</b><br/>
    • Support for multiple languages and character sets<br/>
    • Ensemble models combining multiple algorithms<br/>
    • Active learning with user feedback integration<br/>
    • Advanced NLP techniques (BERT, transformers)<br/>
    • Deployment to cloud platforms (AWS, GCP, Azure)<br/>
    • Real-time model retraining with new data<br/>
    """
    story.append(Paragraph(conclusion_text, body_style))
    story.append(Spacer(1, 0.5*inch))
    
    # Footer
    story.append(Paragraph(
        f"<font size='8' color='gray'>Generated on {date_str} | GitHub: github.com/Brain0927/HW3_Fulong_5114056035</font>",
        ParagraphStyle('Footer', parent=styles['Normal'], alignment=TA_CENTER)
    ))
    
    # Build PDF
    doc.build(story)
    print(f"✅ PDF報告已生成: {pdf_file}")
    print(f"📄 文件大小: {os.path.getsize(pdf_file) / 1024:.1f} KB")

if __name__ == "__main__":
    create_report()
