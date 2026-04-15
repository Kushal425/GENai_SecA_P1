import io
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.enums import TA_LEFT, TA_CENTER


def generate_pdf_report(report: dict) -> bytes:
    """
    Generate a professional PDF from a structured research report dict.
    Returns raw PDF bytes for Streamlit's st.download_button.
    """
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()

    # ── Custom Styles ───────────────────────────────────────────────────────────
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=20,
        textColor=colors.HexColor("#1a1a2e"),
        spaceAfter=10,
        alignment=TA_CENTER,
    )
    label_style = ParagraphStyle(
        "AppLabel",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#7c3aed"),
        spaceAfter=4,
        alignment=TA_CENTER,
    )
    section_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=colors.HexColor("#7c3aed"),
        spaceBefore=18,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "CustomBody",
        parent=styles["Normal"],
        fontSize=10,
        leading=16,
        textColor=colors.HexColor("#1a1a1a"),
        spaceAfter=6,
        alignment=TA_LEFT,
    )
    finding_style = ParagraphStyle(
        "Finding",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        leftIndent=18,
        textColor=colors.HexColor("#1a1a1a"),
        spaceAfter=5,
    )
    source_style = ParagraphStyle(
        "Source",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
        leftIndent=18,
        textColor=colors.HexColor("#2563eb"),
        spaceAfter=4,
    )

    story = []

    # ── App Label ───────────────────────────────────────────────────────────────
    story.append(Paragraph("ResearchScope — Agentic AI Research Report", label_style))
    story.append(Spacer(1, 0.2 * cm))

    # ── Title ───────────────────────────────────────────────────────────────────
    story.append(Paragraph(report.get("title", "Research Report"), title_style))
    story.append(
        HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#7c3aed"))
    )
    story.append(Spacer(1, 0.4 * cm))

    # ── Abstract ────────────────────────────────────────────────────────────────
    story.append(Paragraph("Abstract", section_style))
    story.append(Paragraph(report.get("abstract", "Not available."), body_style))

    # ── Key Findings ────────────────────────────────────────────────────────────
    story.append(Paragraph("Key Findings", section_style))
    findings = report.get("key_findings", [])
    if findings:
        for i, finding in enumerate(findings, 1):
            story.append(Paragraph(f"{i}.  {finding}", finding_style))
    else:
        story.append(Paragraph("No key findings were extracted.", body_style))

    # ── Conclusion ──────────────────────────────────────────────────────────────
    story.append(Paragraph("Conclusion", section_style))
    story.append(Paragraph(report.get("conclusion", "Not available."), body_style))

    # ── Sources ─────────────────────────────────────────────────────────────────
    story.append(Paragraph("Sources", section_style))
    sources = report.get("sources", [])
    if sources:
        for i, source in enumerate(sources, 1):
            title = source.get("title", "Unknown Source")
            url = source.get("url", "")
            if url:
                story.append(
                    Paragraph(
                        f"{i}.  <a href='{url}' color='#2563eb'>{title}</a>",
                        source_style,
                    )
                )
            else:
                story.append(Paragraph(f"{i}.  {title}", source_style))
    else:
        story.append(Paragraph("No sources available.", body_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()
