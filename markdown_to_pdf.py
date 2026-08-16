import markdown2
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import os
import re

def convert_markdown_to_pdf(markdown_file, pdf_file):
    """Convert Markdown file to PDF using markdown2 and reportlab"""
    
    # Read the markdown file
    with open(markdown_file, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    # Convert markdown to HTML first
    html_content = markdown2.markdown(markdown_content, extras=['tables', 'fenced-code-blocks'])
    
    # Create PDF
    doc = SimpleDocTemplate(pdf_file, pagesize=A4, topMargin=0.75*inch, bottomMargin=0.75*inch)
    story = []
    
    # Get sample styles
    styles = getSampleStyleSheet()
    
    # Define custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#0066cc'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#0066cc'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=10,
        alignment=TA_JUSTIFY,
        spaceAfter=12
    )
    
    # Parse HTML tables and content
    # Extract title
    title_match = re.search(r'<h1>(.*?)</h1>', html_content)
    if title_match:
        title = title_match.group(1)
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 0.3*inch))
    
    # Extract sections
    sections = re.split(r'<h2>(.*?)</h2>', html_content)[1:]
    
    for i in range(0, len(sections), 2):
        if i+1 < len(sections):
            section_title = sections[i]
            section_content = sections[i+1]
            
            story.append(Paragraph(section_title, heading_style))
            
            # Extract and process tables
            tables = re.findall(r'<table>(.*?)</table>', section_content, re.DOTALL)
            for table_html in tables:
                # Parse table rows
                rows = re.findall(r'<tr>(.*?)</tr>', table_html, re.DOTALL)
                table_data = []
                
                for row in rows:
                    cells = re.findall(r'<(?:th|td)>(.*?)</(?:th|td)>', row)
                    # Clean HTML tags from cells
                    cleaned_cells = []
                    for cell in cells:
                        cell_text = re.sub(r'<[^>]+>', '', cell).strip()
                        cleaned_cells.append(Paragraph(cell_text, body_style))
                    if cleaned_cells:
                        table_data.append(cleaned_cells)
                
                if table_data:
                    # Create table
                    t = Table(table_data, colWidths=[2*inch]*len(table_data[0]))
                    t.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0066cc')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 10),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')])
                    ]))
                    story.append(t)
                    story.append(Spacer(1, 0.2*inch))
            
            # Extract paragraphs (remove tables from section)
            section_content = re.sub(r'<table>.*?</table>', '', section_content, flags=re.DOTALL)
            paragraphs = re.findall(r'<p>(.*?)</p>', section_content)
            for para in paragraphs:
                para_text = re.sub(r'<[^>]+>', '', para).strip()
                if para_text:
                    story.append(Paragraph(para_text, body_style))
            
            story.append(Spacer(1, 0.3*inch))
    
    # Add footer
    story.append(Spacer(1, 0.5*inch))
    footer = Paragraph("<i>Generated on 2026-08-15 | Bank Marketing Classification Report</i>", body_style)
    story.append(footer)
    
    # Build PDF
    doc.build(story)
    print(f"✅ PDF report generated successfully: {pdf_file}")

if __name__ == "__main__":
    markdown_file = "README_FILLED.md"
    pdf_file = "Bank_Marketing_Classification_Report.pdf"
    
    if os.path.exists(markdown_file):
        convert_markdown_to_pdf(markdown_file, pdf_file)
    else:
        print(f"❌ Error: {markdown_file} not found")
