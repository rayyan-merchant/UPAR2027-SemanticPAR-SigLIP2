"""Render the three-page assignment PDF from its editable structured content."""
from pathlib import Path
import argparse,json
from html import escape
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT,TA_CENTER
from reportlab.lib.pagesizes import letter
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
def build(output):
    content=json.loads((ROOT/'assignment/report_content.json').read_text(encoding='utf-8'))
    title=ParagraphStyle('Title',fontName='Helvetica-Bold',fontSize=18.5,leading=22,textColor=colors.black,spaceAfter=6)
    subtitle=ParagraphStyle('Subtitle',fontName='Helvetica',fontSize=9,leading=12,textColor=colors.HexColor('#444444'),spaceAfter=12)
    heading=ParagraphStyle('Heading',fontName='Helvetica-Bold',fontSize=11.5,leading=14,textColor=colors.black,spaceBefore=7,spaceAfter=5,keepWithNext=True)
    body=ParagraphStyle('Body',fontName='Times-Roman',fontSize=11,leading=14,spaceAfter=7,textColor=colors.black)
    caption=ParagraphStyle('Caption',fontName='Helvetica',fontSize=8.2,leading=10.5,spaceBefore=5,spaceAfter=8)
    th=ParagraphStyle('TH',fontName='Helvetica-Bold',fontSize=8.1,leading=10.4,textColor=colors.white)
    td=ParagraphStyle('TD',fontName='Helvetica',fontSize=8.7,leading=11.2)
    story=[]
    for idx,page in enumerate(content['pages']):
        story += [Paragraph(escape(page['title']),title),Paragraph(escape(page['subtitle']),subtitle)]
        for name,paragraphs in page['sections']:
            story.append(Paragraph(escape(name),heading))
            story += [Paragraph(escape(p),body) for p in paragraphs]
            if name=='Measured calibrated transfer':
                data=[[Paragraph(escape(c),th if i==0 else td) for c in row] for i,row in enumerate(content['table'])]
                table=Table(data,colWidths=[157,70,63,70,70,70],repeatRows=1,hAlign='LEFT')
                table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#303B47')),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#D9D9D9')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F3F5F7')]),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
                story += [table,Paragraph('Local Challenge_Avg is the per-domain harmonic mean of mA (label-balanced accuracy) and instance-F1, then averaged equally over domains. Local F1 uses harmonic mean of mean-image precision/recall; exact organizer edge cases remain unresolved. These are not official hidden-test scores.',caption)]
        if idx<2:story.append(PageBreak())
        else:
            story.append(Paragraph('Evidence and references: repository assignment/EXPERIMENTS_AND_DECISIONS.md and RESULTS.csv; official task github.com/speckean/UPAR-Challenge-2027; competition codabench.org/competitions/18180; foundations Google SigLIP2-Base-NaFlex and timm DINOv3 ViT-B/16 LVD-1689M. Source links and method descriptions are included in the report text.',caption))
    def footer(canvas,doc):
        canvas.saveState();canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#555555'))
        canvas.drawString(46,25,'UPAR 2027 Track 1  |  Assignment report')
        canvas.drawRightString(566,25,f'{doc.page} / 3');canvas.restoreState()
    output.parent.mkdir(parents=True,exist_ok=True)
    SimpleDocTemplate(str(output),pagesize=letter,leftMargin=46,rightMargin=46,topMargin=39,bottomMargin=40,title='Pedestrian attribute recognition under domain shift',author='',subject='UPAR 2027 assignment methodology and experimental results').build(story,onFirstPage=footer,onLaterPages=footer)
    reader=PdfReader(output)
    if len(reader.pages)!=3:raise ValueError(f'Report overflow: expected 3 pages, found {len(reader.pages)}')
    for i,page in enumerate(reader.pages):
        text=page.extract_text()
        if not text or len(text)<1000:raise ValueError(f'Unexpected empty/short page {i+1}')
    print(json.dumps({'pdf':str(output),'pages':len(reader.pages),'text_characters_by_page':[len(p.extract_text()) for p in reader.pages]},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'assignment/UPAR2027_Assignment_Report.pdf');a=p.parse_args();build(a.output)
