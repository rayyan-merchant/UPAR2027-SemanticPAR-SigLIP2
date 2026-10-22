"""Build the readable assignment PDF and matching Markdown from structured content.

No fixed page count: tables, equations and interpretation determine the length.
A Unicode font is embedded; Windows Segoe UI or DejaVu Sans is required.
"""
from pathlib import Path
import argparse,json,re
from html import escape,unescape
import reportlab
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,KeepTogether
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
INK=colors.HexColor('#172B40'); MUTED=colors.HexColor('#526373'); BLUE=colors.HexColor('#174D70')

def register_fonts():
    # Embed a Unicode font: ReportLab's default Vera lacks Greek/math glyphs.
    # Segoe UI ships with Windows; DejaVu Sans is common on Linux.
    candidates=[(Path('C:/Windows/Fonts'),['segoeui.ttf','segoeuib.ttf','segoeuii.ttf']),
                (Path('/usr/share/fonts/truetype/dejavu'),['DejaVuSans.ttf','DejaVuSans-Bold.ttf','DejaVuSans-Oblique.ttf'])]
    try:
        from matplotlib import get_data_path
        candidates.append((Path(get_data_path())/'fonts/ttf',['DejaVuSans.ttf','DejaVuSans-Bold.ttf','DejaVuSans-Oblique.ttf']))
    except ImportError:
        pass
    for folder,names in candidates:
        if all((folder/name).exists() for name in names):
            for name,file in zip(['Report','Report-Bold','Report-Italic'],names):
                pdfmetrics.registerFont(TTFont(name,str(folder/file)))
            pdfmetrics.registerFontFamily('Report',normal='Report',bold='Report-Bold',italic='Report-Italic',boldItalic='Report-Bold')
            return
    raise RuntimeError('Install DejaVu Sans or the optional matplotlib package to supply Unicode report fonts.')

def plain_math(text):
    text=re.sub(r'<sub>(.*?)</sub>',lambda m:'_{'+m[1]+'}',text)
    text=re.sub(r'<super>(.*?)</super>',lambda m:'^{'+m[1]+'}',text)
    return unescape(text)

def markdown(content):
    parts=['# '+content['title'],content['subtitle']]
    if content.get('group_members'):
        parts.append('Group members: '+ '; '.join(x['name']+' ('+x['roll_number']+')' for x in content['group_members']))
    for section in content['pages']:
        parts += ['## '+section['title']]
        for b in section['blocks']:
            kind=b['kind']
            if kind=='heading':parts.append('### '+b['text'])
            elif kind=='table':
                parts.append('\n'.join(['| '+' | '.join(b['headers'])+' |','| '+' | '.join(['---']*len(b['headers']))+' |']+['| '+' | '.join(row)+' |' for row in b['rows']]))
                if b.get('caption'):parts.append('*'+b['caption']+'*')
            elif kind=='equation':
                # The PDF provides typeset subscripts/superscripts; keep a readable
                # matching HTML expression rather than printing raw LaTeX.
                parts += ['<div align="center">'+b['text'].replace('<super>','<sup>').replace('</super>','</sup>')+'</div>',b['note']]
            else:parts.append(b['text'])
    return '\n\n'.join(parts)+'\n'

def build(output):
    register_fonts()
    content=json.loads((ROOT/'assignment/report_content.json').read_text(encoding='utf-8'))
    styles={
        'title':ParagraphStyle('Title',fontName='Report-Bold',fontSize=22,leading=28,textColor=INK,spaceAfter=10),
        'subtitle':ParagraphStyle('Subtitle',fontName='Report',fontSize=9.2,leading=14,textColor=MUTED,spaceAfter=20),
        'chapter':ParagraphStyle('Chapter',fontName='Report-Bold',fontSize=15,leading=20,textColor=BLUE,spaceBefore=18,spaceAfter=8,keepWithNext=True),
        'kicker':ParagraphStyle('Kicker',fontName='Report-Bold',fontSize=7.5,leading=11,textColor=MUTED,spaceAfter=8,keepWithNext=True),
        'heading':ParagraphStyle('Heading',fontName='Report-Bold',fontSize=10.7,leading=15,textColor=INK,spaceBefore=10,spaceAfter=6,keepWithNext=True),
        'body':ParagraphStyle('Body',fontName='Report',fontSize=9.7,leading=14.1,spaceAfter=8,textColor=INK,allowWidows=0,allowOrphans=0),
        'note':ParagraphStyle('Note',fontName='Report',fontSize=8.2,leading=11.8,spaceAfter=9,textColor=MUTED),
        'equation':ParagraphStyle('Equation',fontName='Report',fontSize=9.1,leading=20,alignment=1,textColor=INK),
        'th':ParagraphStyle('TH',fontName='Report-Bold',fontSize=7.6,leading=10.8,textColor=colors.white),
        'td':ParagraphStyle('TD',fontName='Report',fontSize=8,leading=11.6,textColor=INK),
    }
    width=A4[0]-100
    story=[Paragraph(escape(content['title']),styles['title']),Paragraph(escape(content['subtitle']),styles['subtitle'])]
    if content.get('group_members'):
        story.append(Paragraph('<br/>'.join(escape(x['name']+' | '+x['roll_number']) for x in content['group_members']),styles['subtitle']))
    for section in content['pages']:
        story += [Paragraph(escape(section['title']),styles['chapter']),Paragraph(escape(section['kicker']),styles['kicker'])]
        for block in section['blocks']:
            kind=block['kind']
            if kind=='table':
                data=[[Paragraph(escape(str(c)),styles['th'] if i==0 else styles['td']) for c in row] for i,row in enumerate([block['headers']]+block['rows'])]
                ratios=block['widths']; total=sum(ratios)
                tab=Table(data,colWidths=[width*x/total for x in ratios],repeatRows=1,hAlign='LEFT')
                tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),BLUE),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#F0F4F7'),colors.white]),('LINEBELOW',(0,0),(-1,0),.5,BLUE),('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#C6D0D8')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
                story += [Spacer(1,4),tab,Spacer(1,7)]
                if block.get('caption'):story.append(Paragraph(escape(block['caption']),styles['note']))
            elif kind=='equation':
                text=block['text']
                if text.startswith('L ='):text=text.replace('[w<sub>a</sub>','<br/>[w<sub>a</sub>',1)
                box=Table([[Paragraph(text,styles['equation'])]],colWidths=[width])
                box.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#F0F4F7')),('BOX',(0,0),(-1,-1),.4,colors.HexColor('#D5DFE7')),('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8)]))
                group=[box,Spacer(1,5),Paragraph(escape(block['note']),styles['note'])]
                if story and isinstance(story[-1],Paragraph) and story[-1].style.name=='Heading':
                    group.insert(0,story.pop())
                story.append(KeepTogether(group))
            else:
                style=styles['heading' if kind=='heading' else 'note' if kind=='references' else 'body']
                story.append(Paragraph(escape(block['text']),style))
    def footer(canvas,doc):
        canvas.saveState();canvas.setStrokeColor(colors.HexColor('#C6D0D8'));canvas.line(50,40,A4[0]-50,40)
        canvas.setFont('Report',7.4);canvas.setFillColor(MUTED)
        canvas.drawString(50,27,'UPAR 2027 · Research methods and experimental results')
        canvas.drawRightString(A4[0]-50,27,str(doc.page));canvas.restoreState()
    output.parent.mkdir(parents=True,exist_ok=True)
    SimpleDocTemplate(str(output),pagesize=A4,leftMargin=50,rightMargin=50,topMargin=44,bottomMargin=54,title=content['title'],author='',subject='Assignment methodology, controlled evaluation and interpretation').build(story,onFirstPage=footer,onLaterPages=footer)
    (ROOT/'assignment/UPAR2027_Assignment_Report.md').write_text(markdown(content),encoding='utf-8')
    reader=PdfReader(output)
    lengths=[len(p.extract_text() or '') for p in reader.pages]
    if not lengths or min(lengths)<150:raise ValueError('Unexpected empty report page')
    print(json.dumps({'pdf':str(output),'pages':len(reader.pages),'text_characters_by_page':lengths},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'assignment/UPAR2027_Assignment_Report.pdf');a=p.parse_args();build(a.output)
