"""Evidence-led overview: native SVG/ReportLab composition with Matplotlib data panels.
Run locally or from figure_reproduction/scripts; inputs are small, bundled CSVs.
"""
from pathlib import Path
from io import BytesIO
import csv, shutil, subprocess, html, xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pypdf import PdfReader,PdfWriter,Transformation
HERE=Path(__file__).resolve().parent
if (HERE.parent/'data').exists():
 ROOT=HERE.parent; DATA=ROOT/'data'; OUT=ROOT/'output'
else:
 ROOT=HERE.parent;DATA=ROOT/'figure_reproduction/data';OUT=ROOT/'figures'
OUT.mkdir(exist_ok=True)
WORK=OUT/'overview_parts';WORK.mkdir(exist_ok=True)
PT=72/25.4; W=180;H=124
BLUE='#0072B2';ORANGE='#D55E00';INK='#222222';GRAY='#777777'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
 'font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,
 'axes.linewidth':.6,'lines.linewidth':1.25,'pdf.fonttype':42,'ps.fonttype':42,
 'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,
 'xtick.direction':'out','ytick.direction':'out','savefig.facecolor':'white'})
rows=list(csv.DictReader((DATA/'from_zero/evaluation/analysis/CURVE_MEANS_20.csv').open()))
full=list(csv.DictReader((DATA/'from_zero/evaluation/analysis/FULL_BUDGET_CURVE_80.csv').open()))
fig,axs=plt.subplots(1,2,figsize=(W/25.4,70/25.4))
fig.subplots_adjust(left=.11,right=.98,bottom=.19,top=.83,wspace=.43)
for arm,color,marker,label in [('price',BLUE,'o','Price-volume'),('wide',ORANGE,'s','Expanded information')]:
 d=sorted([x for x in rows if x['information_set']==arm and int(x['budget'])>=8],key=lambda x:int(x['budget']))
 assert [int(x['budget']) for x in d]==[8,16,24,26,28,30,32,34,36],arm
 assert len({x['seed'] for x in full if x['information_set']==arm})==4,arm
 for ax,col,scale in [(axs[0],'frontier_best_ic',1),(axs[1],'arithmetic_annual_return',100)]:
  for seed in sorted({x['seed'] for x in full if x['information_set']==arm}):
   a=sorted([x for x in full if x['information_set']==arm and x['seed']==seed and int(x['budget'])>=8],key=lambda x:int(x['budget']))
   assert len(a)==9,(arm,seed)
   ax.plot([int(x['budget']) for x in a],[float(x[col])*scale for x in a],color=color,alpha=.17,lw=.65)
  ax.plot([int(x['budget']) for x in d],[float(x[col])*scale for x in d],color=color,marker=marker,ms=3,label=label)
for ax in axs:
 ax.set_xlim(7,37);ax.set_xticks([8,16,24,36]);ax.set_xlabel('Candidate proposals, including initialization')
 ax.axvline(24,color='#999999',ls=':',lw=.7,zorder=0)
axs[0].set_ylabel('Best training IC');axs[0].set_ylim(.044,.089)
axs[1].set_ylabel('Annual net return (%)');axs[1].set_ylim(8,21)
for ax,letter,title in zip(axs,['b','c'],['Search frontier','Portfolio value']):
 ax.text(-.24,1.17,letter,transform=ax.transAxes,fontweight='bold',fontsize=10)
 ax.set_title(title,loc='left',fontsize=9,pad=12)
axs[0].legend(frameon=False,loc='upper left',fontsize=7,handlelength=2)
for ext in ['pdf','svg']:fig.savefig(WORK/f'budget_panels.{ext}')
plt.close(fig)
font='Helvetica';fontb='Helvetica-Bold'
fp=Path('/System/Library/Fonts/Supplemental/Arial.ttf')
if fp.exists():
 pdfmetrics.registerFont(TTFont('FigureArial',str(fp)));pdfmetrics.registerFont(TTFont('FigureArialBold',str(fp.with_name('Arial Bold.ttf'))));font='FigureArial';fontb='FigureArialBold'

def make_top(variant):
 buf=BytesIO();c=canvas.Canvas(buf,pagesize=(W*PT,H*PT));svg=[]
 def txt(x,y,t,size=8,bold=False):
  c.setFont(fontb if bold else font,size);c.setFillColorRGB(.13,.13,.13);c.drawString(x*PT,(H-y)*PT,t)
  svg.append(f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" font-size="{size/PT:.3f}" font-weight="{"bold" if bold else "normal"}" fill="{INK}">{html.escape(t)}</text>')
 def line(x1,y1,x2,y2,color='#b9b9b9',lw=.2):
  from reportlab.lib.colors import HexColor
  c.setStrokeColor(HexColor(color));c.setLineWidth(lw*PT);c.line(x1*PT,(H-y1)*PT,x2*PT,(H-y2)*PT)
  svg.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{lw}"/>')
 txt(1,5,'a',10,True);txt(9,5,'Three comparisons of the return to search effort',9,True)
 if variant=='A':
  entries=[(1,'Search policy',['Same initial library','Different proposal strategies','Common model and portfolio']),
   (63,'Information supply',['Start without learned formulas','Price-volume or added fields','Include initialization in the budget']),
   (125,'Formula composition',['Same raw inputs in both models','Vary fields allowed in formulas','Separate inputs from transformations'])]
  for x,title,lines in entries:
   txt(x,16,title,8.5,True);line(x,19,x+53,19)
   for j,t in enumerate(lines):txt(x,25+5*j,t,7.5)
  line(1,42,178,42)
  txt(1,48,'Observed budget curves: training progress and investment value can diverge',8,True)
 else:
  txt(2,15,'Comparison',8,True);txt(54,15,'Held common',8,True);txt(113,15,'Varied',8,True)
  for y,a,b,d in [(24,'Search policy','Initial library and evaluator','Proposal strategy'),(32,'Information supply','Final model dimension','Available field dictionary'),(40,'Formula composition','Direct raw-model inputs','Fields usable in formulas')]:
   line(2,y-5,177,y-5);txt(2,y,a,7.5);txt(54,y,b,7.5);txt(113,y,d,7.5)
  txt(2,49,'Measured along the budget: best training score and portfolio return',8,True)
 c.save();return buf.getvalue(),svg
for v in ['A','B']:
 data,top=make_top(v);page=PdfReader(BytesIO(data)).pages[0];lower=PdfReader(WORK/'budget_panels.pdf').pages[0]
 page.merge_transformed_page(lower,Transformation().translate(tx=0,ty=0));wr=PdfWriter();wr.add_page(page)
 name='research_overview' if v=='A' else 'overview_variant_B'
 with (OUT/f'{name}.pdf').open('wb') as f:wr.write(f)
 ns='http://www.w3.org/2000/svg';ET.register_namespace('',ns)
 root=ET.Element('{'+ns+'}svg',{'width':f'{W}mm','height':f'{H}mm','viewBox':f'0 0 {W} {H}'})
 g=ET.SubElement(root,'{'+ns+'}g',{'id':'experimental-comparisons'})
 for s in top:g.append(ET.fromstring(s))
 lower=ET.parse(WORK/'budget_panels.svg').getroot();vb=[float(x) for x in lower.attrib['viewBox'].split()]
 lower.set('x','0');lower.set('y',str(H-70));lower.set('width',str(W));lower.set('height','70');lower.set('id','observed-budget-curves');root.append(lower)
 ET.ElementTree(root).write(OUT/f'{name}.svg',encoding='utf-8',xml_declaration=True)
 pop=shutil.which('pdftoppm')
 if pop:subprocess.run([pop,'-singlefile','-png','-scale-to','1800',str(OUT/f'{name}.pdf'),str(OUT/name)],check=True,capture_output=True)
# Restore the historical long-search evidence as an independent data figure.
d=list(csv.DictReader((DATA/'historical/SEARCH_GENERATION.csv').open()));x=np.array([int(z['generation']) for z in d]);q=np.array([float(z['cumulative_best_prediction_A1']) for z in d]);tokens=np.cumsum([float(z['output_tokens']) for z in d])/1e6
fig,axs=plt.subplots(1,2,figsize=(180/25.4,64/25.4));fig.subplots_adjust(left=.12,right=.97,bottom=.2,top=.85,wspace=.45)
axs[0].step(x,q,where='post',color=BLUE);axs[0].set_ylabel('Best historical selection score');axs[0].ticklabel_format(axis='y',style='plain',useOffset=False)
axs[1].plot(x,tokens,color=BLUE);axs[1].set_ylabel('Cumulative output (million tokens)')
for a,letter,title in zip(axs,['a','b'],['Training frontier','Research effort']):
 a.set_xlim(0,21);a.set_xticks([0,6,13,21]);a.set_xlabel('Search iteration');a.axvline(13,color=GRAY,ls=':',lw=.8);a.set_title(title,loc='left',fontsize=9);a.text(-.27,1.1,letter,transform=a.transAxes,fontweight='bold',fontsize=10)
for ext in ['pdf','svg','png']:fig.savefig(OUT/f'historical_frontier.{ext}',dpi=260)
plt.close(fig)
print('Overview variants and historical frontier saved:',OUT)
