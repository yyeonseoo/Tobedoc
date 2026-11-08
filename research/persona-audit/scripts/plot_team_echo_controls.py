"""Export the two prespecified medium-intensity contrasts and the strong-intensity check."""
from pathlib import Path
import json
from reportlab.graphics.shapes import Drawing,Rect,String,Line,Circle
from reportlab.graphics import renderSVG
from reportlab.lib.colors import HexColor
R=Path(__file__).resolve().parents[1]
data=json.loads((R/'analysis/team-echo-controls-v01/results.json').read_text())['by_source']
W,H=730,355;d=Drawing(W,H);ink='#203040';gray='#516474';bg='#ffffff'
def t(x,y,s,size=10,bold=False,color=ink,anchor='start'):
 d.add(String(x,y,s,fontName='Helvetica-Bold' if bold else 'Helvetica',fontSize=size,fillColor=HexColor(color),textAnchor=anchor))
def line(x,y,z,w,color):
 d.add(Line(x,y,z,w,strokeColor=HexColor(color),strokeWidth=.7))
def mapx(n): return 220+(n+30)*5.65
d.add(Rect(0,0,W,H,fillColor=HexColor(bg),strokeColor=None))
t(18,330,'Does lexical echo distinguish derailment from repetition?',18,True)
t(18,308,'Each marker is one matched question. Positive values indicate more echo boundaries under DER.',10)
offsets=[-6,-3,0,3,6]
for j,g in enumerate(['Claude','GPT-4.1']):
 top=275-j*105;t(18,top,g+' corpus',12,True)
 for i,k in enumerate(['DER2-CIRC2','DER2-PER2','DER3-PER3']):
  y=top-20-i*26;t(24,y-3,k.replace('-',' minus '),10)
  line(mapx(-30),y,mapx(55),y,'#e4e9eb')
  values=data[g]['contrasts'][k]['pp']
  for off,q in zip(offsets,sorted(values)):
   d.add(Circle(mapx(values[q]),y+off,2.9,fillColor=HexColor('#00736e'),strokeColor=None))
line(mapx(0),76,mapx(0),265,'#71808b')
for num in [-30,-10,0,10,30,50]:
 t(mapx(num),50,str(num),9,anchor='middle')
t(430,32,'Within-question difference in echo rate (percentage points)',10,anchor='middle')
t(18,12,'5 questions per row. DER = derailment; CIRC = circumstantiality; PER = perseveration. Existing synthetic texts, 2026-11-08.',9,color=gray)
renderSVG.drawToFile(d,str(R/'share/team-echo-controls-v01/paired-contrasts.svg'))
