"""Build the two-page Korean meeting report."""
from pathlib import Path
from docx import Document
from docx.shared import Pt,Mm,RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
R=Path(__file__).resolve().parents[1]; OUT=R/'share/meeting-20261108'; OUT.mkdir(parents=True,exist_ok=True)
doc=Document();sec=doc.sections[0];sec.page_width=Mm(210);sec.page_height=Mm(297);sec.left_margin=sec.right_margin=Mm(22);sec.top_margin=Mm(21);sec.bottom_margin=Mm(18)
navy='152D39';teal='16695F'
font='Apple SD Gothic Neo'
def style(st,n,col,bold=False):
 st.font.name=font;st.font.size=Pt(n);st.font.color.rgb=RGBColor.from_string(col);st.font.bold=bold
 st.element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),font)
for k,n,col,b in [('Normal',11,'202F34',False),('Title',24,'111111',True),('Heading 1',20,navy,True),('Heading 2',15,navy,True)]:style(doc.styles[k],n,col,b)
doc.styles['Normal'].paragraph_format.space_after=Pt(6);doc.styles['Normal'].paragraph_format.line_spacing=1.31
def p(s):return doc.add_paragraph(s)
def small(s): 
 x=p(s);x.runs[0].font.size=Pt(9);x.runs[0].font.color.rgb=RGBColor.from_string(teal);return x
def h(s):doc.add_heading(s,2)
def tab(rows,widths):
 t=doc.add_table(rows=0,cols=len(widths));t.autofit=False;t.alignment=WD_TABLE_ALIGNMENT.CENTER
 for c,w in zip(t.columns,widths): c.width=Mm(w)
 for i,row in enumerate(rows):
  cs=t.add_row().cells
  for c,w,txt in zip(cs,widths,row):
   c.width=Mm(w);c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
   sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'DDECE8' if i==0 else ('F2F6F6' if i%2 else 'FFFFFF'));c._tc.get_or_add_tcPr().append(sh)
   a=c.paragraphs[0];a.paragraph_format.space_before=Pt(4);a.paragraph_format.space_after=Pt(4);r=a.add_run(txt);r.bold=(i==0);r.font.size=Pt(10)
small('투비닥터 연구팀  /  중간 회의  /  2026년 11월 8일')
doc.add_paragraph('합성 면담 발화의 사고 전환 분석',style='Title')
p('반복을 유도하면 같은 표현은 늘었다. 그러나 반복을 많이 쓴 답변이 곧 생각 사이의 연결이 약한 답변은 아니었다. 또 ‘더 강한 탈선’을 지시해도 실제 발화에서 불명확한 전환이 일관되게 늘지는 않았다.')
h('분석 자료와 판독 방법')
p('같은 가상 인물에 대해 여섯 질문과 19개 조건을 조합한 팀 발화 228개를 확보했다. Claude 114개는 대화 중 시범 작성 뒤 일부 지시를 조정했고, GPT-4.1 114개는 API로 생성했다. 두 자료를 분리해 같은 질문끼리 비교했다. 각 질문·조건당 발화는 하나다.')
p('전환은 “앞뒤 구절은 각각 이해되지만 둘 사이의 관계가 불분명한 곳이 한 군데 이상 있는가”로 정의했다. 기준 조정에 사용한 가족 질문은 강도 비교에서 제외했다. AI가 조건과 출처를 가리고 판독한 뒤 조건표를 연결했다.')
h('강도를 높였을 때의 결과')
tab([['자료 · 조건당 5개','다주제 대조','중간 탈선 지시','강한 탈선 지시'],['Claude','0 / 0 / 5','5 / 0 / 0','5 / 0 / 0'],['GPT-4.1','0 / 0 / 5','0 / 4 / 1','1 / 3 / 1']],[45,39,41,41])
small('각 칸은 ‘불명확한 전환 있음 / 불확실 / 없음’ 순서의 발화 수.')
p('GPT-4.1에서 중간 지시 때 불확실했던 네 질문 중 영화만 ‘있음’으로 바뀌었다. 주말과 성장 지역은 불확실로 남았고, 요리는 ‘없음’이 됐다. Claude는 중간 지시부터 모두 ‘있음’이어서 이 지표로는 추가 강도를 구별할 수 없다.')
doc.add_page_break()
small('투비닥터 연구팀  /  중간 회의  /  결과 해석과 다음 결정')
doc.add_heading('무엇을 측정하고 무엇을 검증할 것인가',1)
h('반복 지표의 쓰임과 한계')
p('강한 반복 지시에서 기본 조건보다 3단어 묶음 반복률이 여섯 질문 모두 높아졌다. 질문별 평균 차이는 Claude +26.94%p, GPT-4.1 +8.44%p였다. 두 주 비교의 다중검정 보정 후 p값은 각각 0.0625로, 5% 유의성에는 이르지 않았다.')
p('별도로 인접 구절에서 같은 단어를 되받는 비율도 측정했다. 중간 반복 지시는 다주제 대조보다 두 자료 모두 5/5 질문에서 높았다. 중간 탈선과 중간 반복을 직접 비교하면 탈선 쪽이 높은 질문은 Claude 5/5, GPT-4.1 1/5였다. 되받기는 문체를 기술하는 데 쓰되, 탈선 단독 지표로는 쓰지 않는다.')
p('원문 예시: 같은 답변에서 ‘야간 택시 일 → 야간 택시 일의 위험’은 단어가 반복돼도 자연스럽다. 반대로 ‘뱀의 허물(skins) → 인체 피부(skin)의 크기’는 생각의 연결이 약하지만 단수·복수 차이 때문에 동일 단어 규칙이 놓쳤다.')
h('본 실험에 남은 질문')
p('“범주명만 제시할 때보다 구체적 지시를 덧붙일 때 불명확한 전환이 늘어나는가?” 별도 서점 사례의 기존 발화 16개에서는 구체 지시+탈선 4개가 모두 불확실했다. 네 조건을 모두 판정할 수 있는 묶음은 0/4였고, 효과의 가능 범위는 −50~+100%p였다. 효과가 없다는 뜻은 아니다.')
h('회의에서 결정할 사항')
p('① 동일 사례·질문·모델로 고정한 본 실험을 실행할지 결정한다. 개발 사례 대조조건 8개와 미사용 가상 사례 4개의 확인용 96개는 아직 생성하지 않았다.')
p('② 전문의가 AI 판정을 보기 전 독립적으로 읽을 표본과 자문 일정을 정한다. 96개 생성 시 사전 선정한 24개를 검토할 계획이다. 현재 전문의 판독은 0건이며, 이 결과를 임상 타당도나 모델 성능 순위로 해석하지 않는다.')
doc.core_properties.title='합성 면담 발화의 사고 전환 분석';doc.core_properties.author='투비닥터 연구팀'
for root in [doc.styles.element,doc.element]:
 for f in root.iter(qn('w:rFonts')):
  for k in list(f.attrib):
   if 'theme' in k.lower(): del f.attrib[k]
  for k in ('ascii','hAnsi','eastAsia','cs'): f.set(qn('w:'+k),font)
for b in list(doc.styles['Title'].element.iter(qn('w:pBdr'))): b.getparent().remove(b)
doc.save(OUT/'중간회의_연구보고서.docx')
