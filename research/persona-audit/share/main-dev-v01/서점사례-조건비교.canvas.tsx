import { Stack, H1, H2, Text, Table } from "cursor/canvas";

export default function DevelopmentComparison() {
  return <Stack gap={16} style={{padding:20}}>
    <H1>서점 사례 | 구체 지시의 효과는 결정되지 않았다</H1>
    <Text>구체 지시 + Derailment의 4건 모두 연결 불확실. 네 조건의 판정이 모두 확정된 질문·seed 블록은 0/4였다.</Text>
    <H2>두 문장 사이에 연결이 불분명한 전환이 있는가?</H2>
    <Table headers={["조건 (각 4건)","있음","없음","불확실"]} rows={[
      ["범주명 · Normal",1,2,1],["범주명 · Derailment",1,3,0],
      ["구체 지시 · Normal",1,2,1],["구체 지시 · Derailment",0,0,4]
    ]}/>
    <Text>불확실 판정을 가능한 양쪽 값으로 바꾸면 상호작용 범위는 −50~+100%p이다. 통계적 신뢰구간이 아니다.</Text>
    <H2>업무 질문에서 불확실했던 한 구간</H2>
    <Text>“I leave around five, take the bus home, and spend the evening with my sister. I don’t really think about the voice when I’m working, but it sometimes comes back when I’m alone.”</Text>
    <Text>근무 중과 혼자일 때의 대비는 이해되지만, 직전의 여동생과 함께한 저녁과의 연결은 약하다.</Text>
    <H2>함께 확인한 결과</H2>
    <Text>첫 문장이 직접 답한 비율은 조건마다 3/4. 질문별 목표 사실은 32회 중 31회 명시, 1회 불확실. 이 개발 사례에서 명백한 목표 사실 모순은 없었다.</Text>
    <Text>출처: 기존 로컬 Qwen-14B 서점 발화 16개 · 2026-11-05 AI 탐색 판독. 가상 사례 1개, 기검토 발화 8개, 독립 전문의 판독 0개.</Text>
  </Stack>;
}
