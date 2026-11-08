# 연구 이력과 인계 전 산출물

현재 후속 연구의 원문·설명서·분석 조건·익명 판독표·코드·결과와 2쪽 회의 보고서는 상위 연구 폴더에 일반 파일로 들어 있다. 원래 로컬 Git 이력은 followup-history.bundle에 보존했다. 새 폴더에서 `git clone followup-history.bundle -b main restored-study`로 회의 보고서 작성까지의 커밋을 복원할 수 있다.

인계 전에 만들어진 로컬 연구 산출물(코드·테스트·실험 입력·실행 기록·결과·이전 연구 보고서)은 pre-handoff-research.tar.xz에 원래 상대경로로 보존했다. 연구 폴더에서 `tar -xJf archive/pre-handoff-research.tar.xz`로 풀면 이전 개발 응답을 참조하는 검산 스크립트도 실행할 수 있다. pre-handoff-inventory.json에는 각 파일의 SHA-256과 제외 내역이 있다. 외부 논문 전문·외부 벤치마크 원문·재생성 가능한 임베딩 캐시·중복 페이지 검수 이미지·시스템 파일은 연구 결과물이 아니므로 배포하지 않았다. 외부 자료는 기록된 원출처를 따른다.
