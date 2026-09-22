# 기술 개요 추출

대상 기술: **{tech_name}** (`{tech_id}`)
선정 논문: {paper_title} ({paper_date})

아래 검색 결과에서 이 기술의 개요를 추출한다.

## 채울 항목

- `principle` — 핵심 접근을 2~5문장으로. 무엇을 바꿔서 KV cache 병목을 다루는지가 드러나야 한다.
- `scope` — 적용 범위와 전제 조건. 어떤 조건에서 성립하는 기법인지.
- `limitations` — **원문에 명시된** 한계만 나열한다. 네가 추론한 한계를 넣지 않는다.
- `validation_env` — 이 기술이 어디에서 검증됐는지 서술한다. 해석·수식 / 시뮬레이션 / 실물 하드웨어 시제품 / 실서비스 트래픽 중 무엇에 해당하는 근거가 원문에 있는지 그대로 옮긴다. (T2 판정의 입력이 된다)
- `measurements` — 원문이 보고한 정량 수치. 각 수치마다 `model_size`, `context_length`, `hardware`, `baseline`, `batch_size`를 채운다. **원문에 없는 조건은 비워 두고** `condition_note`에 "원문 미기재"라고 적는다. 지어내지 않는다.
- `citations` — 위 항목들의 근거.

## 검색 결과

{context}
