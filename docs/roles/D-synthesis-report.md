# 역할 D — 평가 종합 Agent + 보고서 생성 Agent (담당: 김가연)

먼저 읽기: `AGENTS.md` → `docs/CONTRACTS.md` §2, §5, §9 → `docs/CRITERIA.md` §0, §1, §5, §6 → 이 문서.
당신은 앞 단계 결과를 **추가 검색 없이** 받아 종합하고 보고서를 만든다. 입력은 전부 State에서 오며, 이미 스키마 검증과 중복 제거를 거친 상태다.

## 1. 목표

- `run_synthesis(inp, deps) -> SynthesisResult`: 4관점 결과의 일치(`agreements`)·상충(`conflicts`)·공백(`gaps`)을 연결하고, 평가 단위 차이·근거 비대칭·반대 근거 누락을 표시한다.
- `run_report(inp, deps) -> str`: SUMMARY ~ REFERENCE 구조의 마크다운 보고서를 State의 평가 결과와 근거만으로 작성한다.
- `format_reference(e) -> str`: `source_type`별 참고문헌 포맷. REFERENCE는 본문에서 실제 인용된 evidence만.
- `render_pdf(report_md, out_path) -> str`: 마크다운 → PDF.

## 2. 소유 파일

```
src/techeval/agents/synthesis.py         # run_synthesis
src/techeval/agents/report.py            # run_report
src/techeval/report/citation.py          # format_reference, collect_cited_ids, build_reference_section
src/techeval/report/pdf.py               # render_pdf
src/techeval/report/tables.py            # 기술 × 기준 요약표 생성 (마크다운 표)
src/techeval/report/lint.py              # 금칙어·필수 챕터·REFERENCE 일치 검사 (judge 전 자체 검사)
src/techeval/report/sections.py          # 챕터·절 분리/조립 (lint, 문제 챕터만 재생성)
src/techeval/report/templates.py         # 1·2장 고정 텍스트
src/techeval/prompts/synthesis/          # system.md, agreements.md, conflicts.md, gaps.md
src/techeval/prompts/report/             # system.md, summary.md, ch1~ch6.md, ch4_{trl,market,stakeholder,domain}.md, revision.md
tests/agents/test_synthesis.py
tests/agents/test_report.py
tests/report/test_citation.py, test_lint.py, test_pdf.py, test_tables.py
tests/report/d_fixtures.py, make_d_fixtures.py, upstream_samples/   # 픽스처 로더·재생성 스크립트·B·C 샘플
tests/fixtures/synthesis.json
tests/fixtures/report_md.md
```

## 3. 상류 / 하류

- 상류: B(`tech_profiles`, `trl_eval`, `domain_eval`), C(`market_eval`, `stakeholder_eval`), E(`counter_evidence`, `evidence_gap`, `judge_result`, `SynthesisInput`/`ReportInput` 조립, `latest_by_criterion`)
- 하류: E(`evidence_gap` 검사가 `synthesis`를 읽음, judge가 `report_md`를 읽음, `render_pdf` 호출)
- 상류가 없을 때: B·C·E 픽스처(`tests/fixtures/*_eval.json`, `tech_profiles.json`, `counter_evidence.json`, `evidence_gap.json`, `judge_result.json`)로 `SynthesisInput`/`ReportInput`을 조립해 개발한다. 픽스처가 아직 없으면 CONTRACTS 스키마로 직접 최소 샘플을 만들어 `tests/fixtures/`에 두되, 소유 역할의 픽스처가 올라오면 교체한다.

## 4. 구현 순서 (권장)

1. `citation.py`, `tables.py`, `lint.py` — LLM 없이 결정적으로 동작하는 부분부터. 테스트 포함.
2. `run_synthesis` — 입력 정리(기술 × 관점 × 기준 매트릭스) → 구조화 출력 → evidence_id 실존 검증.
3. `tests/fixtures/synthesis.json`, `report_md.md` 커밋 (E가 judge/evidence_gap 개발에 사용).
4. `run_report` — 챕터별 생성(한 번에 전체 생성 금지, 챕터 단위 프롬프트) → 조립 → `lint` → 반환.
5. `render_pdf`, 통합 테스트.

## 5. 세부 요구사항

### 5.1 종합 (`run_synthesis`)
- **추가 검색 금지.** `deps.retriever`/`deps.web_search`를 호출하지 않는다(테스트에서 호출 시 실패하도록 spy 사용).
- 입력 정리: `(tech_id, perspective, criterion_id) → CriterionResult` 딕셔너리와 전체 `evidence_id → Evidence` 인덱스를 만든다. 프롬프트에는 각 기준의 `level`, `content`, `evidence_unit`, `confidence`, evidence_id 목록을 표로 넣는다.
- `conflicts`: 관점 간 상충. **S4(`details.tradeoffs`)를 반드시 기술별 최소 1개 conflict의 출발점으로 사용.** `cause`는 "기준 차이 / 평가 단위 차이(paper vs family) / 근거 유형 차이(official vs paper) / 직접성 차이(D의 L2 외삽)" 중 하나 이상으로 서술하고 **우열로 쓰지 않는다.**
- `agreements`: 2개 이상 관점이 같은 방향을 가리키는 지점.
- `gaps`: `not_public` 결과, 단일 출처(`confidence=low/medium` 근거 1개), inference-only, 평가 단위 불일치, 반대 근거 없음(`counter_evidence`가 특정 기술·기준을 다루지 않음).
- `unit_notes`: TRL(paper)과 시장·이해관계자(family)의 단위 차이가 판정에 미친 영향.
- `evidence_asymmetry_note`: 기술별 evidence 수, `official` 비율, 방향성.
- 모든 `evidence_ids`는 입력에 실존해야 한다. 코드에서 검증하고 없는 id는 제거 후 경고 로그(그 결과 `min_length` 위반이면 raise).
- `counter_evidence`가 있으면 재실행 시 이를 반영(`gaps`의 `opposing_missing` 해소 여부 갱신).

### 5.2 보고서 (`run_report`)
- 챕터 구성·분량은 `docs/CRITERIA.md` §5. 챕터별로 프롬프트를 분리하고 각 챕터 프롬프트에 필요한 데이터만 넣는다.
  - SUMMARY: `synthesis.conflicts` 상위 3~4개 → 3~4문장
  - 1장: 배경(고정 텍스트 템플릿 + 서베이 `io_survey`/`kv_survey` 근거가 `tech_profiles`·`counter_evidence`에 있으면 인용)
  - 2장: 기술 선정(고정 텍스트 템플릿, CRITERIA §7 선정 사유)
  - 3장: `tech_profiles` — 원리·정량 성과(Measurement 조건 병기)·한계, 원문 인용
  - 4장: 4.1~4.4 관점별. 각 절 말미에 `tables.criterion_table(tech_ids, criteria, results)` 로 생성한 **기술 × 기준 요약표**(레벨·신뢰도·근거 단위·evidence_id)
  - 5장: `synthesis.conflicts` 중심, 원인은 기준 차이로
  - 6장: 한계점 4항목 — TRL 공개정보 추정 한계 / 계열 단위 확장 사실(`unit_notes`) / 제안사 자료 의존 비율(`evidence_gap.vendor_source_ratio`) / 적용한 검사(재검색·반대 근거 탐색·검수)와 남은 한계(`gaps`)
  - REFERENCE: `build_reference_section(report_body, evidence_index)` — 본문의 `[E: id]` 각주만 수집
- 본문 인용 표기: `[E: mla-T1-01]`. LLM에게 이 형식을 강제하고, `lint`에서 존재하지 않는 id를 검출한다.
- 수치는 항상 조건 병기: "KV cache 93.3% 감소(DeepSeek-V2 236B, 128K, vs DeepSeek 67B MHA)[E: mla-PROFILE-02]".
- 재생성(`inp.judge_result` 존재): `inp.previous_report_md`를 기반으로 `revision_instructions`를 반영해 **문제 챕터만** 재생성한다. 전체 재작성 금지.
- 출력은 순수 마크다운 문자열. 파일 저장은 E가 한다.

### 5.3 `lint.py` (judge 전 자체 검사, 결정적)
- 필수 챕터 7개 존재(SUMMARY, 1~6, REFERENCE), 4장에 요약표 4개 존재.
- 금칙어: `더 우수`, `더 낫`, `추천`, `권장`, `1위`, `순위`, `종합 점수`, `총점`, `평균 레벨`, `우세`, `열세`. 검출 시 재생성 지시 목록을 반환(예외 아님).
- `[E: id]` 전부 evidence 인덱스에 실존. REFERENCE 항목 = 본문 인용 집합과 정확히 일치.
- 수치 패턴(`\d+(\.\d+)?%|x\b`) 근처 20자 내에 `[E:` 없으면 경고.

### 5.4 `citation.py`
- `format_reference(e)`:
  - paper: `저자(YYYY). 제목. 학술지/arXiv, 권(호), 페이지.` — 없는 필드는 생략, 페이지는 `e.page`
  - patent: `출원인(YYYY-MM). 특허명, 번호, URL`
  - web/official: `기관(YYYY-MM-DD). 제목. 사이트명, URL` — `published_date` 없으면 `(n.d., 확인일 YYYY-MM-DD)`
  - inference/not_public: REFERENCE에 넣지 않는다(본문에서 "추론"/"미공개(검색어, 검색일)"로 표기)
- 같은 논문의 여러 evidence는 REFERENCE에서 1항목으로 병합하고 페이지를 나열.

### 5.5 `pdf.py`
- `markdown` → HTML(표 확장 `tables`, `fenced_code`) → `weasyprint`로 PDF. 한글 폰트 임베드(`assets/fonts/` 또는 시스템 Noto Sans KR/Apple SD Gothic Neo 탐색). weasyprint 설치가 어려우면 `pandoc` 대체 경로를 두되 기본은 파이썬 내 변환.
- 표가 페이지를 넘어가도 깨지지 않게 CSS 지정. 출력 경로는 인자로 받고 생성된 경로를 반환.

## 6. 테스트

- 결정적 부분: `format_reference` 3종, REFERENCE 병합, `lint` 금칙어·챕터·인용 일치, 요약표 생성(30개 결과 → 4표), PDF 생성(파일 크기 > 0, 페이지 수 ≥ 5).
- 스텁 LLM: `run_synthesis`가 retriever/web_search를 호출하지 않음, 존재하지 않는 evidence_id 제거, S4 기반 conflict ≥1. `run_report`가 재생성 시 문제 챕터만 바꿈(다른 챕터 텍스트 동일).
- 통합: 픽스처 전체로 실제 LLM 1회 → `lint` 통과, PDF 생성.

## 7. 완료 기준 (DoD)

- [ ] `citation`, `tables`, `lint` 단위 테스트 통과
- [ ] `synthesis.json`, `report_md.md` 픽스처 커밋 (E 사용)
- [ ] `run_synthesis` 추가 검색 없음, evidence_id 실존 검증, conflict ≥1
- [ ] `run_report` 7챕터 + 요약표 4개 + REFERENCE 일치, 금칙어 0
- [ ] 재생성 경로(문제 챕터만) 동작
- [ ] `render_pdf` 한글 정상 출력

## 8. 하지 말 것

- 검색·추가 근거 수집. State 밖 정보(모델 사전지식)로 사실 서술.
- 레벨 합산·평균·순위·추천. "MLA가 더 성숙/유망/우수".
- REFERENCE에 본문 미인용 항목 넣기, inference/not_public을 참고문헌으로 포맷.
- 보고서를 한 번의 LLM 호출로 통째로 생성(챕터 단위로).
