# 역할 B — 기술 조사 Agent + 도메인 평가 Agent (담당: 노윤성)

먼저 읽기: `AGENTS.md` → `docs/CONTRACTS.md` §2, §3, §5, §9 → `docs/CRITERIA.md` §2.1, §2.4 → 이 문서.
당신은 **논문 원문 RAG를 가장 많이 쓰는** 두 에이전트를 만든다. 기술 조사(T1~T4)는 그래프에서 가장 먼저 실행되어 `tech_profiles`와 `trl_eval`을 만들고, 도메인 평가(D1~D4)는 그 결과를 받아 병렬 단계에서 실행된다.

## 1. 목표

- `run_tech_research(inp, deps) -> TechResearchOutput`: 기술 원리·범위·한계·정량 수치·측정 조건을 원문에서 추출해 `TechProfile`을 만들고, 검증 환경과 공개 구현 자료를 근거로 T1~T4를 평가한다.
- `run_domain_eval(inp, deps) -> list[CriterionResult]`: 배치 크기·컨텍스트 길이·하드웨어 구성 등 논문 실험 조건을 근거로, 대규모 데이터센터 장문맥 추론 도메인에 대한 근거의 **직접성**(D1~D3)과 도입 변경 범위(D4)를 판정한다.
- 수치는 반드시 `Measurement`(모델 규모·컨텍스트·HW·베이스라인·배치)로 구조화한다.

## 2. 소유 파일

```
src/techeval/agents/tech_research.py     # run_tech_research
src/techeval/agents/domain.py            # run_domain_eval
src/techeval/prompts/tech_research/      # system.md, profile.md, T1.md ... T4.md
src/techeval/prompts/domain/             # system.md, D1.md ... D4.md
tests/agents/test_tech_research.py
tests/agents/test_domain.py
tests/fixtures/tech_profiles.json        # TechProfile 2개
tests/fixtures/trl_eval.json             # CriterionResult 8개 (T1~T4 × 2)
tests/fixtures/domain_eval.json          # CriterionResult 8개 (D1~D4 × 2)
```

## 3. 상류 / 하류

- 상류: A(`deps.retriever`), C(`deps.web_search` — T3 공개 구현 확인, D4 서빙 엔진 지원 확인에 보조 사용), E(`schemas.py`, `Deps`, `AgentInput`, `compute_confidence`)
- 하류: C(`inp.tech_profile`, `inp.trl_eval` 참고), D(`tech_profiles`, `trl_eval`, `domain_eval`), E(검사 노드)
- 상류가 없을 때: `StubRetriever`, `stub_web_search`, `FakeStructuredLLM`(E의 conftest)으로 개발한다. E의 `schemas.py`가 아직 없으면 `docs/CONTRACTS.md` §2 코드를 그대로 로컬에 두고 진행하되 커밋하지 않는다.

## 4. 구현 순서 (권장)

1. **픽스처 3개를 먼저 작성해 커밋한다** (실제 논문 내용 기반, 스키마 통과). D와 C가 이걸로 개발한다.
2. `run_tech_research` 골격: 검색 질의 생성 → `retriever.search(doc_ids=[tech.primary_doc_id, ...])` → 청크 묶음 → `TechProfile` 구조화 출력 → T1~T4 구조화 출력 → `Evidence` 실존 검증 → 반환.
3. `run_domain_eval` 골격: `inp.tech_profile.measurements` 우선 활용 + 추가 검색(배치·컨텍스트·HW 키워드) → D1~D4.
4. 프롬프트 튜닝, 통합 테스트.

## 5. 세부 요구사항

### 5.1 기술 조사 (`run_tech_research`)
- 검색 전략: 기준마다 2~4개 질의(한국어 + 영어 키워드 혼용). 예 T2: "실험 환경 GPU 구성", "evaluation setup hardware H800". `inp.rewritten_queries`가 있으면 그것을 **우선** 사용(E의 질의 재작성 결과).
- 검색 범위: 기본 `doc_ids=[tech.primary_doc_id]`. 서베이(`io_survey`, `kv_survey`)는 기술 계열 맥락·한계 보강용으로 2차 검색(`unit="family"` 또는 `medium` 이하). T3(재현성)·T4는 `web_search`로 GitHub·HF·벤더 자료 확인 허용(`unit="paper"` 유지, 웹 출처는 `source_type="official"|"web"`).
- `TechProfile.public_artifacts` 4항목(`code`, `model_or_design`, `eval_scripts_data`, `third_party_reproduction`)을 Y/N/unknown으로 채우고 URL을 `public_artifact_urls`에. → T3 레벨은 **코드로** 계산(Y 개수 3~4→L3, 1~2→L2, 0→L1, 전부 unknown→L0). LLM에게 레벨을 맡기지 않는다.
- T1: `details.trl_band`, `details.estimate`, `details.why_not_higher` 필수. "공개 정보 기반 추정" 문구를 `level_estimate`에 포함.
- T2: `TechProfile.validation_env`에서 L1~L4 판정. 실서비스 트래픽(L4) 주장은 공식 자료 근거 없으면 인정하지 않는다.
- T4: `details.remaining_tasks`, `details.not_public_items` 분리. `not_public_items`에는 각 항목별 검색어를 남긴다.
- 재실행(`inp.missing_criteria` 지정): 해당 기준만 생성하고 `TechProfile`은 `inp.tech_profile`을 갱신(누락 필드만 보강). `retry_count`를 결과에 기록.
- 2회 재검색 후에도 없는 항목은 `level="not_public"` + `not_public_evidence()`.

### 5.2 도메인 평가 (`run_domain_eval`)
- D1~D3 판정은 **수치 크기가 아니라 직접성**이다. 논문 실험 조건이 "대규모 데이터센터 + 장문맥(≥32K 등) + 다중 요청"과 얼마나 맞는지: L3 직접 / L2 간접(`details.extrapolation_logic` 필수) / L1 근거 없음.
- 모든 수치는 `measurements`에 `Measurement`로. `context_length`, `hardware`, `baseline` 중 하나라도 원문에 없으면 `None`으로 두고 `condition_note`에 "원문 미기재".
- D4 체크리스트 6항목(`model_retrain`, `model_convert`, `serving_engine_change`, `hw_replace`, `memory_add`, `other`)은 Y/N/unknown. **프롬프트에 예상 답을 넣지 않는다**(예: "MLA는 재학습 필요"를 힌트로 주지 않음). 서빙 엔진 지원 여부는 `web_search`로 확인 가능(vLLM, SGLang 등).
- 두 기술에 같은 프롬프트 템플릿을 쓴다. 기술별 분기 프롬프트 금지(편향 방지).

### 5.3 공통
- 구조화 출력: `deps.llm.with_structured_output(<Model>)` → `Model.model_validate()` 재검증.
- `confidence`는 `compute_confidence(evidence)`로 계산.
- `evidence_id` 규칙 `{tech_id}-{criterion_id}-{NN}`, `TechProfile`은 `{tech_id}-PROFILE-{NN}`.
- 인용 `quote`는 청크 `text`의 부분 문자열이어야 한다(A의 `to_evidence`가 검증). LLM이 인용문을 다듬으면 실패하므로 프롬프트에 "verbatim" 요구.
- `generated_at`은 `deps.now()`.

## 6. 테스트

- 스텁 기반: 검색이 `doc_ids`를 올바르게 넘기는지, `missing_criteria=["T3"]`일 때 T3만 반환하는지, T3 레벨 계산, D1~D3 `measurements` 비어 있으면 validator 실패, D4 체크리스트 키 6개, evidence의 chunk_id가 스텁 청크에 존재.
- 통합(`integration`): 실제 retriever + LLM으로 `mla` 1회 실행 → 4개 T 결과 + TechProfile 스키마 통과, 8개 D 결과 스키마 통과.

## 7. 완료 기준 (DoD)

- [ ] 픽스처 3개 커밋 및 `tests/test_fixtures.py` 통과 (1순위)
- [ ] `run_tech_research` / `run_domain_eval` 시그니처 CONTRACTS §5와 일치
- [ ] `missing_criteria` 부분 재실행 동작
- [ ] T3 레벨 코드 계산, D1~D3 Measurement 포함, D4 힌트 미주입
- [ ] `not_public` 경로(검색어·검색일 기록) 동작
- [ ] 스텁 테스트 통과, 통합 테스트 1회 이상 실제 통과 확인

## 8. 하지 말 것

- `langgraph` import, State 직접 접근, 두 기술 루프(전부 E 담당).
- 두 기술을 비교하는 문장 생성("MLA가 PIM/CXL보다 성숙"). 각 기술은 독립적으로 평가한다.
- 검색 결과에 없는 수치·URL 생성.
- 다른 논문의 수치를 해당 기술의 근거로 사용(서베이가 다른 기법에 대해 보고한 수치를 MLA 근거로 등). 서베이는 계열 맥락·`limitations`·반대 근거에서만.
