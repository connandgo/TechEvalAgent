# 역할 E — LangGraph / State / 품질 제어 + 통합 (담당: 백승현)

먼저 읽기: `AGENTS.md` → `docs/CONTRACTS.md` 전체 → `docs/CRITERIA.md` §4, §6 → 이 문서.
당신은 공통 인터페이스의 관리자이자 전체 흐름의 소유자다. 이 설계의 차별점은 에이전트 개수가 아니라 **Loop/Branch/검수 구조**이며, 그 흐름 전부가 당신 코드다. 제어 노드는 의견을 생성하지 않고 흐름과 품질만 관리한다.

## 1. 목표

- `schemas.py`, `state.py`, `config.py`, `Deps`/`AgentInput`을 CONTRACTS.md와 1:1로 구현하고 가장 먼저 커밋한다.
- `graph.py`: `StateGraph`, `Send` fan-out(기술별), `operator.add` fan-in, 조건부 Edge, 재시도 상한.
- 제어 노드 6개: 기술 근거 검사 / 질의 재작성 / 관점별 근거 검사 / 근거 비대칭 검사 / 반대 근거 탐색 / 보고서 검수(Judge).
- `scripts/run.py`: 전체 실행 CLI(`--stub`, `--no-cache`, `--out`), 실행 로그·중간 State 덤프.
- 통합·E2E 테스트, 픽스처 검증(`tests/test_fixtures.py`), `FakeStructuredLLM`(`tests/conftest.py`).

## 2. 소유 파일

```
src/techeval/config.py                   # Settings (pydantic-settings 또는 dotenv), 모델 팩토리 get_llm()/get_judge_llm()
src/techeval/schemas.py                  # CONTRACTS §1~§2 전체 + compute_confidence() + TECHNOLOGIES 상수
src/techeval/state.py                    # GraphState + latest_by_criterion() + 초기 State 빌더
src/techeval/agents/_deps.py             # Deps, AgentInput, TechResearchOutput, SynthesisInput, ReportInput
src/techeval/graph.py                    # build_graph(deps) -> CompiledGraph
src/techeval/control/
├── tech_evidence_check.py
├── query_rewrite.py
├── perspective_check.py
├── evidence_gap.py
├── counter_evidence.py
└── judge.py
src/techeval/prompts/control/            # query_rewrite.md, judge.md
scripts/run.py
tests/conftest.py                        # FakeStructuredLLM, deps_stub 픽스처, now() 고정
tests/test_fixtures.py                   # 모든 tests/fixtures/* 스키마 검증
tests/control/                           # 제어 노드 단위 테스트
tests/test_graph.py                      # 스텁 deps로 그래프 흐름·분기·상한 테스트
tests/test_e2e.py                        # integration: 실제 실행
tests/fixtures/state_initial.json, counter_evidence.json, evidence_gap.json, judge_result.json
.env.example
docs/CONTRACTS.md
```

## 3. 상류 / 하류

- 상류: 전원(A retriever, C web_search, B/C/D 에이전트 함수).
- 하류: 전원(schemas, Deps, AgentInput, conftest). **당신의 첫 커밋이 늦으면 모두가 멈춘다.** Day 1에 `schemas.py`, `state.py`, `_deps.py`, `conftest.py`, `test_fixtures.py`, `.env.example`을 올린다.

## 4. 구현 순서 (권장)

1. `schemas.py`, `state.py`, `_deps.py`, `config.py`, `conftest.py`(FakeStructuredLLM), `test_fixtures.py`, `.env.example` → 커밋.
2. `graph.py` 골격: 모든 노드를 **픽스처 반환 스텁**으로 연결해 `scripts/run.py --stub`이 END까지 도달. 조건부 Edge·상한 테스트(`tests/test_graph.py`).
3. 제어 노드 구현 (검사 노드 → 질의 재작성 → 비대칭/반대 근거 → judge).
4. 실제 에이전트 함수를 노드에 연결(각 역할의 PR 머지 순서대로 교체).
5. E2E, 로그·State 덤프, 성능(총 실행 시간·LLM 호출 수) 측정.

## 5. 세부 요구사항

### 5.1 `schemas.py` / `state.py` / `_deps.py`
- CONTRACTS.md 코드를 그대로. validator 포함. `PerspectiveResult = CriterionResult` 별칭 유지.
- `compute_confidence(evidence)`: `inference`/`not_public` 포함 → `low`; 독립 출처(서로 다른 `locator`의 도메인 또는 doc_id) ≥2 이고 `paper`/`official` 포함 → `high`; 그 외 → `medium`.
- `TECHNOLOGIES`: `mla`(primary_doc_id=`deepseek_v2`, family="MLA 계열", aliases 포함), `pim_cxl`(primary_doc_id=`pim_cxl_1m`, family="CXL·PIM/PNM 메모리 계열").
- `latest_by_criterion(results) -> list[CriterionResult]`: `(tech_id, criterion_id)`별 `generated_at` 최대 1개. `tech_profiles`도 `latest_by_tech`.
- `build_initial_state() -> GraphState`: `domain`, `technologies`, `stakeholders`, 빈 리스트/딕셔너리, `retry_counts` 0.

### 5.2 `graph.py`
- 노드: `init`, `tech_research`, `tech_evidence_check`, `query_rewrite`, `market_eval`, `stakeholder_eval`, `domain_eval`, `perspective_check`, `synthesis`, `evidence_gap_check`, `counter_evidence_search`, `report`, `judge`, `render_pdf`.
- fan-out: `Send("tech_research", {"tech": t, ...AgentInput 필드})` × 2. 에이전트 노드는 `AgentInput.model_validate(payload)` → `run_xxx(inp, deps)` → `{"trl_eval": [...], "tech_profiles": [...]}` 형태로 반환(부분 State).
- 병렬 관점: `tech_evidence_check` 충분 시 `[Send("market_eval",..)*2, Send("stakeholder_eval",..)*2, Send("domain_eval",..)*2]` 6개 동시. fan-in 후 `perspective_check`.
- 조건부 Edge:
  - `tech_evidence_check` → 부족 & `retry_counts["trl:<tech>"] < 2` → `query_rewrite` → 해당 기술만 `Send`. 상한이면 부족 항목 `not_public` 확정 후 진행.
  - `perspective_check` → 부족 기준 분기: T → `tech_research`(해당 기술), M/S/D → 해당 에이전트(해당 기술, `missing_criteria` 지정). 기준별 상한 2. 상한 도달 시 `gaps` 후보로 표시하고 진행.
  - `evidence_gap_check` → `needs_counter_search & retry_counts["counter"] < 1` → `counter_evidence_search` → `synthesis` 재실행.
  - `judge` → `not passed & retry_counts["report"] < 1` → `report`(judge_result 전달). 상한이면 그대로 `render_pdf`.
- 종합·보고서 입력 조립 시 `latest_by_criterion` 적용(중복 제거). D는 중복을 모른다.
- `recursion_limit` 명시 설정(예 60). 무한 루프 방지는 `retry_counts`로.
- 노드 시작/종료·분기 결과·재시도 카운트를 `INFO` 로그. `--dump-state` 시 노드별 State를 `outputs/state/<step>_<node>.json`으로 저장.

### 5.3 제어 노드
- **`tech_evidence_check`**: 각 tech의 `TechProfile` 필수 필드(`principle`, `limitations`≥1, `measurements`≥1, `validation_env`) + T1~T4 존재 + 각 evidence `chunk_id`가 `retriever.get_chunk()`에 실존 + `quote` 부분 문자열(V6). 부족하면 `missing_criteria["trl"]`에 기준 또는 `PROFILE:<field>`를 기록.
- **`query_rewrite`**: 부족 항목별 대체 검색어 2~3개. LLM 사용 가능(`prompts/control/query_rewrite.md`), 이전 검색어(`TechProfile.search_queries_used`)를 피한다. 의견 생성 없음.
- **`perspective_check`**: V4~V8 전부. 30개 존재, S2/S3 4주체 각 ≥1, S4 ≥1, D1~D3 measurements, evidence 실존(paper→`get_chunk`, web→캐시된 `WebResult` URL 집합), `confidence == compute_confidence(evidence)`(불일치면 경고 후 재계산 값으로 덮어씀). 결과를 관점·기술별 `missing_criteria`로.
- **`evidence_gap_check`**: 기술별 evidence 수(중복 id 제거), `official` 비율, 기준별로 한쪽 기술만 `not_public`인 항목, `synthesis.gaps` 중 `opposing_missing`. 비율 > 2:1 또는 어느 기술·기준에 반대 방향 근거 0 → `needs_counter_search=True`.
- **`counter_evidence_search`**: `evidence_gap.opposing_missing` 항목별로 `web_search`(계열 단위) + `retriever`(서베이 2편 우선 `doc_ids=["io_survey","kv_survey"]`) 각 1회. 결과를 `Evidence`(`evidence_id="{tech}-COUNTER-{NN}"`)로만 반환. 의견·판정 없음. 1회 상한.
- **`judge`**: `deps.judge_llm`(`JUDGE_MODEL`) 사용. CRITERIA §6 5차원 1/3/5. 입력에는 `report_md` + 평가 결과 요약(기준별 level/evidence_id) + evidence 인덱스를 넣어 "본문 주장 ↔ 근거" 대조가 가능하게 한다. `missing_required`는 D의 `lint` 결과와 합친다. 분량·마크다운 장식 채점 금지를 프롬프트에 명시. `passed = all(score>=3) and not missing_required`.

### 5.4 `config.py` / `.env.example`
- `LLM_PROVIDER`, `LLM_MODEL`, `JUDGE_MODEL`(LLM_MODEL과 같으면 시작 시 경고), `EMBEDDING_MODEL`, `CHROMA_DIR`, `PAPERS_DIR`, `WEB_SEARCH_PROVIDER`, `WEB_SEARCH_API_KEY`, `OUTPUT_DIR`, `LOG_LEVEL`.
- `get_llm()`/`get_judge_llm()`: `langchain.chat_models.init_chat_model(model, model_provider=...)`, `temperature=0`.
- `build_deps(stub: bool) -> Deps`: stub=True면 `StubRetriever`, `stub_web_search`, `FakeStructuredLLM`.

### 5.5 `tests/conftest.py` — `FakeStructuredLLM`
- `with_structured_output(Model)` → 호출 시 `tests/fixtures/`에서 `Model`에 맞는 픽스처를 찾아 반환(기준 id·tech_id는 프롬프트 문자열에서 추출해 매칭). 매칭 실패 시 명확한 에러.
- `invoke(prompt)` 호출 기록을 남겨 각 역할이 "프롬프트에 무엇이 들어갔는지" 검증할 수 있게 한다.
- `deps_stub` 픽스처: `Deps(retriever=StubRetriever(), web_search=stub_web_search, llm=FakeStructuredLLM(), judge_llm=FakeStructuredLLM(), now=lambda: "2026-01-01T00:00:00")`.

### 5.6 `scripts/run.py`
- `--stub`(전부 스텁), `--no-cache`(웹 캐시 무시), `--out outputs/`, `--dump-state`, `--skip-pdf`.
- 종료 시 요약 출력: 노드 실행 횟수, 재시도 횟수, LLM 호출 수, 총 소요 시간, `judge_result` 점수, 산출물 경로.

## 6. 테스트

- `test_fixtures.py`: 모든 픽스처가 스키마 통과 (main 머지 조건).
- `tests/control/`: 각 검사 노드에 "부족한 입력"을 넣었을 때 정확한 `missing_criteria`/`needs_counter_search`/`passed` 산출. `compute_confidence` 케이스. `latest_by_criterion` 중복 제거.
- `test_graph.py`(스텁): (a) 정상 경로 END 도달, (b) T3 누락 픽스처 → `query_rewrite` 경유 후 2회 상한에서 진행, (c) M2 누락 → `market_eval`만 재실행(다른 노드 호출 수 불변), (d) 비대칭 → `counter_evidence_search` 1회만, (e) judge 실패 → `report` 1회만 재실행, (f) `operator.add` 중복이 D 입력에서 제거됨.
- `test_e2e.py`(integration): 실제 실행 1회 → `outputs/report.md`, `report.pdf`, judge 점수 로그.

## 7. 완료 기준 (DoD)

- [ ] Day 1: `schemas.py`, `state.py`, `_deps.py`, `conftest.py`, `test_fixtures.py`, `.env.example` 커밋
- [ ] `run.py --stub` END 도달, `test_graph.py` (a)~(f) 통과
- [ ] 제어 노드 6개 + 단위 테스트
- [ ] 각 역할 PR 머지 시 노드 교체·통합 확인
- [ ] E2E 실제 실행 성공, 실행 요약 출력
- [ ] CONTRACTS.md와 코드 불일치 0 (변경 시 문서 동시 갱신)

## 8. 하지 말 것

- 제어 노드에서 평가 의견·레벨을 생성하거나 수정(재계산 허용 범위: `confidence`만).
- 상한 없는 루프. `retry_counts` 없이 조건부 Edge 작성.
- 에이전트 함수 내부 수정(문제는 해당 역할에 요청). 노드 래핑 코드만 소유.
- 다른 역할과 합의 없이 CONTRACTS 변경.
