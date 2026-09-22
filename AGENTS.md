# TechEvalAgent — KV cache 최적화 기술 다관점 평가 Agentic RAG

이 문서는 이 저장소에서 작업하는 AI 코딩 에이전트(Claude Code, Codex 등)와 개발자를 위한 최상위 하네스다.
원본은 `AGENTS.md`이며 `CLAUDE.md`는 이를 import한다. 내용을 고칠 때는 `AGENTS.md`만 고친다.
작업을 시작하기 전에 **반드시 아래 순서로 읽는다.**

1. 이 문서 (`AGENTS.md`) — 프로젝트 목표, 공통 규칙, 구조, 역할 분담
2. `docs/CONTRACTS.md` — 모든 역할이 공유하는 인터페이스(Pydantic 스키마, State, 함수 시그니처). **변경 금지 영역**
3. `docs/CRITERIA.md` — 평가 기준(15개)과 레벨 정의, 보고서 목차, 검수 기준
4. `docs/roles/<자기 역할>.md` — 자기 역할의 상세 하네스 (소유 파일, 구현 항목, 완료 기준)

다른 역할의 문서는 "내가 무엇을 받고 무엇을 넘기는지" 확인할 때만 읽는다. 다른 역할이 소유한 파일은 수정하지 않는다.

---

## 1. 프로젝트 한 줄 요약

동일한 KV cache 병목을 다루는 두 기술을 **기술 성숙도(TRL) · 시장성 · 이해관계자 · 도메인 적합성** 4관점에서 각각 평가하고,
관점별 평가가 **왜 달라지는지**를 정리한 비교 보고서를 생성하는 LangGraph 기반 Multi-Agent RAG 시스템.

| 접근 | 기술 | `tech_id` | 선정 논문 |
|---|---|---|---|
| SW — KV cache 축소 | DeepSeek-V2의 MLA (Multi-head Latent Attention) | `mla` | DeepSeek-V2 (2024-06) |
| HW — 메모리 계층 확장 | PIM/CXL (Processing-in/near-Memory over CXL) | `pim_cxl` | 1M-Token LLM Inference (2025-10) |

- **공통 전제 도메인**: 대규모 데이터센터의 장문맥 LLM 추론 환경. 모든 관점 평가는 이 전제 아래 수행한다.
- **원문 코퍼스**: 영문 논문 4편 — DeepSeek-V2 원문, PIM/CXL(1M-Token) 원문, 서베이 2편(I/O for LLM inference: storage and memory bottlenecks / A Survey on LLM Acceleration based on KV Cache Management). (`data/papers/`, git 미추적)
- **기술 선정은 Human 고정**이다. 에이전트가 기술을 고르지 않는다.
- **우열·순위·합산 점수를 내지 않는다.** 관점별 레벨은 "현재 공개 근거로 확인되는 단계"일 뿐이며, TRL 차이도 우열이 아니다.
- 설계 원문은 계획서 PDF(`RAG-Design_판교-6반_*.pdf`)이고, 코드 작성에 필요한 부분은 `docs/CRITERIA.md`에 옮겨 두었다. 계획서와 이 저장소 문서가 충돌하면 **저장소 문서가 우선**이고, 충돌을 발견하면 E에게 알린다.

---

## 2. 절대 규칙 (모든 역할 공통)

1. **근거 없는 서술 금지.** 모든 `CriterionResult`는 최소 1개 `Evidence`를 가진다. LLM은 retriever/web_search가 실제로 돌려준 청크·URL만 인용할 수 있다. 프롬프트에 "출처를 지어내지 말라"고 쓰는 것으로 끝내지 말고, **코드에서 evidence의 locator가 실제 검색 결과에 존재하는지 검증**한다.
2. **못 찾은 정보는 `not_public`.** 추측하지 않는다. `level="not_public"`으로 기록하고 `search_query`, `searched_at`, 확인 범위를 Evidence에 남긴다.
3. **신뢰도 규칙 고정.** `high` = 독립 출처 2개 이상 + 논문/공식 자료 포함, `medium` = 출처 1개 또는 2차 자료만, `low` = 추론(inference) 또는 비공개 항목 포함. 코드로 계산한다(LLM이 임의로 정하지 않는다).
4. **평가 단위 표기.** TRL·기술 개요는 논문/구현 단위(`unit="paper"`), 시장·이해관계자는 기술 계열 단위(`unit="family"`)까지 확장한다. 모든 Evidence에 `unit`을 기록한다.
5. **수치는 조건과 함께.** D1~D3 및 기술 개요의 성능 수치는 `Measurement`(모델 규모·컨텍스트 길이·하드웨어·베이스라인·배치)를 반드시 동반한다. 서로 다른 실험의 값을 직접 비교하는 문장을 생성하지 않는다.
6. **중립성.** 관점별 레벨을 합산·평균·순위화하는 코드나 프롬프트를 작성하지 않는다. "더 우수하다", "추천한다" 류 표현은 보고서에서 금지.
7. **State 쓰기 원칙.** 각 에이전트는 자기 State 키에만 쓴다. 근거는 `CriterionResult` 내부에 포함하고 공용 evidence 키에 동시 기록하지 않는다. 같은 에이전트가 두 기술을 처리할 때만 `Send` + `operator.add` reducer를 쓴다.
8. **Pydantic 검증 실패 = 종합 단계로 넘어가지 않는다.** 기준 ID나 근거가 빠진 결과는 예외를 내거나 `missing_criteria`에 올린다. 조용히 기본값으로 채우지 않는다.
9. **생성 모델과 검수 모델 분리.** 실효 보고서 생성 모델(`LLM_MODEL_REPORT`, 비어 있으면 `LLM_MODEL`)과 보고서 검수(`JUDGE_MODEL`)는 서로 다른 모델 설정을 쓴다. 평가 종합 등 다른 에이전트 모델이 `JUDGE_MODEL`과 같은 것은 허용한다.
10. **공통 인터페이스 변경은 합의 후.** `src/techeval/schemas.py`, `src/techeval/state.py`, `docs/CONTRACTS.md`는 E가 관리한다. 필드를 추가·변경하려면 PR 설명에 "CONTRACT CHANGE"를 붙이고 영향받는 역할 전원의 승인을 받는다.

---

## 3. 역할 분담 (5인)

| 역할 | 담당자 | 담당 영역 | 소유 디렉토리 | 하네스 문서 |
|---|---|---|---|---|
| **A** | 이지수 | RAG / Retrieval — PDF 파싱·청킹, BGE-M3, Chroma, Dense+BM25+RRF, `VectorRetriever`, 검색 결과 형식 통일 | `src/techeval/retrieval/`, `scripts/ingest.py`, `data/papers/README.md` | `docs/roles/A-retrieval.md` |
| **B** | 노윤성 | 기술 조사 Agent(T1~T4 TRL) + 도메인 평가 Agent(D1~D4) — 논문 원문 RAG 최다 사용, 측정 조건 구조화 | `src/techeval/agents/tech_research.py`, `src/techeval/agents/domain.py`, `src/techeval/prompts/tech_research/`, `src/techeval/prompts/domain/` | `docs/roles/B-tech-domain.md` |
| **C** | 강준모 | 시장 평가 Agent(M1~M3) + 이해관계자 평가 Agent(S1~S4) + 공통 Web Search Tool, 출처·날짜 관리 | `src/techeval/agents/market.py`, `src/techeval/agents/stakeholder.py`, `src/techeval/tools/web_search.py`, `src/techeval/prompts/market/`, `src/techeval/prompts/stakeholder/` | `docs/roles/C-market-stakeholder.md` |
| **D** | 김가연 | 평가 종합 Agent(agreements/conflicts/gaps) + 보고서 생성 Agent(SUMMARY~REFERENCE) + Citation formatter + PDF 출력 | `src/techeval/agents/synthesis.py`, `src/techeval/agents/report.py`, `src/techeval/report/`, `src/techeval/prompts/synthesis/`, `src/techeval/prompts/report/` | `docs/roles/D-synthesis-report.md` |
| **E** | 백승현 | LangGraph `StateGraph`, State 스키마, `Send`/reducer, Fan-out/Fan-in, 조건부 Edge, 제어 노드(근거 검사·질의 재작성·반대 근거 탐색·보고서 Judge), 전체 통합·실행 테스트 | `src/techeval/graph.py`, `src/techeval/state.py`, `src/techeval/schemas.py`, `src/techeval/config.py`, `src/techeval/control/`, `scripts/run.py`, `tests/test_e2e.py`, `docs/CONTRACTS.md` | `docs/roles/E-graph-control.md` |

의존 관계:

```
A. Retrieval ──────────────┐
                           ├──▶ B. 기술 + 도메인 ──┐
C. Web Search Tool ────────┤                      ├──▶ D. 종합 + 보고서
                           └──▶ C. 시장 + 이해관계자 ┘
E. Graph / State / Control ── 위 전부를 노드로 연결, 제어 노드로 루프·분기
```

병렬 개발 원칙: **모든 역할은 상류 결과가 없어도 스텁/픽스처만으로 자기 모듈을 끝까지 만들 수 있어야 한다.**
각 역할은 `docs/CONTRACTS.md`에 정의된 스텁과 `tests/fixtures/` 샘플을 **가장 먼저** 제공한다(하류가 그것으로 개발한다).

---

## 4. 디렉토리 구조 (전체 파일 트리)

괄호 안은 소유 역할. 여기 없는 파일을 새로 만들 때는 자기 역할의 디렉토리 아래에만 만든다.

```
TechEvalAgent/
├── AGENTS.md                          # 최상위 하네스 원본 (Codex·기타 에이전트가 읽음)
├── CLAUDE.md                          # Claude Code용 — AGENTS.md를 @import
├── README.md
├── .gitignore
├── .env.example                       # (E) 환경변수 템플릿
├── docs/
│   ├── CONTRACTS.md                   # (E) 공통 인터페이스 — 변경은 CONTRACT CHANGE 합의
│   ├── CRITERIA.md                    # 평가 기준·레벨·보고서 목차·검수 기준
│   └── roles/
│       ├── A-retrieval.md
│       ├── B-tech-domain.md
│       ├── C-market-stakeholder.md
│       ├── D-synthesis-report.md
│       └── E-graph-control.md
├── src/techeval/
│   ├── __init__.py
│   ├── config.py                      # (E) Settings, get_llm()/get_judge_llm(), build_deps()
│   ├── schemas.py                     # (E) 공통 Pydantic 모델, 상수, compute_confidence()
│   ├── state.py                       # (E) GraphState, latest_by_criterion(), build_initial_state()
│   ├── graph.py                       # (E) build_graph(): StateGraph·Send·조건부 Edge
│   ├── stub_llm.py                    # (E) FakeStructuredLLM — 픽스처 기반 가짜 LLM (--stub 실행·테스트 공용)
│   ├── retrieval/                     # (A)
│   │   ├── __init__.py
│   │   ├── ingest.py                  #   parse_pdf(), chunk_document(), build_index()
│   │   ├── embedder.py                #   BGE-M3 래퍼
│   │   ├── store.py                   #   Chroma 컬렉션
│   │   ├── bm25.py                    #   BM25 인덱스
│   │   ├── retriever.py               #   RetrievedChunk, BaseRetriever, VectorRetriever, rrf_fuse()
│   │   └── stub.py                    #   StubRetriever
│   ├── tools/                         # (C)
│   │   ├── __init__.py
│   │   ├── web_search.py              #   WebResult, web_search(), not_public_evidence(), 캐시
│   │   └── stub.py                    #   stub_web_search()
│   ├── agents/                        # 순수 함수. langgraph import 금지
│   │   ├── __init__.py
│   │   ├── _deps.py                   # (E) Deps, AgentInput, TechResearchOutput, SynthesisInput, ReportInput
│   │   ├── tech_research.py           # (B) run_tech_research()
│   │   ├── domain.py                  # (B) run_domain_eval()
│   │   ├── market.py                  # (C) run_market_eval()
│   │   ├── stakeholder.py             # (C) run_stakeholder_eval()
│   │   ├── synthesis.py               # (D) run_synthesis()
│   │   └── report.py                  # (D) run_report()
│   ├── control/                       # (E) 제어 노드 — 의견 생성 금지
│   │   ├── __init__.py
│   │   ├── tech_evidence_check.py
│   │   ├── query_rewrite.py
│   │   ├── perspective_check.py
│   │   ├── evidence_gap.py
│   │   ├── counter_evidence.py
│   │   └── judge.py
│   ├── report/                        # (D)
│   │   ├── __init__.py
│   │   ├── citation.py                #   format_reference(), build_reference_section()
│   │   ├── tables.py                  #   기술 × 기준 요약표
│   │   ├── lint.py                    #   금칙어·챕터·인용 일치 검사
│   │   ├── sections.py                #   보고서 챕터·절 분리/조립 (lint·문제 챕터 재생성 공용)
│   │   ├── templates.py               #   1·2장 고정 텍스트, 4장 머리말
│   │   └── pdf.py                     #   render_pdf()
│   └── prompts/                       # 마크다운 프롬프트, 소유자 = 해당 에이전트 소유자
│       ├── tech_research/             # (B) system.md, profile.md, T1.md ~ T4.md
│       ├── domain/                    # (B) system.md, D1.md ~ D4.md
│       ├── market/                    # (C) system.md, M1.md ~ M3.md
│       ├── stakeholder/               # (C) system.md, S1.md ~ S4.md
│       ├── synthesis/                 # (D) system.md, agreements.md, conflicts.md, gaps.md
│       ├── report/                    # (D) system.md, summary.md, ch1.md ~ ch6.md, ch4_{trl,market,stakeholder,domain}.md, revision.md
│       └── control/                   # (E) query_rewrite.md, judge.md
├── scripts/
│   ├── ingest.py                      # (A) 코퍼스 인덱싱 CLI
│   └── run.py                         # (E) 전체 파이프라인 실행 CLI
├── tests/
│   ├── conftest.py                    # (E) deps_stub·fake_llm 픽스처 (FakeStructuredLLM은 src/techeval/stub_llm.py)
│   ├── test_fixtures.py               # (E) 모든 픽스처 스키마 검증
│   ├── test_graph.py                  # (E) 스텁 그래프 흐름·분기·상한
│   ├── test_e2e.py                    # (E) integration
│   ├── fixtures/
│   │   ├── chunks.json                # (A)
│   │   ├── web_results.json           # (C)
│   │   ├── tech_profiles.json         # (B)
│   │   ├── trl_eval.json              # (B)
│   │   ├── domain_eval.json           # (B)
│   │   ├── market_eval.json           # (C)
│   │   ├── stakeholder_eval.json      # (C)
│   │   ├── synthesis.json             # (D)
│   │   ├── report_md.md               # (D)
│   │   ├── counter_evidence.json      # (E)
│   │   ├── evidence_gap.json          # (E)
│   │   ├── judge_result.json          # (E)
│   │   └── state_initial.json         # (E)
│   ├── retrieval/                     # (A)
│   ├── tools/                         # (C)
│   ├── agents/                        # (B, C, D) test_tech_research.py, test_domain.py, test_market.py, test_stakeholder.py, test_synthesis.py, test_report.py
│   ├── control/                       # (E)
│   └── report/                        # (D) test_citation.py, test_lint.py, test_pdf.py, test_tables.py
│       ├── d_fixtures.py              #   D 테스트용 픽스처 로더
│       ├── make_d_fixtures.py         #   synthesis.json / report_md.md 재생성 스크립트
│       └── upstream_samples/          #   C 픽스처(market/stakeholder)가 main에 오기 전까지 D가 쓰는 샘플 (머지 후 삭제)
├── assets/fonts/                      # (D) PDF 한글 폰트 (선택)
├── data/
│   ├── papers/
│   │   ├── README.md                  # (A) 4편 메타(제목·저자·발행·URL·doc_id·파일명) — 추적
│   │   ├── deepseek_v2.pdf            #   git 미추적
│   │   ├── pim_cxl_1m.pdf             #   git 미추적
│   │   ├── io_survey.pdf              #   git 미추적
│   │   └── kv_survey.pdf              #   git 미추적
│   └── chroma/                        # 벡터 저장소 + bm25.pkl — git 미추적
└── outputs/                           # git 미추적
    ├── report.md
    ├── report.pdf
    ├── state/                         # --dump-state 시 노드별 State
    └── web_cache/                     # (C) 웹 검색 캐시
```

`doc_id` 고정값: `deepseek_v2`, `pim_cxl_1m`, `io_survey`, `kv_survey`. 선정 기술의 근거는 원문 2편에서, 배경·기술 계열 맥락·반대 근거는 서베이 2편에서 찾는다.

---

## 5. 기술 스택 및 환경

- Python 3.11+, 패키지 관리 `uv` (`pyproject.toml`/`uv.lock`은 현재 `.gitignore`에 있어 개인 관리. 의존성 목록은 아래를 기준으로 맞춘다)
- 핵심 의존성: `langgraph`, `langchain-core`, `pydantic>=2`, `chromadb`, `FlagEmbedding` 또는 `sentence-transformers`(BGE-M3), `rank-bm25`, `pymupdf`(PDF 파싱), `python-dotenv`, `httpx`
- 보고서: `markdown` + `weasyprint` (PDF 변환, D가 최종 확정)
- 테스트/린트: `pytest`, `ruff`
- LLM: `langchain`의 `init_chat_model`로 provider-agnostic하게 호출. 모델 ID는 코드에 하드코딩하지 않고 `.env`로 주입한다.

`.env` 키 (E가 `config.py`에서 로드, `.env.example`로 공유):

```
LLM_PROVIDER=openai      # with_structured_output이 네이티브 tool calling으로 동작해야 함
OPENAI_API_KEY=
LLM_MODEL=gpt-5-mini     # B·C 및 별도 오버라이드가 없는 에이전트의 기본값
LLM_MODEL_<AGENT>=       # 선택. TECH_RESEARCH/DOMAIN/MARKET/STAKEHOLDER/SYNTHESIS/REPORT 별 모델, 비우면 LLM_MODEL
LLM_MODEL_SYNTHESIS=gpt-5-mini
LLM_MODEL_REPORT=gpt-4o
JUDGE_MODEL=gpt-5-mini   # 보고서 검수용 (LLM_MODEL_REPORT 폴백 LLM_MODEL과 달라야 함)
EMBEDDING_MODEL=BAAI/bge-m3
CHROMA_DIR=data/chroma
PAPERS_DIR=data/papers
WEB_SEARCH_PROVIDER=     # C가 확정 (tavily / serper / 등)
WEB_SEARCH_API_KEY=
OUTPUT_DIR=outputs
```

실행 명령:

```bash
uv sync
uv run python scripts/ingest.py                 # A: 논문 인덱싱 (최초 1회)
uv run python scripts/run.py --stub             # E: 전부 스텁으로 그래프 흐름만 검증
uv run python scripts/run.py                    # E: 실제 실행 -> outputs/report.md, outputs/report.pdf
uv run pytest                                   # 전체 테스트 (스텁 기반, 네트워크·모델 불필요)
uv run pytest -m integration                    # 실제 모델/검색 필요한 테스트
uv run ruff check . && uv run ruff format .
```

---

## 6. 개발 방식

### 6.1 스텁 우선 (Stub-first)
- 각 역할은 실제 구현 전에 **스텁 + 픽스처**를 먼저 커밋한다.
  - A: `StubRetriever` + `tests/fixtures/chunks.json`
  - C: `stub_web_search()` + `tests/fixtures/web_results.json`
  - B: `tests/fixtures/tech_profiles.json`, `trl_eval.json`, `domain_eval.json`
  - C: `tests/fixtures/market_eval.json`, `stakeholder_eval.json`
  - D: `tests/fixtures/synthesis.json`, `report_md.md`
  - E: `tests/fixtures/state_initial.json`, `judge_result.json`
- 픽스처는 `docs/CONTRACTS.md`의 스키마로 `model_validate` 되어야 한다(`tests/test_fixtures.py`가 검증).
- 에이전트 함수는 `deps` 인자로 retriever/web_search/llm을 주입받는다. 모듈 전역에서 실제 모델을 초기화하지 않는다.

### 6.2 에이전트 함수 규약
- 에이전트는 **그래프를 모르는 순수 함수**다: `run_xxx(inp: XxxInput, deps: Deps) -> XxxOutput`.
- LangGraph 노드 래핑, `Send`, State 읽기/쓰기는 전부 E가 `graph.py`에서 한다. 에이전트 파일에서 `langgraph`를 import하지 않는다.
- LLM 구조화 출력은 `llm.with_structured_output(Model)`을 쓰고, 결과를 반드시 Pydantic으로 재검증한다.
- 재시도 시 `inp.missing_criteria`가 주어지면 **그 기준만** 다시 생성한다(전체 재실행 금지). `inp.retry_count`를 프롬프트에 반영해 검색어를 바꾼다.

### 6.3 프롬프트
- `src/techeval/prompts/<agent>/system.md`, `<criterion>.md` 등 마크다운 파일로 관리하고 코드에서 로드한다.
- 프롬프트에 **예상 답을 주입하지 않는다**(예: D4 변경 범위 체크에서 "MLA는 재학습이 필요하다"를 미리 쓰지 않는다).
- 프롬프트에 `docs/CRITERIA.md`의 해당 기준 판정 방식을 그대로 인용한다.
- 출력 언어는 한국어. 인용문(`quote`)은 원문 언어(영문) 그대로 보존한다.

### 6.4 테스트
- 모든 역할: 스텁 기반 단위 테스트 필수. 네트워크·모델 로딩이 필요한 테스트는 `@pytest.mark.integration`.
- 픽스처를 바꾸면 그 픽스처를 소비하는 역할에 PR에서 멘션한다.

### 6.5 코딩 컨벤션
- 타입 힌트 필수, Pydantic v2 (`model_validate`, `model_dump`), `from __future__ import annotations` 사용 안 함(Pydantic 호환 문제 회피).
- 로깅은 `logging.getLogger(__name__)`. `print` 금지(스크립트 제외).
- 파일/함수 이름은 `docs/CONTRACTS.md`의 시그니처를 그대로 따른다. 이름을 바꾸면 하류가 깨진다.
- 예외는 삼키지 않는다. 검색 실패, 파싱 실패는 로그 후 raise 또는 `not_public` 기록.
- 주석·문서는 한국어, 식별자는 영어.

---

## 7. Git 규칙

- `main`은 항상 `uv run pytest`(스텁 테스트)가 통과하는 상태를 유지한다.
- 브랜치: `feat/<역할>-<주제>` (예: `feat/A-hybrid-retriever`, `feat/E-perspective-check`).
- 커밋 메시지: `<type>(<scope>): <요약>` — type은 `feat|fix|docs|test|refactor|chore`, scope는 역할 문자 또는 모듈명. 예: `feat(A): RRF 결합 구현`.
- 공통 인터페이스 변경은 "CONTRACT CHANGE" 라벨 + 전원 승인.
- 다른 역할의 디렉토리를 수정해야 하면 PR 대신 해당 역할에게 요청한다.
- **AI 에이전트는 사용자가 명시적으로 지시하기 전에 commit/push 하지 않는다.**
- `data/papers/*.pdf`, `data/chroma/`, `outputs/`, `.env`는 커밋하지 않는다.

---

## 8. 용어

| 용어 | 뜻 |
|---|---|
| 관점(perspective) | `trl` / `market` / `stakeholder` / `domain` 4개 |
| 기준(criterion) | T1~T4, M1~M3, S1~S4, D1~D4 총 15개. 두 기술에 동일 적용 → 결과 30개 |
| 레벨(level) | 기준별 판정 단계 문자열 (`docs/CRITERIA.md`) |
| 평가 단위(unit) | `paper`(선정 논문·구현 단위) / `family`(기술 계열 단위) |
| 4주체(stakeholders) | 모델 개발사 / 클라우드·서빙 운영사 / 메모리·반도체 벤더 / 투자 업계 — 고정 상수 |
| 제어 노드 | 의견을 생성하지 않고 흐름·품질만 관리하는 노드 (E) |
| 반대 근거(counter evidence) | 한 기술에 유리한 근거만 쌓였을 때 추가로 찾는 반대 방향 근거 |
| not_public | 2회 재검색 후에도 못 찾은 정보. 검색어·검색일 필수 |
