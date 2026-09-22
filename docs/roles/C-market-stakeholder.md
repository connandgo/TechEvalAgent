# 역할 C — 시장 평가 Agent + 이해관계자 평가 Agent + Web Search Tool (담당: 강준모)

먼저 읽기: `AGENTS.md` → `docs/CONTRACTS.md` §2, §4, §5, §9 → `docs/CRITERIA.md` §2.2, §2.3 → 이 문서.
당신은 **외부 웹 검색 비중이 높은** 두 에이전트와, B·E도 함께 쓰는 공통 `web_search` 툴을 만든다. 출처·날짜·근거 관리가 핵심이다.

## 1. 목표

- `web_search()` 공통 툴: 검색 → `WebResult` 통일 형식, 검색어·검색일 기록, 캐시, `not_public_evidence()` 헬퍼.
- `run_market_eval(inp, deps) -> list[CriterionResult]`: M1 성장성 / M2 채택 / M3 생태계. 시장·채택 자료는 웹에서, 제품·생태계가 지원하는 기술 사양은 `retriever`로 원문과 교차 확인.
- `run_stakeholder_eval(inp, deps) -> list[CriterionResult]`: 고정 4주체에 대해 S1 역할 / S2 편익 / S3 부담 / S4 상충. RAG 미사용(웹만).
- 조사 범위는 **기술 계열 단위**(`unit="family"`): MLA 계열, CXL·PIM/PNM 메모리 계열. 논문 단위 근거와 계열 단위 근거를 구분해 표기한다.

## 2. 소유 파일

```
src/techeval/tools/web_search.py         # WebResult, web_search, not_public_evidence, 캐시
src/techeval/tools/stub.py               # stub_web_search
src/techeval/agents/market.py            # run_market_eval
src/techeval/agents/stakeholder.py       # run_stakeholder_eval
src/techeval/prompts/market/             # system.md, M1.md, M2.md, M3.md
src/techeval/prompts/stakeholder/        # system.md, S1.md ... S4.md
tests/tools/test_web_search.py
tests/agents/test_market.py
tests/agents/test_stakeholder.py
tests/fixtures/web_results.json          # dict[str, list[WebResult]]
tests/fixtures/market_eval.json          # CriterionResult 6개
tests/fixtures/stakeholder_eval.json     # CriterionResult 8개
```

## 3. 상류 / 하류

- 상류: A(`deps.retriever` — M3 교차 확인), B(`inp.tech_profile`, `inp.trl_eval` — 기술 사양 참고), E(`schemas.py`, `Deps`, `AgentInput`, `compute_confidence`, `config.WEB_SEARCH_*`)
- 하류: B(T3·D4에서 `web_search` 사용), D(`market_eval`, `stakeholder_eval`), E(`counter_evidence`에서 `web_search` 사용, 검사 노드)
- 상류가 없을 때: `StubRetriever`, B의 `tech_profiles.json`, `FakeStructuredLLM`으로 개발.

## 4. 구현 순서 (권장)

1. **`WebResult`, `stub_web_search`, `tests/fixtures/web_results.json`, `not_public_evidence()`를 가장 먼저 커밋한다.** B와 E가 이것으로 개발한다.
   - 픽스처 키는 검색어 키워드(예 `"MLA vLLM support"`, `"CXL memory expander KV cache"`, `"Samsung PIM HBM"`, `"DeepSeek-V2 adoption"`). 각 5개 이내, `published_date`·`publisher`·`source_kind` 채움.
2. `web_search()` 실제 구현: provider는 `.env`의 `WEB_SEARCH_PROVIDER`로 선택(Tavily 권장, 대안 Serper). 본문 추출(`fetch_content=True`)은 `httpx` + `trafilatura` 또는 `readability` 사용, 최대 4,000자.
3. 캐시: `outputs/web_cache/{sha1(query+params)}.json`. 실행 간 재사용, `--no-cache` 옵션(E의 run.py에서 전달).
4. `run_market_eval`, `run_stakeholder_eval` 골격 → 프롬프트 → 통합 테스트.

## 5. 세부 요구사항

### 5.1 `web_search`
- `published_date` 파싱: 메타태그(`article:published_time`, `date`), URL 패턴, 본문 첫 날짜 순으로 시도. 실패 시 `None`(추정 금지).
- `publisher`: 도메인에서 추출하되 알려진 매핑(`nvidia.com→NVIDIA`, `samsung.com→Samsung`, `arxiv.org→arXiv` ...)을 `PUBLISHER_MAP`으로 유지.
- `source_kind`: 벤더·제안사 공식 도메인 → `official`, 언론 → `news`, arxiv/acm/ieee → `paper`, 개인 블로그/미디엄 → `blog`, 레딧/HN/포럼 → `forum`. **`official` 비율은 E의 근거 비대칭 검사와 보고서 한계점에서 계산되므로 정확히 표기.**
- `recency_days`, `site_filter` 지원. 에러 시 빈 리스트가 아니라 예외를 올린다(호출자가 `not_public` 처리 결정).
- 검색어·검색일(`fetched_at`, `query`)은 모든 결과에 기록.

### 5.2 시장 평가 (`run_market_eval`)
- 조사 범위: `inp.tech.family` + `inp.tech.search_aliases`. 논문 단위 근거(선정 논문 언급)와 계열 단위 근거를 evidence `unit`으로 구분.
- M1: `details.market_figures`에 수치·발행 주체·시점·evidence_id. 단일 전망을 사실로 단정하는 문장 금지("~로 전망된다(출처, 시점)" 형태). L3 확산 / L2 형성 / L1 태동 / L0 미확인.
- M2: `details.adopters`에 채택 주체·단계(research/poc/product/production). **채택 주체 없는 사례는 인정하지 않는다.** 레벨은 확인된 최고 단계로 코드 계산.
- M3: 4항목(`framework`, `vendor_product`, `standardization`, `third_party_research_tools`) Y/N + 근거. 레벨은 Y 개수로 코드 계산(3~4→L3, 1~2→L2, 0→L1, 미확인→L0). 프레임워크 지원(vLLM/SGLang/TensorRT-LLM 등)은 `deps.retriever`로 원문 사양(예: MLA의 KV 형태)과 교차 확인하고 그 청크도 evidence에 포함.
- PIM/CXL은 **논문 시제품과 기술 계열 상용화 근거를 구분**한다(계획서 2.2). 계열 근거로 논문 시제품의 채택을 주장하지 않는다.

### 5.3 이해관계자 평가 (`run_stakeholder_eval`)
- 4주체 상수 `STAKEHOLDERS` 순서 고정. 출력 `details` 키는 주체 id.
- S1: `details.roles[주체] ∈ {decision_maker, affected, supplier, unrelated}` + 근거 evidence_id.
- S2/S3: 주체당 최소 1개. 직접 자료가 없으면 `is_inference=True` + 근거 evidence_id(추론의 출발점이 된 자료) + `Evidence(source_type="inference")` 추가. inference가 포함되면 `confidence`는 자동으로 `low`.
- S4: 최소 1쌍. `details.tradeoffs[].text`는 "A에게 편익인 것이 B에게 부담" 형식을 지킨다. D의 `conflicts`와 보고서 5장의 핵심 입력이다.
- 두 기술에 같은 프롬프트 템플릿, 같은 4주체.

### 5.4 공통
- 구조화 출력 → Pydantic 재검증. `confidence = compute_confidence(evidence)`.
- `evidence_id` 규칙 `{tech_id}-{criterion_id}-{NN}`.
- 웹 evidence의 `quote`는 `snippet` 또는 `content`의 부분 문자열. `accessed_date` = `fetched_at` 날짜.
- 재실행(`inp.missing_criteria`) 시 해당 기준만, `inp.retry_count>0`이면 검색어를 바꾼다(동의어·영문·벤더명 추가).
- 못 찾으면 `not_public` + 검색어·검색일·범위.

## 6. 테스트

- 스텁: `stub_web_search` 키워드 매칭, `published_date` 파서 케이스, `source_kind` 분류, M2/M3 레벨 코드 계산, S2/S3 4주체 각 ≥1 검증, S4 ≥1쌍, `missing_criteria=["M2"]` 부분 실행, 캐시 히트.
- 통합: 실제 검색 provider로 `"CXL memory KV cache"` 5건 이상 반환·날짜 파싱율 ≥60%.

## 7. 완료 기준 (DoD)

- [ ] `WebResult`/`stub_web_search`/`web_results.json`/`not_public_evidence` 커밋 (1순위)
- [ ] 픽스처 `market_eval.json`, `stakeholder_eval.json` 커밋, `test_fixtures` 통과
- [ ] `web_search` 실제 provider 동작 + 캐시
- [ ] M2·M3 레벨 코드 계산, S2/S3 4주체 보장, S4 형식 보장
- [ ] `unit` 구분(paper/family) evidence에 표기
- [ ] 스텁 테스트 통과, 통합 1회 실제 실행 확인

## 8. 하지 말 것

- `langgraph` import, State 직접 접근, 두 기술 루프.
- 검색 결과 없는 기업명·수치·URL 생성. 날짜 추정.
- "시장에서 MLA가 더 유망" 류 비교·우열 서술. 각 기술은 독립 평가.
- 벤더 발표 자료를 `news`나 `paper`로 표시(비대칭 계산이 틀어진다).
