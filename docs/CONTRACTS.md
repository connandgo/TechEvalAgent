# CONTRACTS — 공통 인터페이스 정의

이 문서는 5개 역할이 서로 맞물리는 모든 경계를 정의한다. **여기 정의된 이름·필드·시그니처는 그대로 코드에 옮긴다.**
관리자는 E이며, `src/techeval/schemas.py`와 `src/techeval/state.py`는 이 문서와 1:1로 대응해야 한다.
변경하려면 PR 제목에 `CONTRACT CHANGE`를 붙이고 영향받는 역할 전원의 승인을 받는다.

목차
1. 상수
2. 공통 스키마 (`schemas.py`)
3. Retrieval 계약 (A → B, C, E)
4. Web Search 계약 (C → B, C, E)
5. 에이전트 함수 계약 (B, C, D → E)
6. 제어 노드 계약 (E)
7. State 스키마 (`state.py`)
8. Graph 흐름 (E)
9. 스텁 및 픽스처
10. 검증 규칙 (validator로 강제할 것)

---

## 1. 상수

```python
# src/techeval/schemas.py 상단
from typing import Literal

TechId = Literal["mla", "pim_cxl"]
Perspective = Literal["trl", "market", "stakeholder", "domain"]
CriterionId = Literal[
    "T1", "T2", "T3", "T4",
    "M1", "M2", "M3",
    "S1", "S2", "S3", "S4",
    "D1", "D2", "D3", "D4",
]
Confidence = Literal["high", "medium", "low"]
SourceType = Literal["paper", "web", "official", "patent", "inference", "not_public"]
EvidenceUnit = Literal["paper", "family"]
DocId = Literal["deepseek_v2", "pim_cxl_1m", "io_survey", "kv_survey"]

PERSPECTIVE_CRITERIA: dict[str, list[str]] = {
    "trl": ["T1", "T2", "T3", "T4"],
    "market": ["M1", "M2", "M3"],
    "stakeholder": ["S1", "S2", "S3", "S4"],
    "domain": ["D1", "D2", "D3", "D4"],
}

STAKEHOLDERS: tuple[str, ...] = (
    "model_developer",      # 모델 개발사
    "cloud_serving_operator",  # 클라우드/서빙 운영사
    "memory_semiconductor_vendor",  # 메모리/반도체 벤더
    "investor",             # 투자 업계
)

DOMAIN = "대규모 데이터센터의 장문맥 LLM 추론 환경"

MAX_RETRY_PER_PERSPECTIVE = 2   # 기술 조사 재검색, 관점별 재실행 상한
MAX_COUNTER_EVIDENCE_SEARCH = 1
MAX_REPORT_REGENERATION = 1

TECHNOLOGIES: list["TechRef"]  # 아래 TechRef 정의 후 초기화 (mla, pim_cxl 2개 고정)
```

---

## 2. 공통 스키마 (`src/techeval/schemas.py`)

```python
from datetime import date
from pydantic import BaseModel, Field, model_validator


class TechRef(BaseModel):
    """Human이 고정한 평가 대상 기술. State.technologies에 2개."""
    tech_id: TechId
    name: str                      # "DeepSeek-V2 MLA" / "PIM/CXL"
    approach: Literal["sw", "hw"]
    family: str                    # "MLA 계열" / "CXL·PIM/PNM 계열" — 시장·이해관계자 조사 범위
    primary_doc_id: DocId          # 선정 논문의 doc_id
    paper_title: str
    paper_date: str                # "2024-06" 형식
    search_aliases: list[str] = [] # 검색어 확장용: ["Multi-head Latent Attention", "MLA", ...]


class Measurement(BaseModel):
    """성능 수치 1건. 수치는 반드시 측정 조건과 함께 저장한다 (AGENTS.md 규칙 5)."""
    metric: str                    # "KV cache reduction", "TTFT", "tokens/s", ...
    value: str                     # 원문 표기 그대로 ("93.3%", "2.1x")
    unit: str | None = None
    model_size: str | None = None      # "236B (21B active)"
    context_length: str | None = None  # "128K"
    hardware: str | None = None        # "8x H800"
    baseline: str | None = None        # "DeepSeek 67B (MHA)"
    batch_size: str | None = None
    condition_note: str | None = None  # 그 외 조건
    evidence_id: str               # 이 수치의 출처 Evidence.evidence_id


class Evidence(BaseModel):
    """근거 1건. 모든 CriterionResult / TechProfile / counter_evidence에 포함된다."""
    evidence_id: str               # 규칙: "{tech_id}-{criterion_id|PROFILE|COUNTER}-{2자리}" 예 "mla-T1-01"
    source_type: SourceType
    unit: EvidenceUnit             # paper=논문/구현 단위, family=기술 계열 단위
    quote: str                     # 원문 인용 (원문 언어 그대로). inference/not_public이면 요약 또는 ""
    locator: str                   # 논문: "deepseek_v2 p.4 §2.1.4" / 웹: URL / not_public: "not_found"
    title: str | None = None
    authors: str | None = None     # "DeepSeek-AI" / "Kim, J. et al."
    publisher: str | None = None   # 학술지·사이트명·기관
    published_date: str | None = None  # "2024-06-19" 또는 "2024-06"
    url: str | None = None
    accessed_date: str | None = None   # 웹 자료 확인일 "YYYY-MM-DD"
    doc_id: DocId | None = None    # source_type=paper일 때
    page: int | None = None
    section: str | None = None
    chunk_id: str | None = None    # retriever가 반환한 chunk_id (검증용)
    # not_public 전용
    search_query: str | None = None
    searched_at: str | None = None
    search_scope: str | None = None    # "arxiv, vendor blogs, news 2024-2026"

    @model_validator(mode="after")
    def _check_not_public(self):
        if self.source_type == "not_public" and not (self.search_query and self.searched_at):
            raise ValueError("not_public evidence requires search_query and searched_at")
        if self.source_type in ("paper", "web", "official", "patent") and not self.quote:
            raise ValueError("quoted evidence requires a non-empty quote")
        return self


class CriterionResult(BaseModel):
    """관점 노드 출력의 최소 단위: 기술 1개 × 기준 1개. 계획서 4.3 '관점 노드 출력 계약'."""
    tech_id: TechId
    perspective: Perspective
    criterion_id: CriterionId
    level: str                     # CRITERIA.md의 레벨 문자열 그대로. 예 "TRL 4-6", "L2", "not_public"
    level_estimate: str | None = None  # T1 전용: 구간 내 추정치 "TRL 5 (공개 정보 기반 추정)"
    content: str                   # 2~4문장 평가 내용 (한국어)
    evidence: list[Evidence] = Field(min_length=1)
    confidence: Confidence
    evidence_unit: EvidenceUnit    # 이 판정의 주된 근거 단위
    measurements: list[Measurement] = []   # D1~D3, T 수치 포함 시 필수
    details: dict = {}             # 기준별 구조화 필드 (CRITERIA.md 각 기준의 details 키 참조)
    generated_at: str              # ISO datetime
    retry_count: int = 0

    @model_validator(mode="after")
    def _check_perspective(self):
        if self.criterion_id not in PERSPECTIVE_CRITERIA[self.perspective]:
            raise ValueError(f"{self.criterion_id} does not belong to {self.perspective}")
        if self.level == "not_public" and not any(e.source_type == "not_public" for e in self.evidence):
            raise ValueError("not_public level requires a not_public evidence record")
        if self.criterion_id in ("D1", "D2", "D3") and self.level != "not_public" and not self.measurements:
            raise ValueError(f"{self.criterion_id} requires measurements")
        return self


# 계획서·역할 분담 문서에서 말하는 PerspectiveResult == 기준 단위 결과. 별칭으로 둘 다 허용.
PerspectiveResult = CriterionResult


class TechProfile(BaseModel):
    """기술 조사 에이전트(B)의 1차 산출물. 기술 원리·범위·한계·정량 수치·측정 조건."""
    tech_id: TechId
    principle: str                 # 핵심 접근 (2~5문장)
    scope: str                     # 적용 범위·전제 조건
    limitations: list[str]         # 원문에 명시된 한계
    measurements: list[Measurement]
    validation_env: str            # 검증 환경 서술 (T2 판정의 입력)
    public_artifacts: dict[str, str]  # {"code": "Y|N|unknown", "model_or_design": ..., "eval_scripts_data": ..., "third_party_reproduction": ...}
    public_artifact_urls: dict[str, str] = {}
    evidence: list[Evidence] = Field(min_length=1)
    search_queries_used: list[str] = []
    generated_at: str
    retry_count: int = 0


class Agreement(BaseModel):
    tech_id: TechId | None         # None이면 두 기술 공통
    perspectives: list[Perspective] = Field(min_length=2)
    criterion_ids: list[CriterionId]
    statement: str
    evidence_ids: list[str] = Field(min_length=1)


class Conflict(BaseModel):
    """관점 간 상충. '시장 관점은 즉시 적용성을 높이 평가하나 도메인 관점은 직접 근거 부족으로 상반' 형태."""
    tech_id: TechId
    perspective_a: Perspective
    criterion_a: CriterionId
    perspective_b: Perspective
    criterion_b: CriterionId
    statement: str                 # 무엇이 상충하는가
    cause: str                     # 왜 — 기준 차이 / 평가 단위 차이 / 근거 유형 차이 (우열로 서술 금지)
    evidence_ids: list[str] = Field(min_length=2)


class Gap(BaseModel):
    tech_id: TechId
    criterion_id: CriterionId
    gap_type: Literal["not_public", "unit_mismatch", "single_source", "inference_only", "opposing_missing"]
    description: str


class SynthesisResult(BaseModel):
    agreements: list[Agreement]
    conflicts: list[Conflict] = Field(min_length=1)   # S4가 기술별 최소 1쌍이므로 최소 1
    gaps: list[Gap]
    unit_notes: str                # 평가 단위(논문 vs 계열) 차이가 판정에 미친 영향
    evidence_asymmetry_note: str   # 근거 비대칭 서술
    generated_at: str


class EvidenceGap(BaseModel):
    """근거 비대칭 검사 노드(E) 출력. State.evidence_gap."""
    asymmetry: bool
    evidence_count: dict[str, int]         # {"mla": 23, "pim_cxl": 11}
    vendor_source_ratio: dict[str, float]  # 제안사/벤더 발표 자료 비율 (보고서 한계점에 사용)
    opposing_missing: list[dict]           # [{"tech_id": "mla", "criterion_id": "D1", "reason": "..."}]
    needs_counter_search: bool
    note: str


class JudgeResult(BaseModel):
    """보고서 검수 노드(E) 출력. 5차원 1/3/5점."""
    scores: dict[Literal["evidence", "criteria_compliance", "neutrality", "specificity", "format"], Literal[1, 3, 5]]
    reasons: dict[str, str]
    missing_required: list[str]     # 누락된 필수 기준/챕터
    revision_instructions: list[str]
    passed: bool                    # 모든 점수 >= 3 이고 missing_required 비어 있음
    judged_at: str
```

`details` 키 규약(기준별, CRITERIA.md와 동일):

| 기준 | `details` 필수 키 |
|---|---|
| T1 | `trl_band` ("1-3"/"4-6"/"7-9"), `estimate` (int), `why_not_higher` (str) |
| T2 | `env_level` ("L1"~"L4") |
| T3 | `checklist` {"code","model_or_design","eval_scripts_data","third_party_reproduction": "Y"/"N"/"unknown"} |
| T4 | `remaining_tasks` (list[str]), `not_public_items` (list[str]) |
| M1 | `market_figures` (list[{"figure","publisher","published_date","evidence_id"}]) |
| M2 | `adopters` (list[{"name","stage","evidence_id"}]) |
| M3 | `checklist` {"framework","vendor_product","standardization","third_party_research_tools": "Y"/"N"} |
| S1 | `roles` {stakeholder: "decision_maker"/"affected"/"supplier"/"unrelated"} |
| S2 | `benefits` {stakeholder: [{"text","evidence_id","is_inference"}]} (4주체 각 ≥1) |
| S3 | `burdens` {stakeholder: [{"text","evidence_id","is_inference"}]} (4주체 각 ≥1) |
| S4 | `tradeoffs` (list[{"beneficiary","burdened","text","evidence_ids"}]) ≥1 |
| D1~D3 | `directness` ("L3"/"L2"/"L1"), `extrapolation_logic` (L2일 때 필수) |
| D4 | `checklist` {"model_retrain","model_convert","serving_engine_change","hw_replace","memory_add","other": "Y"/"N"/"unknown"} |

---

## 3. Retrieval 계약 (A 제공)

```python
# src/techeval/retrieval/retriever.py
class RetrievedChunk(BaseModel):
    chunk_id: str                  # "{doc_id}:{page:03d}:{seq:02d}"
    doc_id: DocId
    doc_title: str
    page: int                      # 1-based
    section: str | None            # "2.1.4 Multi-Head Latent Attention" (추출 가능할 때)
    text: str
    score: float                   # 최종 (RRF) 점수, 내림차순
    dense_score: float | None = None
    bm25_score: float | None = None
    chunk_type: Literal["text", "table", "figure_caption"] = "text"

    def to_evidence(self, *, evidence_id: str, quote: str, unit: EvidenceUnit = "paper") -> Evidence:
        """청크를 Evidence로 변환. locator = f"{doc_id} p.{page} §{section}". quote는 text의 부분 문자열이어야 함."""


class BaseRetriever(Protocol):
    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        doc_ids: list[DocId] | None = None,   # None이면 전체 4편
        mode: Literal["hybrid", "dense", "bm25"] = "hybrid",
    ) -> list[RetrievedChunk]: ...

    def get_chunk(self, chunk_id: str) -> RetrievedChunk | None: ...   # evidence 검증용


class VectorRetriever(BaseRetriever):
    """실제 구현. Chroma(dense, BGE-M3) + BM25 → RRF(k=60)."""
    def __init__(self, chroma_dir: str, collection: str = "papers", embedding_model: str = "BAAI/bge-m3"): ...


# src/techeval/retrieval/stub.py
class StubRetriever(BaseRetriever):
    """tests/fixtures/chunks.json을 로드해 query 무관하게 doc_ids 필터 + top_k만 적용해 반환."""
    def __init__(self, fixture_path: str = "tests/fixtures/chunks.json"): ...
```

인덱싱 CLI: `uv run python scripts/ingest.py --papers-dir data/papers --chroma-dir data/chroma [--rebuild]`
- 청킹: 절·표 단위 구조적 청킹, 1,000~1,500 토큰, 페이지·절 메타데이터 보존. 표는 `chunk_type="table"`로 분리.
- 질의와 문서는 같은 모델(BGE-M3)로 임베딩. 임베딩은 Chroma에 저장하고 재계산하지 않는다.
- 한국어 질의 → 영문 문서 교차 언어 검색이 동작해야 한다 (테스트 필수).

---

## 4. Web Search 계약 (C 제공)

```python
# src/techeval/tools/web_search.py
class WebResult(BaseModel):
    title: str
    url: str
    snippet: str
    content: str | None = None     # 본문 추출 성공 시 (최대 N자, C가 결정)
    publisher: str | None = None   # 도메인 또는 기관명
    published_date: str | None = None   # 파싱 가능할 때 "YYYY-MM-DD"
    fetched_at: str                # ISO datetime
    source_kind: Literal["official", "news", "blog", "paper", "forum", "other"] = "other"
    query: str                     # 이 결과를 만든 검색어

    def to_evidence(self, *, evidence_id: str, quote: str, unit: EvidenceUnit) -> Evidence:
        """source_type은 source_kind=official→"official", paper→"paper", 그 외→"web"."""


WebSearchFn = Callable[..., list[WebResult]]

def web_search(
    query: str,
    *,
    max_results: int = 5,
    recency_days: int | None = None,
    site_filter: list[str] | None = None,
    fetch_content: bool = False,
) -> list[WebResult]: ...

def not_public_evidence(*, evidence_id: str, queries: list[str], scope: str, unit: EvidenceUnit) -> Evidence:
    """검색 실패 기록 헬퍼. searched_at은 현재 시각으로 채운다."""


# src/techeval/tools/stub.py
def stub_web_search(query: str, **kwargs) -> list[WebResult]:
    """tests/fixtures/web_results.json에서 query 키워드 매칭(없으면 빈 리스트) 반환."""
```

- 검색 결과에는 항상 검색어·검색일을 남긴다. 캐시(`outputs/web_cache/`)를 두어 같은 검색어를 재실행 시 재호출하지 않는다.
- 제안사·벤더 자료는 `source_kind="official"`로 표시한다(근거 비대칭 검사와 보고서 한계점에서 비율을 계산한다).

---

## 5. 에이전트 함수 계약 (B, C, D 제공 → E가 노드로 감쌈)

에이전트는 **그래프를 모르는 순수 함수**다. `langgraph` import 금지.

```python
# src/techeval/agents/_deps.py (E 제공)
class Deps(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    retriever: BaseRetriever
    web_search: WebSearchFn
    llm: Any                       # langchain BaseChatModel (LLM_MODEL)
    judge_llm: Any | None = None   # JUDGE_MODEL, judge 노드에만 주입
    now: Callable[[], str]         # ISO datetime 생성기 (테스트에서 고정)


class AgentInput(BaseModel):
    tech: TechRef
    domain: str = DOMAIN
    tech_profile: TechProfile | None = None       # 기술 조사 완료 후 관점 에이전트에 전달
    trl_eval: list[CriterionResult] = []          # 도메인·시장 에이전트가 참고 가능
    missing_criteria: list[CriterionId] | None = None  # 재실행 시 이 기준만 생성
    previous_results: list[CriterionResult] = []  # 재실행 시 기존 결과 (부분 갱신용)
    retry_count: int = 0
    rewritten_queries: list[str] = []             # 질의 재작성 노드가 제안한 검색어
```

| 함수 | 소유 | 시그니처 | 반환 |
|---|---|---|---|
| `run_tech_research` | B | `(inp: AgentInput, deps: Deps) -> TechResearchOutput` | `TechProfile` + T1~T4 `CriterionResult` 4개 |
| `run_domain_eval` | B | `(inp: AgentInput, deps: Deps) -> list[CriterionResult]` | D1~D4 4개 (missing 지정 시 그 기준만) |
| `run_market_eval` | C | `(inp: AgentInput, deps: Deps) -> list[CriterionResult]` | M1~M3 3개 |
| `run_stakeholder_eval` | C | `(inp: AgentInput, deps: Deps) -> list[CriterionResult]` | S1~S4 4개 |
| `run_synthesis` | D | `(inp: SynthesisInput, deps: Deps) -> SynthesisResult` | 추가 검색 없음. `deps.retriever`/`web_search` 호출 금지 |
| `run_report` | D | `(inp: ReportInput, deps: Deps) -> str` | `report_md` (마크다운 문자열). 추가 검색 없음 |
| `render_pdf` | D | `(report_md: str, out_path: str) -> str` | 생성된 PDF 경로 |
| `format_reference` | D | `(e: Evidence) -> str` | 참고문헌 1줄 (CRITERIA.md REFERENCE 형식) |

```python
class TechResearchOutput(BaseModel):
    tech_profile: TechProfile
    trl_eval: list[CriterionResult]    # T1~T4


class SynthesisInput(BaseModel):
    technologies: list[TechRef]
    tech_profiles: list[TechProfile]
    trl_eval: list[CriterionResult]
    market_eval: list[CriterionResult]
    stakeholder_eval: list[CriterionResult]
    domain_eval: list[CriterionResult]
    counter_evidence: list[Evidence] = []


class ReportInput(BaseModel):
    technologies: list[TechRef]
    domain: str
    tech_profiles: list[TechProfile]
    trl_eval: list[CriterionResult]
    market_eval: list[CriterionResult]
    stakeholder_eval: list[CriterionResult]
    domain_eval: list[CriterionResult]
    counter_evidence: list[Evidence]
    synthesis: SynthesisResult
    evidence_gap: EvidenceGap
    judge_result: JudgeResult | None = None   # 재생성 시 수정 지시 반영
    previous_report_md: str | None = None
```

공통 요구:
- 반환 리스트의 `criterion_id` 집합은 `missing_criteria`(없으면 해당 관점 전체)와 정확히 일치해야 한다. 부족하면 E의 검사 노드가 재실행을 건다.
- `evidence[].chunk_id`/`url`은 실제로 `deps.retriever`/`deps.web_search`가 돌려준 값이어야 한다. E가 `control/perspective_check.py`에서 `retriever.get_chunk`로 대조한다.
- `confidence`는 `schemas.compute_confidence(evidence: list[Evidence]) -> Confidence` (E 제공 헬퍼)로 계산해 채운다.
- 두 기술을 처리하는 코드는 `inp.tech` 하나만 본다. 두 기술 루프는 E의 `Send`가 담당한다.

---

## 6. 제어 노드 계약 (E)

제어 노드는 별도 의견을 생성하지 않는다. 흐름과 품질만 관리한다.

| 노드 | 파일 | 입력 | 출력(State 키) | 판정 |
|---|---|---|---|---|
| 기술 근거 검사 | `control/tech_evidence_check.py` | `tech_profiles`, `trl_eval` | `missing_criteria["trl"]`, `retry_counts["trl:<tech>"]` | TechProfile 필수 필드(principle/limitations/measurements/validation_env) + T1~T4 + 각 evidence locator 실존 여부 |
| 질의 재작성 | `control/query_rewrite.py` | `missing_criteria`, 이전 검색어 | `rewritten_queries` (Send payload) | 부족 항목별 대체 검색어 2~3개 생성 (LLM 사용 가능, 의견 생성 아님) |
| 관점별 근거 검사 | `control/perspective_check.py` | 4개 `*_eval` | `missing_criteria`, `retry_counts` | 15기준 × 2기술 = 30개 존재, Pydantic 통과, evidence 실존, S2/S3 4주체 각 ≥1, S4 ≥1쌍, D2 measurement |
| 근거 비대칭 검사 | `control/evidence_gap.py` | `synthesis`, 4개 `*_eval` | `evidence_gap` | 기술별 evidence 수 비율 > 2:1 이거나 어느 기술의 반대 근거가 0이면 `needs_counter_search=True` |
| 반대 근거 탐색 | `control/counter_evidence.py` | `evidence_gap` | `counter_evidence` | 부족한 방향으로 `web_search`+`retriever` 1회. 결과는 Evidence 리스트, 의견 없음 |
| 보고서 검수 | `control/judge.py` | `report_md`, `ReportInput` | `judge_result` | `JUDGE_MODEL`로 5차원 1/3/5. 분량·마크다운 장식은 채점 금지 |

---

## 7. State 스키마 (`src/techeval/state.py`)

계획서 5.1 그대로. 키 이름·타입·reducer를 바꾸지 않는다.

```python
import operator
from typing import Annotated, TypedDict


class GraphState(TypedDict, total=False):
    # 초기 입력 (overwrite)
    domain: str
    technologies: list[TechRef]              # Human 고정 (mla, pim_cxl)
    stakeholders: tuple[str, ...]            # STAKEHOLDERS 상수

    # 기술 조사 (append, 기술별 Send)
    tech_profiles: Annotated[list[TechProfile], operator.add]
    trl_eval: Annotated[list[CriterionResult], operator.add]

    # 관점 평가 (append, 기술별 Send) — 병렬 실행, 서로 다른 키
    market_eval: Annotated[list[CriterionResult], operator.add]
    stakeholder_eval: Annotated[list[CriterionResult], operator.add]
    domain_eval: Annotated[list[CriterionResult], operator.add]

    # 제어 (overwrite)
    missing_criteria: dict[str, list[str]]   # {"trl": ["T3"], "market": [], ...} 관점별 (재실행 대상은 기술별로 payload에서 구분)
    retry_counts: dict[str, int]             # {"trl:mla": 1, "market:pim_cxl": 0, "counter": 0, "report": 0}
    counter_evidence: list[Evidence]
    synthesis: SynthesisResult
    evidence_gap: EvidenceGap
    report_md: str
    judge_result: JudgeResult
    report_pdf_path: str
```

- `operator.add` 키는 재실행 시 **중복이 쌓인다**. E는 검사 노드에서 `(tech_id, criterion_id)` 기준 최신(`generated_at` 최대) 것만 취하는 `latest_by_criterion()` 헬퍼를 `state.py`에 두고, 종합·보고서 입력을 만들 때 이 헬퍼를 거친다. D는 이미 중복 제거된 입력만 받는다.
- `retry_counts` 키 형식: `"<perspective>:<tech_id>"`, `"counter"`, `"report"`.
- `missing_criteria`는 관점별 합집합 키(`"trl"`)와 함께 기술별 키(`"trl:mla"`)도 같은 dict에 기록한다(E의 그래프가 재실행 대상을 Send payload로 나눌 때 사용). 하류는 관점별 키만 읽으면 된다.

---

## 8. Graph 흐름 (E)

```
START
 └─ init (technologies, stakeholders, domain 세팅)
 └─ Send × 2 ─▶ tech_research (B)            # 기술별 fan-out
 └─ tech_evidence_check ──부족(retry<2)──▶ query_rewrite ─▶ Send ─▶ tech_research (해당 기술만)
        │ 충분 (또는 retry 상한 → not_public 확정)
        ▼
 ┌─ Send × 2 ─▶ market_eval (C)      ┐
 ├─ Send × 2 ─▶ stakeholder_eval (C) ├─ 병렬 fan-out, fan-in
 └─ Send × 2 ─▶ domain_eval (B)      ┘
        ▼
 perspective_check ──부족 기준 분기──▶ T → tech_research / M·S·D → 해당 에이전트 (해당 기술·기준만, retry<2)
        │ 충분
        ▼
 synthesis (D)
        ▼
 evidence_gap_check ──있음(retry<1)──▶ counter_evidence_search ─▶ synthesis 재실행
        │ 없음
        ▼
 report (D)
        ▼
 judge ──미달(retry<1)──▶ report (judge_result 반영)
        │ 통과 (또는 상한)
        ▼
 render_pdf (D) ─▶ END
```

상한: 기술 조사 재검색 2회, 관점별 재실행 기준별 2회, 반대 근거 탐색 1회, 보고서 재생성 1회. 상한 도달 시 남은 항목은 `not_public`/`gaps`로 확정하고 다음 단계로 진행한다(무한 루프 금지).

---

## 9. 스텁 및 픽스처

| 픽스처 | 제공 | 스키마 | 소비 |
|---|---|---|---|
| `tests/fixtures/chunks.json` | A | `list[RetrievedChunk]` (doc 4편, 각 ≥5청크, 표 청크 포함) | B, C, E |
| `tests/fixtures/web_results.json` | C | `dict[str, list[WebResult]]` (키=검색어 키워드) | B, C, E |
| `tests/fixtures/tech_profiles.json` | B | `list[TechProfile]` (2개) | C, D, E |
| `tests/fixtures/trl_eval.json` | B | `list[CriterionResult]` (8개) | D, E |
| `tests/fixtures/domain_eval.json` | B | `list[CriterionResult]` (8개) | D, E |
| `tests/fixtures/market_eval.json` | C | `list[CriterionResult]` (6개) | D, E |
| `tests/fixtures/stakeholder_eval.json` | C | `list[CriterionResult]` (8개) | D, E |
| `tests/fixtures/counter_evidence.json` | E | `list[Evidence]` | D |
| `tests/fixtures/synthesis.json` | D | `SynthesisResult` | E |
| `tests/fixtures/evidence_gap.json` | E | `EvidenceGap` | D |
| `tests/fixtures/report_md.md` | D | str | E |
| `tests/fixtures/judge_result.json` | E | `JudgeResult` | D |
| `tests/fixtures/state_initial.json` | E | `GraphState` 초기값 | 전원 |

- 픽스처는 **실제 논문 내용 기반**으로 그럴듯하게 작성한다(하류의 프롬프트 튜닝에 쓰이므로). 단, 빈 필드로 스키마를 통과시키지 않는다.
- `tests/test_fixtures.py`(E)가 모든 픽스처를 스키마로 검증한다. 픽스처가 스키마를 깨면 main에 머지되지 않는다.
- 스텁 LLM: `src/techeval/stub_llm.py`(E)의 `FakeStructuredLLM`. `scripts/run.py --stub`과 테스트가 함께 쓰므로 `tests/` 밖에 둔다(`tests/conftest.py`의 `fake_llm`/`deps_stub` 픽스처가 이를 주입). `with_structured_output(Model)` 호출 시 프롬프트에서 `tech_id`(`tech_id: mla` 표기 권장)·`criterion_id`를 추출해 픽스처에서 해당 Model 인스턴스를 돌려준다. `list[CriterionResult]`와 각 역할이 정의한 래퍼 BaseModel(필드가 계약 모델/리스트인 경우)도 지원한다. 매칭 실패 시 `FixtureLookupError`. 각 역할은 이걸로 자기 에이전트 함수의 흐름(검색 호출 → 프롬프트 조립 → 검증)을 테스트하고, `llm.calls`로 프롬프트 내용을 검증한다.

---

## 10. 검증 규칙 (validator 또는 검사 노드로 강제)

| # | 규칙 | 강제 위치 |
|---|---|---|
| V1 | `CriterionResult.evidence` ≥ 1 | schemas validator |
| V2 | `not_public` 레벨 ⇒ not_public evidence(검색어·검색일) 포함 | schemas validator |
| V3 | D1~D3 ⇒ measurements ≥ 1 (not_public 제외) | schemas validator |
| V4 | S2/S3 ⇒ 4주체 각 ≥1, S4 ⇒ ≥1쌍 | perspective_check |
| V5 | evidence.chunk_id가 retriever에 실존, url이 web 결과에 실존 | perspective_check, tech_evidence_check |
| V6 | evidence.quote가 해당 chunk.text의 부분 문자열(공백 정규화 후) | perspective_check (paper 출처) |
| V7 | `confidence` = `compute_confidence(evidence)` 와 일치 | perspective_check |
| V8 | 30개 (tech × criterion) 모두 존재 | perspective_check |
| V9 | 보고서 REFERENCE는 본문에서 실제 인용된 evidence_id만 포함 | judge + D 단위 테스트 |
| V10 | 보고서에 우열·추천·순위 표현 금지 (금칙어 리스트: "더 우수", "추천", "1위", "종합 점수" 등) | judge(neutrality) + D 단위 테스트 |
| V11 | 관점별 레벨을 합산·평균한 값이 State/보고서에 존재하지 않음 | judge |
