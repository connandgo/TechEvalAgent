"""공통 Pydantic 스키마 및 상수.

`docs/CONTRACTS.md` §1~§2와 1:1로 대응한다. 필드·이름을 바꾸려면 CONTRACT CHANGE 합의가 필요하다.
"""

import logging
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. 상수
# ---------------------------------------------------------------------------

TechId = Literal["mla", "pim_cxl"]
Perspective = Literal["trl", "market", "stakeholder", "domain"]
CriterionId = Literal[
    "T1",
    "T2",
    "T3",
    "T4",
    "M1",
    "M2",
    "M3",
    "S1",
    "S2",
    "S3",
    "S4",
    "D1",
    "D2",
    "D3",
    "D4",
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

ALL_CRITERIA: tuple[str, ...] = tuple(c for cs in PERSPECTIVE_CRITERIA.values() for c in cs)

# criterion_id → perspective 역방향 조회
CRITERION_PERSPECTIVE: dict[str, str] = {c: p for p, cs in PERSPECTIVE_CRITERIA.items() for c in cs}

STAKEHOLDERS: tuple[str, ...] = (
    "model_developer",  # 모델 개발사
    "cloud_serving_operator",  # 클라우드/서빙 운영사
    "memory_semiconductor_vendor",  # 메모리/반도체 벤더
    "investor",  # 투자 업계
)

DOMAIN = "대규모 데이터센터의 장문맥 LLM 추론 환경"

MAX_RETRY_PER_PERSPECTIVE = 2  # 기술 조사 재검색, 관점별 재실행 상한
MAX_COUNTER_EVIDENCE_SEARCH = 1
MAX_REPORT_REGENERATION = 1

DOC_IDS: tuple[str, ...] = ("deepseek_v2", "pim_cxl_1m", "io_survey", "kv_survey")
SURVEY_DOC_IDS: tuple[str, ...] = ("io_survey", "kv_survey")


# ---------------------------------------------------------------------------
# 2. 공통 스키마
# ---------------------------------------------------------------------------


class TechRef(BaseModel):
    """Human이 고정한 평가 대상 기술. State.technologies에 2개."""

    tech_id: TechId
    name: str  # "DeepSeek-V2 MLA" / "PIM/CXL"
    approach: Literal["sw", "hw"]
    family: str  # "MLA 계열" / "CXL·PIM/PNM 계열" — 시장·이해관계자 조사 범위
    primary_doc_id: DocId  # 선정 논문의 doc_id
    paper_title: str
    paper_date: str  # "2024-06" 형식
    search_aliases: list[str] = []  # 검색어 확장용


class Measurement(BaseModel):
    """성능 수치 1건. 수치는 반드시 측정 조건과 함께 저장한다 (AGENTS.md 규칙 5)."""

    metric: str  # "KV cache reduction", "TTFT", "tokens/s", ...
    value: str  # 원문 표기 그대로 ("93.3%", "2.1x")
    unit: str | None = None
    model_size: str | None = None  # "236B (21B active)"
    context_length: str | None = None  # "128K"
    hardware: str | None = None  # "8x H800"
    baseline: str | None = None  # "DeepSeek 67B (MHA)"
    batch_size: str | None = None
    condition_note: str | None = None  # 그 외 조건
    evidence_id: str  # 이 수치의 출처 Evidence.evidence_id


class Evidence(BaseModel):
    """근거 1건. 모든 CriterionResult / TechProfile / counter_evidence에 포함된다."""

    evidence_id: str  # 규칙: "{tech_id}-{criterion_id|PROFILE|COUNTER}-{2자리}" 예 "mla-T1-01"
    source_type: SourceType
    unit: EvidenceUnit  # paper=논문/구현 단위, family=기술 계열 단위
    quote: str  # 원문 인용 (원문 언어 그대로). inference/not_public이면 요약 또는 ""
    locator: str  # 논문: "deepseek_v2 p.4 §2.1.4" / 웹: URL / not_public: "not_found"
    title: str | None = None
    authors: str | None = None  # "DeepSeek-AI" / "Kim, J. et al."
    publisher: str | None = None  # 학술지·사이트명·기관
    published_date: str | None = None  # "2024-06-19" 또는 "2024-06"
    url: str | None = None
    accessed_date: str | None = None  # 웹 자료 확인일 "YYYY-MM-DD"
    doc_id: DocId | None = None  # source_type=paper일 때
    page: int | None = None
    section: str | None = None
    chunk_id: str | None = None  # retriever가 반환한 chunk_id (검증용)
    # not_public 전용
    search_query: str | None = None
    searched_at: str | None = None
    search_scope: str | None = None  # "arxiv, vendor blogs, news 2024-2026"

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
    level: str  # CRITERIA.md의 레벨 문자열 그대로. 예 "TRL 4-6", "L2", "not_public"
    level_estimate: str | None = None  # T1 전용: 구간 내 추정치 "TRL 5 (공개 정보 기반 추정)"
    content: str  # 2~4문장 평가 내용 (한국어)
    evidence: list[Evidence] = Field(min_length=1)
    confidence: Confidence
    evidence_unit: EvidenceUnit  # 이 판정의 주된 근거 단위
    measurements: list[Measurement] = []  # D1~D3, T 수치 포함 시 필수
    details: dict = {}  # 기준별 구조화 필드 (CRITERIA.md 각 기준의 details 키 참조)
    generated_at: str  # ISO datetime
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
    principle: str  # 핵심 접근 (2~5문장)
    scope: str  # 적용 범위·전제 조건
    limitations: list[str]  # 원문에 명시된 한계
    measurements: list[Measurement]
    validation_env: str  # 검증 환경 서술 (T2 판정의 입력)
    public_artifacts: dict[str, str]  # {"code": "Y|N|unknown", "model_or_design": ..., ...}
    public_artifact_urls: dict[str, str] = {}
    evidence: list[Evidence] = Field(min_length=1)
    search_queries_used: list[str] = []
    generated_at: str
    retry_count: int = 0


class Agreement(BaseModel):
    tech_id: TechId | None  # None이면 두 기술 공통
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
    statement: str  # 무엇이 상충하는가
    cause: str  # 왜 — 기준 차이 / 평가 단위 차이 / 근거 유형 차이 (우열로 서술 금지)
    evidence_ids: list[str] = Field(min_length=2)


class Gap(BaseModel):
    tech_id: TechId
    criterion_id: CriterionId
    gap_type: Literal["not_public", "unit_mismatch", "single_source", "inference_only", "opposing_missing"]
    description: str


class SynthesisResult(BaseModel):
    agreements: list[Agreement]
    conflicts: list[Conflict] = Field(min_length=1)  # S4가 기술별 최소 1쌍이므로 최소 1
    gaps: list[Gap]
    unit_notes: str  # 평가 단위(논문 vs 계열) 차이가 판정에 미친 영향
    evidence_asymmetry_note: str  # 근거 비대칭 서술
    generated_at: str


class EvidenceGap(BaseModel):
    """근거 비대칭 검사 노드(E) 출력. State.evidence_gap."""

    asymmetry: bool
    evidence_count: dict[str, int]  # {"mla": 23, "pim_cxl": 11}
    vendor_source_ratio: dict[str, float]  # 제안사/벤더 발표 자료 비율 (보고서 한계점에 사용)
    opposing_missing: list[dict]  # [{"tech_id": "mla", "criterion_id": "D1", "reason": "..."}]
    needs_counter_search: bool
    note: str


JudgeDimension = Literal["evidence", "criteria_compliance", "neutrality", "specificity", "format"]
JUDGE_DIMENSIONS: tuple[str, ...] = ("evidence", "criteria_compliance", "neutrality", "specificity", "format")


class JudgeResult(BaseModel):
    """보고서 검수 노드(E) 출력. 5차원 1/3/5점."""

    scores: dict[JudgeDimension, Literal[1, 3, 5]]
    reasons: dict[str, str]
    missing_required: list[str]  # 누락된 필수 기준/챕터
    revision_instructions: list[str]
    passed: bool  # 모든 점수 >= 3 이고 missing_required 비어 있음
    judged_at: str


# ---------------------------------------------------------------------------
# 고정 기술 목록 (Human 선정)
# ---------------------------------------------------------------------------

TECHNOLOGIES: list[TechRef] = [
    TechRef(
        tech_id="mla",
        name="DeepSeek-V2 MLA",
        approach="sw",
        family="MLA 계열",
        primary_doc_id="deepseek_v2",
        paper_title="DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model",
        paper_date="2024-06",
        search_aliases=[
            "Multi-head Latent Attention",
            "MLA",
            "DeepSeek-V2",
            "latent KV compression",
            "low-rank KV joint compression",
        ],
    ),
    TechRef(
        tech_id="pim_cxl",
        name="PIM/CXL",
        approach="hw",
        family="CXL·PIM/PNM 메모리 계열",
        primary_doc_id="pim_cxl_1m",
        paper_title="1M-Token LLM Inference with PIM/CXL Memory Expansion",
        paper_date="2025-10",
        search_aliases=[
            "Processing-in-Memory",
            "PIM",
            "Processing-near-Memory",
            "PNM",
            "CXL memory expansion",
            "CXL-PNM",
            "KV cache offloading CXL",
        ],
    ),
]


def get_tech(tech_id: str) -> TechRef:
    """tech_id로 TechRef를 찾는다. 없으면 KeyError."""
    for t in TECHNOLOGIES:
        if t.tech_id == tech_id:
            return t
    raise KeyError(f"unknown tech_id: {tech_id}")


# ---------------------------------------------------------------------------
# 신뢰도 계산 (AGENTS.md 규칙 3 — LLM이 아니라 코드로 계산)
# ---------------------------------------------------------------------------


def _source_key(e: Evidence) -> str | None:
    """독립 출처 판별 키. 논문은 doc_id, 웹류는 URL/locator의 도메인."""
    if e.source_type == "paper":
        return f"doc:{e.doc_id}" if e.doc_id else f"loc:{e.locator}"
    raw = e.url or e.locator
    host = urlparse(raw).netloc.lower() if "://" in raw else ""
    if host:
        host = host.removeprefix("www.")
        return f"host:{host}"
    return f"loc:{raw}" if raw else None


def compute_confidence(evidence: list[Evidence]) -> Confidence:
    """CRITERIA §1 신뢰도 규칙.

    - `inference` 또는 `not_public` 근거가 하나라도 있으면 `low`
    - 독립 출처(doc_id 또는 URL 도메인 기준) 2개 이상이고 `paper`/`official`을 포함하면 `high`
    - 그 외 `medium`
    """
    if not evidence:
        return "low"
    if any(e.source_type in ("inference", "not_public") for e in evidence):
        return "low"
    keys = {k for k in (_source_key(e) for e in evidence) if k}
    has_primary = any(e.source_type in ("paper", "official") for e in evidence)
    if len(keys) >= 2 and has_primary:
        return "high"
    return "medium"
