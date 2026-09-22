"""(C) market_eval.json / stakeholder_eval.json 생성 스크립트.

픽스처를 손으로 고치지 않고 이 스크립트로 다시 만든다. 규칙:

- 웹 근거의 `url`은 `web_results.json`에 실재하는 URL만 쓴다 (E의 V5 대조 대상).
- 논문 근거의 `chunk_id`는 `chunks.json`에 실재해야 하고 `quote`는 해당 청크
  `text`의 부분 문자열이어야 한다 (V5·V6).
- `confidence`는 `compute_confidence()`로 계산한다 (V7). 손으로 적지 않는다.
- `evidence_id`는 `{tech_id}-{criterion_id}-{NN}`.
- 우열·순위 서술 금지. 두 기술은 각각 독립으로 기술한다.

실행: uv run python tests/fixtures/_make_c_fixtures.py
"""

import json
from pathlib import Path

from techeval.schemas import STAKEHOLDERS, CriterionResult, Evidence, compute_confidence

FIXTURES = Path(__file__).resolve().parent
GENERATED_AT = "2026-09-20T10:00:00+00:00"

# web_results.json에서 URL -> (publisher, published_date, source_type) 를 끌어오기 위한 인덱스
_WEB = json.loads((FIXTURES / "web_results.json").read_text(encoding="utf-8"))
_BY_URL = {r["url"]: r for results in _WEB.values() for r in results}
_SOURCE_TYPE = {"official": "official", "paper": "paper"}
_CHUNKS = {
    c["chunk_id"]: c
    for c in json.loads((FIXTURES / "chunks.json").read_text(encoding="utf-8"))
}


def web(evidence_id: str, url: str, quote: str, unit: str = "family") -> Evidence:
    """web_results.json의 실제 결과 1건을 Evidence로 만든다."""
    r = _BY_URL[url]
    body = r.get("content") or r["snippet"]
    assert quote in body, f"{evidence_id}: quote가 {url} 의 본문/스니펫에 없다"
    return Evidence(
        evidence_id=evidence_id,
        source_type=_SOURCE_TYPE.get(r["source_kind"], "web"),
        unit=unit,
        quote=quote,
        locator=url,
        title=r["title"],
        publisher=r.get("publisher"),
        published_date=r.get("published_date"),
        url=url,
        accessed_date=r["fetched_at"][:10],
    )


def paper(evidence_id: str, chunk_id: str, quote: str, unit: str = "paper") -> Evidence:
    """chunks.json의 실제 청크를 Evidence로 만든다. quote는 원문 부분 문자열."""
    c = _CHUNKS[chunk_id]
    assert quote in c["text"], f"{evidence_id}: quote가 청크 {chunk_id} 원문에 없다"
    return Evidence(
        evidence_id=evidence_id,
        source_type="paper",
        unit=unit,
        quote=quote,
        locator=f"{c['doc_id']} p.{c['page']}"
        + (f" §{c['section']}" if c.get("section") else ""),
        title=c["doc_title"],
        doc_id=c["doc_id"],
        page=c["page"],
        section=c.get("section"),
        chunk_id=chunk_id,
    )


def inference(
    evidence_id: str, basis: str, quote: str, unit: str = "family"
) -> Evidence:
    """직접 자료가 없어 추론한 항목. basis = 추론의 출발점이 된 evidence_id."""
    return Evidence(
        evidence_id=evidence_id,
        source_type="inference",
        unit=unit,
        quote=quote,
        locator=f"inference from {basis}",
    )


def result(
    tech_id: str,
    perspective: str,
    criterion_id: str,
    level: str,
    content: str,
    evidence: list[Evidence],
    evidence_unit: str,
    details: dict,
) -> CriterionResult:
    return CriterionResult(
        tech_id=tech_id,
        perspective=perspective,
        criterion_id=criterion_id,
        level=level,
        content=content,
        evidence=evidence,
        confidence=compute_confidence(evidence),
        evidence_unit=evidence_unit,
        details=details,
        generated_at=GENERATED_AT,
    )


# --- URL 상수 (web_results.json에 실재) -----------------------------------------

VLLM_DOCS = "https://docs.vllm.ai/en/latest/models/supported_models.html"
VLLM_REPO = "https://github.com/vllm-project/vllm"
SGLANG = "https://docs.sglang.ai/backend/attention_backend.html"
KV_SURVEY = "https://arxiv.org/abs/2412.19442"
DS_PAPER = "https://arxiv.org/abs/2405.04434"
DS_PRICING = "https://api-docs.deepseek.com/quick_start/pricing"
DS_HF = "https://huggingface.co/deepseek-ai/DeepSeek-V2"
NEXTPLATFORM = "https://www.nextplatform.com/2025/04/08/open-weight-moe-models-move-into-production-serving-stacks/"
SEMIANALYSIS = (
    "https://semianalysis.com/2025/05/12/the-economics-of-long-context-inference/"
)
REUTERS_CLOUD = "https://www.reuters.com/technology/cloud-providers-expand-ai-inference-capacity-2026-02-11/"
REUTERS_MEM = "https://www.reuters.com/technology/memory-makers-raise-capital-spending-ai-dram-2025-12-03/"
REGISTER_INVEST = (
    "https://www.theregister.com/2026/01/19/memory_centric_ai_hardware_investment/"
)
CXL_SPEC = "https://computeexpresslink.org/cxl-specification/"
ASTERA = "https://www.astera-labs.com/products/leo-cxl-smart-memory-controllers/"
STH_CXL = "https://www.servethehome.com/cxl-memory-expansion-what-is-actually-shipping/"
SAMSUNG_PIM = "https://semiconductor.samsung.com/dram/hbm/hbm-pim/"
SKH_AIM = "https://news.skhynix.com/sk-hynix-aim-accelerator-card-generative-ai/"
EETIMES = "https://www.eetimes.com/processing-in-memory-parts-remain-sampling-stage-for-ai-servers/"
TRENDFORCE = "https://www.trendforce.com/presscenter/news/20250318-cxl-memory-module-outlook.html"
MNM = "https://www.marketsandmarkets.com/Market-Reports/compute-express-link-market.asp"
JEDEC = "https://www.jedec.org/news/pressreleases/jedec-publishes-standards-ai-memory-modules"
CXL_JEDEC = "https://computeexpresslink.org/blog/cxl-consortium-jedec-collaboration/"
PNM_PAPER = "https://arxiv.org/abs/2511.00321"


def market_results() -> list[CriterionResult]:
    out: list[CriterionResult] = []

    # --- MLA (SW — KV cache 축소 계열) ---------------------------------------
    out.append(
        result(
            "mla",
            "market",
            "M1",
            "L2",
            "장문맥 추론의 메모리·비용 부담을 수요 동인으로 지목한 2차 자료가 확인된다. "
            "다만 MLA를 포함한 KV cache 축소 기법 계열을 따로 떼어 집계한 정량 시장 전망은 "
            "확인되지 않았고, 확인된 수치는 장문맥 추론 비용 일반에 대한 것이다. "
            "단일 매체의 추정이므로 사실로 단정하지 않는다.",
            [
                web(
                    "mla-M1-01",
                    SEMIANALYSIS,
                    "memory footprint per session therefore dominates cost per token at 128K contexts",
                ),
                web(
                    "mla-M1-02",
                    REUTERS_CLOUD,
                    "memory pricing has become a material input to inference service margins",
                ),
            ],
            "family",
            {
                "market_figures": [
                    {
                        "figure": "장문맥 요청의 KV cache 비용 비중 상승 (정량 전망 아님, 매체 분석)",
                        "publisher": "SemiAnalysis",
                        "published_date": "2025-05-12",
                        "evidence_id": "mla-M1-01",
                    },
                    {
                        "figure": "클라우드 사업자의 추론 처리 용량 증설 (금액 미명시)",
                        "publisher": "Reuters",
                        "published_date": "2026-02-11",
                        "evidence_id": "mla-M1-02",
                    },
                ]
            },
        )
    )
    out.append(
        result(
            "mla",
            "market",
            "M2",
            "L4",
            "제안사인 DeepSeek이 MLA를 적용한 모델을 유료 API로 운영 중인 것이 공식 가격 문서에서 "
            "확인된다. 오픈 웨이트 배포로 제3자 서빙 스택에서의 구동도 2차 자료에서 확인되나, "
            "해당 자료는 구체적 운영 주체를 명시하지 않아 별도 채택 사례로 세지 않았다.",
            [
                web(
                    "mla-M2-01",
                    DS_PRICING,
                    "serves MLA-based models through a commercial inference API with published per-token pricing",
                ),
                web(
                    "mla-M2-02",
                    DS_HF,
                    "Weights are released under the DeepSeek model license",
                ),
                web(
                    "mla-M2-03",
                    NEXTPLATFORM,
                    "deploying MLA-based open-weight models in production",
                ),
            ],
            "family",
            {
                "adopters": [
                    {
                        "name": "DeepSeek",
                        "stage": "production",
                        "evidence_id": "mla-M2-01",
                    },
                    {
                        "name": "DeepSeek (오픈 웨이트 배포)",
                        "stage": "product",
                        "evidence_id": "mla-M2-02",
                    },
                ]
            },
        )
    )
    out.append(
        result(
            "mla",
            "market",
            "M3",
            "L2",
            "오픈소스 서빙 프레임워크 2종이 MLA 전용 어텐션 백엔드를 문서화하고 있으며, "
            "이 백엔드가 전제하는 압축 latent KV 표현은 원문 2.1.2절의 저랭크 결합 압축 설명과 "
            "일치한다. 반면 MLA를 지원 사양으로 내건 하드웨어 벤더 제품이나 표준화 활동은 "
            "확인되지 않았다.",
            [
                web(
                    "mla-M3-01",
                    VLLM_DOCS,
                    "vLLM implements a dedicated MLA (Multi-head Latent Attention) backend",
                ),
                web("mla-M3-02", SGLANG, "SGLang supports MLA-based models"),
                paper(
                    "mla-M3-03",
                    "deepseek_v2:007:12",
                    "low-rank joint compression for keys and values",
                ),
                web(
                    "mla-M3-04",
                    KV_SURVEY,
                    "low-rank latent compression such as Multi-head Latent Attention",
                ),
            ],
            "family",
            {
                "checklist": {
                    "framework": "Y",
                    "vendor_product": "N",
                    "standardization": "N",
                    "third_party_research_tools": "Y",
                }
            },
        )
    )

    # --- PIM/CXL (HW — 메모리 계층 확장 계열) ---------------------------------
    out.append(
        result(
            "pim_cxl",
            "market",
            "M1",
            "L2",
            "CXL 메모리 모듈에 대한 정량 시장 전망이 조사기관 2곳에서 제시된다. 두 자료 모두 "
            "발행 주체의 추정치이며 산출 근거가 공개돼 있지 않아 사실로 단정하지 않는다. "
            "수요 동인으로는 메모리 제조사의 설비 투자 확대가 언론 보도에서 확인된다.",
            [
                web(
                    "pim_cxl-M1-01",
                    TRENDFORCE,
                    "projects the CXL memory module market to reach USD 2.1 billion by 2028",
                ),
                web(
                    "pim_cxl-M1-02",
                    MNM,
                    "estimates a compound annual growth rate in the mid-thirties for CXL-attached memory through 2030",
                ),
                web(
                    "pim_cxl-M1-03",
                    REUTERS_MEM,
                    "AI-specific DRAM products require dedicated process and packaging investment",
                ),
            ],
            "family",
            {
                "market_figures": [
                    {
                        "figure": "CXL 메모리 모듈 출하 확대 전망 (조사기관 추정)",
                        "publisher": "TrendForce",
                        "published_date": "2025-03-18",
                        "evidence_id": "pim_cxl-M1-01",
                    },
                    {
                        "figure": "CXL 시장 규모 전망 (조사기관 추정, 유료 보고서로 수치 비공개)",
                        "publisher": "MarketsandMarkets",
                        "published_date": "2025-08-01",
                        "evidence_id": "pim_cxl-M1-02",
                    },
                ]
            },
        )
    )
    out.append(
        result(
            "pim_cxl",
            "market",
            "M2",
            "L3",
            "CXL 메모리 확장 컨트롤러는 벤더의 양산 제품으로 확인된다. 반면 연산 기능을 메모리에 "
            "둔 PIM/PNM 부품은 벤더 샘플링 단계에 머문다는 보도가 있다. 선정 논문의 CXL-PNM "
            "시제품은 계열 제품의 양산 사실과 구분해 기록한다 — 계열 근거로 논문 시제품의 "
            "채택을 주장하지 않는다.",
            [
                web(
                    "pim_cxl-M2-01",
                    ASTERA,
                    "Leo CXL memory controllers are shipping in production platforms",
                ),
                web(
                    "pim_cxl-M2-02",
                    STH_CXL,
                    "deployments reported so far target capacity-bound in-memory workloads rather than LLM KV cache offload",
                ),
                web(
                    "pim_cxl-M2-03",
                    SAMSUNG_PIM,
                    "DRAM with in-bank programmable compute units",
                ),
                web(
                    "pim_cxl-M2-04",
                    EETIMES,
                    "no vendor has disclosed volume production for LLM inference servers",
                ),
            ],
            "family",
            {
                "adopters": [
                    {
                        "name": "Astera Labs (CXL 메모리 확장 컨트롤러)",
                        "stage": "product",
                        "evidence_id": "pim_cxl-M2-01",
                    },
                    {
                        "name": "Samsung (HBM-PIM)",
                        "stage": "poc",
                        "evidence_id": "pim_cxl-M2-03",
                    },
                    {
                        "name": "SK hynix (AiM)",
                        "stage": "poc",
                        "evidence_id": "pim_cxl-M2-04",
                    },
                ]
            },
        )
    )
    out.append(
        result(
            "pim_cxl",
            "market",
            "M3",
            "L2",
            "CXL은 컨소시엄 사양과 JEDEC 협업으로 표준화 활동이 확인되고, 메모리 확장 컨트롤러가 "
            "벤더 제품으로 존재한다. 반면 오픈소스 서빙 프레임워크가 CXL 부착 메모리나 근접 연산 "
            "장치를 KV cache 저장 계층으로 문서화한 사례는 확인되지 않았다. 선정 논문이 기술하는 "
            "CXL-PNM 확장 플랫폼은 원문 근거로 함께 기록한다.",
            [
                web(
                    "pim_cxl-M3-01",
                    CXL_SPEC,
                    "CXL 3.1 defines Type-3 memory device semantics, fabric management and memory pooling",
                ),
                web(
                    "pim_cxl-M3-02",
                    CXL_JEDEC,
                    "a liaison agreement covering memory device standards",
                ),
                web(
                    "pim_cxl-M3-03",
                    ASTERA,
                    "support memory expansion and pooling for server hosts",
                ),
                paper(
                    "pim_cxl-M3-04",
                    "pim_cxl_1m:004:08",
                    "scalable, high-capacity, and high-bandwidth memory expansion platform",
                ),
            ],
            "family",
            {
                "checklist": {
                    "framework": "N",
                    "vendor_product": "Y",
                    "standardization": "Y",
                    "third_party_research_tools": "N",
                }
            },
        )
    )
    return out


def stakeholder_results() -> list[CriterionResult]:
    out: list[CriterionResult] = []

    # --- MLA ------------------------------------------------------------------
    out.append(
        result(
            "mla",
            "stakeholder",
            "S1",
            "assigned",
            "모델 구조를 바꾸는 기법이므로 채택 여부를 정하는 쪽은 모델 개발사다. 서빙 운영사는 "
            "백엔드 지원 여부에 따라 영향을 받고, 메모리·반도체 벤더는 이 기법이 자사 부품을 "
            "요구하지 않으므로 공급 관계가 성립하지 않는다. 투자 업계는 공개 자료에서 이 기법을 "
            "단독 투자 대상으로 다루지 않아 영향받는 쪽으로 기록한다.",
            [
                web(
                    "mla-S1-01",
                    DS_HF,
                    "the model card documents the MLA projection dimensions required by serving frameworks",
                ),
                web(
                    "mla-S1-02",
                    VLLM_DOCS,
                    "Enabling the backend requires a model config that declares MLA-specific projection dimensions",
                ),
                inference(
                    "mla-S1-03",
                    "mla-S1-02",
                    "서빙 스택 변경 없이도 동작하는 범위에 대한 추론",
                ),
            ],
            "family",
            {
                "roles": {
                    "model_developer": "decision_maker",
                    "cloud_serving_operator": "affected",
                    "memory_semiconductor_vendor": "unrelated",
                    "investor": "affected",
                }
            },
        )
    )
    out.append(
        result(
            "mla",
            "stakeholder",
            "S2",
            "narrative",
            "모델 개발사와 서빙 운영사의 편익은 공개 자료로 확인되며, 벤더와 투자 업계의 편익은 "
            "직접 자료가 없어 추론으로 기록한다.",
            [
                web(
                    "mla-S2-01",
                    VLLM_DOCS,
                    "reduces paged KV cache footprint relative to the standard MHA backend",
                ),
                web(
                    "mla-S2-02",
                    DS_PRICING,
                    "serves MLA-based models through a commercial inference API",
                ),
                inference(
                    "mla-S2-03",
                    "mla-S2-01",
                    "부품 교체를 요구하지 않는다는 점에서 파생되는 추론",
                ),
                inference(
                    "mla-S2-04",
                    "mla-S2-02",
                    "운영 비용 구조에서 파생되는 투자 관점 추론",
                ),
            ],
            "family",
            {
                "benefits": {
                    "model_developer": [
                        {
                            "text": "동일 하드웨어에서 KV cache 점유를 줄여 더 긴 컨텍스트나 더 큰 배치를 다룰 여지를 얻는다.",
                            "evidence_id": "mla-S2-01",
                            "is_inference": False,
                        }
                    ],
                    "cloud_serving_operator": [
                        {
                            "text": "지원 백엔드가 있는 서빙 스택에서 별도 하드웨어 증설 없이 모델을 제공할 수 있다.",
                            "evidence_id": "mla-S2-02",
                            "is_inference": False,
                        }
                    ],
                    "memory_semiconductor_vendor": [
                        {
                            "text": "부품 사양 변경을 요구하지 않으므로 기존 제품 라인을 그대로 공급할 수 있다.",
                            "evidence_id": "mla-S2-03",
                            "is_inference": True,
                        }
                    ],
                    "investor": [
                        {
                            "text": "하드웨어 증설 없이 적용 가능한 기법이므로 추가 설비 투자 없이 단가 구조가 바뀔 여지가 있다.",
                            "evidence_id": "mla-S2-04",
                            "is_inference": True,
                        }
                    ],
                }
            },
        )
    )
    out.append(
        result(
            "mla",
            "stakeholder",
            "S3",
            "narrative",
            "모델 구조 변경을 전제하므로 모델 개발사와 서빙 운영사에 전환 부담이 발생한다. "
            "벤더와 투자 업계의 부담은 직접 자료가 없어 추론으로 기록한다.",
            [
                web(
                    "mla-S3-01",
                    VLLM_DOCS,
                    "models without those fields fall back to the generic attention backend",
                ),
                web(
                    "mla-S3-02",
                    SGLANG,
                    "the absorbed projection variant used during decoding",
                ),
                inference("mla-S3-03", "mla-S3-01", "부품 수요 측면에서 파생되는 추론"),
                inference(
                    "mla-S3-04",
                    "mla-S3-02",
                    "구현 성숙도 불확실성에서 파생되는 투자 관점 추론",
                ),
            ],
            "family",
            {
                "burdens": {
                    "model_developer": [
                        {
                            "text": "MLA 전용 투영 차원을 선언한 모델 구성이 필요하므로 기존 모델에 사후 적용할 수 없다.",
                            "evidence_id": "mla-S3-01",
                            "is_inference": False,
                        }
                    ],
                    "cloud_serving_operator": [
                        {
                            "text": "디코딩 경로가 일반 어텐션과 달라 백엔드별로 별도 검증이 필요하다.",
                            "evidence_id": "mla-S3-02",
                            "is_inference": False,
                        }
                    ],
                    "memory_semiconductor_vendor": [
                        {
                            "text": "메모리 용량 증설로 해결하던 수요의 일부가 소프트웨어로 흡수될 여지가 있다.",
                            "evidence_id": "mla-S3-03",
                            "is_inference": True,
                        }
                    ],
                    "investor": [
                        {
                            "text": "구조 변경이 모델 재설계를 전제하므로 적용 시점과 범위를 외부에서 가늠하기 어렵다.",
                            "evidence_id": "mla-S3-04",
                            "is_inference": True,
                        }
                    ],
                }
            },
        )
    )
    out.append(
        result(
            "mla",
            "stakeholder",
            "S4",
            "narrative",
            "모델 개발사가 얻는 메모리 절감이 다른 주체에게는 부담으로 돌아오는 지점이 두 곳 "
            "확인된다. 두 항목 모두 직접 자료가 아닌 추론을 포함한다.",
            [
                web(
                    "mla-S4-01",
                    VLLM_DOCS,
                    "reduces paged KV cache footprint relative to the standard MHA backend",
                ),
                web(
                    "mla-S4-02",
                    SGLANG,
                    "the absorbed projection variant used during decoding",
                ),
                inference("mla-S4-03", "mla-S4-01", "수요 이전 방향에 대한 추론"),
            ],
            "family",
            {
                "tradeoffs": [
                    {
                        "beneficiary": "model_developer",
                        "burdened": "memory_semiconductor_vendor",
                        "text": "모델 개발사에게 편익인 메모리 점유 축소가, 메모리·반도체 벤더에게는 용량 증설 수요가 줄어드는 부담이 된다.",
                        "evidence_ids": ["mla-S4-01", "mla-S4-03"],
                    },
                    {
                        "beneficiary": "model_developer",
                        "burdened": "cloud_serving_operator",
                        "text": "모델 개발사에게 편익인 구조 변경이, 서빙 운영사에게는 전용 백엔드를 따로 검증해야 하는 부담이 된다.",
                        "evidence_ids": ["mla-S4-02"],
                    },
                ]
            },
        )
    )

    # --- PIM/CXL --------------------------------------------------------------
    out.append(
        result(
            "pim_cxl",
            "stakeholder",
            "S1",
            "assigned",
            "메모리 계층 자체를 바꾸는 접근이므로 부품을 공급하는 메모리·반도체 벤더가 공급자이고, "
            "장비를 도입할지 정하는 쪽은 데이터센터를 운영하는 클라우드·서빙 운영사다. 모델 "
            "개발사는 모델 구조 변경 없이 영향을 받는 쪽이며, 투자 업계는 메모리 설비 투자 "
            "보도에서 의사결정 주체로 등장한다.",
            [
                web(
                    "pim_cxl-S1-01",
                    ASTERA,
                    "Leo CXL memory controllers are shipping in production platforms",
                ),
                web(
                    "pim_cxl-S1-02",
                    STH_CXL,
                    "deployments reported so far target capacity-bound in-memory workloads",
                ),
                web(
                    "pim_cxl-S1-03",
                    REUTERS_MEM,
                    "AI-specific DRAM products require dedicated process and packaging investment",
                ),
            ],
            "family",
            {
                "roles": {
                    "model_developer": "affected",
                    "cloud_serving_operator": "decision_maker",
                    "memory_semiconductor_vendor": "supplier",
                    "investor": "decision_maker",
                }
            },
        )
    )
    out.append(
        result(
            "pim_cxl",
            "stakeholder",
            "S2",
            "narrative",
            "벤더·운영사·투자 업계의 편익은 공개 자료로 확인되며, 모델 개발사의 편익은 직접 "
            "자료가 없어 추론으로 기록한다.",
            [
                web(
                    "pim_cxl-S2-01",
                    ASTERA,
                    "appears to the host as additional system memory, so that applications use it without modification",
                ),
                web(
                    "pim_cxl-S2-02",
                    REUTERS_MEM,
                    "AI-specific DRAM products require dedicated process and packaging investment",
                ),
                web(
                    "pim_cxl-S2-03",
                    REGISTER_INVEST,
                    "memory-centric inference hardware as a longer-horizon position than accelerator supply",
                ),
                inference(
                    "pim_cxl-S2-04",
                    "pim_cxl-S2-01",
                    "모델 변경 불필요라는 점에서 파생되는 추론",
                ),
            ],
            "family",
            {
                "benefits": {
                    "model_developer": [
                        {
                            "text": "호스트에 추가 시스템 메모리로 보이는 방식이므로 모델 구조를 바꾸지 않고도 용량 여유를 얻을 수 있다.",
                            "evidence_id": "pim_cxl-S2-04",
                            "is_inference": True,
                        }
                    ],
                    "cloud_serving_operator": [
                        {
                            "text": "GPU 부착 메모리 한계를 넘어 용량을 늘리는 선택지가 생긴다.",
                            "evidence_id": "pim_cxl-S2-01",
                            "is_inference": False,
                        }
                    ],
                    "memory_semiconductor_vendor": [
                        {
                            "text": "메모리 수요가 늘어나는 방향이므로 설비 투자 확대의 근거가 된다.",
                            "evidence_id": "pim_cxl-S2-02",
                            "is_inference": False,
                        }
                    ],
                    "investor": [
                        {
                            "text": "메모리 중심 하드웨어가 별도 투자 대상으로 다뤄진다.",
                            "evidence_id": "pim_cxl-S2-03",
                            "is_inference": False,
                        }
                    ],
                }
            },
        )
    )
    out.append(
        result(
            "pim_cxl",
            "stakeholder",
            "S3",
            "narrative",
            "장비 도입과 부품 양산에 따르는 부담이 공개 자료에서 확인되고, 모델 개발사의 부담은 "
            "추론으로 기록한다.",
            [
                web(
                    "pim_cxl-S3-01",
                    EETIMES,
                    "PIM and PNM devices are sampled to selected customers",
                ),
                web(
                    "pim_cxl-S3-02",
                    STH_CXL,
                    "CXL Type-3 expansion modules are available from multiple vendors",
                ),
                web(
                    "pim_cxl-S3-03",
                    REUTERS_MEM,
                    "demand visibility for in-memory compute parts remains limited to sampling customers",
                ),
                inference(
                    "pim_cxl-S3-04", "pim_cxl-S3-01", "검증 부담에서 파생되는 추론"
                ),
            ],
            "family",
            {
                "burdens": {
                    "model_developer": [
                        {
                            "text": "부품이 샘플링 단계에 머물러 있어 해당 환경에서의 동작을 사전에 확인하기 어렵다.",
                            "evidence_id": "pim_cxl-S3-04",
                            "is_inference": True,
                        }
                    ],
                    "cloud_serving_operator": [
                        {
                            "text": "실제 출하되는 구성이 제한적이어서 도입 범위를 넓히기 어렵다.",
                            "evidence_id": "pim_cxl-S3-02",
                            "is_inference": False,
                        }
                    ],
                    "memory_semiconductor_vendor": [
                        {
                            "text": "설비 투자 확대가 선행되어야 하므로 수요가 확정되기 전에 비용이 발생한다.",
                            "evidence_id": "pim_cxl-S3-03",
                            "is_inference": False,
                        }
                    ],
                    "investor": [
                        {
                            "text": "부품이 샘플링 단계에 머물러 회수 시점을 가늠하기 어렵다.",
                            "evidence_id": "pim_cxl-S3-01",
                            "is_inference": False,
                        }
                    ],
                }
            },
        )
    )
    out.append(
        result(
            "pim_cxl",
            "stakeholder",
            "S4",
            "narrative",
            "벤더와 운영사 사이, 그리고 벤더와 투자 업계 사이에서 한쪽의 편익이 다른 쪽의 부담이 "
            "되는 지점이 확인된다.",
            [
                web(
                    "pim_cxl-S4-01",
                    REUTERS_MEM,
                    "AI-specific DRAM products require dedicated process and packaging investment",
                ),
                web(
                    "pim_cxl-S4-02",
                    STH_CXL,
                    "CXL Type-3 expansion modules are available from multiple vendors",
                ),
                web(
                    "pim_cxl-S4-03",
                    EETIMES,
                    "PIM and PNM devices are sampled to selected customers",
                ),
            ],
            "family",
            {
                "tradeoffs": [
                    {
                        "beneficiary": "memory_semiconductor_vendor",
                        "burdened": "cloud_serving_operator",
                        "text": "메모리·반도체 벤더에게 편익인 메모리 수요 확대가, 클라우드·서빙 운영사에게는 장비 도입 비용 증가라는 부담이 된다.",
                        "evidence_ids": ["pim_cxl-S4-01", "pim_cxl-S4-02"],
                    },
                    {
                        "beneficiary": "memory_semiconductor_vendor",
                        "burdened": "investor",
                        "text": "메모리·반도체 벤더에게 편익인 선제적 설비 투자가, 투자 업계에는 수요 확정 전 자본이 묶이는 부담이 된다.",
                        "evidence_ids": ["pim_cxl-S4-01", "pim_cxl-S4-03"],
                    },
                ]
            },
        )
    )
    return out


def _check(results: list[CriterionResult], expected_ids: list[str]) -> None:
    """생성 직후 자체 검증 — E의 perspective_check가 볼 항목을 미리 본다."""
    got = [f"{r.tech_id}/{r.criterion_id}" for r in results]
    assert got == expected_ids, f"기준 집합 불일치: {got}"
    for r in results:
        assert r.confidence == compute_confidence(r.evidence), r.criterion_id
        for e in r.evidence:
            assert e.evidence_id.startswith(f"{r.tech_id}-{r.criterion_id}-"), (
                e.evidence_id
            )
        if r.criterion_id in ("S2", "S3"):
            key = "benefits" if r.criterion_id == "S2" else "burdens"
            for s in STAKEHOLDERS:
                assert r.details[key].get(s), (
                    f"{r.tech_id}/{r.criterion_id}: {s} 비어 있음"
                )
        if r.criterion_id == "S4":
            assert r.details["tradeoffs"], r.criterion_id


def main() -> None:
    market = market_results()
    stakeholder = stakeholder_results()
    _check(market, [f"{t}/{c}" for t in ("mla", "pim_cxl") for c in ("M1", "M2", "M3")])
    _check(
        stakeholder,
        [f"{t}/{c}" for t in ("mla", "pim_cxl") for c in ("S1", "S2", "S3", "S4")],
    )

    for name, results in (
        ("market_eval.json", market),
        ("stakeholder_eval.json", stakeholder),
    ):
        path = FIXTURES / name
        payload = [r.model_dump(mode="json", exclude_none=False) for r in results]
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"{name}: {len(results)}건")


if __name__ == "__main__":
    main()
