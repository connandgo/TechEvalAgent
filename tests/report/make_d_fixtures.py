"""D 픽스처(synthesis.json, report_md.md) 재생성 스크립트.

실제 `run_synthesis` / `run_report`에 절별 응답을 미리 써 둔 스크립트 LLM을 넣어 돌리므로, 픽스처가 코드의
조립·요약표·REFERENCE·lint 규칙을 그대로 따른다. 상류 픽스처(B·C 샘플, E 픽스처)가 바뀌면 다시 실행한다.

    uv run python -m tests.report.make_d_fixtures
"""

import json
import logging

from techeval.agents._deps import Deps, ReportInput, SynthesisInput
from techeval.agents.report import run_report
from techeval.agents.synthesis import SynthesisDraft, run_synthesis
from techeval.report.lint import lint_report
from techeval.schemas import DOMAIN, TECHNOLOGIES, Evidence, EvidenceGap, TechProfile
from tests.report.d_fixtures import FIXTURES, load, load_evals, load_index

logger = logging.getLogger(__name__)

DRAFT = {
    "agreements": [
        {
            "tech_id": "mla",
            "perspectives": ["trl", "market"],
            "criterion_ids": ["T1", "M2"],
            "statement": "선정 논문의 서비스 배포 서술(T1)과 계열 단위 상용 운영 사례(M2)가 모두 운용 단계의 근거를 가리킨다.",
            "evidence_ids": ["mla-T1-01", "mla-M2-01"],
        },
        {
            "tech_id": "pim_cxl",
            "perspectives": ["trl", "domain"],
            "criterion_ids": ["T2", "D2"],
            "statement": "검증 환경이 시뮬레이션(T2 L2)이라는 판정과 동시 처리 근거의 직접성 L2(D2)가 같은 근거 한계를 가리킨다.",
            "evidence_ids": ["pim_cxl-T2-01", "pim_cxl-D2-01"],
        },
        {
            "tech_id": None,
            "perspectives": ["stakeholder", "domain"],
            "criterion_ids": ["S3", "D4"],
            "statement": "두 기술 모두 변경 범위(D4)에서 Y로 기록된 항목이 이해관계자 부담(S3)으로 이어진다.",
            "evidence_ids": [
                "mla-D4-01",
                "mla-S3-01",
                "pim_cxl-D4-01",
                "pim_cxl-S3-01",
            ],
        },
    ],
    "conflicts": [
        {
            "tech_id": "mla",
            "perspective_a": "stakeholder",
            "criterion_a": "S4",
            "perspective_b": "market",
            "criterion_b": "M2",
            "statement": "이해관계자 관점은 운영사의 배치 확대 편익이 개발사의 구조 변경 부담 위에서 성립한다고 보나(S4), "
            "시장 관점은 DeepSeek 자사 모델의 상용 운영을 근거로 M2를 L4로 판정한다.",
            "cause": "평가 단위·기준 차이: M2는 계열 단위 채택 사례에 원개발사 자체 운영을 포함해 판정하고, S4는 제3자 개발사가 "
            "기존 모델을 MLA로 전환할 때의 부담을 다룬다.",
            "evidence_ids": ["mla-S4-01", "mla-S4-02", "mla-M2-01", "mla-M2-02"],
        },
        {
            "tech_id": "mla",
            "perspective_a": "trl",
            "criterion_a": "T1",
            "perspective_b": "domain",
            "criterion_b": "D1",
            "statement": "TRL은 서비스 배포를 근거로 TRL 7-9 구간으로 추정하나, 도메인 D1은 128K 장문맥 메모리 실측이 없어 직접성 L2로 판정한다.",
            "cause": "기준 차이·직접성 차이: T1은 구현의 검증 환경을, D1은 우리 도메인 조건에서의 수치 직접성을 묻는다.",
            "evidence_ids": ["mla-T1-01", "mla-D1-01"],
        },
        {
            "tech_id": "pim_cxl",
            "perspective_a": "stakeholder",
            "criterion_a": "S4",
            "perspective_b": "market",
            "criterion_b": "M2",
            "statement": "시장 관점은 CXL 메모리 모듈 제품화를 근거로 M2를 L3로 판정하나, S4는 운영사가 설비 교체와 지연 증가를 "
            "감수해야 벤더의 편익이 성립한다고 본다.",
            "cause": "근거 유형·평가 단위 차이: M2는 벤더 발표(official)로 계열 단위 제품화를 확인한 것이고, S4는 KV cache용 PNM "
            "도입 시 운영사 부담을 다룬다.",
            "evidence_ids": [
                "pim_cxl-S4-01",
                "pim_cxl-S4-02",
                "pim_cxl-M2-01",
                "pim_cxl-M2-02",
            ],
        },
        {
            "tech_id": "pim_cxl",
            "perspective_a": "trl",
            "criterion_a": "T1",
            "perspective_b": "market",
            "criterion_b": "M3",
            "statement": "TRL은 시뮬레이션 평가를 근거로 TRL 4-6 구간 하단으로 추정하나, 시장 M3는 표준·벤더 제품·서베이를 근거로 L3로 판정한다.",
            "cause": "평가 단위 차이: T1은 선정 논문 구현(paper), M3는 CXL·PIM/PNM 계열 전체(family)를 본다.",
            "evidence_ids": ["pim_cxl-T1-01", "pim_cxl-M3-01", "pim_cxl-M3-02"],
        },
    ],
    "gaps": [
        {
            "tech_id": "mla",
            "criterion_id": "D2",
            "gap_type": "opposing_missing",
            "description": "처리량 근거가 원저자 측정뿐이다. counter_evidence(mla-COUNTER-02)는 속도 향상이 서빙 엔진의 MLA 커널 지원에 달려 있다고 지적하나, 처리량 이득이 줄어드는 조건을 측정한 제3자 근거는 없다.",
        },
    ],
    "unit_notes": "MLA는 TRL(paper)과 시장 M2(family)가 모두 원개발사 DeepSeek의 자료에 기대어 단위 차이가 판정을 크게 벌리지 않는다. "
    "PIM/CXL은 선정 논문 구현이 시뮬레이션 단계(TRL 4-6)인 반면, 시장·이해관계자는 CXL 메모리 모듈 등 계열 제품을 근거로 "
    "M2 L3·M3 L3가 나왔다. 따라서 PIM/CXL의 시장 레벨은 선정 논문 설계의 채택이 아니라 계열 단위 확장의 결과다.",
    "evidence_asymmetry_note": "MLA는 원저자 논문 인용 비중이 높고, PIM/CXL은 시장·이해관계자 근거에서 벤더(official) 발표 비중이 높다. "
    "반대 근거 탐색으로 4건(서베이 2건, 웹 2건)이 추가되어 pim_cxl M2의 채택 단계 반례는 보완되었으나, mla D2의 처리량 반례는 여전히 없다.",
}

TEXTS = {
    "SUMMARY": """\
MLA는 TRL 관점에서 서비스 배포를 근거로 TRL 7-9 구간(추정 TRL 7)으로 판정되지만, 도메인 관점의 D1은 128K 장문맥 메모리 실측이 없어 직접성 L2로 판정되어, 두 판정은 기준이 묻는 질문(검증 환경 대 도메인 조건 직접성)의 차이로 엇갈린다[E: mla-T1-01, mla-D1-01]. MLA의 시장 채택(M2 L4)은 원개발사의 자체 상용 운영에 기대고 있어, 제3자 개발사가 기존 모델을 MLA로 전환하는 부담을 다루는 이해관계자 상충(S4)과 평가 단위가 다르다[E: mla-M2-01, mla-S4-02]. PIM/CXL은 선정 논문 설계가 시뮬레이션 단계(TRL 4-6)에 있으나 시장 관점은 CXL 메모리 모듈 제품화와 표준을 근거로 M2 L3·M3 L3로 판정해, 논문 단위와 계열 단위의 차이가 레벨 차이로 나타난다[E: pim_cxl-T1-01, pim_cxl-M2-01, pim_cxl-M3-01]. PIM/CXL의 이해관계자 상충(S4)은 메모리 벤더의 판매 편익이 서빙 운영사의 설비 교체·지연 부담과 맞물린다는 점이며, 이 부담은 도메인 D4의 HW 교체·메모리 추가 항목과 같은 지점을 가리킨다[E: pim_cxl-S4-01, pim_cxl-S4-02, pim_cxl-D4-01].""",
    "1": """\
서베이 근거도 이 구조를 뒷받침한다. `io_survey`는 CXL 연결 메모리가 KV cache 용량을 늘리는 대신 로컬 DRAM보다 접근 지연이 크다고 정리하며[E: pim_cxl-PROFILE-04], KV cache를 CXL 링크 너머로 내리면 추가 지연을 메모리 측 연산이나 프리페치로 가리지 못할 경우 디코딩 처리량이 떨어질 수 있다고 지적한다[E: pim_cxl-COUNTER-01]. `kv_survey`는 MLA가 어텐션 구조를 바꾸고 모델을 처음부터 학습해야 하므로 MHA·GQA 기반의 기존 사전학습 모델에 바로 적용하기 어렵다고 정리한다[E: mla-COUNTER-01]. 즉 SW 축소는 모델 변경 비용을, HW 확장은 메모리 계층의 지연·대역폭 비용을 동반한다는 점이 두 갈래를 가르는 축이다.""",
    "2": """\
`kv_survey`는 MLA를 어텐션 구조를 바꾸고 처음부터 학습해야 하는 구조 수준의 KV cache 압축으로 분류한다[E: mla-COUNTER-01]. 양자화 계열 후보는 모델 구조를 유지한 채 저장 정밀도를 낮추는 쪽에 속하므로, 대조성 기준(두 기술이 서로 다른 변경 범위를 전제하는가)을 드러내는 SW 측 사례로 구조 변경형인 MLA를 고정했다.""",
    "3.mla": """\
**핵심 접근.** MLA는 Key·Value를 저차원 잠재 벡터로 함께 압축해 추론 시 KV cache에 잠재 벡터만 저장한다. 원문은 이를 "MLA guarantees efficient inference through significantly compressing the Key-Value (KV) cache into a latent vector"라고 요약한다[E: mla-PROFILE-01]. 적용 범위는 DeepSeek-V2(총 236B, 토큰당 21B 활성 MoE, 128K 컨텍스트)의 사전학습 단계부터 설계된 어텐션 구조다[E: mla-PROFILE-01].

**정량 성과.** 선정 논문은 KV cache 93.3% 감소(DeepSeek-V2 총 236B·21B 활성, vs DeepSeek 67B MHA)를 보고한다[E: mla-PROFILE-02]. 최대 생성 처리량은 5.76x(단일 노드 8x H800, FP8 파라미터와 평균 6-bit KV 양자화 적용, 실서비스 요청 길이 분포, vs DeepSeek 67B)로 보고되었다[E: mla-PROFILE-04].

**검증 환경과 공개 자료.** 처리량은 실물 8x H800 노드에서 실서비스 요청 길이 분포로 측정되었고, 모델 가중치와 추론 코드가 공개되어 있다[E: mla-PROFILE-04, mla-PROFILE-05]. 평가 스크립트·데이터 공개는 확인되지 않았다[E: mla-PROFILE-05].

**명시된 한계.** 원문은 "RoPE is incompatible with low-rank KV compression"이라고 밝히며, 이를 위해 분리된 RoPE 키(decoupled RoPE)라는 별도 설계를 둔다[E: mla-PROFILE-03]. 또한 보고된 효율 수치의 베이스라인은 DeepSeek 67B(MHA)로, 같은 규모 모델 사이의 비교가 아니다[E: mla-PROFILE-02].""",
    "3.pim_cxl": """\
**핵심 접근.** 이 접근은 장문맥 요청의 KV cache를 GPU HBM 밖의 CXL 연결 메모리로 옮기고, 어텐션에 필요한 토큰 페이지 선택 연산을 메모리 근처 연산 장치(PNM)에서 수행해 GPU로 옮기는 데이터 양을 줄인다[E: pim_cxl-PROFILE-01]. 적용 범위는 1M 토큰급 장문맥 추론이며 CXL 메모리 확장과 PNM 장치를 갖춘 서버 구성을 전제한다[E: pim_cxl-PROFILE-01].

**정량 성과.** 선정 논문은 시뮬레이션(GPU + CXL-PNM 모델링, 베이스라인 GPU HBM 단독 구성)에서 1M 토큰 컨텍스트 추론을 유지했다고 보고한다[E: pim_cxl-PROFILE-02]. 이 값은 픽스처용 예시로 원문 확인이 필요하다.

**검증 환경과 공개 자료.** 평가는 사이클 수준 시뮬레이터로 CXL 메모리와 PNM 장치를 모델링해 수행되었으며 실물 시제품 측정이 아니다[E: pim_cxl-PROFILE-02]. 코드·설계·평가 스크립트·제3자 재현은 모두 공개가 확인되지 않았다.

**명시된 한계.** 원문은 CXL 링크의 추가 지연이 페이지 선택 정확도가 낮을 때 병목으로 남는다고 밝힌다[E: pim_cxl-PROFILE-03]. 계열 수준에서도 `io_survey`는 CXL 연결 메모리가 용량을 늘리는 대신 로컬 DRAM보다 접근 지연이 크다고 정리한다[E: pim_cxl-PROFILE-04].""",
    "4.trl": """\
**DeepSeek-V2 MLA**

- **T1 현재 TRL — TRL 7-9 (추정 TRL 7, 공개 정보 기반 추정)**: 서비스 배포를 위해 파라미터를 FP8로 변환했다는 원문 서술과 공개 체크포인트를 근거로 실제 운용 환경 시연 이상 구간으로 추정했다[E: mla-T1-01, mla-T1-02]. 한 단계 위로 올리지 못한 이유는 실서비스 배포는 확인되나 운영 규모·안정성 지표가 공개되지 않아 TRL 8-9 근거가 없기 때문이다. 신뢰도 high, 근거 단위 paper.
- **T2 검증 환경 — L3**: 생성 처리량은 실물 8x H800 노드에서 실서비스 요청 길이 분포를 재현해 측정했으며, 실서비스 트래픽 자체의 측정은 공개되지 않았다[E: mla-T2-01]. 신뢰도 medium(단일 출처), 근거 단위 paper.
- **T3 재현성 — L3**: 코드 Y, 모델/설계 Y, 평가 스크립트·데이터 N, 제3자 재현 Y(SGLang의 MLA 구현)다[E: mla-T3-01, mla-T3-02]. 신뢰도 high.
- **T4 다음 과제**: 남은 과제는 기존 MHA/GQA 모델 적용 경로 검증과 운영 규모 지표 공개다[E: mla-T4-01]. 서비스 운영 중 KV cache 메모리 실측치는 미공개(검색어: DeepSeek-V2 production KV cache memory usage, 검색일: 2026-09-20)다[E: mla-T4-02]. 신뢰도 low.

**PIM/CXL**

- **T1 현재 TRL — TRL 4-6 (추정 TRL 4, 공개 정보 기반 추정)**: 선정 논문은 CXL 메모리와 PNM 장치를 시뮬레이터로 모델링해 1M 토큰 추론을 평가했다[E: pim_cxl-T1-01]. 실물 CXL-PNM 시제품 측정 결과가 공개되지 않아 TRL 5 이상으로 올리지 못했다. 신뢰도 medium, 근거 단위 paper.
- **T2 검증 환경 — L2**: 사이클 수준 시뮬레이션으로 평가되었다[E: pim_cxl-T2-01]. 신뢰도 medium.
- **T3 재현성 — L1**: 코드 N, 설계 N, 평가 스크립트·데이터 N, 제3자 재현 N이며, 구현 코드는 미공개(검색어: CXL PNM 1M token LLM inference code github, 검색일: 2026-09-20)로 기록했다[E: pim_cxl-T3-01, pim_cxl-T3-02]. 신뢰도 low.
- **T4 다음 과제**: 남은 과제는 실물 시제품 측정과 서빙 엔진 통합이며, 시뮬레이터 설정과 구현 코드는 공개되지 않았다[E: pim_cxl-T4-01, pim_cxl-T4-02]. 신뢰도 low.

두 기술의 TRL 판정 차이는 검증 환경의 차이(실물 노드 측정 대 시뮬레이션)에서 나오며, 현재 공개 근거로 확인되는 단계의 차이다.""",
    "4.market": """\
이 절의 판정은 선정 논문이 아니라 기술 계열 단위(MLA 계열 / CXL·PIM/PNM 메모리 계열)로 확장한 결과다.

**DeepSeek-V2 MLA**

- **M1 성장성 — not_public**: MLA 계열을 따로 구분한 정량 시장 전망은 미공개(검색어: Multi-head Latent Attention market forecast, 검색일: 2026-09-20, 범위: 시장조사 기관 보고서·뉴스 2024-2026)다[E: mla-M1-01]. 신뢰도 low.
- **M2 채택 — L4 상용 운영**: 채택 주체는 DeepSeek로, MLA를 적용한 DeepSeek-V2를 서비스에 배포했고 후속 모델 DeepSeek-V3도 MLA를 채택했다[E: mla-M2-01, mla-M2-02]. 두 근거 모두 원개발사 자료다. 신뢰도 high.
- **M3 생태계 — L2**: 프레임워크 Y(SGLang의 MLA 최적화), 벤더 제품 N, 표준화 N, 제3자 연구·도구 Y(TransMLA)다[E: mla-M3-01, mla-M3-02]. 신뢰도 high.

**PIM/CXL**

- **M1 성장성 — L2 형성**: Yole Group(2024-03)은 CXL 메모리 시장이 2028년 약 150억 달러 규모로 성장할 것으로 전망했으며, 단일 기관 전망이므로 사실로 단정하지 않는다[E: pim_cxl-M1-01]. 신뢰도 medium(단일 출처).
- **M2 채택 — L3 제품화**: 채택 주체는 Samsung(CMM-D)과 SK hynix(CMM-DDR5)로, CXL 메모리 확장 모듈을 제품으로 출시했다[E: pim_cxl-M2-01, pim_cxl-M2-02]. 다만 LLM KV cache용 PNM 채택은 연구 단계로, 계열 단위 제품화와 구분된다. 신뢰도 high.
- **M3 생태계 — L3**: 프레임워크 N, 벤더 제품 Y, 표준화 Y(CXL Consortium), 제3자 연구·도구 Y(io_survey)다[E: pim_cxl-M3-01, pim_cxl-M3-02, pim_cxl-M3-03]. 신뢰도 high.

MLA 계열의 채택 근거는 원개발사 자체 운영에, PIM/CXL 계열의 채택 근거는 메모리 벤더 발표(official)에 기대고 있어, 두 기술의 시장 레벨은 근거 유형이 서로 다른 상태에서 나온 판정이다.""",
    "4.stakeholder": """\
이 절은 기술 계열 단위로, 모델 개발사 / 클라우드·서빙 운영사 / 메모리·반도체 벤더 / 투자 업계 4주체를 고정해 판정했다.

**DeepSeek-V2 MLA**

- **S1 의사결정**: 모델 개발사는 어텐션 구조를 정하는 결정권자, 서빙 운영사는 영향받는 자이며, 메모리·반도체 벤더와 투자 업계는 수요 변화로 영향받는 자로 두었다(후자 둘은 추론)[E: mla-S1-01, mla-S1-02].
- **S2 편익**: 모델 개발사는 서비스 배포 시 KV cache 부담이 줄고, 운영사는 더 큰 배치를 처리할 수 있다[E: mla-S2-01]. 메모리·반도체 벤더 쪽에서는 서빙 프레임워크 최적화 수요가 확인되며[E: mla-S2-02], 투자 업계에는 메모리 증설 없이 처리량을 넓힐 여지가 편익이 될 수 있다(추론)[E: mla-S2-03].
- **S3 부담**: 모델 개발사는 기존 모델의 구조 변환·재학습 부담을 진다[E: mla-S3-01]. 운영사는 MLA 전용 커널 지원·검증 부담을(추론)[E: mla-S3-03], 메모리·반도체 벤더와 투자 업계는 메모리 용량 수요 감소 가능성과 구조 변경 비용 대비 효과의 불확실성을 부담한다(추론)[E: mla-S3-02].
- **S4 상충**: 클라우드·서빙 운영사가 얻는 배치 확대 편익은 모델 개발사의 모델 구조 변경·재학습 부담을 전제로 한다[E: mla-S4-01, mla-S4-02].

**PIM/CXL**

- **S1 의사결정**: 서빙 운영사는 서버 구성 변경을 결정하는 결정권자, 메모리·반도체 벤더는 공급자, 모델 개발사와 투자 업계는 영향받는 자로 두었다(모델 개발사 역할은 추론)[E: pim_cxl-S1-01, pim_cxl-S1-02].
- **S2 편익**: 메모리·반도체 벤더는 CXL 메모리 모듈 판매 기회를[E: pim_cxl-S2-01], 운영사는 HBM 용량을 넘는 컨텍스트 수용 여지를 얻는다[E: pim_cxl-S2-02]. 모델 개발사(구조 변경 없는 장문맥 지원)와 투자 업계(메모리 계층 확장 시장 진입)의 편익은 추론이다[E: pim_cxl-S2-03].
- **S3 부담**: 메모리·반도체 벤더는 CXL 추가 지연을 줄이는 설계 부담을 진다[E: pim_cxl-S3-01]. 운영사의 설비 도입·서빙 경로 변경과 모델 개발사의 생성 지연 영향은 추론이며[E: pim_cxl-S3-02], 투자 업계는 단일 전망에 기댄 수요 불확실성을 부담한다(추론)[E: pim_cxl-S3-03].
- **S4 상충**: 메모리·반도체 벤더의 CXL 메모리 판매 편익은 클라우드·서빙 운영사의 서버·메모리 교체와 지연 증가 부담으로 이어진다[E: pim_cxl-S4-01, pim_cxl-S4-02].

두 기술 모두 편익을 얻는 주체와 부담을 지는 주체가 다르며, MLA에서는 그 부담이 모델 쪽(개발사)에, PIM/CXL에서는 인프라 쪽(운영사)에 놓인다.""",
    "4.domain": """\
**DeepSeek-V2 MLA**

- **D1 장문맥 병목 — L2(간접)**: KV cache 93.3% 감소(DeepSeek-V2 총 236B·21B 활성, vs DeepSeek 67B MHA, 토큰당 KV cache 기준)가 보고되었다[E: mla-D1-01]. 128K 장문맥에서의 메모리 실측은 없으며, 토큰당 KV 원소 수 감소는 컨텍스트 길이와 무관한 비율이므로 장문맥에도 같은 비율을 적용할 수 있다고 외삽했다. 신뢰도 medium.
- **D2 동시 처리 — L3(직접)**: 단일 노드 8x H800에서 실서비스 요청 길이 분포로 측정한 생성 처리량이 50K tokens/s를 넘었다(총 236B·21B 활성, FP8 + KV 6-bit 양자화)[E: mla-D2-01]. 신뢰도 medium.
- **D3 지연·에너지·비용 — L2(간접)**: prompt 입력 처리량 100K tokens/s 초과(단일 노드 8x H800, 총 236B·21B 활성)가 보고되었으나 TTFT/TPOT, 토큰당 에너지, 총비용은 보고되지 않았다[E: mla-D3-01]. 입력 처리량에서 prefill 지연·토큰당 비용 감소를 외삽했을 뿐 직접 측정은 없다. 신뢰도 medium.
- **D4 변경 범위**: 모델 재학습 Y, 모델 변환 미확인, 서빙 엔진 수정 Y, HW 교체 N, 메모리 추가 N, 기타 N[E: mla-D4-01, mla-D4-02].

**PIM/CXL**

- **D1 장문맥 병목 — L3(직접)**: 1M 토큰 컨텍스트의 KV cache를 CXL 메모리 계층으로 옮겨 GPU HBM 단독 구성의 용량을 넘는 추론을 시뮬레이션(GPU + CXL-PNM)으로 보였다[E: pim_cxl-D1-01]. 신뢰도 medium.
- **D2 동시 처리 — L2(간접)**: 시뮬레이션(GPU + CXL-PNM)에서 KV 용량 확대가 동시 요청 수 증가로 이어졌고, 실서버에서도 같은 방향일 것으로 외삽했다[E: pim_cxl-D2-01]. 실제 서빙 배치 측정은 없다. 신뢰도 medium.
- **D3 지연·에너지·비용 — L2(간접)**: 원문은 CXL 링크 추가 지연이 페이지 선택 정확도가 낮을 때 병목이 된다고 서술한다[E: pim_cxl-D3-01]. 시뮬레이터의 링크 지연 모델이 실제 장치와 같다고 가정한 외삽이며, 에너지·총비용은 보고되지 않았다.
- **D4 변경 범위**: 모델 재학습 N, 모델 변환 N, 서빙 엔진 수정 Y, HW 교체 Y, 메모리 추가 Y, 기타 미확인[E: pim_cxl-D4-01].

두 기술의 수치는 서로 다른 실험 조건에서 나온 값이므로 직접 비교하지 않는다. 도메인 관점의 판정 차이는 수치 크기가 아니라 도메인 조건과의 직접성(실측 대 외삽)과 변경 범위 항목의 차이다.""",
    "5": """\
**MLA — 이해관계자 상충(S4)과 시장 채택(M2).** 이해관계자 관점은 서빙 운영사의 배치 확대 편익이 모델 개발사의 구조 변경·재학습 부담 위에서 성립한다고 보지만, 시장 관점은 원개발사 DeepSeek의 자체 상용 운영을 근거로 M2를 L4로 판정한다[E: mla-S4-01, mla-S4-02, mla-M2-01, mla-M2-02]. 이 차이는 평가 단위와 기준의 차이에서 나온다. M2는 계열 단위 채택 사례에 원개발사 자체 운영을 포함하고, S4는 제3자 개발사가 기존 모델을 MLA로 전환할 때의 부담을 다룬다. 원개발사 밖의 채택 근거(변환 연구)는 아직 연구 단계다[E: mla-M3-02].

**MLA — TRL(T1)과 도메인 장문맥 병목(D1).** TRL 관점은 서비스 배포를 근거로 TRL 7-9 구간으로 추정하지만, 도메인 D1은 128K 장문맥 메모리 실측이 없어 직접성 L2로 판정한다[E: mla-T1-01, mla-D1-01]. T1은 구현이 어떤 환경까지 검증됐는지를, D1은 우리 도메인 조건에서 수치가 직접 확인되는지를 묻는다. 두 판정은 기준의 질문이 다르기 때문에 엇갈린다.

**PIM/CXL — 이해관계자 상충(S4)과 시장 채택(M2).** 시장 관점은 CXL 메모리 모듈 제품화를 근거로 M2를 L3로 판정하지만, S4는 운영사가 서버·메모리 교체와 지연 증가를 감수해야 벤더의 편익이 성립한다고 본다[E: pim_cxl-M2-01, pim_cxl-M2-02, pim_cxl-S4-01, pim_cxl-S4-02]. M2의 근거는 벤더 발표(official)로 확인한 계열 단위 제품화이고, S4는 KV cache용 PNM 도입 시 운영사의 부담을 다루므로 근거 유형과 평가 단위가 다르다.

**PIM/CXL — TRL(T1)과 시장 생태계(M3).** TRL 관점은 시뮬레이션 평가를 근거로 TRL 4-6 구간 하단으로 추정하지만, 시장 M3는 CXL 표준·벤더 제품·서베이를 근거로 생태계를 L3로 판정한다[E: pim_cxl-T1-01, pim_cxl-M3-01, pim_cxl-M3-02]. T1은 선정 논문 구현을, M3는 CXL·PIM/PNM 계열 전체를 보기 때문에 생기는 평가 단위 차이다.

**평가 단위와 근거 비대칭이 해석에 주는 제약.** MLA는 TRL과 시장 채택이 모두 원개발사 자료에 기대어 단위 차이가 판정을 크게 벌리지 않는다. PIM/CXL은 선정 논문 설계가 시뮬레이션 단계인 반면 시장·이해관계자 판정은 계열 제품을 근거로 삼으므로, 시장 레벨은 선정 논문 설계의 채택이 아니라 계열 단위 확장의 결과로 읽어야 한다. 반대 근거 탐색으로 추가된 자료는 이 해석을 보강한다. GQA 모델을 MLA로 바꾸려면 추가 미세조정이 필요하고 속도 향상이 서빙 엔진의 커널 지원에 달려 있다는 근거[E: mla-COUNTER-02]는 MLA의 S3·D4 부담과 같은 지점을, CXL 메모리 확장이 AI 추론에서는 아직 평가 시스템 중심이라는 근거[E: pim_cxl-COUNTER-02]는 PIM/CXL M2가 계열 제품 출시 단계이지 실서비스 운영 단계가 아님을 가리킨다.

두 기술의 평가는 어느 관점의 기준으로 보느냐에 따라 달라진다. 모델 구조 변경 비용을 묻는 기준(D4·S3·S4)에서는 MLA의 부담이, 인프라 변경과 검증 환경을 묻는 기준(T1·T2·D4)에서는 PIM/CXL의 부담이 드러난다[E: mla-D4-01, pim_cxl-D4-01, pim_cxl-T2-01].""",
    "6": """\
1. **공개 정보만으로 TRL을 추정한 한계.** 두 기술의 T1은 공개 자료 기반 추정이다. MLA는 실서비스 배포가 확인되나 운영 규모·안정성 지표가 공개되지 않아 TRL 7-9 구간 하단(추정 TRL 7)에 두었고[E: mla-T1-01], PIM/CXL은 실물 시제품 측정 결과가 공개되지 않아 TRL 4-6 구간 하단(추정 TRL 4)에 두었다[E: pim_cxl-T1-01]. 비공개 운영 자료가 공개되면 판정이 달라질 수 있다.
2. **평가 단위 확장.** 시장·이해관계자 관점은 선정 논문이 아니라 MLA 계열 / CXL·PIM/PNM 메모리 계열 단위로 확장해 평가했다. 특히 PIM/CXL의 시장 레벨은 CXL 메모리 모듈 등 계열 제품을 근거로 하므로 선정 논문 설계의 채택 단계와 같지 않다.
3. **제안사·벤더 자료 의존 비율.** evidence_gap 기준 벤더(official) 자료 비율은 DeepSeek-V2 MLA 35.0%, PIM/CXL 55.0%다(근거 비대칭 검사 산출값). 고유 evidence 수도 MLA 23건, PIM/CXL 11건으로 비대칭이며, PIM/CXL 시장·이해관계자 근거의 벤더 자료 의존이 크다는 점을 감안해 읽어야 한다.
4. **적용한 검사와 남은 한계.** 기술 조사·관점별 근거 검사, 근거 비대칭 검사, 반대 근거 탐색(4건 반영), 별도 검수 모델의 보고서 검수를 적용했다. 그럼에도 not_public 항목(MLA M1, 운영 중 KV cache 실측치, PIM/CXL 구현 코드), 단일 출처·추론 의존 판정, 반대 근거가 없는 기준이 남아 있으며 아래 표에 정리했다.""",
}


class ScriptedLLM:
    """run_synthesis: with_structured_output(SynthesisDraft) → DRAFT. run_report: invoke → 절별 TEXTS."""

    def with_structured_output(self, model):
        assert model is SynthesisDraft

        class _S:
            def invoke(self, messages):
                return SynthesisDraft.model_validate(DRAFT)

        return _S()

    def invoke(self, messages):
        human = messages[-1][1]
        head = human.splitlines()[0]
        if "SUMMARY" in head:
            key = "SUMMARY"
        elif "1. 분석 배경" in head:
            key = "1"
        elif "2. 기술 선정" in head:
            key = "2"
        elif "3. 기술 개요" in head:
            key = "3.mla" if "tech_id=mla," in human else "3.pim_cxl"
        elif "4. 관점별 평가" in head:
            key = "4." + next(
                p
                for p in ("trl", "market", "stakeholder", "domain")
                if f"({p})\n" in human
            )
        elif "5. 시사점" in head:
            key = "5"
        elif "6. 한계점" in head:
            key = "6"
        else:
            raise KeyError(head)

        class R:
            content = TEXTS[key]

        return R()


def _no_search(*a, **k):
    raise AssertionError("D 에이전트는 검색을 호출하지 않는다")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    deps = Deps(
        retriever=None,
        web_search=_no_search,
        llm=ScriptedLLM(),
        now=lambda: "2026-09-20T10:30:00",
    )
    profiles = load("tech_profiles.json", TechProfile)
    evals = load_evals()
    counter = load("counter_evidence.json", Evidence)
    syn = run_synthesis(
        SynthesisInput(
            technologies=TECHNOLOGIES,
            tech_profiles=profiles,
            counter_evidence=counter,
            **evals,
        ),
        deps,
    )
    (FIXTURES / "synthesis.json").write_text(
        json.dumps(syn.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    inp = ReportInput(
        technologies=TECHNOLOGIES,
        domain=DOMAIN,
        tech_profiles=profiles,
        counter_evidence=counter,
        synthesis=syn,
        evidence_gap=load("evidence_gap.json", EvidenceGap),
        **evals,
    )
    md = run_report(inp, deps)
    (FIXTURES / "report_md.md").write_text(md, encoding="utf-8")
    res = lint_report(md, load_index())
    logger.info(
        "lint passed=%s errors=%s missing=%s",
        res.passed,
        res.errors,
        res.missing_required,
    )


if __name__ == "__main__":
    main()
