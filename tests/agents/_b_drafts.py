"""B 테스트·픽스처 생성용 LLM 응답 초안.

인용은 전부 tests/fixtures/chunks.json(A) / web_results.json(C) 의 실제 문자열이다.
quote 를 고치면 to_evidence 검증에서 폐기되므로, 청크·웹 픽스처가 바뀌면 여기도 맞춘다.
"""

import copy

ARXIV_MLA = "https://arxiv.org/abs/2405.04434"
ARXIV_PIM = "https://arxiv.org/abs/2511.00321"
GH_MLA = "https://github.com/deepseek-ai/DeepSeek-V2"
HF_MLA = "https://huggingface.co/deepseek-ai/DeepSeek-V2"
GH_VLLM = "https://github.com/vllm-project/vllm"
VLLM_MODELS = "https://docs.vllm.ai/en/latest/models/supported_models.html"
VLLM_DOCS = "https://docs.vllm.ai/en/latest/"
EETIMES = "https://www.eetimes.com/processing-in-memory-parts-remain-sampling-stage-for-ai-servers/"
SKH = "https://news.skhynix.com/sk-hynix-aim-accelerator-card-generative-ai/"

Q_MLA_93 = "DeepSeek-V2 reduces KV cache per token by 93.3% compared with DeepSeek 67B while supporting a 128K context length"
Q_MLA_H800 = "trained and served on clusters of NVIDIA H800 GPUs"
Q_MLA_GPUH = (
    "DeepSeek 67B requires 300.6K GPU hours, while DeepSeek-V2 needs only 172.8K GPU"
)
Q_PIM_GAIN = "Reported gains reach up to 21.9x throughput and up to 60x lower energy per token relative to the stated baseline, for models up to 405B parameters at 1M-token context"
Q_PIM_MEAS = "reports results measured on a CXL-PNM system"


def C(ref: str, quote: str, source: str = "chunk") -> dict:
    return {"source": source, "ref": ref, "quote": quote}


def W(url: str, quote: str) -> dict:
    return C(url, quote, "web")


MLA = {
    "PROFILE": {
        "principle": "키와 값을 저차원 잠재 벡터로 공동 압축해 생성 중 캐시할 데이터를 줄인다. 어텐션 구조 자체를 바꾸는 접근이다.",
        "scope": "모델 아키텍처 수준의 변경이며, 논문은 236B(활성 21B) MoE 모델과 128K 컨텍스트에서 보고한다.",
        "limitations": [
            "공개 저장소에 학습 파이프라인과 벤치마크 하네스가 포함되지 않는다"
        ],
        "validation_env": "NVIDIA H800 클러스터에서 학습·서빙했다고 보고한다. 실서비스 트래픽 검증은 확인되지 않는다.",
        "measurements": [
            {
                "metric": "KV cache per token reduction",
                "value": "93.3%",
                "model_size": "236B (21B active)",
                "context_length": "128K",
                "baseline": "DeepSeek 67B",
                "citation_index": 2,
            },
            {
                "metric": "training GPU hours per trillion tokens",
                "value": "172.8K",
                "unit": "GPU hours",
                "hardware": "H800 cluster",
                "baseline": "DeepSeek 67B: 300.6K",
                "citation_index": 4,
            },
        ],
        "citations": [
            C(
                "deepseek_v2:006:09",
                "Equipped with low-rank key-value joint compression, MLA achieves better performance than MHA, but requires a significantly smaller amount of KV cache.",
            ),
            C(
                "deepseek_v2:007:12",
                "The core of MLA is the low-rank joint compression for keys and values to reduce KV",
            ),
            W(ARXIV_MLA, Q_MLA_93),
            W(ARXIV_MLA, Q_MLA_H800),
            C("deepseek_v2:016:36", Q_MLA_GPUH),
        ],
    },
    "T1": {
        "level": "TRL 4-6",
        "level_estimate": "TRL 6 (공개 정보 기반 추정)",
        "content": "H800 클러스터에서 학습·서빙했고 가중치가 공개돼 관련 환경 검증 단계로 본다. 실서비스 트래픽 운영 근거는 공개 자료에 없다.",
        "details": {
            "trl_band": "4-6",
            "estimate": 6,
            "why_not_higher": "지속 트래픽 하의 실서비스 운영 실적이 공개 자료에서 확인되지 않아 7-9로 올리지 못한다.",
        },
        "citations": [
            W(ARXIV_MLA, Q_MLA_H800),
            W(
                HF_MLA,
                "Model weights and configuration for DeepSeek-V2 are published under an open licence",
            ),
        ],
    },
    "T2": {
        "level": "L3",
        "content": "실물 H800 GPU 클러스터에서 학습·서빙했다고 보고한다. 실서비스 트래픽 검증은 원문에 없다.",
        "details": {"env_level": "L3"},
        "citations": [W(ARXIV_MLA, Q_MLA_H800)],
    },
    "T3": {
        "checklist": {
            "code": "Y",
            "model_or_design": "Y",
            "eval_scripts_data": "N",
            "third_party_reproduction": "Y",
        },
        "urls": {
            "code": GH_MLA,
            "model_or_design": HF_MLA,
            "third_party_reproduction": GH_VLLM,
        },
        "content": "공식 저장소에 추론 코드가 있고 가중치가 공개돼 있으며 vLLM이 독립 구현을 유지한다. 벤치마크 하네스는 포함되지 않는다.",
        "citations": [
            W(
                GH_MLA,
                "Contains the model configuration, inference example code for the MLA attention path",
            ),
            W(
                HF_MLA,
                "Model weights and configuration for DeepSeek-V2 are published under an open licence",
            ),
            W(
                GH_MLA,
                "the benchmark harness used for the numbers reported in the paper are not included",
            ),
            W(
                GH_VLLM,
                "vLLM carries an independently written MLA attention backend for DeepSeek-V2 style models",
            ),
        ],
    },
    "T4": {
        "level": "narrative",
        "content": "실서비스 트래픽 검증이 남아 있다. 벤치마크 하네스와 배포 규모는 공개 자료에서 확인되지 않는다.",
        "details": {
            "remaining_tasks": ["지속 트래픽 하의 다중 테넌트 서빙 검증"],
            "not_public_items": [
                "벤치마크 하네스 (검색어: DeepSeek-V2 code availability github)",
                "실서비스 배포 규모 (검색어: DeepSeek-V2 MLA production deployment)",
            ],
        },
        "citations": [
            W(
                GH_MLA,
                "the benchmark harness used for the numbers reported in the paper are not included",
            )
        ],
    },
    "D1": {
        "level": "L2",
        "content": "토큰당 KV cache 93.3% 감소가 128K 컨텍스트에서 보고된다. 다중 요청 조건은 명시되지 않아 간접 근거로 본다.",
        "details": {
            "directness": "L2",
            "extrapolation_logic": "토큰당 캐시 감소는 요청 수와 컨텍스트 길이에 선형으로 비례하므로 다중 요청 장문맥 조건에도 같은 비율이 적용된다고 본다.",
        },
        "measurements": [
            {
                "metric": "KV cache per token reduction",
                "value": "93.3%",
                "model_size": "236B (21B active)",
                "context_length": "128K",
                "baseline": "DeepSeek 67B",
                "citation_index": 0,
            }
        ],
        "citations": [
            W(ARXIV_MLA, Q_MLA_93),
            C(
                "deepseek_v2:006:09",
                "requires a significantly smaller amount of KV cache",
            ),
        ],
    },
    "D2": {
        "level": "L2",
        "content": "캐시 감소가 동시 세션 수용량을 늘린다는 논리는 성립하지만 배치 크기·동시 세션 수치는 원문에 없다.",
        "details": {
            "directness": "L2",
            "extrapolation_logic": "토큰당 캐시가 줄면 같은 메모리에 더 많은 세션을 담을 수 있으나, 실제 배치 처리량은 원문에 미기재라 외삽이다.",
        },
        "measurements": [
            {
                "metric": "KV cache per token reduction",
                "value": "93.3%",
                "model_size": "236B (21B active)",
                "context_length": "128K",
                "baseline": "DeepSeek 67B",
                "citation_index": 0,
            }
        ],
        "citations": [W(ARXIV_MLA, Q_MLA_93)],
    },
    "D3": {
        "level": "L2",
        "content": "학습 GPU 시간 절감이 보고되나 추론 지연·토큰당 에너지·전송 오버헤드는 공개 자료에서 확인되지 않는다.",
        "details": {
            "directness": "L2",
            "extrapolation_logic": "학습 비용 수치는 추론 총비용의 직접 근거가 아니며, 캐시 감소가 메모리 대역폭 요구를 줄인다는 간접 논리로만 연결된다.",
        },
        "measurements": [
            {
                "metric": "training GPU hours per trillion tokens",
                "value": "172.8K",
                "unit": "GPU hours",
                "hardware": "H800 cluster",
                "baseline": "DeepSeek 67B: 300.6K",
                "citation_index": 0,
            }
        ],
        "citations": [C("deepseek_v2:016:36", Q_MLA_GPUH)],
    },
    "D4": {
        "level": "checklist",
        "content": "vLLM 문서는 MLA 전용 백엔드를 지원 목록에 두고 있어 서빙 엔진 수정은 불필요하다. 재학습·변환 여부는 주어진 근거에 없다.",
        "details": {
            "checklist": {
                "model_retrain": "unknown",
                "model_convert": "unknown",
                "serving_engine_change": "N",
                "hw_replace": "N",
                "memory_add": "N",
                "other": "unknown",
            }
        },
        "citations": [
            W(
                VLLM_MODELS,
                "vLLM implements a dedicated MLA (Multi-head Latent Attention) backend for DeepSeek-V2 and DeepSeek-V3 style models",
            ),
            W(ARXIV_MLA, Q_MLA_H800),
        ],
    },
}

PIM = {
    "PROFILE": {
        "principle": "KV cache를 CXL로 확장한 메모리에 두고, 토큰 페이지 선택 연산을 메모리 옆 가속기(PNM)로 이관한다. GPU 한계를 넘는 메모리·연산 조율이 핵심이다.",
        "scope": "1M 토큰급 장문맥에서 GPU 부착 메모리 한계를 넘는 조건을 전제한다. 논문은 최대 405B 모델까지 보고한다.",
        "limitations": ["저장소·RTL·시뮬레이터 설정이 논문에 링크돼 있지 않다"],
        "validation_env": "CXL-PNM 시스템에서 측정한 결과를 보고한다. 양산 실리콘·실서비스 트래픽 검증은 확인되지 않는다.",
        "measurements": [
            {
                "metric": "throughput gain",
                "value": "up to 21.9x",
                "model_size": "up to 405B",
                "context_length": "1M",
                "baseline": "stated baseline (paper)",
                "citation_index": 1,
            },
            {
                "metric": "energy per token reduction",
                "value": "up to 60x lower",
                "model_size": "up to 405B",
                "context_length": "1M",
                "baseline": "stated baseline (paper)",
                "citation_index": 1,
            },
        ],
        "citations": [
            C(
                "pim_cxl_1m:004:08",
                "The proposed CXL-PNM architecture realizes a scalable, high-capacity, and high-bandwidth memory expansion platform tailored for the LLM inference.",
            ),
            W(ARXIV_PIM, Q_PIM_GAIN),
            W(ARXIV_PIM, Q_PIM_MEAS),
            C(
                "pim_cxl_1m:002:02",
                "offering coherent, low-latency access to external memory beyond the physical limits of GPU-attached",
            ),
        ],
    },
    "T1": {
        "level": "TRL 4-6",
        "level_estimate": "TRL 4 (공개 정보 기반 추정)",
        "content": "CXL-PNM 시스템에서 측정한 결과를 보고해 구성요소 검증 단계로 본다. 벤더 부품은 샘플링 단계라 상위 단계 근거가 없다.",
        "details": {
            "trl_band": "4-6",
            "estimate": 4,
            "why_not_higher": "실제 운용 환경 시연 근거가 없고, 벤더 PIM/PNM 부품 평가가 샘플링 프로그램에 한정된다는 보도만 있어 7-9로 올리지 못한다.",
        },
        "citations": [
            W(ARXIV_PIM, Q_PIM_MEAS),
            W(EETIMES, "evaluation is limited to vendor sampling programmes"),
        ],
    },
    "T2": {
        "level": "L3",
        "content": "CXL-PNM 시스템에서 측정한 결과를 보고한다. 실서비스 트래픽 검증은 확인되지 않는다.",
        "details": {"env_level": "L3"},
        "citations": [W(ARXIV_PIM, Q_PIM_MEAS)],
    },
    "T3": {
        "checklist": {
            "code": "N",
            "model_or_design": "Y",
            "eval_scripts_data": "N",
            "third_party_reproduction": "N",
        },
        "urls": {"model_or_design": ARXIV_PIM},
        "content": "논문이 설계를 명세하지만 저장소·RTL·시뮬레이터 설정은 링크돼 있지 않다. 독립 그룹의 재현은 보고되지 않는다.",
        "citations": [
            W(
                ARXIV_PIM,
                "The paper specifies the PNM accelerator design, the token page selection offload, and the hybrid GPU-PNM parallelization strategy",
            ),
            W(
                ARXIV_PIM,
                "No repository, RTL, or simulator configuration is linked from the paper.",
            ),
            W(EETIMES, "evaluation is limited to vendor sampling programmes"),
        ],
    },
    "T4": {
        "level": "narrative",
        "content": "양산 실리콘 검증과 서빙 프레임워크 통합이 남아 있다. 구현 코드는 공개 자료에서 확인되지 않는다.",
        "details": {
            "remaining_tasks": ["양산 CXL 메모리 확장 실리콘에서의 검증"],
            "not_public_items": [
                "구현 코드·시뮬레이터 설정 (검색어: PIM/CXL code availability github)"
            ],
        },
        "citations": [
            W(
                ARXIV_PIM,
                "No repository, RTL, or simulator configuration is linked from the paper.",
            )
        ],
    },
    "D1": {
        "level": "L3",
        "content": "1M 토큰 컨텍스트에서 KV cache를 CXL 확장 메모리로 이전한 결과가 직접 보고된다.",
        "details": {"directness": "L3"},
        "measurements": [
            {
                "metric": "throughput gain",
                "value": "up to 21.9x",
                "model_size": "up to 405B",
                "context_length": "1M",
                "baseline": "stated baseline (paper)",
                "citation_index": 0,
            }
        ],
        "citations": [
            W(ARXIV_PIM, Q_PIM_GAIN),
            C(
                "pim_cxl_1m:004:08",
                "high-bandwidth memory expansion platform tailored for the LLM inference",
            ),
        ],
    },
    "D2": {
        "level": "L2",
        "content": "In-server 구성별 처리량 비교가 있으나 배치 크기·동시 세션 수는 명시되지 않는다.",
        "details": {
            "directness": "L2",
            "extrapolation_logic": "처리량 향상은 다중 요청 처리와 관련되나 배치 크기가 원문에 미기재라 외삽이다.",
        },
        "measurements": [
            {
                "metric": "throughput gain",
                "value": "up to 21.9x",
                "model_size": "up to 405B",
                "context_length": "1M",
                "baseline": "stated baseline (paper)",
                "citation_index": 0,
            }
        ],
        "citations": [
            W(ARXIV_PIM, Q_PIM_GAIN),
            C(
                "pim_cxl_1m:009:14",
                "Figure 10 compares the throughput across three In-server configurations",
            ),
        ],
    },
    "D3": {
        "level": "L2",
        "content": "토큰당 에너지 절감이 보고되나 TTFT/TPOT와 총비용은 확인되지 않는다.",
        "details": {
            "directness": "L2",
            "extrapolation_logic": "에너지 수치만 있고 지연·비용은 미기재라 도메인 총비용에는 간접적으로만 연결된다.",
        },
        "measurements": [
            {
                "metric": "energy per token reduction",
                "value": "up to 60x lower",
                "model_size": "up to 405B",
                "context_length": "1M",
                "baseline": "stated baseline (paper)",
                "citation_index": 0,
            }
        ],
        "citations": [W(ARXIV_PIM, Q_PIM_GAIN)],
    },
    "D4": {
        "level": "checklist",
        "content": "vLLM 문서는 CXL 부착 계층을 지원하지 않아 서빙 엔진 수정이 필요하고, 확장 메모리 추가를 전제한다.",
        "details": {
            "checklist": {
                "model_retrain": "unknown",
                "model_convert": "unknown",
                "serving_engine_change": "Y",
                "hw_replace": "unknown",
                "memory_add": "Y",
                "other": "unknown",
            }
        },
        "citations": [
            W(
                VLLM_DOCS,
                "The documented storage tiers do not include a CXL-attached or near-memory compute device",
            ),
            C(
                "pim_cxl_1m:004:08",
                "memory expansion platform tailored for the LLM inference",
            ),
        ],
    },
}


def drafts(tech_id: str) -> dict:
    return copy.deepcopy(MLA if tech_id == "mla" else PIM)
