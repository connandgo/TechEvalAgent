# KV cache 최적화 기술 다관점 평가 보고서

- 평가 도메인: 대규모 데이터센터의 장문맥 LLM 추론 환경
- 대상 기술: DeepSeek-V2 MLA / PIM/CXL
- 생성 시각: 2026-09-20T10:30:00

## SUMMARY

MLA는 TRL 관점에서 H800 클러스터 학습·서빙을 근거로 TRL 4-6(추정 6)으로 판정되지만, 시장 관점은 제안사의 유료 API 운영을 공식 가격 문서로 확인해 M2를 L4로 판정해, 공개 운영 실적을 요구하는 T1과 상용 운영 사실을 보는 M2의 기준 차이로 엇갈린다[E: mla-T1-01][E: mla-M2-01]. MLA의 이해관계자 상충(S4)은 모델 개발사의 구조 변경이 서빙 운영사에게 전용 백엔드 검증 부담이 된다고 보지만, 도메인 D4는 vLLM의 MLA 백엔드 지원을 근거로 서빙 엔진 수정을 N으로 기록해, 코드 수정 여부와 검증 부담이라는 기준 차이가 드러난다[E: mla-S4-02][E: mla-D4-01]. PIM/CXL은 선정 논문 설계가 CXL-PNM 시스템 측정 단계(TRL 4-6, 추정 4)이나, 시장 M2는 CXL 메모리 확장 컨트롤러 양산을 근거로 L3로 판정해 논문 단위와 계열 부품 단위의 차이가 레벨 차이로 나타난다[E: pim_cxl-T1-01][E: pim_cxl-M2-01]. PIM/CXL의 S4 상충인 벤더의 메모리 수요 확대와 운영사의 장비 도입 비용 부담은, 현재 CXL 메모리 확장 배포가 KV cache 오프로드가 아닌 용량 중심 워크로드에 머문다는 보도와 함께 읽어야 한다[E: pim_cxl-S4-01][E: pim_cxl-M2-02]. 시장·이해관계자 판정은 기술 계열 단위로 확장한 결과이고 PIM/CXL 근거는 벤더 자료 의존이 커서, 두 기술의 시장 레벨을 선정 논문 설계의 채택 단계로 읽어서는 안 된다[E: pim_cxl-M2-04].

## 1. 분석 배경

Transformer 기반 LLM은 디코딩 단계에서 앞선 토큰의 Key·Value 벡터를 KV cache로 보관해 재계산을 피한다. KV cache의 크기는 층 수·헤드 차원·컨텍스트 길이·동시 요청 수에 비례해 커지므로, 긴 입력과 많은 동시 세션이 겹치는 추론 환경에서는 가속기 메모리의 용량과 대역폭이 계산보다 먼저 한계에 닿는다(일반 구조 설명).

이 병목을 다루는 접근은 크게 두 갈래로 나뉜다. 하나는 SW 측에서 KV cache 자체를 줄이는 방법(어텐션 구조 재설계, 양자화, 토큰 선별 등)이고, 다른 하나는 HW 측에서 메모리 계층을 넓히거나 연산 일부를 메모리 쪽으로 옮기는 방법(CXL 메모리 확장, 메모리 근처 연산 등)이다. 본 보고서는 두 갈래에서 기술을 하나씩 골라 같은 기준으로 평가하고, 관점에 따라 평가가 어떻게 달라지는지를 정리한다.

평가 대상 환경은 **대규모 데이터센터의 장문맥 LLM 추론 환경**으로 고정한다. 이 환경은 컨텍스트 길이와 동시 처리량이 함께 커지는 조건이어서 KV cache 부담이 가장 직접적으로 드러나고, SW 축소와 HW 확장이 서로 다른 비용 구조(모델 변경 대 인프라 변경)를 요구한다는 차이가 판정에 반영되기 쉽다.

서베이 근거도 이 구조를 뒷받침한다. `io_survey`는 CXL 연결 메모리의 접근 지연이 로컬 DRAM보다 몇 배 높아, KV cache를 CXL 링크 너머로 내리면 추가 지연을 메모리 측 연산이나 프리페치로 가리지 못할 경우 디코딩 처리량이 떨어질 수 있다고 지적한다[E: pim_cxl-COUNTER-01]. `kv_survey`는 MLA가 어텐션 구조를 바꾸고 모델을 처음부터 학습해야 하므로 MHA·GQA 기반의 기존 사전학습 모델에 바로 적용하기 어렵다고 정리한다[E: mla-COUNTER-01]. 즉 SW 축소는 모델 변경 비용을, HW 확장은 메모리 계층의 지연·대역폭 비용을 동반한다는 점이 두 갈래를 가르는 축이다.

## 2. 기술 선정

기술 선정은 사람이 고정했으며 에이전트가 선택하지 않았다. 선정 기준은 **대표성**(각 접근을 대표하는 설계인가), **대조성**(두 기술이 서로 다른 변경 범위를 전제하는가), **자료 확보 가능성**(원문·공개 구현·계열 자료를 확보할 수 있는가) 세 가지다.

| 접근 | 기술 | 선정 논문 | 발행 | 기술 계열(시장·이해관계자 조사 범위) |
|---|---|---|---|---|
| SW — KV cache 축소 | DeepSeek-V2 MLA | DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model | 2024-06 | MLA 계열 |
| HW — 메모리 계층 확장 | PIM/CXL | 1M-Token LLM Inference with PIM/CXL Memory Expansion | 2025-10 | CXL·PIM/PNM 메모리 계열 |

- **DeepSeek-V2 MLA**: 대표성 — 어텐션 구조를 재설계해 KV를 저차원 잠재 공간으로 압축하는 SW 접근이다. 선정 논문은 KV cache per token reduction 93.3% (모델: 236B (21B active), 컨텍스트: 128K, 베이스라인: DeepSeek 67B, 조건: 원문 미기재: hardware, batch_size)를 보고한다[E: mla-PROFILE-03]. 대조성 — 기존 GPU·HBM 인프라에서 운용할 수 있으나 모델 구조와 서빙 최적화가 필요하다. 자료 확보 — 공개 모델·논문·서빙 프레임워크 자료가 있다.
- **PIM/CXL**: 대표성 — KV cache를 CXL 메모리로 확장하고 토큰 페이지 선택 연산을 메모리 측 가속기로 옮기는 HW 접근이다. 대조성 — 서버 HW·메모리 계층·서빙 경로 변경을 전제한다. 자료 확보 — 논문 실험과 CXL·PIM/PNM 벤더 발표·표준화·후속 연구가 있다. 논문 시제품과 기술 계열 전체의 상용화 근거는 구분해 기록했다.

비교 후보(TurboQuant·KIVI 등 KV cache 양자화·압축 계열)는 `kv_survey`의 KV cache 관리 기법 분류를 참고해 검토했으며, 위 세 기준을 함께 충족하는 조합으로 두 기술을 고정했다.

`kv_survey`는 MLA를 어텐션 구조를 바꾸고 처음부터 학습해야 하는 구조 수준의 KV cache 압축으로 분류한다[E: mla-COUNTER-01]. 양자화 계열 후보는 모델 구조를 유지한 채 저장 정밀도를 낮추는 쪽에 속하므로, 대조성 기준(두 기술이 서로 다른 변경 범위를 전제하는가)을 드러내는 SW 측 사례로 구조 변경형인 MLA를 고정했다.

## 3. 기술 개요

### 3.1 DeepSeek-V2 MLA

**핵심 접근.** MLA는 키와 값을 저차원 잠재 벡터로 공동 압축해 생성 중 캐시할 데이터를 줄이는, 어텐션 구조 자체를 바꾸는 접근이다. 원문은 "The core of MLA is the low-rank joint compression for keys and values to reduce KV"라고 설명한다[E: mla-PROFILE-02]. 적용 범위는 모델 아키텍처 수준의 변경이며, 논문은 236B(활성 21B) MoE 모델과 128K 컨텍스트에서 결과를 보고한다[E: mla-PROFILE-03].

**정량 성과.** 선정 논문은 토큰당 KV cache 93.3% 감소(236B·활성 21B, 128K 컨텍스트, vs DeepSeek 67B; 하드웨어·배치는 원문 미기재)를 보고한다[E: mla-PROFILE-03]. 학습 비용은 1조 토큰당 172.8K GPU hours(H800 클러스터, vs DeepSeek 67B 300.6K GPU hours; 모델 규모·컨텍스트·배치는 원문 미기재)로 보고되었으며, 이는 학습 단계 수치로 추론 지표와 구분된다[E: mla-PROFILE-05].

**검증 환경과 공개 자료.** 논문은 NVIDIA H800 클러스터에서 학습·서빙했다고 보고하며, 실서비스 트래픽 검증은 확인되지 않는다[E: mla-PROFILE-04]. 공식 저장소에 추론 예제 코드가 있고 가중치가 공개되어 있으며 vLLM이 독립 구현을 유지하지만, 평가 스크립트·데이터는 포함되지 않았다[E: mla-T3-01][E: mla-T3-03][E: mla-T3-04].

**명시된 한계.** 공개 저장소에는 학습 파이프라인과 논문 수치에 쓰인 벤치마크 하네스가 포함되지 않는다[E: mla-T3-03]. 또한 보고된 KV cache 감소의 베이스라인은 DeepSeek 67B로, 같은 규모 모델 사이의 비교가 아니다[E: mla-PROFILE-03].

**정량 성과와 측정 조건** (서로 다른 실험의 값이므로 행 사이를 직접 비교하지 않는다)

| 지표 | 값 | 모델 규모 | 컨텍스트 | 하드웨어 | 베이스라인 | 근거 |
|---|---|---|---|---|---|---|
| KV cache per token reduction | 93.3% | 236B (21B active) | 128K | — | DeepSeek 67B | [E: mla-PROFILE-03] |
| training GPU hours per trillion tokens | 172.8K GPU hours | — | — | H800 cluster | DeepSeek 67B: 300.6K | [E: mla-PROFILE-05] |

### 3.2 PIM/CXL

**핵심 접근.** 이 접근은 KV cache를 CXL로 확장한 메모리에 두고, 토큰 페이지 선택 연산을 메모리 옆 가속기(PNM)로 이관해 GPU 부착 메모리의 물리적 한계를 넘는 메모리·연산 조율을 한다[E: pim_cxl-PROFILE-04]. 원문은 이를 "a scalable, high-capacity, and high-bandwidth memory expansion platform tailored for the LLM inference"로 설명한다[E: pim_cxl-PROFILE-01]. 적용 범위는 1M 토큰급 장문맥이며, 논문은 최대 405B 모델까지 보고한다[E: pim_cxl-PROFILE-02].

**정량 성과.** 논문이 제시한 베이스라인 대비 처리량 최대 21.9x(최대 405B 모델, 1M 토큰 컨텍스트; 하드웨어·배치는 원문 미기재)를 보고한다[E: pim_cxl-PROFILE-02]. 같은 조건에서 토큰당 에너지는 최대 60x 낮다고 보고되었다[E: pim_cxl-PROFILE-02]. 두 값은 모두 "최대" 수치이며, 베이스라인 구성은 원문 표기를 그대로 옮겼다.

**검증 환경과 공개 자료.** 결과는 CXL-PNM 시스템에서 측정한 것으로 보고되며, 양산 실리콘·실서비스 트래픽 검증은 확인되지 않는다[E: pim_cxl-PROFILE-03]. 논문은 PNM 가속기 설계를 명세하지만 저장소·RTL·시뮬레이터 설정은 링크하지 않았고, 제3자 재현도 보고되지 않았다[E: pim_cxl-T3-01][E: pim_cxl-T3-02].

**명시된 한계.** 저장소·RTL·시뮬레이터 설정이 논문에 링크돼 있지 않아 외부에서 재현할 경로가 없다[E: pim_cxl-T3-02].

**정량 성과와 측정 조건** (서로 다른 실험의 값이므로 행 사이를 직접 비교하지 않는다)

| 지표 | 값 | 모델 규모 | 컨텍스트 | 하드웨어 | 베이스라인 | 근거 |
|---|---|---|---|---|---|---|
| throughput gain | up to 21.9x | up to 405B | 1M | — | stated baseline (paper) | [E: pim_cxl-PROFILE-02] |
| energy per token reduction | up to 60x lower | up to 405B | 1M | — | stated baseline (paper) | [E: pim_cxl-PROFILE-02] |

## 4. 관점별 평가

각 절은 두 기술에 같은 기준을 적용한 결과다. 절 말미 요약표의 레벨은 기준별 판정 단계이며 관점·기준이 다른 레벨끼리 더하거나 묶어 하나의 값으로 만들지 않는다. 근거 단위는 논문·구현(paper)과 기술 계열(family)로 구분해 적었다.

### 4.1 기술 성숙도(TRL)

**DeepSeek-V2 MLA**

- **T1 현재 TRL — TRL 4-6 (추정 TRL 6, 공개 정보 기반 추정)**: H800 클러스터에서 학습·서빙했고 가중치가 공개돼 관련 환경 검증 단계로 보았다[E: mla-T1-01][E: mla-T1-02]. 한 단계 위로 올리지 못한 이유는 지속 트래픽 하의 실서비스 운영 실적이 공개 자료에서 확인되지 않기 때문이다. 신뢰도 high, 근거 단위 paper.
- **T2 검증 환경 — L3**: 실물 H800 GPU 클러스터에서 학습·서빙했다고 보고하며, 실서비스 트래픽 검증은 원문에 없다[E: mla-T2-01]. 신뢰도 medium, 근거 단위 paper.
- **T3 재현성 — L3**: 코드 Y(공식 저장소의 추론 예제), 모델/설계 Y(가중치 공개), 평가 스크립트·데이터 N(벤치마크 하네스 미포함), 제3자 재현 Y(vLLM의 독립 MLA 백엔드)다[E: mla-T3-01][E: mla-T3-02][E: mla-T3-03][E: mla-T3-04]. 신뢰도 medium.
- **T4 다음 과제**: 남은 과제는 지속 트래픽 하의 다중 테넌트 서빙 검증이다[E: mla-T4-01]. 공개되지 않은 항목은 벤치마크 하네스(검색어: DeepSeek-V2 code availability github)와 실서비스 배포 규모(검색어: DeepSeek-V2 MLA production deployment)다. 신뢰도 medium.

**PIM/CXL**

- **T1 현재 TRL — TRL 4-6 (추정 TRL 4, 공개 정보 기반 추정)**: CXL-PNM 시스템에서 측정한 결과를 보고해 구성요소 검증 단계로 보았다[E: pim_cxl-T1-01]. 한 단계 위로 올리지 못한 이유는 실제 운용 환경 시연 근거가 없고, 벤더 PIM/PNM 부품 평가가 샘플링 프로그램에 한정된다는 보도만 있기 때문이다[E: pim_cxl-T1-02]. 신뢰도 high, 근거 단위 paper.
- **T2 검증 환경 — L3**: CXL-PNM 시스템에서 측정한 결과를 보고하며, 실서비스 트래픽 검증은 확인되지 않는다[E: pim_cxl-T2-01]. 신뢰도 medium.
- **T3 재현성 — L2**: 코드 N, 모델/설계 Y(논문의 PNM 가속기·페이지 선택 이관 설계 명세), 평가 스크립트·데이터 N, 제3자 재현 N이다[E: pim_cxl-T3-01][E: pim_cxl-T3-02][E: pim_cxl-T3-03]. 신뢰도 high.
- **T4 다음 과제**: 남은 과제는 양산 CXL 메모리 확장 실리콘에서의 검증이며, 구현 코드·시뮬레이터 설정은 공개되지 않았다(검색어: PIM/CXL code availability github)[E: pim_cxl-T4-01]. 신뢰도 medium.

두 기술은 T1 구간(TRL 4-6)과 T2 검증 환경(L3)이 같지만 구간 내 추정치와 재현성 항목이 다르며, 이는 공개 자료로 확인되는 가중치·제3자 구현 유무의 차이다.

**표 4.1 기술 × 기준 요약 — 기술 성숙도(TRL)**

| 기준 | 기술 | 레벨 | 신뢰도 | 근거 단위 | 근거 |
|---|---|---|---|---|---|
| T1 현재 TRL | DeepSeek-V2 MLA | TRL 4-6 (추정: TRL 6 (공개 정보 기반 추정)) | high | 논문·구현(paper) | [E: mla-T1-01] [E: mla-T1-02] |
| T1 현재 TRL | PIM/CXL | TRL 4-6 (추정: TRL 4 (공개 정보 기반 추정)) | high | 논문·구현(paper) | [E: pim_cxl-T1-01] [E: pim_cxl-T1-02] |
| T2 검증 환경 | DeepSeek-V2 MLA | L3 | medium | 논문·구현(paper) | [E: mla-T2-01] |
| T2 검증 환경 | PIM/CXL | L3 | medium | 논문·구현(paper) | [E: pim_cxl-T2-01] |
| T3 재현성 | DeepSeek-V2 MLA | L3 | medium | 논문·구현(paper) | [E: mla-T3-01] [E: mla-T3-02] [E: mla-T3-03] [E: mla-T3-04] |
| T3 재현성 | PIM/CXL | L2 | high | 논문·구현(paper) | [E: pim_cxl-T3-01] [E: pim_cxl-T3-02] [E: pim_cxl-T3-03] |
| T4 다음 과제 | DeepSeek-V2 MLA | narrative | medium | 논문·구현(paper) | [E: mla-T4-01] |
| T4 다음 과제 | PIM/CXL | narrative | medium | 논문·구현(paper) | [E: pim_cxl-T4-01] |

레벨은 기준별 판정 단계일 뿐이며, 관점·기준이 다른 레벨끼리 더하거나 묶어 하나의 값으로 만들지 않는다.

### 4.2 시장성

이 절의 판정은 선정 논문이 아니라 기술 계열 단위(MLA 계열 / CXL·PIM/PNM 메모리 계열)로 확장한 결과다.

**DeepSeek-V2 MLA**

- **M1 성장성 — L2 형성**: 128K 컨텍스트에서 세션당 메모리 점유가 토큰당 비용을 좌우한다는 매체 분석(SemiAnalysis, 2025-05-12)과, 메모리 가격이 추론 서비스 마진의 주요 변수가 되었다는 보도(Reuters, 2026-02-11)가 수요 동인으로 확인된다[E: mla-M1-01][E: mla-M1-02]. 다만 MLA를 포함한 KV cache 축소 기법 계열을 따로 집계한 정량 시장 전망은 확인되지 않았고, 단일 매체 추정이므로 사실로 단정하지 않는다. 신뢰도 medium.
- **M2 채택 — L4 상용 운영**: 채택 주체는 DeepSeek로, MLA 기반 모델을 토큰당 가격이 공개된 유료 API로 운영하고 있다(공식 가격 문서, 2025-02-20)[E: mla-M2-01]. 오픈 웨이트 배포로 제3자 서빙 스택에서 구동한 사례도 보도되었으나 운영 주체가 명시되지 않아 별도 채택 사례로 세지 않았다[E: mla-M2-02][E: mla-M2-03]. 신뢰도 high.
- **M3 생태계 — L2**: 프레임워크 Y(vLLM·SGLang의 MLA 전용 백엔드), 벤더 제품 N, 표준화 N, 제3자 연구·도구 Y(KV cache 관리 서베이의 분류)다[E: mla-M3-01][E: mla-M3-02][E: mla-M3-04]. 백엔드가 전제하는 압축 latent KV 표현은 원문의 저랭크 결합 압축 설명과 일치한다[E: mla-M3-03]. 신뢰도 high.

**PIM/CXL**

- **M1 성장성 — L2 형성**: TrendForce(2025-03-18)는 CXL 메모리 모듈 시장이 2028년 21억 달러에 이를 것으로 전망했고, MarketsandMarkets(2025-08-01)는 2030년까지 CXL 부착 메모리의 연평균 성장률을 30%대 중반으로 추정했다[E: pim_cxl-M1-01][E: pim_cxl-M1-02]. 두 수치 모두 발행 주체의 추정치이고 산출 근거가 공개되지 않아 사실로 단정하지 않는다. 수요 동인으로는 메모리 제조사의 설비 투자 확대 보도가 있다[E: pim_cxl-M1-03]. 신뢰도 medium.
- **M2 채택 — L3 제품화**: CXL 메모리 확장 컨트롤러는 Astera Labs의 양산 제품으로 출하되고 있다[E: pim_cxl-M2-01]. 반면 연산 기능을 메모리에 둔 PIM/PNM 부품(Samsung HBM-PIM, SK hynix AiM)은 PoC·샘플링 단계이며, LLM 추론 서버용 양산을 밝힌 벤더는 없다[E: pim_cxl-M2-03][E: pim_cxl-M2-04]. 현재 CXL 메모리 확장 배포도 KV cache 오프로드가 아니라 용량 중심 인메모리 워크로드를 겨냥한다[E: pim_cxl-M2-02]. 선정 논문의 CXL-PNM 시제품은 계열 제품의 양산 사실과 구분해 기록한다. 신뢰도 high.
- **M3 생태계 — L2**: 프레임워크 N(오픈소스 서빙 프레임워크가 CXL 계층을 문서화한 사례 없음), 벤더 제품 Y(메모리 확장 컨트롤러), 표준화 Y(CXL 3.1 사양, JEDEC 협업), 제3자 연구·도구 N이다[E: pim_cxl-M3-01][E: pim_cxl-M3-02][E: pim_cxl-M3-03]. 선정 논문의 CXL-PNM 확장 플랫폼은 원문 근거로 함께 기록했다[E: pim_cxl-M3-04]. 신뢰도 high.

MLA 계열의 채택 근거는 제안사 자체 유료 API에, PIM/CXL 계열의 채택 근거는 메모리 확장 컨트롤러라는 계열 부품의 양산에 기대고 있어, 두 기술의 시장 레벨은 서로 다른 대상의 채택 단계를 가리킨다.

**표 4.2 기술 × 기준 요약 — 시장성**

| 기준 | 기술 | 레벨 | 신뢰도 | 근거 단위 | 근거 |
|---|---|---|---|---|---|
| M1 성장성 | DeepSeek-V2 MLA | L2 | medium | 기술 계열(family) | [E: mla-M1-01] [E: mla-M1-02] |
| M1 성장성 | PIM/CXL | L2 | medium | 기술 계열(family) | [E: pim_cxl-M1-01] [E: pim_cxl-M1-02] [E: pim_cxl-M1-03] |
| M2 채택 | DeepSeek-V2 MLA | L4 | high | 기술 계열(family) | [E: mla-M2-01] [E: mla-M2-02] [E: mla-M2-03] |
| M2 채택 | PIM/CXL | L3 | high | 기술 계열(family) | [E: pim_cxl-M2-01] [E: pim_cxl-M2-02] [E: pim_cxl-M2-03] [E: pim_cxl-M2-04] |
| M3 생태계 | DeepSeek-V2 MLA | L2 | high | 기술 계열(family) | [E: mla-M3-01] [E: mla-M3-02] [E: mla-M3-03] [E: mla-M3-04] |
| M3 생태계 | PIM/CXL | L2 | high | 기술 계열(family) | [E: pim_cxl-M3-01] [E: pim_cxl-M3-02] [E: pim_cxl-M3-03] [E: pim_cxl-M3-04] |

레벨은 기준별 판정 단계일 뿐이며, 관점·기준이 다른 레벨끼리 더하거나 묶어 하나의 값으로 만들지 않는다.

### 4.3 이해관계자

이 절은 기술 계열 단위로, 모델 개발사 / 클라우드·서빙 운영사 / 메모리·반도체 벤더 / 투자 업계 4주체를 고정해 판정했다.

**DeepSeek-V2 MLA**

- **S1 의사결정**: 모델 구조를 바꾸는 기법이므로 채택을 정하는 결정권자는 모델 개발사다[E: mla-S1-01]. 서빙 운영사는 MLA 전용 투영 차원을 선언한 모델 구성과 백엔드 지원 여부에 따라 영향받는 자이고[E: mla-S1-02], 메모리·반도체 벤더는 자사 부품을 요구하지 않아 무관, 투자 업계는 영향받는 자로 두었다(추론)[E: mla-S1-03].
- **S2 편익**: 모델 개발사는 KV cache 점유를 줄여 더 긴 컨텍스트나 더 큰 배치를 다룰 여지를 얻고[E: mla-S2-01], 서빙 운영사는 지원 백엔드가 있는 스택에서 하드웨어 증설 없이 모델을 제공할 수 있다[E: mla-S2-02]. 벤더(기존 제품 라인 유지)와 투자 업계(설비 투자 없는 단가 구조 변화)의 편익은 추론이다[E: mla-S2-03][E: mla-S2-04].
- **S3 부담**: 모델 개발사는 MLA 전용 구성이 필요해 기존 모델에 사후 적용할 수 없고[E: mla-S3-01], 서빙 운영사는 디코딩 경로가 일반 어텐션과 달라 백엔드별 검증이 필요하다[E: mla-S3-02]. 벤더(용량 증설 수요 일부의 소프트웨어 흡수)와 투자 업계(적용 시점·범위 불확실)의 부담은 추론이다[E: mla-S3-03][E: mla-S3-04].
- **S4 상충**: 모델 개발사에게 편익인 메모리 점유 축소가 메모리·반도체 벤더에게는 용량 증설 수요가 줄어드는 부담이 되고(추론 포함)[E: mla-S4-01][E: mla-S4-03], 모델 개발사에게 편익인 구조 변경이 서빙 운영사에게는 전용 백엔드를 따로 검증해야 하는 부담이 된다[E: mla-S4-02].

**PIM/CXL**

- **S1 의사결정**: 서빙 운영사는 장비 도입을 정하는 결정권자, 메모리·반도체 벤더는 공급자, 모델 개발사는 모델 구조 변경 없이 영향받는 자이며, 투자 업계는 메모리 설비 투자 보도에서 결정권자로 등장한다[E: pim_cxl-S1-01][E: pim_cxl-S1-02][E: pim_cxl-S1-03].
- **S2 편익**: 서빙 운영사는 GPU 부착 메모리 한계를 넘어 용량을 늘리는 선택지를 얻고[E: pim_cxl-S2-01], 벤더는 메모리 수요 확대를 설비 투자 근거로 삼으며[E: pim_cxl-S2-02], 투자 업계에서는 메모리 중심 하드웨어가 별도 투자 대상으로 다뤄진다[E: pim_cxl-S2-03]. 모델 개발사의 편익(모델 구조 변경 없는 용량 여유)은 추론이다[E: pim_cxl-S2-04].
- **S3 부담**: 서빙 운영사는 실제 출하 구성이 제한적이어서 도입 범위를 넓히기 어렵고[E: pim_cxl-S3-02], 벤더는 수요 확정 전 설비 투자 비용을 먼저 부담하며[E: pim_cxl-S3-03], 투자 업계는 부품이 샘플링 단계에 머물러 회수 시점을 가늠하기 어렵다[E: pim_cxl-S3-01]. 모델 개발사의 부담(동작 사전 확인의 어려움)은 추론이다[E: pim_cxl-S3-04].
- **S4 상충**: 메모리·반도체 벤더에게 편익인 메모리 수요 확대가 클라우드·서빙 운영사에게는 장비 도입 비용 증가라는 부담이 되고[E: pim_cxl-S4-01][E: pim_cxl-S4-02], 벤더에게 편익인 선제적 설비 투자가 투자 업계에는 수요 확정 전 자본이 묶이는 부담이 된다[E: pim_cxl-S4-01][E: pim_cxl-S4-03].

두 기술 모두 편익을 얻는 주체와 부담을 지는 주체가 다르며, MLA에서는 모델 개발사의 편익이 벤더·운영사의 부담으로, PIM/CXL에서는 벤더의 편익이 운영사·투자 업계의 부담으로 이어진다.

**표 4.3 기술 × 기준 요약 — 이해관계자**

| 기준 | 기술 | 레벨 | 신뢰도 | 근거 단위 | 근거 |
|---|---|---|---|---|---|
| S1 의사결정 | DeepSeek-V2 MLA | assigned (모델 개발사 결정권자, 서빙 운영사 영향, 메모리 벤더 무관, 투자 업계 영향) | low | 기술 계열(family) | [E: mla-S1-01] [E: mla-S1-02] [E: mla-S1-03] |
| S1 의사결정 | PIM/CXL | assigned (모델 개발사 영향, 서빙 운영사 결정권자, 메모리 벤더 공급자, 투자 업계 결정권자) | high | 기술 계열(family) | [E: pim_cxl-S1-01] [E: pim_cxl-S1-02] [E: pim_cxl-S1-03] |
| S2 편익 | DeepSeek-V2 MLA | narrative | low | 기술 계열(family) | [E: mla-S2-01] [E: mla-S2-02] [E: mla-S2-03] [E: mla-S2-04] |
| S2 편익 | PIM/CXL | narrative | low | 기술 계열(family) | [E: pim_cxl-S2-01] [E: pim_cxl-S2-02] [E: pim_cxl-S2-03] [E: pim_cxl-S2-04] |
| S3 부담 | DeepSeek-V2 MLA | narrative | low | 기술 계열(family) | [E: mla-S3-01] [E: mla-S3-02] [E: mla-S3-03] [E: mla-S3-04] |
| S3 부담 | PIM/CXL | narrative | low | 기술 계열(family) | [E: pim_cxl-S3-01] [E: pim_cxl-S3-02] [E: pim_cxl-S3-03] [E: pim_cxl-S3-04] |
| S4 상충 | DeepSeek-V2 MLA | narrative | low | 기술 계열(family) | [E: mla-S4-01] [E: mla-S4-02] [E: mla-S4-03] |
| S4 상충 | PIM/CXL | narrative | medium | 기술 계열(family) | [E: pim_cxl-S4-01] [E: pim_cxl-S4-02] [E: pim_cxl-S4-03] |

레벨은 기준별 판정 단계일 뿐이며, 관점·기준이 다른 레벨끼리 더하거나 묶어 하나의 값으로 만들지 않는다.

### 4.4 도메인 적합성

**DeepSeek-V2 MLA**

- **D1 장문맥 병목 — L2(간접)**: 토큰당 KV cache 93.3% 감소(236B·활성 21B, 128K 컨텍스트, vs DeepSeek 67B; 하드웨어·배치 원문 미기재)가 보고되었다[E: mla-D1-01]. 다중 요청 조건은 명시되지 않아, 토큰당 캐시 감소가 요청 수와 컨텍스트 길이에 선형으로 비례한다는 외삽 논리로 장문맥 다중 요청 조건에 연결했다[E: mla-D1-02]. 신뢰도 high.
- **D2 동시 처리 — L2(간접)**: 같은 토큰당 KV cache 93.3% 감소(236B·활성 21B, 128K, vs DeepSeek 67B) 외에 배치 크기·동시 세션 수치는 원문에 없다[E: mla-D2-01]. 같은 메모리에 더 많은 세션을 담을 수 있다는 것은 외삽이며 실제 배치 처리량은 미기재다. 신뢰도 medium.
- **D3 지연·에너지·비용 — L2(간접)**: 학습 비용 1조 토큰당 172.8K GPU hours(H800 클러스터, vs DeepSeek 67B 300.6K)만 보고되고 추론 지연·토큰당 에너지·전송 오버헤드는 확인되지 않는다[E: mla-D3-01]. 학습 비용은 추론 총비용의 직접 근거가 아니어서, 캐시 감소가 메모리 대역폭 요구를 줄인다는 간접 논리로만 연결했다. 신뢰도 medium.
- **D4 변경 범위**: 모델 재학습 미확인, 모델 변환 미확인, 서빙 엔진 수정 N(vLLM이 MLA 전용 백엔드 지원), HW 교체 N, 메모리 추가 N, 기타 미확인이다[E: mla-D4-01][E: mla-D4-02].

**PIM/CXL**

- **D1 장문맥 병목 — L3(직접)**: 1M 토큰 컨텍스트에서 KV cache를 CXL 확장 메모리로 옮긴 결과가 직접 보고되며, 논문 베이스라인 대비 처리량 최대 21.9x(최대 405B, 1M; 하드웨어·배치 원문 미기재)다[E: pim_cxl-D1-01][E: pim_cxl-D1-02]. 신뢰도 high.
- **D2 동시 처리 — L2(간접)**: In-server 구성별 처리량 비교(최대 21.9x, 최대 405B, 1M)가 있으나 배치 크기·동시 세션 수는 명시되지 않아 다중 요청 처리로의 연결은 외삽이다[E: pim_cxl-D2-01][E: pim_cxl-D2-02]. 신뢰도 high.
- **D3 지연·에너지·비용 — L2(간접)**: 토큰당 에너지 최대 60x 절감(최대 405B, 1M, 논문 베이스라인 대비)이 보고되나 TTFT/TPOT와 총비용은 확인되지 않아, 도메인 총비용에는 간접적으로만 연결된다[E: pim_cxl-D3-01]. 신뢰도 medium.
- **D4 변경 범위**: 모델 재학습 미확인, 모델 변환 미확인, 서빙 엔진 수정 Y(vLLM 문서의 저장 계층에 CXL 부착·근접 연산 장치 없음), HW 교체 미확인, 메모리 추가 Y, 기타 미확인이다[E: pim_cxl-D4-01][E: pim_cxl-D4-02].

두 기술의 수치는 서로 다른 실험 조건에서 나온 값이므로 직접 비교하지 않는다. 도메인 관점의 판정 차이는 수치 크기가 아니라 도메인 조건과의 직접성(실측 대 외삽)과 변경 범위 항목의 차이다.

**표 4.4 기술 × 기준 요약 — 도메인 적합성**

| 기준 | 기술 | 레벨 | 신뢰도 | 근거 단위 | 근거 |
|---|---|---|---|---|---|
| D1 장문맥 병목 | DeepSeek-V2 MLA | L2 | high | 논문·구현(paper) | [E: mla-D1-01] [E: mla-D1-02] |
| D1 장문맥 병목 | PIM/CXL | L3 | high | 논문·구현(paper) | [E: pim_cxl-D1-01] [E: pim_cxl-D1-02] |
| D2 동시 처리 | DeepSeek-V2 MLA | L2 | medium | 논문·구현(paper) | [E: mla-D2-01] |
| D2 동시 처리 | PIM/CXL | L2 | high | 논문·구현(paper) | [E: pim_cxl-D2-01] [E: pim_cxl-D2-02] |
| D3 지연·에너지·비용 | DeepSeek-V2 MLA | L2 | medium | 논문·구현(paper) | [E: mla-D3-01] |
| D3 지연·에너지·비용 | PIM/CXL | L2 | medium | 논문·구현(paper) | [E: pim_cxl-D3-01] |
| D4 변경 범위 | DeepSeek-V2 MLA | checklist (필요: 없음; 미확인: 재학습, 모델 변환, 기타) | high | 논문·구현(paper) | [E: mla-D4-01] [E: mla-D4-02] |
| D4 변경 범위 | PIM/CXL | checklist (필요: 서빙 엔진 수정, 메모리 추가; 미확인: 재학습, 모델 변환, HW 교체, 기타) | high | 논문·구현(paper) | [E: pim_cxl-D4-01] [E: pim_cxl-D4-02] |

레벨은 기준별 판정 단계일 뿐이며, 관점·기준이 다른 레벨끼리 더하거나 묶어 하나의 값으로 만들지 않는다.

## 5. 시사점

**MLA — TRL(T1)과 시장 채택(M2).** TRL 관점은 실서비스 트래픽 운영 근거가 공개 자료에 없어 TRL 4-6(추정 6)으로 두지만, 시장 관점은 제안사가 MLA 기반 모델을 유료 API로 운영한다는 공식 가격 문서를 근거로 M2를 L4로 판정한다[E: mla-T1-01][E: mla-M2-01]. T1은 논문·구현 단위의 운영 실적(트래픽·규모) 공개를 요구하고, M2는 채택 주체의 상용 운영 사실을 제안사 공식 문서로 확인한다. 두 판정은 같은 기술을 서로 다른 질문과 근거 유형으로 본 결과다.

**MLA — 이해관계자 상충(S4)과 변경 범위(D4).** 이해관계자 관점은 모델 개발사의 구조 변경이 서빙 운영사에게 전용 백엔드를 따로 검증해야 하는 부담이 된다고 보지만[E: mla-S4-02], 도메인 D4는 vLLM이 MLA 전용 백엔드를 지원한다는 근거로 서빙 엔진 수정을 N으로 기록한다[E: mla-D4-01]. D4는 서빙 엔진 코드를 고쳐야 하는지를, S4는 이미 있는 백엔드라도 디코딩 경로가 달라 운영사가 따로 검증해야 하는 부담을 묻는 기준 차이다.

**PIM/CXL — 이해관계자 상충(S4)과 시장 채택(M2).** 시장 M2는 CXL 메모리 확장 컨트롤러 양산을 근거로 L3로 판정하지만, S4는 벤더의 메모리 수요 확대가 운영사의 장비 도입 비용 부담으로 이어진다고 본다[E: pim_cxl-M2-01][E: pim_cxl-S4-01][E: pim_cxl-S4-02]. 운영사 측 근거는 현재 CXL 메모리 확장 배포가 KV cache 오프로드가 아닌 용량 중심 워크로드를 겨냥한다고 전한다[E: pim_cxl-M2-02]. M2는 벤더 제품 출하(official)로 계열의 채택 단계를 보고, S4는 운영사가 KV cache 용도로 도입할 때의 부담을 제3자 보도(web)로 보므로 근거 유형과 기준이 다르다.

**PIM/CXL — TRL(T1)과 시장 채택(M2).** TRL 관점은 CXL-PNM 시스템 측정과 벤더 부품이 샘플링 단계라는 보도를 근거로 TRL 4-6(추정 4)으로 두지만, 시장 M2는 CXL 메모리 확장 컨트롤러가 양산 출하 중이라는 벤더 발표를 근거로 L3로 판정한다[E: pim_cxl-T1-01][E: pim_cxl-T1-02][E: pim_cxl-M2-01]. T1은 선정 논문의 CXL-PNM 설계를, M2는 계열 부품 중 메모리 확장 컨트롤러의 제품화를 보는 평가 단위 차이다. PIM/PNM 부품 자체가 샘플링 단계라는 점은 T1·M2·S3에서 같은 방향으로 확인된다[E: pim_cxl-M2-04][E: pim_cxl-S3-01].

**평가 단위와 근거 비대칭이 해석에 주는 제약.** PIM/CXL의 M2 L3는 선정 논문 설계의 채택이 아니라 계열 부품의 제품화로 읽어야 한다. 반대 근거 탐색으로 추가된 자료는 이 해석을 보강한다. GQA 모델을 MLA로 바꾸려면 추가 미세조정이 필요하고 속도 향상이 서빙 엔진의 커널 지원에 달려 있다는 근거[E: mla-COUNTER-02]는 MLA의 S3·S4 부담과 같은 지점을, CXL 메모리 확장이 AI 추론에서는 아직 평가 시스템 중심이라는 근거[E: pim_cxl-COUNTER-02]는 PIM/CXL M2가 KV cache 용도의 실서비스 운영 단계가 아님을 가리킨다.

두 기술의 평가는 어느 관점의 기준으로 보느냐에 따라 달라진다. 모델 구조 변경을 묻는 기준(S3·S4)에서는 MLA의 부담이 개발사·운영사에, 인프라·부품 단계를 묻는 기준(M2·S3·D4)에서는 PIM/CXL의 부담이 운영사·벤더에 드러난다[E: mla-S3-01][E: pim_cxl-S3-02][E: pim_cxl-D4-01].

## 6. 한계점

1. **공개 정보만으로 TRL을 추정한 한계.** 두 기술의 T1은 공개 자료 기반 추정이다. MLA는 지속 트래픽 하의 실서비스 운영 실적이 공개 자료에서 확인되지 않아 TRL 4-6(추정 TRL 6)에 두었고[E: mla-T1-01], PIM/CXL은 실제 운용 환경 시연 근거가 없고 벤더 부품 평가가 샘플링 단계라는 보도만 있어 TRL 4-6(추정 TRL 4)에 두었다[E: pim_cxl-T1-02]. 비공개 운영 자료가 공개되면 판정이 달라질 수 있다.
2. **평가 단위 확장.** 시장·이해관계자 관점은 선정 논문이 아니라 MLA 계열 / CXL·PIM/PNM 메모리 계열 단위로 확장해 평가했다. 특히 PIM/CXL의 M2 L3는 메모리 확장 컨트롤러라는 계열 부품의 양산을 근거로 하므로 선정 논문 설계의 채택 단계와 같지 않다. 두 기술의 M3는 계열 판정에 원문(논문 단위) 근거를 함께 썼다.
3. **제안사·벤더 자료 의존 비율.** evidence_gap 기준 벤더(official) 자료 비율은 DeepSeek-V2 MLA 35.0%, PIM/CXL 55.0%다(근거 비대칭 검사 산출값). 고유 evidence 수도 MLA 23건, PIM/CXL 11건으로 비대칭이며, PIM/CXL 시장·이해관계자 근거의 벤더 자료 의존이 크다는 점을 감안해 읽어야 한다.
4. **적용한 검사와 남은 한계.** 기술 조사·관점별 근거 검사, 근거 비대칭 검사, 반대 근거 탐색(4건 반영), 별도 검수 모델의 보고서 검수를 적용했다. 그럼에도 not_public 항목(MLA 벤치마크 하네스·배포 규모, PIM/CXL 구현 코드·시뮬레이터 설정), 단일 출처 판정(T2, D2·D3 등), 평가 단위 혼합(두 기술의 M3), 반대 근거가 없는 기준(mla D2)이 남아 있으며 아래 표에 정리했다.

**남은 근거 공백 (synthesis.gaps)**

| 기술 | 기준 | 공백 유형 | 내용 |
|---|---|---|---|
| DeepSeek-V2 MLA | T2 | single_source | T2 인용 출처가 1개뿐(confidence=medium) |
| DeepSeek-V2 MLA | T4 | not_public | T4 판정 일부 항목이 not_public으로 기록됨: 벤치마크 하네스 (검색어: DeepSeek-V2 code availability github); 실서비스 배포 규모 (검색어: DeepSeek-V2 MLA production deployment) |
| PIM/CXL | T2 | single_source | T2 인용 출처가 1개뿐(confidence=medium) |
| PIM/CXL | T4 | not_public | T4 판정 일부 항목이 not_public으로 기록됨: 구현 코드·시뮬레이터 설정 (검색어: PIM/CXL code availability github) |
| DeepSeek-V2 MLA | M3 | unit_mismatch | M3 family 단위 판정에 paper 단위 근거가 섞임 |
| PIM/CXL | M3 | unit_mismatch | M3 family 단위 판정에 paper 단위 근거가 섞임 |
| DeepSeek-V2 MLA | D2 | single_source | D2 인용 출처가 1개뿐(confidence=medium) |
| DeepSeek-V2 MLA | D3 | single_source | D3 인용 출처가 1개뿐(confidence=medium) |
| PIM/CXL | D3 | single_source | D3 인용 출처가 1개뿐(confidence=medium) |
| DeepSeek-V2 MLA | D2 | opposing_missing | 동시 처리 근거가 원저자 보고뿐이다. counter_evidence(mla-COUNTER-02)는 속도 향상이 서빙 엔진의 MLA 커널 지원에 달려 있다고 지적하나, 처리량 이득이 줄어드는 조건을 측정한 제3자 근거는 없다. |

## REFERENCE

1. DeepSeek-AI(2024). DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model. arXiv, pp. 6, 7, 16, https://arxiv.org/abs/2405.04434. (근거 ID: mla-T1-01, mla-PROFILE-03, mla-PROFILE-02, mla-PROFILE-05, mla-PROFILE-04, mla-T2-01, mla-M3-03, mla-D1-01, mla-D1-02, mla-D2-01, mla-D3-01, mla-D4-02)
2. DeepSeek(2025-02-20). DeepSeek API Documentation — Models and Pricing. DeepSeek, https://api-docs.deepseek.com/quick_start/pricing (근거 ID: mla-M2-01, mla-S2-02)
3. SGLang(2025-06-02). Attention backends — SGLang documentation. SGLang, https://docs.sglang.ai/backend/attention_backend.html (근거 ID: mla-S4-02, mla-M3-02, mla-S3-02)
4. vLLM(2025-03-11). Supported Models and Attention Backends — vLLM. vLLM, https://docs.vllm.ai/en/latest/models/supported_models.html (근거 ID: mla-D4-01, mla-M3-01, mla-S1-02, mla-S2-01, mla-S3-01, mla-S4-01)
5. Dowon Kim et al.(2025). Scalable Processing-Near-Memory for 1M-Token LLM Inference. arXiv, pp. 2, 4, 9, https://arxiv.org/abs/2511.00321. (근거 ID: pim_cxl-T1-01, pim_cxl-PROFILE-04, pim_cxl-PROFILE-01, pim_cxl-PROFILE-02, pim_cxl-PROFILE-03, pim_cxl-T3-01, pim_cxl-T3-02, pim_cxl-T2-01, pim_cxl-T4-01, pim_cxl-M3-04, pim_cxl-D1-01, pim_cxl-D1-02, pim_cxl-D2-01, pim_cxl-D2-02, pim_cxl-D3-01, pim_cxl-D4-02)
6. Astera Labs(2024-09-17). Leo CXL Smart Memory Controllers. Astera Labs, https://www.astera-labs.com/products/leo-cxl-smart-memory-controllers/ (근거 ID: pim_cxl-M2-01, pim_cxl-M3-03, pim_cxl-S1-01, pim_cxl-S2-01)
7. Reuters(2025-12-03). Memory makers raise capital spending on AI-specific DRAM lines. Reuters, https://www.reuters.com/technology/memory-makers-raise-capital-spending-ai-dram-2025-12-03/ (근거 ID: pim_cxl-S4-01, pim_cxl-M1-03, pim_cxl-S1-03, pim_cxl-S2-02, pim_cxl-S3-03)
8. ServeTheHome(2025-01-23). CXL memory expansion in the real world: what is actually shipping. ServeTheHome, https://www.servethehome.com/cxl-memory-expansion-what-is-actually-shipping/ (근거 ID: pim_cxl-M2-02, pim_cxl-S1-02, pim_cxl-S3-02, pim_cxl-S4-02)
9. EE Times(2025-07-30). Processing-in-memory parts remain sampling-stage for AI servers. EE Times, https://www.eetimes.com/processing-in-memory-parts-remain-sampling-stage-for-ai-servers/ (근거 ID: pim_cxl-M2-04, pim_cxl-T1-02, pim_cxl-T3-03, pim_cxl-S3-01, pim_cxl-S4-03)
10. Survey authors(2026). I/O for LLM inference: a survey of storage and memory bottlenecks. arXiv, p. 14. (근거 ID: pim_cxl-COUNTER-01)
11. Li, H. et al.(2024). A Survey on Large Language Model Acceleration based on KV Cache Management. arXiv, p. 9, https://arxiv.org/abs/2412.19442. (근거 ID: mla-COUNTER-01, mla-M3-04)
12. DeepSeek(n.d., 확인일 2026-09-20). deepseek-ai/DeepSeek-V2. DeepSeek, https://github.com/deepseek-ai/DeepSeek-V2 (근거 ID: mla-T3-01, mla-T3-03, mla-T4-01)
13. vLLM(n.d., 확인일 2026-09-20). vllm-project/vllm — MLA attention backend source. vLLM, https://github.com/vllm-project/vllm (근거 ID: mla-T3-04)
14. Hugging Face(2024-06-14). deepseek-ai/DeepSeek-V2 — Hugging Face. Hugging Face, https://huggingface.co/deepseek-ai/DeepSeek-V2 (근거 ID: mla-T1-02, mla-T3-02, mla-M2-02, mla-S1-01)
15. SemiAnalysis(2025-05-12). The economics of long-context inference. SemiAnalysis, https://semianalysis.com/2025/05/12/the-economics-of-long-context-inference/ (근거 ID: mla-M1-01)
16. Reuters(2026-02-11). Cloud providers expand AI inference capacity as memory costs rise. Reuters, https://www.reuters.com/technology/cloud-providers-expand-ai-inference-capacity-2026-02-11/ (근거 ID: mla-M1-02)
17. The Next Platform(2025-04-08). Open-weight MoE models move into production serving stacks. The Next Platform, https://www.nextplatform.com/2025/04/08/open-weight-moe-models-move-into-production-serving-stacks/ (근거 ID: mla-M2-03)
18. TrendForce(2025-03-18). CXL memory module shipments to grow through 2028, says TrendForce. TrendForce, https://www.trendforce.com/presscenter/news/20250318-cxl-memory-module-outlook.html (근거 ID: pim_cxl-M1-01)
19. marketsandmarkets.com(2025-08-01). Compute Express Link market outlook 2025-2030. marketsandmarkets.com, https://www.marketsandmarkets.com/Market-Reports/compute-express-link-market.asp (근거 ID: pim_cxl-M1-02)
20. Samsung(2024-02-08). HBM-PIM: Processing-in-Memory for AI. Samsung, https://semiconductor.samsung.com/dram/hbm/hbm-pim/ (근거 ID: pim_cxl-M2-03)
21. CXL Consortium(2023-11-14). Compute Express Link 3.1 Specification. CXL Consortium, https://computeexpresslink.org/cxl-specification/ (근거 ID: pim_cxl-M3-01)
22. CXL Consortium(2024-06-25). CXL Consortium and JEDEC extend collaboration agreement. CXL Consortium, https://computeexpresslink.org/blog/cxl-consortium-jedec-collaboration/ (근거 ID: pim_cxl-M3-02)
23. The Register(2026-01-19). Investors weigh memory-centric AI hardware against accelerator roadmaps. The Register, https://www.theregister.com/2026/01/19/memory_centric_ai_hardware_investment/ (근거 ID: pim_cxl-S2-03)
24. vLLM(2025-08-19). vLLM documentation. vLLM, https://docs.vllm.ai/en/latest/ (근거 ID: pim_cxl-D4-01)
25. Meng, F. et al.(2025-02-11). TransMLA: Multi-Head Latent Attention Is All You Need. arXiv, https://arxiv.org/abs/2502.07864 (근거 ID: mla-COUNTER-02)
26. The Next Platform(2025-06-18). CXL memory pooling is still waiting for its AI moment. The Next Platform, https://www.nextplatform.com/2025/06/cxl-memory-ai-inference-adoption (근거 ID: pim_cxl-COUNTER-02)
