# 역할 A — RAG / Retrieval (담당: 이지수)

먼저 읽기: `AGENTS.md` → `docs/CONTRACTS.md` §3, §9 → 이 문서.
당신은 다른 모든 에이전트가 공통으로 쓰는 **검색 기반**을 만든다. 하류(B, C, E)는 당신의 `BaseRetriever` 인터페이스와 `RetrievedChunk` 형식만 알고 있으며, 그 뒤의 구현은 자유다.

## 1. 목표

- 논문 PDF 4편을 절·표 단위로 구조적 청킹하여 BGE-M3로 임베딩하고 Chroma에 저장한다.
- Dense(Chroma) + BM25 결과를 RRF로 결합하는 `VectorRetriever.search()`를 제공한다.
- 한국어/한·영 혼용 질의로 영문 논문을 검색할 수 있어야 한다(교차 언어).
- MLA, PIM, PNM, CXL, TTFT 같은 약어의 정확 일치를 BM25가 보완한다.
- 검색 결과 형식(`RetrievedChunk`)을 통일하고, `to_evidence()`로 `Evidence` 변환을 제공한다.

## 2. 소유 파일

```
src/techeval/retrieval/
├── __init__.py          # VectorRetriever, StubRetriever, RetrievedChunk, BaseRetriever export
├── ingest.py            # parse_pdf(), chunk_document(), build_index()
├── embedder.py          # BGE-M3 래퍼 (embed_documents / embed_query, 배치, 캐시)
├── store.py             # Chroma 컬렉션 생성·업서트·질의
├── bm25.py              # BM25 인덱스 빌드·질의·직렬화 (rank-bm25)
├── retriever.py         # RetrievedChunk, BaseRetriever, VectorRetriever, rrf_fuse()
└── stub.py              # StubRetriever
scripts/ingest.py        # CLI
data/papers/README.md    # 4편의 출처(제목·저자·URL·doc_id·파일명) — PDF 자체는 커밋하지 않음
tests/retrieval/         # 단위 테스트
tests/fixtures/chunks.json
```

## 3. 상류 / 하류

- 상류: 없음. (E가 `config.py`에서 `CHROMA_DIR`, `PAPERS_DIR`, `EMBEDDING_MODEL`을 제공하지만, 없어도 인자로 받으면 됨)
- 하류: B(`run_tech_research`, `run_domain_eval`), C(`run_market_eval` 교차 확인), E(`control/*` evidence 검증, `counter_evidence`)

## 4. 구현 순서 (권장)

1. **`RetrievedChunk`, `BaseRetriever`, `StubRetriever`, `tests/fixtures/chunks.json`을 가장 먼저 커밋한다.** 하류가 이것으로 개발을 시작한다.
   - 픽스처: 4편 각각 ≥5개 청크, 표 청크(`chunk_type="table"`) ≥2개, 실제 논문 문장으로 작성(MLA 93.3% KV 감소, PIM/CXL 실험 조건, 서베이의 KV 관리 기법 분류 등). `chunk_id` 규칙 `{doc_id}:{page:03d}:{seq:02d}` 준수.
2. `ingest.py`: `pymupdf`로 페이지별 텍스트 + 블록 추출. 절 제목 감지(숫자 패턴 `^\d+(\.\d+)*\s+[A-Z]`), 표는 캡션 `Table N` 기준으로 별도 청크. 청크 1,000~1,500 토큰(BGE-M3 토크나이저 기준), 절 경계 우선, 초과 시 오버랩 100~150 토큰.
3. `embedder.py`: `FlagEmbedding.BGEM3FlagModel` 또는 `sentence-transformers`. dense 벡터만 필수(sparse/colbert는 선택). 질의·문서 동일 모델. 임베딩 결과는 Chroma에 저장하고 재계산하지 않는다.
4. `store.py`: Chroma persistent client, 컬렉션 `papers`, metadata에 `doc_id, doc_title, page, section, chunk_type, seq` 저장. `doc_ids` 필터는 Chroma `where`로 처리.
5. `bm25.py`: 토크나이즈는 소문자화 + 영숫자/하이픈 토큰 유지(약어 보존). 인덱스는 `data/chroma/bm25.pkl`로 직렬화.
6. `retriever.py`: `rrf_fuse(dense_ranked, bm25_ranked, k=60)`. `mode` 인자로 dense/bm25 단독도 지원(디버깅용). `get_chunk(chunk_id)`는 Chroma에서 id로 조회.
7. `scripts/ingest.py`: `--papers-dir --chroma-dir --rebuild`. 완료 시 doc별 청크 수를 출력.

## 5. 세부 요구사항

- `RetrievedChunk.to_evidence(evidence_id, quote, unit)`: `quote`가 `text`의 부분 문자열(공백 정규화 후)이 아니면 `ValueError`. `locator = f"{doc_id} p.{page}" + (f" §{section}" if section else "")`. `source_type="paper"`, `doc_id/page/section/chunk_id` 채움. `title`은 doc_title, `authors/published_date`는 `data/papers/README.md` 기반 메타 테이블(`DOC_META` 딕셔너리)에서 채운다.
- 검색 결과는 `score` 내림차순, 동일 청크 중복 없음.
- 모델 로딩은 `VectorRetriever.__init__`에서 lazy하게. import 시점에 로딩 금지.
- 로그: 검색어, mode, doc_ids, 반환 청크 id 목록을 `DEBUG`로.
- 성능 목표: 검색 1회 < 1초 (모델 로딩 제외, M1/M2 Mac 기준).

## 6. 테스트 (`tests/retrieval/`)

- 단위(스텁, 모델 불필요): `rrf_fuse` 순위 결합, 청킹 경계·토큰 길이 범위, 표 청크 분리, `to_evidence` 부분 문자열 검사, `StubRetriever` 필터.
- 통합(`@pytest.mark.integration`, 실제 인덱스 필요):
  - 한국어 질의 "MLA의 KV cache 감소율" → `deepseek_v2` 청크가 top-3에 포함
  - "CXL 메모리로 KV cache 확장 실험 하드웨어" → `pim_cxl_1m` 포함
  - "PNM" 정확 약어 질의 → bm25 모드에서 해당 약어 포함 청크가 1위
  - `doc_ids=["kv_survey"]` 필터 시 다른 doc 미포함
  - 서베이 2편은 분량이 크므로 청크 수·인덱싱 시간을 로그로 확인(수백 청크 예상)

## 7. 완료 기준 (DoD)

- [ ] `StubRetriever` + `chunks.json` 커밋 (1순위)
- [ ] `uv run python scripts/ingest.py` 로 4편 인덱싱 성공, 청크 수 로그
- [ ] `VectorRetriever.search()` hybrid/dense/bm25 동작, `get_chunk()` 동작
- [ ] 교차 언어 통합 테스트 통과
- [ ] `data/papers/README.md`에 4편 메타(제목·저자·발행·URL·doc_id·파일명) 기재
- [ ] `tests/retrieval/` 단위 테스트 통과, ruff 통과

## 8. 하지 말 것

- 청크에 LLM 요약을 섞지 않는다(원문 보존). 근거 인용은 원문이어야 한다.
- 논문 전체를 하나의 벡터로 만들지 않는다.
- `RetrievedChunk` 필드 이름을 바꾸지 않는다(CONTRACT CHANGE 필요).
- 논문 PDF를 커밋하지 않는다.

## 9. 현재 구현·전달 기준

- 현재 로컬 논문 파일은 `paper/`에 두고, 인덱싱 CLI에는 항상 경로를 명시한다.
  ```bash
  uv run python scripts/ingest.py --papers-dir paper --chroma-dir data/chroma
  ```
- `data/chroma/`와 논문 PDF는 로컬 산출물이며 커밋하지 않는다. 팀원은 각자 같은 명령으로 인덱스를 생성한다.
- `tests/fixtures/chunks.json`은 네 논문 각각 5개 청크(표 청크 포함)를 제공한다. 각 `chunk_id`와 fixture `text`는 실제 PDF를 다시 파싱한 결과에 존재하는지 검사한다.
- B/C/E는 `BaseRetriever.search()`와 `get_chunk()`만 사용한다. 논문 Evidence는 `RetrievedChunk.to_evidence()`로 만들고, E의 검사 노드는 `chunk_id`와 quote substring을 대조한다.
