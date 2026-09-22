# TechEvalAgent 논문 코퍼스

PDF 원문은 로컬 인덱싱 전용이며 저장소에 커밋하지 않는다. 아래 스크립트로 공개
원문 PDF를 내려받을 수 있다. 기존 PDF는 건너뛰며, 재다운로드가 필요할 때만 `--force`를 사용한다.

```bash
uv run python scripts/download_papers.py
```

| doc_id | 파일명 | 원문 |
|---|---|---|
| `deepseek_v2` | `Deepseek_v2.pdf` | DeepSeek-AI (2024-06-19). [arXiv:2405.04434](https://arxiv.org/abs/2405.04434) |
| `pim_cxl_1m` | `PIM:CXL_KVcache.pdf` | Kim et al. (2025-10-31). [arXiv:2511.00321](https://arxiv.org/abs/2511.00321) |
| `io_survey` | `LLM_storage_HW_survey.pdf` | Chowdhury (2026-07-24). [DOI:10.1007/s10462-026-11651-1](https://doi.org/10.1007/s10462-026-11651-1), [공개 preprint PDF](https://www.researchsquare.com/article/rs-9036613/latest.pdf) |
| `kv_survey` | `KV_manage_survey.pdf` | Li et al. (2025-07-30). [arXiv:2412.19442](https://arxiv.org/abs/2412.19442) |

실행 예시:

```bash
uv run python scripts/ingest.py --papers-dir data/papers --chroma-dir data/chroma --rebuild
```
