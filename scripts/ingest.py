"""논문 PDF 4편을 Chroma와 BM25 인덱스로 만드는 CLI."""

import argparse
import logging
from pathlib import Path

from techeval.retrieval.ingest import build_index

DOCUMENTS = (
    (
        "deepseek_v2",
        "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model",
        "Deepseek_v2.pdf",
    ),
    (
        "pim_cxl_1m",
        "Scalable Processing-Near-Memory for 1M-Token LLM Inference",
        "PIM:CXL_KVcache.pdf",
    ),
    (
        "io_survey",
        "I/O for LLM Inference: A Survey of Storage and Memory Bottlenecks",
        "LLM_storage_HW_survey.pdf",
    ),
    (
        "kv_survey",
        "A Survey on Large Language Model Acceleration based on KV Cache Management",
        "KV_manage_survey.pdf",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--papers-dir", type=Path, required=True, help="PDF 4편이 있는 디렉터리"
    )
    parser.add_argument(
        "--chroma-dir", type=Path, required=True, help="Chroma와 BM25를 저장할 디렉터리"
    )
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    args = parse_args()
    documents = []
    for doc_id, title, filename in DOCUMENTS:
        path = args.papers_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing paper for {doc_id}: {path}")
        documents.append((doc_id, title, path))
    counts = build_index(
        documents, chroma_dir=args.chroma_dir, embedding_model=args.embedding_model
    )
    for doc_id, count in counts.items():
        print(f"{doc_id}: {count} chunks")


if __name__ == "__main__":
    main()
