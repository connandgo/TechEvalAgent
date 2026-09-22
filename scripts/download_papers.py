"""평가 코퍼스 PDF 4편을 ``data/papers/``에 내려받는 CLI.

기본 실행:
    uv run python scripts/download_papers.py

기존 파일은 건너뛴다. 원본을 다시 받으려면 ``--force``를 사용한다.
"""

import argparse
import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_PAPERS_DIR = Path("data/papers")
USER_AGENT = "TechEvalAgent/1.0 (academic-paper-downloader)"
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Paper:
    """로컬 파일명과 공개 PDF 원문 URL."""

    doc_id: str
    filename: str
    url: str


PAPERS: tuple[Paper, ...] = (
    Paper("deepseek_v2", "Deepseek_v2.pdf", "https://arxiv.org/pdf/2405.04434"),
    Paper("pim_cxl_1m", "PIM:CXL_KVcache.pdf", "https://arxiv.org/pdf/2511.00321"),
    Paper(
        "io_survey",
        "LLM_storage_HW_survey.pdf",
        "https://www.researchsquare.com/article/rs-9036613/latest.pdf",
    ),
    Paper("kv_survey", "KV_manage_survey.pdf", "https://arxiv.org/pdf/2412.19442"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--papers-dir",
        type=Path,
        default=DEFAULT_PAPERS_DIR,
        help="PDF를 저장할 디렉터리 (기본: data/papers)",
    )
    parser.add_argument("--force", action="store_true", help="기존 PDF도 다시 다운로드")
    parser.add_argument(
        "--dry-run", action="store_true", help="다운로드 없이 대상 목록만 출력"
    )
    parser.add_argument(
        "--timeout", type=float, default=60.0, help="논문 1편당 연결 제한 시간(초)"
    )
    return parser.parse_args()


def _is_pdf(path: Path) -> bool:
    """PDF 매직 바이트로 다운로드 결과를 최소 검증한다."""
    if not path.is_file():
        return False
    with path.open("rb") as file:
        return file.read(5) == b"%PDF-"


def download_paper(
    paper: Paper, papers_dir: Path, *, force: bool, timeout: float
) -> str:
    """논문 한 편을 원자적으로 저장하고 ``downloaded`` 또는 ``skipped``를 반환한다."""
    destination = papers_dir / paper.filename
    if destination.exists() and not force:
        if _is_pdf(destination):
            return "skipped"
        raise ValueError(
            f"기존 파일이 PDF가 아닙니다. 확인 후 --force로 다시 받으세요: {destination}"
        )

    request = Request(paper.url, headers={"User-Agent": USER_AGENT})
    temp_path: Path | None = None
    try:
        with (
            urlopen(request, timeout=timeout) as response,
            tempfile.NamedTemporaryFile(dir=papers_dir, delete=False) as temp_file,
        ):
            temp_path = Path(temp_file.name)
            shutil.copyfileobj(response, temp_file)
        if not _is_pdf(temp_path):
            raise ValueError(
                "PDF 응답이 아닙니다. 공개 원문 접근 권한 또는 URL을 확인하세요."
            )
        temp_path.replace(destination)
        return "downloaded"
    except (HTTPError, URLError, OSError, TimeoutError, ValueError) as exc:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"{paper.doc_id} 다운로드 실패 ({paper.url}): {exc}"
        ) from exc


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    if args.timeout <= 0:
        raise ValueError("--timeout은 0보다 커야 합니다.")

    if args.dry_run:
        for paper in PAPERS:
            print(f"{paper.doc_id}: {args.papers_dir / paper.filename} <- {paper.url}")
        return 0

    args.papers_dir.mkdir(parents=True, exist_ok=True)
    failed = 0
    for paper in PAPERS:
        try:
            status = download_paper(
                paper, args.papers_dir, force=args.force, timeout=args.timeout
            )
            print(f"{paper.doc_id}: {status} ({args.papers_dir / paper.filename})")
        except RuntimeError as exc:
            failed += 1
            logger.error("%s", exc)

    if failed:
        logger.error(
            "%d편 다운로드 실패. 오류를 해결한 뒤 같은 명령을 다시 실행하세요.", failed
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
