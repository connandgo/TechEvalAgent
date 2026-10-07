"""마크다운 보고서 → PDF. 기본 경로는 markdown + weasyprint, weasyprint를 못 쓰면 pandoc으로 대체한다."""

import logging
import re
import shutil
import subprocess
from pathlib import Path

import markdown

from techeval.report.citation import to_numbered_citations

logger = logging.getLogger(__name__)

FONT_DIR = Path(__file__).resolve().parents[3] / "assets" / "fonts"
FONT_EXTS = (".ttf", ".otf", ".ttc", ".woff", ".woff2")
# assets/fonts/에 폰트가 없을 때 fontconfig로 찾을 시스템 한글 폰트 (앞에서부터 우선)
SYSTEM_KO_FONTS = ("Noto Sans KR", "Noto Sans CJK KR", "Apple SD Gothic Neo", "NanumGothic", "Malgun Gothic")

BASE_CSS = """
@page { size: A4; margin: 14mm 14mm 15mm 14mm;  /* Agent 실습: 보고서 10장 이내 */
        @bottom-center { content: counter(page) " / " counter(pages); font-size: 8pt; color: #666; } }
html { font-family: %(font_stack)s; font-size: 9.5pt; line-height: 1.45; color: #111; }
h1 { font-size: 18pt; margin: 0 0 6mm; }
h2 { font-size: 13pt; margin: 5mm 0 2mm; border-bottom: 1px solid #999; padding-bottom: 1mm;
     break-after: avoid; }
h3 { font-size: 11pt; margin: 3.5mm 0 1.5mm; break-after: avoid; }
p, li { orphans: 2; widows: 2; }
table { border-collapse: collapse; width: 100%%; margin: 2mm 0; font-size: 8pt;
        table-layout: auto; }
thead { display: table-header-group; }
tr { break-inside: avoid; page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 0.9mm 1.4mm; vertical-align: top;
         word-break: keep-all; overflow-wrap: break-word; }
th { background: #f0f0f0; white-space: nowrap; }
td:first-child { white-space: nowrap; }  /* 기준명·지표명·기술명: 한글이 글자 단위로 끊기지 않게 */
code, pre { font-family: "Menlo", "Consolas", monospace; font-size: 8.5pt; }
pre { white-space: pre-wrap; background: #f7f7f7; padding: 2mm; }
"""


def _local_font_faces() -> tuple[str, list[str]]:
    """assets/fonts/의 폰트를 @font-face로 임베드한다. (css, family 이름 목록)"""
    if not FONT_DIR.is_dir():
        return "", []
    faces, families = [], []
    for path in sorted(FONT_DIR.iterdir()):
        if path.suffix.lower() not in FONT_EXTS:
            continue
        family = f"EmbeddedKo-{path.stem}"
        weight = "bold" if "bold" in path.stem.lower() else "normal"
        faces.append(f'@font-face {{ font-family: "{family}"; src: url("{path.as_uri()}"); font-weight: {weight}; }}')
        families.append(family)
    return "\n".join(faces), families


def build_html(report_md: str) -> str:
    body = markdown.markdown(report_md, extensions=["tables", "fenced_code", "sane_lists"])
    faces, local = _local_font_faces()
    font_stack = ", ".join(f'"{f}"' for f in (*local, *SYSTEM_KO_FONTS)) + ", sans-serif"
    css = faces + BASE_CSS % {"font_stack": font_stack}
    return (
        f'<!DOCTYPE html>\n<html lang="ko"><head><meta charset="utf-8"><style>{css}</style></head>'
        f"<body>{body}</body></html>"
    )


def _render_weasyprint(html: str, out: Path) -> None:
    from weasyprint import HTML  # 시스템 pango가 없으면 OSError

    HTML(string=html, base_url=str(Path.cwd())).write_pdf(str(out))


def _render_pandoc(report_md: str, out: Path) -> None:
    if shutil.which("pandoc") is None:
        raise RuntimeError("weasyprint와 pandoc 모두 사용할 수 없어 PDF를 만들 수 없다")
    subprocess.run(
        [
            "pandoc",
            "-f",
            "gfm",
            "-o",
            str(out),
            "--pdf-engine=xelatex",
            "-V",
            f"mainfont={SYSTEM_KO_FONTS[2]}",
        ],
        input=report_md.encode("utf-8"),
        check=True,
    )


def render_pdf(report_md: str, out_path: str) -> str:
    """report_md를 out_path에 PDF로 저장하고 그 경로를 반환한다.

    PDF는 독자용 최종본이다: 검수용 `[E: evidence_id]`를 REFERENCE 번호 인용 `[1, 3]`으로 바꿔 렌더링하고,
    같은 내용의 마크다운을 PDF 옆에 `<이름>_final.md`로 함께 저장한다(예: report.pdf → report_final.md).
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    final_md = to_numbered_citations(report_md)
    out.with_name(f"{out.stem}_final.md").write_text(final_md, encoding="utf-8")
    try:
        # `[1, *]`·`[*]`의 별표가 마크다운 기울임 기호로 먹혀 `[1, ]`로 보이지 않도록 PDF 변환 직전에만 이스케이프
        _render_weasyprint(build_html(re.sub(r"(?<=[\[ ])\*(?=\])", r"\\*", final_md)), out)
    except (ImportError, OSError) as exc:
        logger.warning("weasyprint 사용 불가(%s) — pandoc으로 대체", exc)
        _render_pandoc(final_md, out)
    logger.info("PDF 생성: %s (%d bytes)", out, out.stat().st_size)
    return str(out)
