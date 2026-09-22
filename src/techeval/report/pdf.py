"""마크다운 보고서 → PDF. 기본 경로는 markdown + weasyprint, weasyprint를 못 쓰면 pandoc으로 대체한다."""

import logging
import shutil
import subprocess
from pathlib import Path

import markdown

logger = logging.getLogger(__name__)

FONT_DIR = Path(__file__).resolve().parents[3] / "assets" / "fonts"
FONT_EXTS = (".ttf", ".otf", ".ttc", ".woff", ".woff2")
# assets/fonts/에 폰트가 없을 때 fontconfig로 찾을 시스템 한글 폰트 (앞에서부터 우선)
SYSTEM_KO_FONTS = (
    "Noto Sans KR",
    "Noto Sans CJK KR",
    "Apple SD Gothic Neo",
    "NanumGothic",
    "Malgun Gothic",
)

BASE_CSS = """
@page { size: A4; margin: 20mm 18mm 20mm 18mm;
        @bottom-center { content: counter(page) " / " counter(pages); font-size: 8pt; color: #666; } }
html { font-family: %(font_stack)s; font-size: 10pt; line-height: 1.6; color: #111; }
h1 { font-size: 18pt; margin: 0 0 6mm; }
h2 { font-size: 14pt; margin: 8mm 0 3mm; border-bottom: 1px solid #999; padding-bottom: 1mm;
     break-after: avoid; }
h3 { font-size: 11.5pt; margin: 5mm 0 2mm; break-after: avoid; }
p, li { orphans: 2; widows: 2; }
table { border-collapse: collapse; width: 100%%; margin: 3mm 0; font-size: 8.5pt;
        table-layout: auto; }
thead { display: table-header-group; }
tr { break-inside: avoid; page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 1.2mm 1.6mm; vertical-align: top;
         overflow-wrap: anywhere; }
th { background: #f0f0f0; }
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
    """report_md를 out_path에 PDF로 저장하고 그 경로를 반환한다."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        _render_weasyprint(build_html(report_md), out)
    except (ImportError, OSError) as exc:
        logger.warning("weasyprint 사용 불가(%s) — pandoc으로 대체", exc)
        _render_pandoc(report_md, out)
    logger.info("PDF 생성: %s (%d bytes)", out, out.stat().st_size)
    return str(out)
