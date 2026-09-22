from pathlib import Path

from weasyprint import HTML

from techeval.report.pdf import build_html, render_pdf
from tests.report.d_fixtures import load


def test_render_fixture_report(tmp_path):
    report_md = load("report_md.md")
    out = render_pdf(report_md, str(tmp_path / "sub" / "report.pdf"))
    path = Path(out)
    assert path.exists() and path.stat().st_size > 0
    assert path.read_bytes().startswith(b"%PDF")
    pages = HTML(string=build_html(report_md)).render().pages
    assert len(pages) >= 5
