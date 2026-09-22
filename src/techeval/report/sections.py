"""보고서 마크다운을 챕터·절 단위로 나누고 다시 합친다. lint와 재생성(문제 챕터만)이 공유한다."""

import re

from pydantic import BaseModel

HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$")
REQUIRED_CHAPTERS: tuple[str, ...] = (
    "SUMMARY",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "REFERENCE",
)
CH4_SUBSECTIONS: tuple[str, ...] = ("4.1", "4.2", "4.3", "4.4")


class Section(BaseModel):
    key: str  # "title", "SUMMARY", "1", "3.1", "4.2", "REFERENCE" ...
    heading: str  # 제목 줄 원문(개행 포함). title 머리말이면 빈 문자열
    body: str  # 다음 ##/### 제목 전까지의 원문

    @property
    def chapter(self) -> str:
        return self.key.split(".")[0]

    @property
    def text(self) -> str:
        return self.heading + self.body


def _key(level: int, title: str) -> str | None:
    if level == 1:
        return "title"
    upper = title.upper()
    for fixed in ("SUMMARY", "REFERENCE"):
        if upper.startswith(fixed):
            return fixed
    if level == 2:
        m = re.match(r"(\d+)\.", title)
    else:
        m = re.match(r"(\d+\.\d+)\b", title)
    return m.group(1) if m else None


def split_sections(report_md: str) -> list[Section]:
    """#/##/### 제목마다 절을 끊는다. 번호 없는 ### 이하 제목은 앞 절 본문에 포함된다."""
    sections: list[Section] = []
    current = Section(key="title", heading="", body="")
    for line in report_md.splitlines(keepends=True):
        m = HEADING_RE.match(line.rstrip("\n"))
        key = _key(len(m.group(1)), m.group(2)) if m else None
        if key is None:
            current.body += line
            continue
        if current.heading or current.body:
            sections.append(current)
        current = Section(
            key=key, heading=line if line.endswith("\n") else line + "\n", body=""
        )
    if current.heading or current.body:
        sections.append(current)
    return sections


def join_sections(sections: list[Section]) -> str:
    return "".join(s.text for s in sections)
