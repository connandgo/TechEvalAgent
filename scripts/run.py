"""전체 파이프라인 실행 CLI (E).

uv run python scripts/run.py --stub                 # 전부 스텁으로 그래프 흐름만 검증
uv run python scripts/run.py                        # 실제 실행 -> outputs/report.md, outputs/report.pdf
옵션: --no-cache (웹 캐시 무시) --out outputs/ --dump-state --skip-pdf --allow-stub-fallback --log-level INFO
"""

import argparse
import logging
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from techeval.config import build_deps, load_settings  # noqa: E402
from techeval.graph import GraphConfig, build_graph, invoke_config, load_agents, state_to_json  # noqa: E402

logger = logging.getLogger("techeval.run")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TechEvalAgent 파이프라인 실행")
    p.add_argument("--stub", action="store_true", help="검색기·웹검색·LLM·에이전트를 전부 스텁으로 실행")
    p.add_argument("--no-cache", action="store_true", help="웹 검색 캐시(outputs/web_cache)를 무시")
    p.add_argument("--out", default=None, help="출력 디렉토리 (기본: OUTPUT_DIR 또는 outputs)")
    p.add_argument("--dump-state", action="store_true", help="노드별 State를 <out>/state/<step>_<node>.json 으로 저장")
    p.add_argument("--skip-pdf", action="store_true", help="PDF 렌더링 생략 (report.md만 저장)")
    p.add_argument(
        "--allow-stub-fallback", action="store_true", help="아직 없는 역할의 에이전트를 스텁으로 대체해 실행"
    )
    p.add_argument("--log-level", default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = load_settings()
    if args.log_level:
        logging.getLogger().setLevel(args.log_level.upper())
    out_dir = Path(args.out or settings.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    deps = build_deps(stub=args.stub, settings=settings, use_cache=not args.no_cache)
    agents, stubbed = load_agents(stub=args.stub, allow_stub_fallback=args.allow_stub_fallback)
    cfg = GraphConfig(output_dir=str(out_dir), skip_pdf=args.skip_pdf, stub=args.stub)
    graph = build_graph(deps, agents, cfg)

    node_counts: Counter[str] = Counter()
    step = 0
    final_state: dict = {}
    t0 = time.perf_counter()
    for mode, chunk in graph.stream({}, config=invoke_config(cfg), stream_mode=["updates", "values"]):
        if mode == "updates":
            for node in chunk:
                step += 1
                node_counts[node] += 1
                if args.dump_state:
                    state_dir = out_dir / "state"
                    state_dir.mkdir(exist_ok=True)
                    (state_dir / f"{step:03d}_{node}.json").write_text(state_to_json(final_state), encoding="utf-8")
        else:
            final_state = chunk
    elapsed = time.perf_counter() - t0

    llm_calls = getattr(deps.llm, "call_count", None)
    judge_calls = getattr(deps.judge_llm, "call_count", None)
    retry = final_state.get("retry_counts", {})
    jr = final_state.get("judge_result")

    print("\n=== 실행 요약 ===")
    print(f"소요 시간: {elapsed:.1f}s   노드 실행: {sum(node_counts.values())}회")
    for node, n in sorted(node_counts.items()):
        print(f"  {node:<24} {n}")
    print("재시도:", {k: v for k, v in retry.items() if v})
    fmt = lambda n: "n/a" if n is None else str(n)  # noqa: E731
    print(f"LLM 호출: {fmt(llm_calls)} (judge {fmt(judge_calls)})")
    if stubbed:
        print("스텁 대체:", ", ".join(stubbed))
    if jr is not None:
        print(f"judge: passed={jr.passed} scores={jr.scores} missing={jr.missing_required}")
    print(
        f"산출물: {out_dir / 'report.md'}",
        f"/ {final_state.get('report_pdf_path')}" if final_state.get("report_pdf_path") else "",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
