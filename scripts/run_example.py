#!/usr/bin/env python3
"""Smoke test — verify all renamed modules load correctly and logging works.

Runs a single question through the full pipeline to confirm:
1. All verbose-named imports resolve
2. Logger writes .log files to disk
3. The ConstrainedMultiHopSearchAgent can be instantiated
4. A real question can be answered end-to-end
"""

import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))


def test_module_imports():
    """Phase 1: Verify all renamed modules can be imported."""
    print("=" * 60)
    print("  Phase 1: Module Imports")
    print("=" * 60)

    modules = {
        "constants": "src.agents.search_agent_shared_constants_and_stopwords",
        "query_gen": "src.agents.search_query_parsing_and_generation",
        "scoring": "src.agents.search_result_relevance_scoring_and_ranking",
        "answer": "src.agents.llm_answer_cleaning_and_candidate_extraction",
        "agent": "src.agents.constrained_multi_hop_search_agent",
        "search_abc": "src.search.abstract_search_client_interface",
        "mcp_transport": "src.search.model_context_protocol_transport_clients",
        "iqs": "src.search.alibaba_iqs_search_client",
        "brave": "src.search.brave_web_search_client",
        "hybrid": "src.search.language_aware_hybrid_search_dispatcher",
        "pipeline_prompts": "src.pipeline_llm_prompt_factory_functions",
        "pipeline": "src.question_answering_pipeline_builder",
        "eval_baseline": "scripts.evaluation_baseline_loader_and_regression_checker",
    }

    ok_count = 0
    for label, mod in modules.items():
        try:
            __import__(mod)
            print(f"  OK  {label:20s} <- {mod}")
            ok_count += 1
        except Exception as exc:
            print(f"  FAIL {label:20s} <- {mod}  ({exc})")

    print(f"\n  Result: {ok_count}/{len(modules)} modules imported\n")
    return ok_count == len(modules)


def test_backward_compat():
    """Phase 2: Verify old import paths still work via shims."""
    print("=" * 60)
    print("  Phase 2: Backward Compatibility")
    print("=" * 60)

    checks = [
        ("src.agents.search_constants", "CN_STOPWORDS"),
        ("src.agents.query_generation", "parse_question"),
        ("src.agents.evidence_scoring", "rank_and_filter"),
        ("src.agents.answer_processing", "process_llm_answer"),
        ("src.agents.constrained_search", "ConstrainedSearchAgent"),
        ("src.search.base", "SearchClient"),
        ("src.search.mcp_clients", "MCPHttpClient"),
        ("src.search.client", "IQSSearchClient"),
        ("src.pipeline_prompts", "make_knowledge_fn"),
        ("src.pipeline", "Pipeline"),
        ("scripts.eval_baseline", "load_best_baseline"),
    ]

    ok_count = 0
    for mod, name in checks:
        try:
            m = __import__(mod, fromlist=[name])
            obj = getattr(m, name)
            print(f"  OK  {mod}.{name}")
            ok_count += 1
        except Exception as exc:
            print(f"  FAIL {mod}.{name}  ({exc})")

    print(f"\n  Result: {ok_count}/{len(checks)} aliases work\n")
    return ok_count == len(checks)


def test_logging():
    """Phase 3: Verify logging writes a file to disk."""
    print("=" * 60)
    print("  Phase 3: Logging")
    print("=" * 60)

    from src.utils.logger_config import get_logger, MultiHopLogger

    log_dir = MultiHopLogger._log_dir
    print(f"  Log directory: {log_dir}")

    logger = get_logger("smoke_test", "smoke_test.log")
    logger.info("Smoke test started")
    logger.warning("This is a test warning")
    logger.info("Smoke test completed")

    log_file = log_dir / "smoke_test.log"
    if log_file.exists():
        size = log_file.stat().st_size
        lines = log_file.read_text(encoding="utf-8").strip().split("\n")
        print(f"  OK  Log file exists: {log_file}")
        print(f"      Size: {size} bytes, Lines: {len(lines)}")
        for line in lines[-3:]:
            print(f"      | {line}")
        print()
        return True
    else:
        print(f"  FAIL Log file NOT found: {log_file}\n")
        return False


def test_pipeline_end_to_end():
    """Phase 4: Run a single question through the real pipeline."""
    print("=" * 60)
    print("  Phase 4: End-to-End Pipeline")
    print("=" * 60)

    from src.question_answering_pipeline_builder import (
        build_question_answering_pipeline,
        answer_single_question,
    )

    try:
        print("  Building pipeline...")
        pipeline = build_question_answering_pipeline()
        print(f"  OK  Pipeline built: LLM={type(pipeline.llm).__name__}, Search={type(pipeline.search).__name__}")
    except Exception as exc:
        print(f"  FAIL Cannot build pipeline: {exc}\n")
        return False

    question = "Who wroteerta?"
    print(f"  Question: {question}")
    start = time.time()

    try:
        result = answer_single_question(pipeline, question, use_mcp=True)
        elapsed = time.time() - start
        answer = result.get("answer", "")
        steps = len(result.get("reasoning_steps", []))
        traces = len(result.get("mcp_results", []))
        print(f"  OK  Answer: {answer}")
        print(f"      Reasoning steps: {steps}, Search traces: {traces}")
        print(f"      Elapsed: {elapsed:.1f}s\n")
        return True
    except Exception as exc:
        print(f"  FAIL Pipeline error: {exc}\n")
        return False


def main():
    print("\n" + "=" * 60)
    print("  ChainReasoner Smoke Test")
    print("=" * 60 + "\n")

    results = {}
    results["imports"] = test_module_imports()
    results["compat"] = test_backward_compat()
    results["logging"] = test_logging()
    results["pipeline"] = test_pipeline_end_to_end()

    # Summary
    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    total_pass = sum(results.values())
    total = len(results)
    print(f"\n  Total: {total_pass}/{total} passed")

    if total_pass == total:
        print("  All tests passed!\n")
    else:
        print("  Some tests failed.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
