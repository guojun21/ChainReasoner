"""Backward-compatibility shim — moved to evaluation_baseline_loader_and_regression_checker.py."""
from scripts.evaluation_baseline_loader_and_regression_checker import (  # noqa: F401
    load_evaluation_results_from_json_or_jsonl_file as load_results_file,
    scan_output_directory_for_highest_scoring_baseline as load_best_baseline,
    detect_answer_regressions_against_baseline as check_regression,
)
