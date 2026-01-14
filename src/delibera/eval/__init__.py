"""Evaluation framework for Delibera.

This module provides systematic evaluation of Delibera runs against
golden expectations and metric constraints.
"""

from delibera.eval.compare import compare_metrics
from delibera.eval.loader import EvalSuiteLoadError, load_eval_suite
from delibera.eval.metrics import extract_metrics
from delibera.eval.models import EvalCase, EvalResult, EvalSuite
from delibera.eval.runner import run_eval_case, run_eval_suite

__all__ = [
    "EvalSuite",
    "EvalCase",
    "EvalResult",
    "load_eval_suite",
    "EvalSuiteLoadError",
    "extract_metrics",
    "run_eval_case",
    "run_eval_suite",
    "compare_metrics",
]
