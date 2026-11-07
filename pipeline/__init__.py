"""Single offline entry point that chains the repository's pipeline stages."""

from .unified import run_case, run_all_cases

__all__ = ["run_case", "run_all_cases"]
