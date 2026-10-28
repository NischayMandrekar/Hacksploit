import importlib.util
from pathlib import Path


_SPEC = importlib.util.spec_from_file_location(
    "benchmark_jobs", Path(__file__).resolve().parents[1] / "scripts" / "benchmark_jobs.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_BENCHMARK = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BENCHMARK)
percentile = _BENCHMARK.percentile


def test_percentile_interpolates_between_observations():
    assert percentile([0.0, 10.0], 0.5) == 5.0
    assert percentile([0.0, 10.0], 0.95) == 9.5


def test_percentile_handles_empty_and_single_value_inputs():
    assert percentile([], 0.5) is None
    assert percentile([3.25], 0.99) == 3.25


def test_percentile_is_order_independent():
    assert percentile([9.0, 1.0, 5.0], 0.5) == 5.0
