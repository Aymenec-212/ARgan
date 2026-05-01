"""Tests for Tier 3 — rgan.explain.

Run with:  uv run --extra explain pytest tests/test_explain.py -m explain
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from rgan.utils.dependencies import require_explain


def _explain_available() -> bool:
    try:
        require_explain()
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.explain


# ------------------------------------------------------------------
# Import guard
# ------------------------------------------------------------------


class TestImportGuard:
    def test_require_explain_does_not_crash_when_available(self) -> None:
        if not _explain_available():
            pytest.skip("explain extras not installed")
        require_explain()


# ------------------------------------------------------------------
# Functional test (requires extras)
# ------------------------------------------------------------------


@pytest.mark.skipif(not _explain_available(), reason="explain extras not installed")
class TestComputeShap:
    @pytest.mark.slow()
    def test_shap_values_shape(self) -> None:
        from rgan.explain import compute_shap_values

        X, y = make_classification(
            n_samples=80, n_features=5, random_state=42
        )
        X = X.astype(np.float32)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        sv = compute_shap_values(model, X, n_partitions=2)
        # Shape must match input
        assert sv.shape[0] == X.shape[0]
        assert sv.shape[-1] == X.shape[1]
