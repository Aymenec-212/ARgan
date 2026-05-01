"""Tests for Tier 2 — rgan.automl.

Run with:  uv run --extra automl pytest tests/test_automl.py -m automl
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification

from rgan.utils.dependencies import require_automl


def _automl_available() -> bool:
    try:
        require_automl()
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.automl


# ------------------------------------------------------------------
# Import guard
# ------------------------------------------------------------------


class TestImportGuard:
    def test_require_automl_does_not_crash_when_available(self) -> None:
        if not _automl_available():
            pytest.skip("automl extras not installed")
        require_automl()  # should not raise


# ------------------------------------------------------------------
# Functional test (requires extras)
# ------------------------------------------------------------------


@pytest.mark.skipif(not _automl_available(), reason="automl extras not installed")
class TestRunSearch:
    @pytest.mark.slow()
    def test_returns_fitted_model(self) -> None:
        from rgan.automl import run_search

        X, y = make_classification(
            n_samples=100,
            n_features=5,
            weights=[0.9, 0.1],
            random_state=42,
        )
        X = X.astype(np.float32)

        model, results = run_search(
            X,
            y,
            num_samples=2,
            cv=2,
            gan_space={
                "latent_dim": [16],
                "epochs": [3],
                "lambda_sim": [0.1],
                "lr_g": [1e-3],
            },
            lgb_space={
                "n_estimators": [10],
                "max_depth": [3],
                "learning_rate": [0.1],
                "num_leaves": [7],
            },
            random_state=42,
        )

        # Model should be fitted and predict
        preds = model.predict(X)
        assert preds.shape == (len(X),)
