"""Parallelised SHAP value computation using Ray Tasks.

Example::

    from rgan.explain import compute_shap_values, plot_summary
    shap_values = compute_shap_values(model, X_test)
    fig = plot_summary(shap_values, X_test)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator

from rgan.utils.dependencies import lazy_import, require_explain

__all__ = ["compute_shap_values", "plot_summary"]


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_shap_values(
    model: BaseEstimator,
    X: NDArray[np.floating[Any]],
    *,
    n_partitions: int = 4,
    ray_address: str | None = None,
    feature_names: list[str] | None = None,
) -> NDArray[np.floating[Any]]:
    """Compute SHAP values for *model* on data *X* using parallel Ray tasks.

    Parameters
    ----------
    model : BaseEstimator
        A fitted scikit-learn-compatible estimator.
    X : np.ndarray
        Feature matrix of shape ``(n_samples, n_features)``.
    n_partitions : int
        Number of data chunks to distribute across Ray workers.
    ray_address : str | None
        Existing Ray cluster address (``None`` = local).
    feature_names : list[str] | None
        Optional feature names forwarded to the SHAP explainer.

    Returns
    -------
    np.ndarray
        SHAP values with shape ``(n_samples, n_features)``.
    """
    require_explain()

    ray = lazy_import("ray")
    shap = lazy_import("shap")

    if not ray.is_initialized():
        ray.init(address=ray_address, ignore_reinit_error=True)

    # Put model + background into the object store once.
    model_ref = ray.put(model)
    # Use a small subsample as background for the explainer.
    bg_idx = np.random.choice(len(X), size=min(100, len(X)), replace=False)
    background_ref = ray.put(X[bg_idx])

    partitions = np.array_split(X, n_partitions)

    @ray.remote  # type: ignore[misc]
    def _shap_partition(
        chunk: NDArray[np.floating[Any]],
        m_ref: Any,
        bg_ref: Any,
    ) -> NDArray[np.floating[Any]]:
        """Compute SHAP values on a single partition."""
        import shap as _shap  # noqa: F811

        m = ray.get(m_ref) if not isinstance(m_ref, BaseEstimator) else m_ref
        bg = ray.get(bg_ref) if not isinstance(bg_ref, np.ndarray) else bg_ref

        explainer = _shap.Explainer(m, bg)
        sv = explainer(chunk)
        return np.asarray(sv.values, dtype=np.float64)

    futures = [
        _shap_partition.remote(part, model_ref, background_ref) for part in partitions
    ]
    results: list[NDArray[np.floating[Any]]] = ray.get(futures)

    return np.vstack(results)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def plot_summary(
    shap_values: NDArray[np.floating[Any]],
    X: NDArray[np.floating[Any]],
    *,
    feature_names: list[str] | None = None,
    max_display: int = 20,
) -> Any:
    """Create a SHAP beeswarm summary plot and return the ``matplotlib`` Figure.

    Parameters
    ----------
    shap_values : np.ndarray
        Output of :func:`compute_shap_values`.
    X : np.ndarray
        Original feature matrix (used for colour mapping).
    feature_names : list[str] | None
        Column names.
    max_display : int
        Maximum features to display.

    Returns
    -------
    matplotlib.figure.Figure
    """
    require_explain()

    shap = lazy_import("shap")
    plt = lazy_import("matplotlib.pyplot")

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    fig.tight_layout()
    return fig
