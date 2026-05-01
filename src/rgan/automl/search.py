"""Distributed hyperparameter search via Ray Tune.

Trains a downstream classifier (default: LightGBM) on oversampled data,
using Ray Tune to search over both GAN and classifier hyper-parameters.

Example::

    from rgan.automl import run_search
    best_model, results = run_search(X_train, y_train)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score

from rgan.resample import RGANSampler
from rgan.utils.dependencies import lazy_import, require_automl

__all__ = ["run_search"]


# ---------------------------------------------------------------------------
# Default search space
# ---------------------------------------------------------------------------

_DEFAULT_GAN_SPACE: dict[str, Any] = {
    "latent_dim": [32, 64, 128],
    "epochs": [200, 300, 500],
    "lambda_sim": [0.0, 0.05, 0.1, 0.2],
    "lr_g": [1e-4, 5e-4, 1e-3],
}

_DEFAULT_LGB_SPACE: dict[str, Any] = {
    "n_estimators": [100, 200, 500],
    "max_depth": [3, 5, 7, -1],
    "learning_rate": [0.01, 0.05, 0.1],
    "num_leaves": [15, 31, 63],
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_search(
    X: NDArray[np.floating[Any]],
    y: NDArray[np.integer[Any]],
    *,
    gan_space: dict[str, Any] | None = None,
    lgb_space: dict[str, Any] | None = None,
    num_samples: int = 10,
    cv: int = 5,
    scoring: str = "f1_macro",
    random_state: int | None = None,
    ray_address: str | None = None,
    resources_per_trial: dict[str, float] | None = None,
) -> tuple[BaseEstimator, Any]:
    """Run a distributed hyper-parameter search and return the best model.

    Parameters
    ----------
    X, y : array-like
        Training data (imbalanced).
    gan_space / lgb_space : dict | None
        Override default search spaces.
    num_samples : int
        Number of Tune trials.
    cv : int
        Cross-validation folds for evaluation.
    scoring : str
        Scikit-learn scoring metric.
    random_state : int | None
        Reproducibility seed.
    ray_address : str | None
        Address of an existing Ray cluster (``None`` starts a local one).
    resources_per_trial : dict | None
        Ray resource request per trial, e.g. ``{"cpu": 2, "gpu": 0}``.

    Returns
    -------
    best_model : BaseEstimator
        The best LightGBM classifier found, already fitted on the full
        (resampled) training set.
    results : ray.tune.ResultGrid
        Full Tune results for further analysis.
    """
    require_automl()

    ray = lazy_import("ray")
    tune = lazy_import("ray.tune")
    lgb = lazy_import("lightgbm")

    _gan_space = gan_space or _DEFAULT_GAN_SPACE
    _lgb_space = lgb_space or _DEFAULT_LGB_SPACE

    if not ray.is_initialized():
        ray.init(address=ray_address, ignore_reinit_error=True)

    # ── Trainable ──

    def _objective(config: dict[str, Any]) -> None:
        """Single Tune trial: resample → fit LightGBM → report CV score."""
        sampler = RGANSampler(
            latent_dim=config["latent_dim"],
            epochs=config["epochs"],
            lambda_sim=config["lambda_sim"],
            lr_g=config["lr_g"],
            random_state=random_state,
        )
        X_res, y_res = sampler.fit_resample(X, y)

        clf = lgb.LGBMClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            learning_rate=config["learning_rate"],
            num_leaves=config["num_leaves"],
            verbose=-1,
            random_state=random_state,
        )
        score = float(cross_val_score(clf, X_res, y_res, cv=cv, scoring=scoring).mean())
        tune.report({"score": score})  # type: ignore[attr-defined]

    # ── Search space ──

    search_space = {
        "latent_dim": tune.choice(_gan_space["latent_dim"]),  # type: ignore[attr-defined]
        "epochs": tune.choice(_gan_space["epochs"]),  # type: ignore[attr-defined]
        "lambda_sim": tune.choice(_gan_space["lambda_sim"]),  # type: ignore[attr-defined]
        "lr_g": tune.choice(_gan_space["lr_g"]),  # type: ignore[attr-defined]
        "n_estimators": tune.choice(_lgb_space["n_estimators"]),  # type: ignore[attr-defined]
        "max_depth": tune.choice(_lgb_space["max_depth"]),  # type: ignore[attr-defined]
        "learning_rate": tune.choice(_lgb_space["learning_rate"]),  # type: ignore[attr-defined]
        "num_leaves": tune.choice(_lgb_space["num_leaves"]),  # type: ignore[attr-defined]
    }

    # ── Run ──

    tuner = tune.Tuner(  # type: ignore[attr-defined]
        tune.with_resources(  # type: ignore[attr-defined]
            _objective,
            resources=resources_per_trial or {"cpu": 1},
        ),
        param_space=search_space,
        tune_config=tune.TuneConfig(  # type: ignore[attr-defined]
            num_samples=num_samples,
            metric="score",
            mode="max",
        ),
    )
    results = tuner.fit()

    # ── Refit best ──

    best_config = results.get_best_result().config
    best_sampler = RGANSampler(
        latent_dim=best_config["latent_dim"],
        epochs=best_config["epochs"],
        lambda_sim=best_config["lambda_sim"],
        lr_g=best_config["lr_g"],
        random_state=random_state,
    )
    X_best, y_best = best_sampler.fit_resample(X, y)

    best_model = lgb.LGBMClassifier(
        n_estimators=best_config["n_estimators"],
        max_depth=best_config["max_depth"],
        learning_rate=best_config["learning_rate"],
        num_leaves=best_config["num_leaves"],
        verbose=-1,
        random_state=random_state,
    )
    best_model.fit(X_best, y_best)

    return best_model, results
