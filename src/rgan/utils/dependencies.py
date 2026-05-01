"""Centralized import guards for optional (Tier 2 / Tier 3) dependencies.

Every module that depends on Ray, LightGBM, SHAP, etc. must call the
appropriate ``require_*`` function **inside the function body** that needs
the dependency — never at module level.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["require_automl", "require_explain", "lazy_import"]


def _check_packages(packages: list[str], extra_name: str) -> None:
    """Attempt to import *packages*; raise a helpful ``ImportError`` on failure."""
    missing: list[str] = []
    for pkg in packages:
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        raise ImportError(
            f"Missing packages: {', '.join(missing)}. "
            f"Install them with:  pip install rgan[{extra_name}]  "
            f"(or:  uv pip install rgan[{extra_name}])"
        )


def require_automl() -> None:
    """Guard: ensures ``ray`` and ``lightgbm`` are importable."""
    _check_packages(["ray", "lightgbm"], extra_name="automl")


def require_explain() -> None:
    """Guard: ensures ``shap``, ``matplotlib``, and ``ray`` are importable."""
    _check_packages(["shap", "matplotlib", "ray"], extra_name="explain")


def lazy_import(module_path: str) -> Any:  # noqa: ANN401
    """Import and return *module_path* at call time (not at module load).

    Example::

        lgb = lazy_import("lightgbm")
        model = lgb.LGBMClassifier()
    """
    return importlib.import_module(module_path)
