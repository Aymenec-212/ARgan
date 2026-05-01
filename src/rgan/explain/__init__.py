"""Tier 3 — Explainability via parallel SHAP.

Requires ``pip install rgan[explain]``.
"""

from rgan.explain.shap_utils import compute_shap_values, plot_summary

__all__ = ["compute_shap_values", "plot_summary"]
