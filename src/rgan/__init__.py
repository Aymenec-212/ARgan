"""R-GAN — Regularized GAN for tabular oversampling.

Quick start::

    from rgan import RGANSampler
    sampler = RGANSampler(epochs=300, random_state=42)
    X_res, y_res = sampler.fit_resample(X, y)

Tier 2 (AutoML) and Tier 3 (XAI) are available as sub-packages and
require additional extras::

    pip install rgan[automl]   # enables rgan.automl
    pip install rgan[explain]  # enables rgan.explain
    pip install rgan[all]      # both
"""

from rgan.resample import RGANSampler

__all__ = ["RGANSampler"]
__version__ = "0.1.0"
