"""Tier 1 — Core GAN-based oversampling engine.

This package is always available with a base ``pip install rgan`` and
has **no optional dependencies**.

Public API::

    from rgan.resample import RGANSampler
"""

from rgan.resample.sampler import RGANSampler

__all__ = ["RGANSampler"]
