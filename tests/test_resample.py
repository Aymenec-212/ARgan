"""Tests for Tier 1 — rgan.resample (core engine).

Run with:  uv run pytest tests/test_resample.py
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification

from rgan.resample import RGANSampler
from rgan.resample._gan import (
    Discriminator,
    Generator,
    compute_gradient_penalty,
    similarity_loss,
    train_gan,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture()
def imbalanced_data() -> tuple[np.ndarray, np.ndarray]:
    """Create a small imbalanced binary dataset."""
    X, y = make_classification(
        n_samples=200,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        weights=[0.9, 0.1],
        random_state=42,
    )
    return X.astype(np.float32), y


# ------------------------------------------------------------------
# Unit tests — network components
# ------------------------------------------------------------------


class TestGenerator:
    def test_output_shape(self) -> None:
        gen = Generator(latent_dim=16, output_dim=5)
        import torch

        z = torch.randn(8, 16)
        out = gen(z)
        assert out.shape == (8, 5)

    def test_deterministic_with_seed(self) -> None:
        import torch

        torch.manual_seed(0)
        gen = Generator(latent_dim=16, output_dim=5)
        z = torch.randn(4, 16)
        out1 = gen(z).detach()

        torch.manual_seed(0)
        gen2 = Generator(latent_dim=16, output_dim=5)
        out2 = gen2(z).detach()
        np.testing.assert_array_almost_equal(out1.numpy(), out2.numpy())


class TestDiscriminator:
    def test_output_shape(self) -> None:
        disc = Discriminator(input_dim=5)
        import torch

        x = torch.randn(8, 5)
        out = disc(x)
        assert out.shape == (8, 1)


class TestLosses:
    def test_gradient_penalty_is_nonneg(self) -> None:
        import torch

        disc = Discriminator(input_dim=3)
        real = torch.randn(4, 3)
        fake = torch.randn(4, 3)
        gp = compute_gradient_penalty(disc, real, fake, torch.device("cpu"))
        assert gp.item() >= 0

    def test_similarity_loss_range(self) -> None:
        import torch

        a = torch.randn(8, 5)
        b = torch.randn(8, 5)
        sim = similarity_loss(a, b)
        assert -1.0 <= sim.item() <= 1.0


# ------------------------------------------------------------------
# Integration — training loop
# ------------------------------------------------------------------


class TestTrainGan:
    @pytest.mark.slow()
    def test_train_returns_generator(self, imbalanced_data: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = imbalanced_data
        X_min = X[y == 1]
        gen = train_gan(X_min, epochs=5, batch_size=8, random_state=42)
        assert isinstance(gen, Generator)
        assert not gen.training  # should be in eval mode


# ------------------------------------------------------------------
# Integration — sampler contract
# ------------------------------------------------------------------


class TestRGANSampler:
    def test_fit_resample_shape(
        self, imbalanced_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, y = imbalanced_data
        sampler = RGANSampler(epochs=5, batch_size=8, random_state=42)
        X_res, y_res = sampler.fit_resample(X, y)
        assert X_res.shape[0] == y_res.shape[0]
        assert X_res.shape[1] == X.shape[1]
        # Should have more samples than original
        assert X_res.shape[0] >= X.shape[0]

    def test_balanced_classes(
        self, imbalanced_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, y = imbalanced_data
        sampler = RGANSampler(epochs=5, batch_size=8, random_state=42)
        _, y_res = sampler.fit_resample(X, y)
        unique, counts = np.unique(y_res, return_counts=True)
        # Default strategy ("auto") should balance classes
        assert counts[0] == counts[1]

    def test_generators_stored(
        self, imbalanced_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, y = imbalanced_data
        sampler = RGANSampler(epochs=5, batch_size=8, random_state=42)
        sampler.fit_resample(X, y)
        assert len(sampler.generators_) > 0
