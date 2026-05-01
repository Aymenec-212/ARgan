"""PyTorch network definitions for R-GAN.

Implements:
  - **Generator** with configurable hidden layers.
  - **Discriminator** with Spectral Normalization (SNGAN).
  - **Training loop** combining WGAN-GP gradient penalty with a
    cosine-similarity regularisation term ("Similarity Loss").
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

__all__ = [
    "Generator",
    "Discriminator",
    "compute_gradient_penalty",
    "similarity_loss",
    "train_gan",
]

# ---------------------------------------------------------------------------
# Network architectures
# ---------------------------------------------------------------------------


class Generator(nn.Module):
    """Fully-connected generator for tabular data.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the noise vector *z*.
    output_dim : int
        Number of features in the generated samples.
    hidden_dims : tuple[int, ...]
        Widths of hidden layers (default: two layers of 128 units).
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = latent_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, h_dim), nn.BatchNorm1d(h_dim), nn.ReLU()])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:  # noqa: D401
        """Map noise *z* → generated sample."""
        return self.net(z)


class Discriminator(nn.Module):
    """Fully-connected critic with **Spectral Normalization** on every linear layer.

    Parameters
    ----------
    input_dim : int
        Number of features in the input samples.
    hidden_dims : tuple[int, ...]
        Widths of hidden layers (default: two layers of 128 units).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend(
                [nn.utils.spectral_norm(nn.Linear(in_dim, h_dim)), nn.LeakyReLU(0.2)]
            )
            in_dim = h_dim
        # Critic outputs a scalar (no sigmoid — Wasserstein formulation)
        layers.append(nn.utils.spectral_norm(nn.Linear(in_dim, 1)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        """Critic score for *x*."""
        return self.net(x)


# ---------------------------------------------------------------------------
# Loss utilities
# ---------------------------------------------------------------------------


def compute_gradient_penalty(
    discriminator: Discriminator,
    real: Tensor,
    fake: Tensor,
    device: torch.device,
    lambda_gp: float = 10.0,
) -> Tensor:
    """WGAN-GP gradient penalty (Gulrajani et al., 2017).

    Interpolates between *real* and *fake* samples, evaluates the critic,
    and penalises deviations of the gradient norm from 1.
    """
    alpha = torch.rand(real.size(0), 1, device=device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp = discriminator(interpolated)
    gradients = torch.autograd.grad(
        outputs=d_interp,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
    )[0]
    grad_norm = gradients.norm(2, dim=1)
    return lambda_gp * ((grad_norm - 1.0) ** 2).mean()


def similarity_loss(real: Tensor, fake: Tensor) -> Tensor:
    """Cosine-similarity regularisation between real and generated batches.

    Encourages the generator to capture the *direction* of the real
    feature vectors, not just the magnitude.  Returns a scalar ∈ [-1, 1]
    that should be **maximised** (i.e. negate when adding to generator loss).
    """
    return nn.functional.cosine_similarity(real.mean(dim=0), fake.mean(dim=0), dim=0)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_gan(
    X_minority: np.ndarray,
    *,
    latent_dim: int = 64,
    g_hidden: tuple[int, ...] = (128, 128),
    d_hidden: tuple[int, ...] = (128, 128),
    epochs: int = 300,
    batch_size: int = 64,
    lr_g: float = 1e-4,
    lr_d: float = 1e-4,
    n_critic: int = 5,
    lambda_gp: float = 10.0,
    lambda_sim: float = 0.1,
    device: torch.device | None = None,
    random_state: int | None = None,
) -> Generator:
    """Train the R-GAN generator and return it in eval mode.

    Parameters
    ----------
    X_minority : np.ndarray, shape (n_samples, n_features)
        The minority-class samples to learn from.
    latent_dim : int
        Size of the noise vector.
    g_hidden / d_hidden : tuple[int, ...]
        Hidden-layer widths for Generator / Discriminator.
    epochs : int
        Number of full passes over the data.
    batch_size : int
        Mini-batch size.
    lr_g / lr_d : float
        Adam learning rates for G / D.
    n_critic : int
        Discriminator updates per generator update (WGAN-GP convention).
    lambda_gp : float
        Gradient-penalty coefficient.
    lambda_sim : float
        Similarity-loss coefficient (0 disables).
    device : torch.device | None
        Compute device; defaults to CUDA if available.
    random_state : int | None
        Seed for reproducibility.

    Returns
    -------
    Generator
        Trained generator in ``eval()`` mode.
    """
    if random_state is not None:
        torch.manual_seed(random_state)
        np.random.seed(random_state)  # noqa: NPY002

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_features = X_minority.shape[1]
    real_tensor = torch.tensor(X_minority, dtype=torch.float32, device=device)
    dataset = TensorDataset(real_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    gen = Generator(latent_dim, n_features, g_hidden).to(device)
    disc = Discriminator(n_features, d_hidden).to(device)
    opt_g = torch.optim.Adam(gen.parameters(), lr=lr_g, betas=(0.0, 0.9))
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr_d, betas=(0.0, 0.9))

    gen.train()
    disc.train()

    for _epoch in range(epochs):
        for (real_batch,) in loader:
            # ── Discriminator step ──
            for _ in range(n_critic):
                z = torch.randn(real_batch.size(0), latent_dim, device=device)
                fake_batch = gen(z).detach()
                d_loss = (
                    disc(fake_batch).mean()
                    - disc(real_batch).mean()
                    + compute_gradient_penalty(disc, real_batch, fake_batch, device, lambda_gp)
                )
                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()

            # ── Generator step ──
            z = torch.randn(real_batch.size(0), latent_dim, device=device)
            fake_batch = gen(z)
            g_loss = -disc(fake_batch).mean()
            if lambda_sim > 0:
                g_loss -= lambda_sim * similarity_loss(real_batch, fake_batch)
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

    gen.eval()
    return gen
