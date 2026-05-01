"""``imbalanced-learn``-compatible oversampler powered by R-GAN.

Usage::

    from rgan.resample import RGANSampler
    sampler = RGANSampler(epochs=300, random_state=42)
    X_res, y_res = sampler.fit_resample(X, y)

The sampler also plugs into ``imblearn.pipeline.Pipeline`` and any
scikit-learn workflow that expects a ``BaseSampler``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from imblearn.base import BaseSampler
from numpy.typing import NDArray
from sklearn.utils import check_random_state

from rgan.resample._gan import Generator, train_gan

__all__ = ["RGANSampler"]


class RGANSampler(BaseSampler):  # type: ignore[misc]
    """Oversampler that trains an R-GAN per minority class.

    Parameters
    ----------
    latent_dim : int
        Size of the generator's noise vector.
    g_hidden / d_hidden : tuple[int, ...]
        Hidden-layer widths for Generator / Discriminator.
    epochs : int
        Training epochs per minority class.
    batch_size : int
        Mini-batch size during GAN training.
    lr_g / lr_d : float
        Learning rates for Adam optimisers.
    n_critic : int
        Discriminator steps per generator step.
    lambda_gp : float
        WGAN-GP gradient-penalty coefficient.
    lambda_sim : float
        Similarity-loss coefficient (0 disables it).
    device : str | None
        ``"cpu"``, ``"cuda"``, or ``None`` (auto-detect).
    random_state : int | None
        Seed for reproducibility.
    sampling_strategy : str | dict | float
        Passed through to ``BaseSampler``.  ``"auto"`` balances all
        minority classes to the majority count.
    """

    # imbalanced-learn uses _sampling_type to enforce contracts.
    _sampling_type = "over-sampling"

    def __init__(
        self,
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
        device: str | None = None,
        random_state: int | None = None,
        sampling_strategy: str | dict[int, int] | float = "auto",
    ) -> None:
        super().__init__(sampling_strategy=sampling_strategy)
        self.latent_dim = latent_dim
        self.g_hidden = g_hidden
        self.d_hidden = d_hidden
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp
        self.lambda_sim = lambda_sim
        self.device = device
        self.random_state = random_state

        # Populated after fit
        self.generators_: dict[Any, Generator] = {}

    # ------------------------------------------------------------------
    # imbalanced-learn contract
    # ------------------------------------------------------------------

    def _fit_resample(
        self,
        X: NDArray[np.floating[Any]],
        y: NDArray[np.integer[Any]],
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.integer[Any]]]:
        """Train one generator per minority class and produce synthetic samples.

        This method is called internally by ``BaseSampler.fit_resample``.
        """
        rng = check_random_state(self.random_state)
        torch_device = torch.device(
            self.device if self.device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        X_res = [X.copy()]
        y_res = [y.copy()]
        self.generators_ = {}

        sampling_dict = self._validate_sampling_strategy()

        for class_label, n_samples in sampling_dict.items():
            X_class = X[y == class_label]
            seed = int(rng.randint(0, 2**31))

            gen = train_gan(
                X_class,
                latent_dim=self.latent_dim,
                g_hidden=self.g_hidden,
                d_hidden=self.d_hidden,
                epochs=self.epochs,
                batch_size=min(self.batch_size, len(X_class)),
                lr_g=self.lr_g,
                lr_d=self.lr_d,
                n_critic=self.n_critic,
                lambda_gp=self.lambda_gp,
                lambda_sim=self.lambda_sim,
                device=torch_device,
                random_state=seed,
            )
            self.generators_[class_label] = gen

            # Generate synthetic samples
            with torch.no_grad():
                z = torch.randn(n_samples, self.latent_dim, device=torch_device)
                synthetic = gen(z).cpu().numpy()

            X_res.append(synthetic)
            y_res.append(np.full(n_samples, class_label, dtype=y.dtype))

        return np.vstack(X_res), np.concatenate(y_res)

    def _validate_sampling_strategy(self) -> dict[Any, int]:
        """Resolve ``sampling_strategy`` into ``{class: n_to_generate}``."""
        # BaseSampler stores the validated dict in sampling_strategy_
        return dict(self.sampling_strategy_)
