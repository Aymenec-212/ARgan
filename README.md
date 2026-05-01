# R-GAN

Regularized GAN for tabular oversampling. Combines Spectral Normalization (SNGAN), Wasserstein loss with gradient penalty (WGAN-GP), and a cosine-similarity regularisation term to generate synthetic minority-class samples for imbalanced classification tasks.

## Installation

Requires Python ≥ 3.10.

```bash
# Core only (local CPU/GPU oversampling)
uv add rgan

# With distributed AutoML (Ray Tune + LightGBM)
uv add "rgan[automl]"

# With explainability (parallel SHAP via Ray)
uv add "rgan[explain]"

# Everything
uv add "rgan[all]"
```

Or with pip:

```bash
pip install rgan
pip install "rgan[all]"
```

## Quickstart

```python
from sklearn.datasets import make_classification
from rgan import RGANSampler

X, y = make_classification(
    n_samples=1000, n_features=10,
    weights=[0.95, 0.05], random_state=42,
)

sampler = RGANSampler(epochs=300, random_state=42)
X_resampled, y_resampled = sampler.fit_resample(X, y)
```

`RGANSampler` implements `imblearn.base.BaseSampler` and plugs directly into `imblearn.pipeline.Pipeline` or any scikit-learn workflow.

## Architecture

The library is organised into three independent tiers to avoid dependency bloat:

| Tier | Package | Dependencies | Purpose |
|------|---------|-------------|---------|
| 1 | `rgan.resample` | torch, numpy, sklearn, imblearn | GAN training + oversampling |
| 2 | `rgan.automl` | + ray[tune], lightgbm | Distributed hyperparameter search |
| 3 | `rgan.explain` | + shap, matplotlib, ray | Parallel SHAP computation |

Tiers 2 and 3 are lazy-loaded. Importing `rgan` on a base installation will never trigger a `ModuleNotFoundError` for Ray or SHAP.

## Methodology

The generator is a fully-connected network mapping latent noise *z* → synthetic samples. The discriminator (critic) uses **spectral normalisation** on every linear layer to enforce Lipschitz continuity.

Training objective:

- **Discriminator:** Wasserstein loss + gradient penalty (λ_gp = 10)
- **Generator:** −E[D(G(z))] − λ_sim · cos_sim(μ_real, μ_fake)

The similarity loss encourages the generator to match the directional structure of the real feature space, not just the magnitude.

## Development

```bash
git clone https://github.com/YOUR_ORG/rgan.git
cd rgan
uv sync --extra dev

# Run core tests
uv run pytest tests/test_resample.py

# Lint + format
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Type check
uv run mypy src/rgan/
```

## License

MIT
