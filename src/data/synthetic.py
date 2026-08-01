"""Synthetic 2-D toy datasets for score-based diffusion prototyping.

Each dataset is a low-dimensional manifold embedded in ``R^2``. Data is
standardized to zero mean and unit variance per feature so it sits roughly
inside a ``[-2, 2]`` box, matching the support of the terminal noise
distribution ``N(0, I)`` that the forward diffusion process integrates
toward. This scale-alignment is what lets an epsilon-prediction network
learn a well-conditioned score across all timesteps.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from sklearn.datasets import make_moons, make_swiss_roll
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

SUPPORTED_DATASETS: Tuple[str, ...] = ("moons", "swiss_roll")


def get_2d_dataset(
    dataset_name: str = "moons",
    n_samples: int = 10_000,
    batch_size: int = 256,
    *,
    seed: Optional[int] = None,
) -> Tuple[DataLoader, StandardScaler]:
    """Builds a standardized 2-D toy dataset wrapped in a shuffled DataLoader.

    Two-manifold options are provided:

    * ``"moons"``      : ``sklearn.datasets.make_moons`` with ``noise=0.05``.
    * ``"swiss_roll"`` : the classical Swiss roll, projected from its native
      3-D embedding onto its two informative axes ``(x, z)`` so the spiral
      manifold is recovered in ``R^2``.

    Args:
        dataset_name: One of ``{"moons", "swiss_roll"}``.
        n_samples: Number of points to draw. Must be a positive int.
        batch_size: Mini-batch size for the returned loader. Must be positive.
        seed: Optional RNG seed forwarded to sklearn and the loader's shuffle
            generator for reproducible sampling. ``None`` yields fresh
            randomness each call.

    Returns:
        Tuple ``(loader, scaler)`` where:
            * ``loader`` yields ``(x,)`` tuples with ``x`` of shape
              ``(batch_size, 2)`` and dtype ``torch.float32``. Shuffling is
              enabled.
            * ``scaler`` is the fitted ``StandardScaler``, retained so that
              generated samples can be inverse-transformed back onto the
              original manifold for visualization and metrics.

    Raises:
        ValueError: On invalid ``dataset_name``, ``n_samples``, or
            ``batch_size``.
    """
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"dataset_name={dataset_name!r} not in {SUPPORTED_DATASETS}.")
    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError(f"n_samples must be a positive int, got {n_samples!r}")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(f"batch_size must be a positive int, got {batch_size!r}")

    x_np = _sample_manifold(dataset_name, n_samples=n_samples, seed=seed)
    assert x_np.ndim == 2 and x_np.shape == (
        n_samples,
        2,
    ), f"Sampler returned shape {x_np.shape}, expected ({n_samples}, 2)"

    scaler = StandardScaler().fit(x_np)
    x_std = scaler.transform(x_np).astype(np.float32, copy=False)

    # Post-scale sanity: catches silent regressions if the scaler is ever
    # swapped out. Tolerances are loose enough for float32 rounding.
    assert np.allclose(
        x_std.mean(axis=0), 0.0, atol=1e-5
    ), f"StandardScaler mean drift: {x_std.mean(axis=0)}"
    assert np.allclose(
        x_std.std(axis=0), 1.0, atol=1e-3
    ), f"StandardScaler std drift: {x_std.std(axis=0)}"

    x_tensor = torch.from_numpy(x_std).contiguous()
    assert (
        x_tensor.dtype == torch.float32
    ), f"Expected float32 tensor, got {x_tensor.dtype}"
    assert x_tensor.shape == (
        n_samples,
        2,
    ), f"Expected shape ({n_samples}, 2), got {tuple(x_tensor.shape)}"

    dataset = TensorDataset(x_tensor)

    generator: Optional[torch.Generator] = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=generator,
    )
    return loader, scaler


def _sample_manifold(
    dataset_name: str,
    n_samples: int,
    seed: Optional[int],
) -> np.ndarray:
    """Draws raw (unscaled) 2-D samples from the chosen manifold.

    Args:
        dataset_name: One of ``{"moons", "swiss_roll"}``. Assumed pre-validated
            by the caller.
        n_samples: Number of points to draw.
        seed: Optional RNG seed forwarded to sklearn.

    Returns:
        Float64 ndarray of shape ``(n_samples, 2)``.
    """
    if dataset_name == "moons":
        x, _ = make_moons(n_samples=n_samples, noise=0.05, random_state=seed)
        return np.asarray(x, dtype=np.float64)

    if dataset_name == "swiss_roll":
        # make_swiss_roll returns (n_samples, 3) with axes (x, y, z). ``y`` is
        # the transverse height of the roll while (x, z) parameterize the
        # spiral. Dropping the height axis recovers the 2-D spiral manifold.
        x3d, _ = make_swiss_roll(n_samples=n_samples, random_state=seed)
        return np.asarray(x3d[:, [0, 2]], dtype=np.float64)

    raise ValueError(f"unreachable: dataset_name={dataset_name!r}")
