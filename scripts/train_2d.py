"""Training script for the 2-D DDPM toy diffusion model.

Stitches together the three building blocks in this repo:
    * ``src.data.synthetic.get_2d_dataset``   -- standardized 2-D manifold data
    * ``src.diffusion.scheduler.DiffusionScheduler`` -- forward-process buffers
    * ``src.models.mlp.TimeConditionedMLP``   -- eps-prediction backbone

Optimizes the standard DDPM epsilon-prediction objective

    L(theta) = E_{x_0, t, eps} || eps_theta(x_t, t) - eps ||_2^2

with ``x_t = sqrt(bar_alpha_t) * x_0 + sqrt(1 - bar_alpha_t) * eps``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm.auto import tqdm

from src.data.synthetic import get_2d_dataset
from src.diffusion.scheduler import DiffusionScheduler
from src.models.mlp import TimeConditionedMLP


@dataclass
class TrainConfig:
    """Hyperparameters for the 2-D DDPM training run.

    Attributes:
        dataset_name: Toy manifold to fit. One of ``{"moons", "swiss_roll"}``.
        n_samples: Number of points drawn from the manifold.
        num_epochs: Number of full passes over the dataset.
        batch_size: Mini-batch size.
        lr: AdamW learning rate.
        weight_decay: AdamW L2 weight-decay coefficient.
        num_timesteps: Number of diffusion steps ``T``.
        in_dim: Data dimensionality (2 for the toy problem).
        hidden_dim: Constant hidden width of the score MLP.
        num_layers: Total number of linear layers in the score MLP.
        grad_clip: Max L2 gradient norm applied every step.
        seed: RNG seed threaded through torch, CUDA, and the dataloader.
    """

    dataset_name: str = "moons"
    n_samples: int = 10_000
    num_epochs: int = 50
    batch_size: int = 256
    lr: float = 2e-4
    weight_decay: float = 0.0
    num_timesteps: int = 1000
    in_dim: int = 2
    hidden_dim: int = 256
    num_layers: int = 4
    grad_clip: float = 1.0
    seed: int = 0


def select_device() -> torch.device:
    """Chooses the best available device: ``cuda > mps > cpu``.

    Returns:
        A ``torch.device`` selected by availability. Preference order matches
        typical performance ranking on the platforms this repo targets
        (NVIDIA workstation, Apple Silicon laptop, CPU fallback).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Seeds torch (and CUDA if present) for reproducible training.

    Args:
        seed: Non-negative integer seed.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(cfg: TrainConfig) -> List[float]:
    """Runs the 2-D DDPM training loop.

    Args:
        cfg: Hyperparameter bundle. See ``TrainConfig``.

    Returns:
        List of per-epoch mean MSE losses, length ``cfg.num_epochs``.
    """
    set_seed(cfg.seed)
    device = select_device()
    print(
        f"[train_2d] device={device}  dataset={cfg.dataset_name}  T={cfg.num_timesteps}"
    )

    loader, _scaler = get_2d_dataset(
        dataset_name=cfg.dataset_name,
        n_samples=cfg.n_samples,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
    )

    # Both are nn.Module: .to(device) moves all registered buffers, so the
    # scheduler's beta / alpha / alpha_cumprod tensors follow the model.
    scheduler = DiffusionScheduler(num_timesteps=cfg.num_timesteps).to(device)
    model = TimeConditionedMLP(
        in_dim=cfg.in_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[train_2d] model params={n_params:,}  lr={cfg.lr:g}  batch_size={cfg.batch_size}"
    )

    epoch_losses: List[float] = []
    model.train()

    for epoch in range(1, cfg.num_epochs + 1):
        running_loss = 0.0
        n_batches = 0

        pbar = tqdm(
            loader,
            desc=f"epoch {epoch:02d}/{cfg.num_epochs}",
            leave=False,
        )
        for (x_0,) in pbar:
            x_0 = x_0.to(device, non_blocking=True)
            batch_size = x_0.shape[0]

            # Uniform t ~ U{0, T-1} per sample: unbiased Monte Carlo estimate
            # of the timestep expectation in the DDPM training objective.
            t = torch.randint(
                low=0,
                high=cfg.num_timesteps,
                size=(batch_size,),
                device=device,
                dtype=torch.long,
            )
            eps = torch.randn_like(x_0)

            x_t, eps = scheduler.add_noise(x_0, t, noise=eps)
            eps_pred = model(x_t, t)

            loss = F.mse_loss(eps_pred, eps)

            # Tripwire: NaN losses corrupt every subsequent step silently.
            # Fail loudly so the run is investigated, not left to burn compute.
            assert not torch.isnan(loss).item(), (
                f"NaN loss at epoch {epoch}, batch {n_batches}. "
                "Investigate learning rate, schedule endpoints, or input scale."
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            optimizer.step()

            loss_value = loss.item()
            running_loss += loss_value
            n_batches += 1
            pbar.set_postfix(loss=f"{loss_value:.4f}")

        avg_loss = running_loss / max(n_batches, 1)
        epoch_losses.append(avg_loss)
        print(
            f"[train_2d] epoch {epoch:02d}/{cfg.num_epochs}  "
            f"avg_loss={avg_loss:.6f}"
        )

    return epoch_losses


def main() -> None:
    """CLI entry point: trains with the default ``TrainConfig``."""
    cfg = TrainConfig()
    train(cfg)


if __name__ == "__main__":
    main()
