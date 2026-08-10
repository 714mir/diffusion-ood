"""Time-conditioned MLP score/eps-prediction network for 2-D diffusion.

The architecture is a stack of linear layers of constant width ``hidden_dim``,
each with a shared time embedding additively injected before the ``SiLU``
nonlinearity. The final linear layer has no post-activation and no time
injection: its output is directly the epsilon prediction consumed by the
DDPM training objective.

The time embedding is the standard sinusoidal positional encoding from
Vaswani et al. 2017 ("Attention Is All You Need"), applied to scalar diffusion
timesteps and post-processed with a small ``Linear -> SiLU -> Linear`` MLP.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class TimeConditionedMLP(nn.Module):
    """Time-conditioned MLP for 2-D score / eps-prediction.

    A stack of ``num_layers`` linear layers of constant width ``hidden_dim``.
    A shared time embedding of shape ``(B, hidden_dim)`` is added to the
    hidden state at every layer *except* the output, followed by ``SiLU``.

    Attributes:
        in_dim: Data dimensionality (2 for the toy 2-D problems).
        hidden_dim: Constant width of every hidden linear layer. Must be even.
        num_layers: Total number of linear layers (input + hidden + output).
        max_period: Maximum period of the sinusoidal time embedding.
        time_freqs: Precomputed geometric frequencies, shape
            ``(hidden_dim // 2,)``, registered as a buffer so it migrates with
            ``.to(device)``.
    """

    def __init__(
        self,
        in_dim: int = 2,
        hidden_dim: int = 256,
        num_layers: int = 4,
        max_period: float = 10_000.0,
    ) -> None:
        """Initializes the network and precomputes the sinusoidal frequencies.

        Args:
            in_dim: Data dimensionality (input and output). Must be positive.
            hidden_dim: Constant hidden width. Must be positive and even so
                the sinusoidal embedding splits cleanly into sin/cos halves.
            num_layers: Total number of linear layers (``input`` + hidden
                blocks + ``output``). Must be in ``{3, 4}`` per the spec.
            max_period: Largest period represented by the sinusoidal
                embedding. Must be a positive float.

        Raises:
            ValueError: On any argument that violates the above constraints.
        """
        super().__init__()

        if not isinstance(in_dim, int) or in_dim <= 0:
            raise ValueError(f"in_dim must be a positive int, got {in_dim!r}")
        if not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be a positive int, got {hidden_dim!r}"
            )
        if hidden_dim % 2 != 0:
            raise ValueError(
                f"hidden_dim must be even for sinusoidal embedding, "
                f"got {hidden_dim}"
            )
        if num_layers not in (3, 4):
            raise ValueError(
                f"num_layers must be 3 or 4, got {num_layers!r}"
            )
        if not (isinstance(max_period, (int, float)) and max_period > 0):
            raise ValueError(
                f"max_period must be a positive number, got {max_period!r}"
            )

        self.in_dim: int = in_dim
        self.hidden_dim: int = hidden_dim
        self.num_layers: int = num_layers
        self.max_period: float = float(max_period)

        # Precompute geometric frequencies once:
        #   freq_i = max_period ** (-i / (hidden_dim / 2))
        #         = exp(-log(max_period) * i / (hidden_dim / 2))
        # Stored as a buffer so it moves with .to(device) and never gets
        # recomputed inside the training loop.
        half = hidden_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, dtype=torch.float32)
            / half
        )
        self.register_buffer("time_freqs", freqs)

        # Time-conditioning MLP: (B, hidden_dim) -> (B, hidden_dim).
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Main backbone. ``num_layers`` counts every nn.Linear:
        #   input layer          (in_dim  -> hidden_dim)
        #   num_layers - 2 hidden blocks (hidden_dim -> hidden_dim)
        #   output layer         (hidden_dim -> in_dim)
        self.input_layer = nn.Linear(in_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [
                nn.Linear(hidden_dim, hidden_dim)
                for _ in range(num_layers - 2)
            ]
        )
        self.output_layer = nn.Linear(hidden_dim, in_dim)

        self.activation = nn.SiLU()

    def _sinusoidal_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Computes the sinusoidal positional encoding of scalar timesteps.

        Args:
            t: Float tensor of shape ``(B,)`` containing (possibly-continuous)
                diffusion timesteps.

        Returns:
            Float32 tensor of shape ``(B, hidden_dim)``. The first
            ``hidden_dim // 2`` columns are the sine components; the
            remaining ``hidden_dim // 2`` columns are the cosine components.
        """
        assert t.dim() == 1, (
            f"Expected 1-D timestep tensor, got shape {tuple(t.shape)}"
        )
        assert t.is_floating_point(), (
            f"Sinusoidal embedding expects float t, got dtype {t.dtype}"
        )
        assert t.device == self.time_freqs.device, (
            f"t on {t.device} but time_freqs on {self.time_freqs.device}"
        )

        # (B, 1) * (1, half) -> (B, half)
        args = t.unsqueeze(1) * self.time_freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        assert emb.shape == (t.shape[0], self.hidden_dim), (
            f"Sinusoidal embedding shape {tuple(emb.shape)} != "
            f"({t.shape[0]}, {self.hidden_dim})"
        )
        return emb

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Predicts the noise ``epsilon_theta(x, t)``.

        Args:
            x: Noisy input of shape ``(B, in_dim)``. Must be a floating tensor
                (``float32`` in the default training loop).
            t: Timestep index tensor of shape ``(B,)``. May be either
                ``torch.long`` (as produced by ``DiffusionScheduler``) or a
                floating tensor. It is cast to float internally before the
                sinusoidal embedding.

        Returns:
            Tensor of shape ``(B, in_dim)`` matching ``x``'s dtype and device.
            Interpreted as the predicted noise for the standard DDPM
            epsilon-prediction objective.
        """
        assert x.dim() == 2, (
            f"Expected x of shape (B, {self.in_dim}), got {tuple(x.shape)}"
        )
        assert x.shape[1] == self.in_dim, (
            f"Expected feature dim {self.in_dim}, got {x.shape[1]}"
        )
        assert x.is_floating_point(), (
            f"x must be a floating tensor, got dtype {x.dtype}"
        )
        assert t.dim() == 1, (
            f"Expected t of shape (B,), got {tuple(t.shape)}"
        )
        assert t.shape[0] == x.shape[0], (
            f"Batch mismatch: x has {x.shape[0]} rows but t has "
            f"{t.shape[0]} entries"
        )
        assert t.device == x.device, (
            f"t on {t.device} but x on {x.device}"
        )

        t_float = t.float() if not t.is_floating_point() else t
        t_emb = self._sinusoidal_embedding(t_float)  # (B, hidden_dim)
        t_emb = self.time_mlp(t_emb)                 # (B, hidden_dim)

        h = self.input_layer(x)                      # (B, hidden_dim)
        h = self.activation(h + t_emb)

        for layer in self.hidden_layers:
            h = layer(h)                             # (B, hidden_dim)
            h = self.activation(h + t_emb)

        out = self.output_layer(h)                   # (B, in_dim)

        assert out.shape == x.shape, (
            f"Output shape {tuple(out.shape)} != input shape {tuple(x.shape)}"
        )
        return out


if __name__ == "__main__":
    # Smoke test: verify shapes, device migration behavior, and that different
    # timesteps really produce different predictions (i.e. time conditioning
    # is wired in, not silently ignored).
    torch.manual_seed(0)

    net = TimeConditionedMLP(in_dim=2, hidden_dim=256, num_layers=4)
    x = torch.randn(5, 2)
    t = torch.tensor([0, 250, 500, 750, 999], dtype=torch.long)

    with torch.no_grad():
        eps_pred = net(x, t)

    n_params = sum(p.numel() for p in net.parameters())
    print(f"TimeConditionedMLP parameters : {n_params:,}")
    print(f"x.shape                       : {tuple(x.shape)}")
    print(f"t                             : {t.tolist()}")
    print(f"eps_pred.shape                : {tuple(eps_pred.shape)}")

    # Confirm t actually conditions the output: predicting on the SAME x with
    # two different t's must produce non-identical outputs.
    x_single = torch.randn(1, 2)
    with torch.no_grad():
        y0 = net(x_single, torch.tensor([0], dtype=torch.long))
        y1 = net(x_single, torch.tensor([999], dtype=torch.long))
    print(f"||eps(x, t=0) - eps(x, t=999)|| = {(y0 - y1).norm().item():.4f}"
          " (should be > 0)")
