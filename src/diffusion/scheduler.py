"""Forward-diffusion variance scheduler (DDPM).

Implements the closed-form marginal ``q(x_t | x_0)`` so any diffusion timestep
can be sampled in a single vectorized step:

    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps,   eps ~ N(0, I)

Reference:
    Ho, Jain, Abbeel. "Denoising Diffusion Probabilistic Models." (2020).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn


class DiffusionScheduler(nn.Module):
    """Pre-computes and applies the DDPM forward-diffusion variance schedule.

    All schedule tensors (``betas``, ``alphas``, ``alphas_cumprod``, and the
    two precomputed square-root variants used in ``add_noise``) are stored as
    registered buffers, so they migrate with ``.to(device)`` and appear in the
    module's ``state_dict``.

    Attributes:
        num_timesteps: Total number of diffusion steps ``T``.
        schedule_type: Name of the variance schedule (currently ``"linear"``).
        betas: Per-step variance ``beta_t``, shape ``(T,)``, dtype ``float32``.
        alphas: ``1 - beta_t``, shape ``(T,)``, dtype ``float32``.
        alphas_cumprod: Cumulative product ``bar{alpha}_t``, shape ``(T,)``.
        sqrt_alphas_cumprod: ``sqrt(bar{alpha}_t)``, shape ``(T,)``.
        sqrt_one_minus_alphas_cumprod: ``sqrt(1 - bar{alpha}_t)``, shape ``(T,)``.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        schedule_type: str = "linear",
    ) -> None:
        """Initializes the scheduler and pre-computes all schedule buffers.

        Args:
            num_timesteps: Total number of diffusion steps ``T``. Must be a
                positive integer.
            schedule_type: Variance schedule name. Only ``"linear"`` is
                implemented; other values raise ``NotImplementedError``.

        Raises:
            ValueError: If ``num_timesteps`` is not a positive integer.
            NotImplementedError: If ``schedule_type`` is not supported.
        """
        super().__init__()

        if not isinstance(num_timesteps, int) or num_timesteps <= 0:
            raise ValueError(
                f"num_timesteps must be a positive int, got {num_timesteps!r}"
            )

        self.num_timesteps: int = num_timesteps
        self.schedule_type: str = schedule_type

        betas = self._build_beta_schedule(num_timesteps, schedule_type)

        assert betas.shape == (num_timesteps,), (
            f"Expected betas shape ({num_timesteps},), got {tuple(betas.shape)}"
        )
        assert betas.dtype == torch.float32, (
            f"Expected float32 betas, got {betas.dtype}"
        )
        assert torch.all(betas > 0) and torch.all(betas < 1), (
            "All beta_t must satisfy 0 < beta_t < 1 for a valid schedule."
        )

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Clamp before sqrt so ``sqrt(1 - alpha_bar_0)`` is safe under fp16 and
        # never returns a NaN gradient. The clamp floor is small enough to be
        # a strict no-op in fp32 for the schedules we ship.
        one_minus_alphas_cumprod = (1.0 - alphas_cumprod).clamp(min=1e-20)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(one_minus_alphas_cumprod),
        )

    @staticmethod
    def _build_beta_schedule(
        num_timesteps: int,
        schedule_type: str,
    ) -> torch.Tensor:
        """Builds the ``beta_t`` schedule as a 1-D float32 tensor.

        Args:
            num_timesteps: Number of steps ``T``.
            schedule_type: One of ``{"linear"}``. Linear uses the Ho et al.
                2020 defaults ``beta_1 = 1e-4`` and ``beta_T = 2e-2``.

        Returns:
            Tensor of shape ``(T,)`` and dtype ``torch.float32``.

        Raises:
            NotImplementedError: If ``schedule_type`` is not ``"linear"``.
        """
        if schedule_type == "linear":
            beta_start, beta_end = 1e-4, 2e-2
            return torch.linspace(
                beta_start, beta_end, num_timesteps, dtype=torch.float32
            )

        raise NotImplementedError(
            f"schedule_type={schedule_type!r} is not implemented. "
            "Supported: 'linear'."
        )

    @staticmethod
    def _extract(
        a: torch.Tensor,
        t: torch.Tensor,
        x_shape: torch.Size,
    ) -> torch.Tensor:
        """Gathers per-sample coefficients and reshapes them for broadcasting.

        Given a 1-D schedule buffer ``a`` of shape ``(T,)`` and long indices
        ``t`` of shape ``(B,)``, this returns a tensor of shape
        ``(B, 1, 1, ..., 1)`` with ``len(x_shape) - 1`` trailing singleton
        dimensions. Multiplying the result by a data tensor of shape
        ``x_shape`` then broadcasts unambiguously across every non-batch axis,
        avoiding any reliance on implicit broadcasting.

        Args:
            a: Schedule buffer of shape ``(T,)``.
            t: Per-sample timestep indices, shape ``(B,)``, dtype
                ``torch.long``.
            x_shape: Target data shape, e.g. ``(B, D)`` for the 2D toy problem
                or ``(B, C, H, W)`` for images.

        Returns:
            Coefficient tensor of shape ``(B, 1, ..., 1)`` with
            ``len(x_shape)`` total dimensions.
        """
        assert a.dim() == 1, (
            f"Schedule buffer must be 1-D, got shape {tuple(a.shape)}"
        )
        assert t.dim() == 1, (
            f"Timestep indices must be 1-D, got shape {tuple(t.shape)}"
        )
        assert t.dtype == torch.long, (
            f"Timestep indices must be torch.long, got {t.dtype}"
        )
        assert a.device == t.device, (
            f"Buffer on {a.device} but indices on {t.device}"
        )
        assert len(x_shape) >= 1, "x_shape must have at least one (batch) dim"
        assert t.shape[0] == x_shape[0], (
            f"Batch mismatch: t has {t.shape[0]} entries but "
            f"x_shape[0]={x_shape[0]}"
        )

        batch_size = t.shape[0]
        gathered = a.gather(0, t)
        view_shape = (batch_size,) + (1,) * (len(x_shape) - 1)
        return gathered.view(view_shape)

    def add_noise(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples ``x_t`` from ``q(x_t | x_0)`` via the closed-form jump.

        Implements the reparameterized marginal

            x_t = sqrt(bar{alpha}_t) * x_0 + sqrt(1 - bar{alpha}_t) * eps,

        which corrupts a clean sample ``x_0`` to its noisy state at any
        timestep ``t`` in a single vectorized step.

        Args:
            x_0: Clean data of shape ``x_shape``, e.g. ``(B, D)`` or
                ``(B, C, H, W)``. Must be a floating tensor.
            t: Timestep indices of shape ``(B,)``, dtype ``torch.long``, with
                every entry in ``[0, num_timesteps - 1]``.
            noise: Optional pre-sampled Gaussian noise of shape ``x_0.shape``,
                matching ``x_0``'s dtype and device. If ``None``,
                ``torch.randn_like(x_0)`` is drawn. Passing this explicitly
                enables deterministic unit tests and lets callers reuse a
                fixed epsilon draw.

        Returns:
            Tuple ``(x_t, noise)`` where:
                * ``x_t`` has shape ``x_0.shape`` and matches ``x_0``'s dtype
                  and device. It is the noisy sample at timestep ``t``.
                * ``noise`` is the epsilon tensor actually used. It is
                  returned so it can serve as the regression target for an
                  epsilon-prediction denoiser.
        """
        assert x_0.is_floating_point(), (
            f"x_0 must be a floating tensor, got dtype {x_0.dtype}"
        )
        assert t.dtype == torch.long, f"t must be torch.long, got {t.dtype}"
        assert t.device == x_0.device, (
            f"t on {t.device} but x_0 on {x_0.device}"
        )
        assert t.shape == (x_0.shape[0],), (
            f"Expected t.shape ({x_0.shape[0]},), got {tuple(t.shape)}"
        )
        # Range check: catches off-by-one errors when callers sample t. One
        # device->host sync per call, negligible next to a training step.
        t_min = int(t.min())
        t_max = int(t.max())
        assert 0 <= t_min and t_max < self.num_timesteps, (
            f"t values must lie in [0, {self.num_timesteps - 1}], got "
            f"[{t_min}, {t_max}]"
        )

        if noise is None:
            noise = torch.randn_like(x_0)
        else:
            assert noise.shape == x_0.shape, (
                f"noise.shape {tuple(noise.shape)} != x_0.shape "
                f"{tuple(x_0.shape)}"
            )
            assert noise.device == x_0.device, (
                f"noise on {noise.device} but x_0 on {x_0.device}"
            )
            assert noise.dtype == x_0.dtype, (
                f"noise dtype {noise.dtype} != x_0 dtype {x_0.dtype}"
            )

        sqrt_alpha_bar = self._extract(
            self.sqrt_alphas_cumprod, t, x_0.shape
        )
        sqrt_one_minus_alpha_bar = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )

        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise

        assert x_t.shape == x_0.shape, (
            "internal broadcasting bug: x_t shape does not match x_0"
        )
        return x_t, noise
