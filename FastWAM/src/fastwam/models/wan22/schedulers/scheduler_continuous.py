import torch


_ACTION_PREDICTION_TYPE_ALIASES = {
    "v_prediction": "v_prediction",
    "velocity": "v_prediction",
    "v_pred": "v_prediction",
    "x_prediction": "x_prediction",
    "x_pred": "x_prediction",
    "sample": "x_prediction",
    "x_prediction_v_loss": "x_prediction_v_loss",
    "x_pred_v_loss": "x_prediction_v_loss",
    "abot": "x_prediction_v_loss",
    "abot_aml": "x_prediction_v_loss",
}


def normalize_action_prediction_type(prediction_type: str) -> str:
    """Return the canonical action-head output/loss parameterization."""
    key = str(prediction_type).lower()
    if key not in _ACTION_PREDICTION_TYPE_ALIASES:
        supported = ", ".join(("v_prediction", "x_prediction", "x_prediction_v_loss"))
        raise ValueError(
            f"Unsupported action_prediction_type={prediction_type!r}; use one of: {supported}."
        )
    return _ACTION_PREDICTION_TYPE_ALIASES[key]


def _sigma_like(timestep: torch.Tensor, sample: torch.Tensor, num_train_timesteps: int) -> torch.Tensor:
    sigma = (timestep / float(num_train_timesteps)).to(device=sample.device, dtype=sample.dtype)
    if sigma.ndim == 0:
        return sigma
    return sigma.view(-1, *([1] * (sample.ndim - 1)))


def action_prediction_loss_token(
    prediction: torch.Tensor,
    clean_action: torch.Tensor,
    noisy_action: torch.Tensor,
    velocity_target: torch.Tensor,
    timestep: torch.Tensor,
    num_train_timesteps: int,
    prediction_type: str,
    x_pred_v_loss_eps: float,
) -> torch.Tensor:
    """Return per-token MSE for the selected action output parameterization.

    FastWAM uses ``a_sigma = (1-sigma) * a_clean + sigma * eps``. Therefore
    the x-pred/v-loss velocity is ``(a_sigma-a_hat_clean)/sigma``.
    """
    if prediction_type == "v_prediction":
        error = prediction.float() - velocity_target.float()
    elif prediction_type == "x_prediction":
        error = prediction.float() - clean_action.float()
    elif prediction_type == "x_prediction_v_loss":
        sigma = _sigma_like(timestep, noisy_action, num_train_timesteps).float()
        denominator = sigma.clamp_min(float(x_pred_v_loss_eps))
        pred_velocity = (noisy_action.float() - prediction.float()) / denominator
        target_velocity = (noisy_action.float() - clean_action.float()) / denominator
        error = pred_velocity - target_velocity
    else:
        raise ValueError(f"Unsupported canonical action prediction type: {prediction_type!r}")
    return error.square().mean(dim=2)


def action_prediction_step(
    sample: torch.Tensor,
    prediction: torch.Tensor,
    timestep: torch.Tensor,
    delta: torch.Tensor,
    num_train_timesteps: int,
    prediction_type: str,
) -> torch.Tensor:
    """Advance an action flow step for velocity or clean-action outputs."""
    if prediction_type == "v_prediction":
        return WanContinuousFlowMatchScheduler.step(prediction, delta, sample)
    if prediction_type not in {"x_prediction", "x_prediction_v_loss"}:
        raise ValueError(f"Unsupported canonical action prediction type: {prediction_type!r}")

    sigma = _sigma_like(timestep, sample, num_train_timesteps)
    delta = delta.to(device=sample.device, dtype=sample.dtype)
    if delta.ndim != 0:
        delta = delta.view(-1, *([1] * (sample.ndim - 1)))
    next_sigma = sigma + delta
    retained_noisy_weight = next_sigma / sigma.clamp_min(torch.finfo(sample.dtype).eps)
    return retained_noisy_weight * sample + (1 - retained_noisy_weight) * prediction.to(sample.dtype)


class WanContinuousFlowMatchScheduler:
    """Continuous-time Flow-Matching scheduler with shift-based sampling."""

    def __init__(self, num_train_timesteps: int = 1000, shift: float = 5.0, eps: float = 1e-10):
        if num_train_timesteps <= 0:
            raise ValueError(f"`num_train_timesteps` must be positive, got {num_train_timesteps}")
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}")
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.eps = float(eps)
        self._y_min, self._weight_norm_const = self._precompute_training_weight_stats()

    @staticmethod
    def _phi(u: torch.Tensor, shift: float) -> torch.Tensor:
        return shift * u / (1.0 + (shift - 1.0) * u)

    def _precompute_training_weight_stats(self) -> tuple[float, float]:
        steps = self.num_train_timesteps
        u_grid = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)[:-1]
        t_grid = self._phi(u_grid, self.shift) * float(steps)
        y_grid = torch.exp(-2.0 * ((t_grid - (steps / 2.0)) / steps) ** 2)
        y_min = float(y_grid.min().item())
        y_shifted_grid = y_grid - y_min
        norm_const = float(y_shifted_grid.mean().item())
        return y_min, norm_const

    def sample_training_t(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError(f"`batch_size` must be positive, got {batch_size}")
        u = torch.rand((batch_size,), device=device, dtype=torch.float32)
        sigma = self._phi(u, self.shift)
        timestep = sigma * float(self.num_train_timesteps)
        return timestep.to(dtype=dtype)

    def training_weight(self, timestep: torch.Tensor) -> torch.Tensor:
        t = timestep.to(dtype=torch.float32)
        steps = float(self.num_train_timesteps)
        y = torch.exp(-2.0 * ((t - (steps / 2.0)) / steps) ** 2)
        y_shifted = y - self._y_min
        weight = y_shifted / (self._weight_norm_const + self.eps)
        if weight.numel() == 1:
            return weight.reshape(())
        return weight

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        sigma = (timestep / float(self.num_train_timesteps)).to(
            original_samples.device, dtype=original_samples.dtype
        )
        if sigma.ndim == 0:
            return (1 - sigma) * original_samples + sigma * noise
        sigma = sigma.view(-1, *([1] * (original_samples.ndim - 1)))
        return (1 - sigma) * original_samples + sigma * noise

    @staticmethod
    def training_target(sample: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        del timestep
        return noise - sample

    def build_inference_schedule(
        self,
        num_inference_steps: int,
        device: torch.device,
        dtype: torch.dtype,
        shift_override: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be positive, got {num_inference_steps}")
        shift = self.shift if shift_override is None else float(shift_override)
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}")

        u_steps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device, dtype=torch.float32)
        sigma_steps = self._phi(u_steps, shift)
        timesteps = sigma_steps[:-1] * float(self.num_train_timesteps)
        deltas = sigma_steps[1:] - sigma_steps[:-1]
        return timesteps.to(dtype=dtype), deltas.to(dtype=dtype)

    @staticmethod
    def step(model_output: torch.Tensor, delta: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        delta = delta.to(sample.device, dtype=sample.dtype)
        if delta.ndim == 0:
            return sample + model_output * delta
        delta = delta.view(-1, *([1] * (sample.ndim - 1)))
        return sample + model_output * delta
