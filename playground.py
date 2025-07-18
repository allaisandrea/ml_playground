from typing import Callable
import dataclasses
import datetime
import logging
import os
import json
import time
import torch
import numpy as np
import functools
import matplotlib
import mlflow
import fsspec


# Utilities ###################################################################

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_OUTPUT_ROOT = "/home/ubuntu/data/diffusion_playground/models"


def get_fs(path: str) -> fsspec.AbstractFileSystem:
    if path.startswith("s3://"):
        return fsspec.filesystem("s3")
    else:
        return fsspec.filesystem("file")


def init_run(
    tag: str,
    output_root: str,
    logger: logging.Logger,
    mlflow_local_path: str | None = None,
):
    logging.basicConfig(
        format="%(asctime)s:%(levelname)s:%(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    logger.setLevel(logging.INFO)
    if mlflow_local_path is None:
        mlflow.set_tracking_uri("databricks")
    else:
        mlflow.set_tracking_uri(f"file://{mlflow_local_path}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{tag}"
    output_path = os.path.join(output_root, run_name)
    run = mlflow.start_run(
        run_name=run_name,
        experiment_id="730119036412452",
    )
    logger.info("Starting run %s", run_name)
    return run_name, output_path


class Timer:
    def __init__(self):
        self.start_time = None
        self.reset()

    def start(self):
        assert self.start_time is None
        self.start_time = time.perf_counter()

    def stop(self):
        assert self.start_time is not None
        end_time = time.perf_counter()
        self.update(end_time - self.start_time)
        self.start_time = None

    def pause(self):
        assert self.start_time is not None
        end_time = time.perf_counter()
        self.total_time += end_time - self.start_time
        self.start_time = None

    def split(self):
        assert self.start_time is not None
        split_time = time.perf_counter()
        self.update(split_time - self.start_time)
        self.start_time = split_time

    def reset(self):
        self.count = 0
        self.total_time = 0.0

    def update(self, elapsed: float):
        self.count += 1
        self.total_time += elapsed

    def mean(self):
        return self.total_time / max(1, self.count)


class HistogramNd(torch.nn.Module):
    def __init__(
        self,
        range_min: tuple[float, ...],
        range_max: tuple[float, ...],
        n_bins: tuple[int, ...],
        device: torch.device,
        initial_value: float = 0.0,
    ):
        super().__init__()
        assert len(range_min) == len(range_max) == len(n_bins)
        self.device = device
        self.range_min = torch.tensor(range_min, device=device)
        self.range_max = torch.tensor(range_max, device=device)
        self.n_bins = torch.tensor(n_bins, device=device, dtype=torch.int32)
        self.stride = torch.flip(
            torch.cumprod(
                torch.tensor((1, *self.n_bins[1:]), device=device),
                dim=0,
            ),
            dims=(0,),
        )
        self.total_bins = torch.prod(self.n_bins).item()
        self.register_buffer("_sum", torch.zeros(self.total_bins, device=device))
        self.register_buffer("_oob_sum", torch.zeros([], device=device))
        self.reset(initial_value)

    def reset(self, initial_value: float = 0.0):
        self._sum.fill_(initial_value)
        self._oob_sum.zero_()

    def forward(self, values: torch.Tensor, weights: torch.Tensor | None = None):
        assert values.ndim == 2
        assert values.shape[1] == self.range_min.shape[0]
        if weights is None:
            weights = torch.ones(values.shape[0], device=values.device)
        assert weights.ndim == 1
        assert weights.shape[0] == values.shape[0]
        bin_index = torch.floor(
            (values - self.range_min)
            / (self.range_max - self.range_min)
            * self.n_bins.float()
        ).long()
        bin_index_is_valid = torch.all(
            (bin_index >= 0) & (bin_index < self.n_bins), dim=1
        )
        bin_index = torch.where(bin_index_is_valid[:, None], bin_index, 0)
        bin_index = torch.sum(bin_index * self.stride, dim=1)
        weights_valid = weights * bin_index_is_valid
        weights_oob = weights * ~bin_index_is_valid
        self._sum = torch.scatter_add(
            self._sum,
            dim=0,
            index=bin_index,
            src=weights_valid,
        )
        self._oob_sum += torch.sum(weights_oob)

    def result(self):
        """Normalized so that result.mean() == 1"""
        total = torch.sum(self._sum) + self._oob_sum
        return self.total_bins * (self._sum / total).view(*self.n_bins)

    def average_count(self):
        return self._sum.mean()

    def oob_fraction(self):
        return self._oob_sum / (self._sum.sum() + self._oob_sum)


# Distributions ###############################################################


def standard_normal_log_pdf(x: torch.Tensor):
    """Log-probability of the standard normal distribution.

    x: [*ldims, d]

    returns: [*ldims]
    """
    return -torch.sum(torch.square(x), dim=-1) / 2 - x.shape[-1] * np.log(2 * np.pi) / 2


class CheckerboardDistribution:
    """A 2D checkerboard distribution."""

    def __init__(self, num_blocks: int, range_: float):
        """Initialize a 2D checkerboard distribution.

        num_blocks: the checkerboard contains num_blocks x num_blocks blocks.
        Each block contains two white squares and two black squares.

        range_: The checkerboard range is [-range_, range_] x [-range_, range_].
        """
        self.num_blocks = num_blocks
        self.range_ = range_

    def sample(self, sample_size: int, generator: torch.Generator) -> torch.Tensor:
        """Sample from the checkerboard distribution.

        returns: [sample_size, 2]
        """
        i_block = torch.randint(
            0,
            self.num_blocks,
            (sample_size, 2),
            device=generator.device,
            generator=generator,
        )
        i_subblock = torch.randint(
            0, 2, (sample_size, 1), device=generator.device, generator=generator
        )
        dx = torch.rand((sample_size, 2), device=generator.device, generator=generator)
        x = 2 * (i_block + i_subblock * 0.5) + dx
        return (x / self.num_blocks - 1) * self.range_

    def range_min(self) -> tuple[float, float]:
        return -self.range_, -self.range_

    def range_max(self) -> tuple[float, float]:
        return self.range_, self.range_

    def get_pdf(self, num_bins_per_subblock: int, device: torch.device) -> torch.Tensor:
        """Get the exact checkerboard pdf, normalized so that pdf.mean() == 1.

        returns: [n_bins, n_bins] where
        n_bins = 2 * num_blocks * num_bins_per_subblock
        """
        n_bins = 2 * self.num_blocks * num_bins_per_subblock
        bin_index = torch.arange(n_bins, device=device)
        bin_index_x = bin_index[None, :].expand(n_bins, n_bins)
        bin_index_y = bin_index[:, None].expand(n_bins, n_bins)
        subblock_index_x = bin_index_x // num_bins_per_subblock
        subblock_index_y = bin_index_y // num_bins_per_subblock
        is_subblock_x_even = subblock_index_x % 2 == 0
        is_subblock_y_even = subblock_index_y % 2 == 0
        is_in_subblock = is_subblock_x_even == is_subblock_y_even
        return is_in_subblock.float() * 2

    def get_nll(self):
        """Perfect model NLL for samples from this distribution"""

        return np.log(2 * np.square(self.range_))

    def compute_relative_entropy(
        self,
        get_sample: Callable[[int, torch.Generator], torch.Tensor],
        n_bins_per_subblock: int,
        n_samples: int,
        batch_size: int,
    ) -> tuple[float, np.ndarray]:
        """Log to MLFlow the relative entropy of the sample distribution

        get_sample: (batch_size, generator) -> sample[batch_size, 2]

        n_bins_per_subblock: number of histogram bins per checkerboard subblock

        n_samples: number of samples to take

        batch_size: batch size

        returns:
            * relative_entropy
            * rgb_pdf: [n_bins, n_bins, 3], rgb visualization of the sample pdf
        """
        true_pdf = self.get_pdf(n_bins_per_subblock, device="cuda")
        range_min = self.range_min()
        range_max = self.range_max()
        sample_pdf = HistogramNd(
            range_min=range_min,
            range_max=range_max,
            n_bins=true_pdf.shape,
            device=torch.device("cuda"),
            initial_value=1.0,
        )

        generator = torch.Generator("cuda")
        n_batches = (n_samples + batch_size - 1) // batch_size
        for _ in range(n_batches):
            sample_pdf(get_sample(batch_size, generator))

        cmap = matplotlib.colormaps["inferno"]
        normalized_pdf = sample_pdf.result()
        normalized_pdf = torch.minimum(
            normalized_pdf / 4.0,
            torch.tensor(1.0, device=normalized_pdf.device),
        )
        normalized_pdf = normalized_pdf[None, None, :, :]
        normalized_pdf = torch.nn.functional.interpolate(
            normalized_pdf, scale_factor=8, mode="nearest-exact"
        )
        normalized_pdf = normalized_pdf.squeeze(0).squeeze(0)
        rgb_pdf = cmap(normalized_pdf.cpu().numpy())
        relative_entropy = compute_relative_entropy(true_pdf, sample_pdf.result())
        return (
            relative_entropy,
            sample_pdf.average_count(),
            sample_pdf.oob_fraction(),
            rgb_pdf,
        )

    def compute_and_log_relative_entropy(
        self,
        get_sample: Callable[[int, torch.Generator], torch.Tensor],
        n_bins_per_subblock: int,
        n_samples: int,
        batch_size: int,
        tag: str,
        step: int,
    ):
        """Log to MLFlow the relative entropy of the sample distribution

        get_sample: (batch_size, generator) -> sample[batch_size, 2]

        n_bins_per_subblock: number of histogram bins per checkerboard subblock

        n_samples: number of samples to take

        batch_size: batch size
        """
        logger.info("Computing relative entropy for %s at step %d", tag, step)
        relative_entropy, average_count, oob_fraction, rgb_pdf = (
            self.compute_relative_entropy(
                get_sample, n_bins_per_subblock, n_samples, batch_size
            )
        )
        mlflow.log_metric(
            f"relative_entropy_{tag}",
            relative_entropy,
            step=step,
        )
        mlflow.log_metric(
            f"relative_entropy_{tag}_average_count",
            average_count,
            step=step,
        )
        mlflow.log_metric(
            f"relative_entropy_{tag}_oob_fraction",
            oob_fraction,
            step=step,
        )
        mlflow.log_image(
            rgb_pdf,
            key=f"density_estimate_{tag}",
            step=step,
        )


def compute_relative_entropy(pdf_1: torch.Tensor, pdf_2: torch.Tensor):
    """AKA Kullback-Leibler divergence.

    pdf_1, pdf_2: PDFs normalized such that pdf.mean() == 1
    """
    assert pdf_1.shape == pdf_2.shape
    return torch.mean(torch.where(pdf_1 > 0, pdf_1 * torch.log(pdf_1 / pdf_2), 0.0))


# Models ######################################################################


class Mlp(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, inner_dim: int, n_layers: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.inner_dim = inner_dim
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(input_dim, inner_dim),
                    torch.nn.ReLU(),
                ),
            ]
            + [
                torch.nn.Sequential(
                    torch.nn.Linear(inner_dim, inner_dim),
                    torch.nn.ReLU(),
                )
                for _ in range(n_layers)
            ]
            + [torch.nn.Linear(inner_dim, output_dim)]
        )

    def forward(self, x: torch.Tensor):
        x = self.layers[0](x)
        for layer in self.layers[1:-1]:
            x = x + layer(x)
        return self.layers[-1](x)


class SimpleModel(torch.nn.Module):
    """A simple MLP for modeling diffusion in two dimensions."""

    def __init__(self, n_channels: int, n_layers: int):
        super().__init__()
        self.n_channels = n_channels
        self.n_layers = n_layers
        self.mlp = Mlp(3, 2, n_channels, n_layers)

    def forward(self, t: torch.Tensor, x: torch.Tensor):
        """Forward pass of the model.

        t: [batch_size], in the range [0, 1]
        x: [batch_size, 2]

        returns: [batch_size, 2]
        """
        return self.mlp(torch.cat([t[:, None], x], dim=1))

    def save(self, path: str):
        fs = get_fs(path)
        fs.makedirs(path, exist_ok=True)
        config = {
            "type": "SimpleModel",
            "version": 1,
            "n_channels": self.n_channels,
            "n_layers": self.n_layers,
        }
        with fs.open(os.path.join(path, "config.json"), "w") as f:
            f.write(json.dumps(config))
        with fs.open(os.path.join(path, "state_dict.pth"), "wb") as f:
            torch.save(self.state_dict(), f)

    @staticmethod
    def load(path: str):
        fs = get_fs(path)
        with fs.open(os.path.join(path, "config.json"), "r") as f:
            config = json.load(f)
        if config["type"] != "SimpleModel":
            raise ValueError(f"Wrong model type: {config['type']}")
        if config["version"] != 1:
            raise ValueError(f"Wrong model version: {config['version']}")
        model = SimpleModel(
            n_channels=config["n_channels"], n_layers=config["n_layers"]
        )
        with fs.open(os.path.join(path, "state_dict.pth"), "rb") as f:
            model.load_state_dict(torch.load(f, weights_only=True))
        return model


# Inductive moment matching ####################################################


@dataclasses.dataclass
class NoiseSchedule:
    alpha: Callable[[torch.Tensor], torch.Tensor]
    sigma: Callable[[torch.Tensor], torch.Tensor]
    time_range: tuple[float, float]

    @classmethod
    def flow_matching(cls):
        def alpha(t: torch.Tensor):
            return 1 - t

        def sigma(t: torch.Tensor):
            return t

        return cls(alpha=alpha, sigma=sigma, time_range=(0.0, 1.0))


def ddim_interpolate(
    x_t: torch.Tensor,
    x_0: torch.Tensor,
    s: torch.Tensor,
    t: torch.Tensor,
    noise_schedule: NoiseSchedule,
):
    """DDIM interpolation

    x_t: [*ldim, d] The conditional noisy point at time t
    x_0: [*ldim, d] the clean data point
    s: [*ldim] in [0, 1] the required interpolation time between 0 and t
    t: [*ldim] in [0, 1] the conditional interpolation time
    noise_schedule: the noise_schedule
    returns: [batch_size, d] the interpolated point at time s
    """
    assert x_0.ndim >= 2
    assert t.shape == s.shape
    assert x_0.shape == x_t.shape
    assert t.shape == x_t.shape[:-1]
    t = t.unsqueeze(-1)
    s = s.unsqueeze(-1)
    alpha_t = noise_schedule.alpha(t)
    alpha_s = noise_schedule.alpha(s)
    sigma_t = noise_schedule.sigma(t)
    sigma_s = noise_schedule.sigma(s)
    return (alpha_s - alpha_t * sigma_s / sigma_t) * x_0 + (sigma_s / sigma_t) * x_t


def diffusion_interpolate(
    x_t: torch.Tensor,
    x_0: torch.Tensor,
    s: torch.Tensor,
    t: torch.Tensor,
    noise_schedule: NoiseSchedule,
    generator: torch.Generator,
):
    assert x_0.ndim >= 2
    assert t.shape == s.shape
    assert x_0.shape == x_t.shape
    assert t.shape == x_t.shape[:-1]
    t = t.unsqueeze(-1)
    s = s.unsqueeze(-1)
    alpha_t = noise_schedule.alpha(t)
    alpha_s = noise_schedule.alpha(s)
    sigma_t = noise_schedule.sigma(t)
    sigma_s = noise_schedule.sigma(s)
    corr = alpha_t * sigma_s / (alpha_s * sigma_t)
    mean = alpha_s * x_0 + sigma_s * corr / sigma_t * (x_t - alpha_t * x_0)
    stddev = sigma_s * torch.sqrt(1 - torch.square(corr))
    return mean + stddev * torch.randn(
        x_0.shape, generator=generator, device=x_0.device
    )


class ImmModel(torch.nn.Module):
    def __init__(
        self, n_channels: int, n_layers: int, condition_on_s: bool, enforce_bc: bool
    ):
        super().__init__()
        self.condition_on_s = condition_on_s
        self.enforce_bc = enforce_bc
        self.mlp = Mlp(4, 2, n_channels, n_layers)

    def forward(
        self,
        x_t: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
    ):
        """Forward pass of the IMM model.

        x_t: [*ldim, d] the noisy point at time t
        s: [*ldim] the intermediate time
        t: [*ldim] the time of the noisy point
        returns: [*ldim, d] the sampled point at time = 0
        """
        assert x_t.ndim >= 2
        assert t.shape == x_t.shape[:-1]
        assert s.shape == x_t.shape[:-1]
        if not self.condition_on_s:
            s = torch.zeros_like(s)
        x_t_flat = x_t.reshape(-1, x_t.shape[-1])
        t_flat = t.reshape(-1, 1)
        s_flat = s.reshape(-1, 1)
        x_0_flat = self.mlp(torch.cat([t_flat, s_flat, x_t_flat], dim=1))
        if self.enforce_bc:
            x_0_flat = t_flat * x_0_flat + (1 - t_flat) * x_t_flat
        return x_0_flat.view(*x_t.shape[:-1], x_0_flat.shape[-1])


def imm_compute_loss(
    sample_a: torch.Tensor,
    sample_b: torch.Tensor,
    kernel: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
):
    """Compute the MMD loss between two samples.
    sample_a: [n_groups, group_size, d]
    sample_b: [n_groups, group_size, d]
    returns: scalar, the MMD loss
    """
    assert sample_a.ndim == 3
    assert sample_a.shape == sample_b.shape
    # [n_groups, group_size, group_size]
    k_aa = kernel(sample_a[:, :, None], sample_a[:, None, :])
    k_bb = kernel(sample_b[:, :, None], sample_b[:, None, :])
    k_ab = kernel(sample_a[:, :, None], sample_b[:, None, :])
    return torch.mean(k_aa) + torch.mean(k_bb) - 2 * torch.mean(k_ab)


class LaplacianKernel:
    def __init__(self, sigma: float):
        self.sigma = sigma

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        return torch.exp(-(x - y).norm(dim=-1) / self.sigma)


# ODE / SDE solvers ##########################################################


def compute_divergence_mc(
    f: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    n_samples: int,
    generator: torch.Generator,
):
    """Compute the divergence of a vector field.

    f: x -> v
        x: [*ldims, d]
        v: [*ldims, d]
    x: [*ldims, d]
    n_samples: number of samples to take
    generator: torch.Generator

    returns: [*ldims]: sum_i d_i f_i(x)

    The computation is approximate, with error O(1/sqrt(n_samples)).
    """
    x = x.detach()
    x.requires_grad = True
    n_dim = x.shape[-1]
    result = torch.zeros(x.shape[:-1], device=x.device)
    for i in range(n_samples):
        z = torch.randn(n_dim, device=x.device, generator=generator)
        z = z.expand_as(x)
        v = f(x)
        g = torch.autograd.grad(v, x, z)[0]
        result += torch.sum(z * g, dim=-1)
    return result / n_samples


def integrate_flow(
    velocity_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    t_start: torch.Tensor,
    t_end: torch.Tensor,
    x_start: torch.Tensor,
    n_steps: int,
    log_p_start: torch.Tensor | None = None,
    num_divergence_samples: int = 100,
    generator: torch.Generator = None,
    step_callback: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None], None
    ]
    | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """ltegrate flow ODE

    Integrates the ODE:

        dx/dt = v(t, x)

        d/dt log p(x) = -div v(t, x)

    using Euler's method.

    velocity_function: (t, x) -> v
        t: [batch_size] in [t_start, t_end]
        x: [batch_size, d]
        v: [batch_size, d]

    t_start: [batch_size] the start time

    t_end: [batch_size] the end time

    x_start: [batch_size, d] the initial vector

    n_steps: number integration steps

    log_p_start: [batch_size] the log-probability of the initial vector.
        If provided, the log-likelihood is computed.

    num_divergence_samples: number of samples to take for the divergence
        estimate. Only used if computing the log likelihood.

    generator: torch.Generator: Generator to estimate the divergence. Only
        used if computing the log likelihood.

    step_callback: (t, x_t, v_t, log_p_t) -> None
        t: [batch_size] the current time
        x_t: [batch_size, d] the current vector
        v_t: [batch_size, d] the current velocity
        log_p_t: [batch_size] the log-probability of the current vector.
            If None, the log-likelihood is not computed.

    returns:
        * [batch_size, d]: the final vector
        * [batch_size]: the log-likelihood of the final vector, if requested
          else None.
    """

    assert x_start.ndim == 2, x_start.shape
    batch_size, _ = x_start.shape
    assert t_start.shape == (batch_size,), (t_start.shape, batch_size)
    assert t_end.shape == (batch_size,), (t_end.shape, batch_size)

    if log_p_start is not None:
        if generator is None:
            raise ValueError("A generator must be provided if log_p_start is provided")
        log_p_t = log_p_start
    else:
        log_p_t = None

    dt = (t_end - t_start) / n_steps
    x_t = x_start
    for i in range(n_steps):
        t = t_start + i * dt
        with torch.no_grad():
            v_t = velocity_function(t, x_t)
            if v_t.shape != x_t.shape:
                raise ValueError(f"v_t.shape {v_t.shape} != x_t.shape {x_t.shape}")

        if log_p_t is not None:
            div_v = compute_divergence_mc(
                functools.partial(velocity_function, t),
                x_t,
                num_divergence_samples,
                generator,
            )
            assert div_v.shape == (batch_size,)
            log_p_t = log_p_t - div_v * dt

        x_t = x_t + dt[:, None] * v_t

        if step_callback is not None:
            step_callback(t, x_t, v_t, log_p_t)

    return x_t, log_p_t


def update_ema(
    ema_model: torch.nn.Module, model: torch.nn.Module, update_weight: float
):
    ema_params = dict(ema_model.named_parameters())
    model_params = dict(model.named_parameters())
    if not ema_params.keys() == model_params.keys():
        raise ValueError(
            "EMA model and model must have the same parameters:\n"
            f"EMA - model: {ema_params.keys() - model_params.keys()}\n"
            f"model - EMA: {model_params.keys() - ema_params.keys()}"
        )
    for key, ema_param in ema_params.items():
        param = model_params[key]
        if ema_param.data.shape != param.data.shape:
            raise ValueError(
                f"EMA parameter and model parameter shape mismatch for {key}:\n"
                f"EMA: {ema_param.data.shape}\n"
                f"model: {param.data.shape}"
            )
        ema_param.data.mul_(1.0 - update_weight)
        ema_param.data.add_(param.data * update_weight)


# Flow matching ###############################################################


def flow_matching_compute_loss(
    model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x_0: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """Compute the flow matching loss.

    model: (diffusion_time, noisy_signal) -> (estimate of the velocity field)
        diffusion_time: [batch_size], in the range [0, 1]
        noisy_signal: [batch_size, *dims]
        estimate of the velocity field: [batch_size, *dims]

    x_0: [batch_size, *dims] clean signal

    generator: torch.Generator

    returns: scalar, the flow matching loss
    """
    device = x_0.device
    t = torch.rand(x_0.shape[0], device=device, generator=generator)
    t_bc = t.unsqueeze(1).expand_as(x_0.view(x_0.shape[0], -1)).view_as(x_0)
    z = torch.randn(x_0.shape, generator=generator, device=device)
    x_t = (1 - t_bc) * x_0 + t_bc * z
    u = z - x_0
    v = model(t, x_t)
    return torch.mean(torch.square(u - v))


def flow_matching_sample(
    model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x_1: torch.Tensor,
    n_steps: int,
) -> torch.Tensor:
    x_0, _ = integrate_flow(
        velocity_function=model,
        t_start=torch.ones(x_1.shape[0], device=x_1.device),
        t_end=torch.zeros(x_1.shape[0], device=x_1.device),
        x_start=x_1,
        n_steps=n_steps,
    )
    return x_0


def flow_matching_compute_log_likelihood(
    model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x_0: torch.Tensor,
    n_steps: int,
    generator: torch.Generator,
    num_divergence_samples: int = 100,
) -> torch.Tensor:
    """Compute the log likelihood of the data

    model: (t, x) -> v
        t: [batch_size] in [0, 1]
        x: [batch_size, d]
        v: [batch_size, d]
    x_0: [batch_size, d] the sample from the data distribution
    n_steps: number integration steps
    generator: torch.Generator

    returns:
       * x_0_backward: [batch_size, d]: x_0 after forward and backward
         integration. This should equal x_0 up to integration error.
       * log_p_0: [batch_size] the log likelihood of x_0.
       * x_1: [batch_size, d]
       * log_p_1: [batch_size] the log likelihood of x_1.
    """
    x_1, _ = integrate_flow(
        velocity_function=model,
        t_start=torch.zeros(x_0.shape[0], device=x_0.device),
        t_end=torch.ones(x_0.shape[0], device=x_0.device),
        x_start=x_0,
        n_steps=n_steps,
        generator=generator,
    )
    log_p_1 = standard_normal_log_pdf(x_1)
    x_0_backward, log_p_0 = integrate_flow(
        velocity_function=model,
        t_start=torch.ones(x_1.shape[0], device=x_0.device),
        t_end=torch.zeros(x_1.shape[0], device=x_0.device),
        x_start=x_1,
        log_p_start=log_p_1,
        n_steps=n_steps,
        generator=generator,
        num_divergence_samples=num_divergence_samples,
    )
    return x_0_backward, log_p_0, x_1, log_p_1
