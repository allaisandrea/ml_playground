from typing import Any
import argparse
import logging
import functools
import torch
import mlflow
import numpy as np
import playground
from playground import Timer

logger = logging.getLogger(__name__)


def get_sample(
    batch_size: int,
    generator: torch.Generator,
    n_sample_steps: int,
    power: float,
    model: playground.ImmModel,
    noise_schedule: playground.NoiseSchedule,
):
    t_steps = np.power(1 - np.arange(n_sample_steps + 1) / n_sample_steps, power)
    x_0 = torch.zeros((batch_size, 2), device=generator.device)
    # t1_value < t2_value, t2_value: [1, 0.9,..., 0.1]
    for t1_value, t2_value in zip(t_steps[1:], t_steps[:-1]):
        t1 = torch.full((batch_size,), t1_value, device=generator.device)
        t2 = torch.full((batch_size,), t2_value, device=generator.device)
        x_1 = torch.randn(batch_size, 2, device=generator.device, generator=generator)
        x_t = (1 - t2[:, None]) * x_0 + t2[:, None] * x_1
        x_0 = model(x_t, t1, t2, noise_schedule)
    return x_0


def main(
    tag: str,
    output_root: str,
    n_steps: int,
    batch_size: int,
    log_every: int,
    n_sample: int,
    learning_rate: float,
    n_particles: int,
    dt: float,
    mlflow_local_path: str | None,
    match_at: str,
    loss_type: str,
    kernel_radius: float,
    condition_on_s: bool,
    args: dict[str, Any],
):
    if batch_size % n_particles != 0:
        raise ValueError(
            f"batch_size ({batch_size}) must be divisible by n_particles ({n_particles})"
        )
    group_size = batch_size // n_particles
    run_name, output_path = playground.init_run(
        tag, output_root, logger, mlflow_local_path
    )
    mlflow.log_params(args)

    noise_schedule = playground.NoiseSchedule.flow_matching()
    checkerboard = playground.CheckerboardDistribution(num_blocks=2, range_=4.0)
    model = playground.ImmModel(
        n_channels=1024, n_layers=4, condition_on_s=condition_on_s
    ).cuda()

    generator = torch.Generator("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    step_timer = Timer()
    sample_timer = Timer()
    step_timer.start()
    for step in range(1, n_steps + 1):
        optimizer.zero_grad()
        x_0 = checkerboard.sample(group_size * n_particles, generator).reshape(
            group_size, n_particles, 2
        )
        x_1 = torch.randn(group_size, n_particles, 2, device=generator.device, generator=generator)
        r = (1 - dt) * torch.rand(
            (group_size, 1), device=generator.device, generator=generator
        ).expand(group_size, n_particles)
        t = torch.minimum(r + dt, torch.ones_like(r))
        x_r = r[..., None] * x_1 + (1 - r[..., None]) * x_0
        x_t = t[..., None] * x_1 + (1 - t[..., None]) * x_0
        if match_at == "s":
            s = r * torch.rand(
                (group_size, 1), device=generator.device, generator=generator
            ).expand(group_size, n_particles)
        elif match_at == "0":
            s = torch.zeros_like(r)
        else:
            raise ValueError(f"Invalid match_at: {match_at}")
        with torch.no_grad():
            x_0_target = model(x_r, s, r, noise_schedule)
        x_0_pred = model(x_t, s, t, noise_schedule)

        x_s_target = s[..., None] * x_1 + (1 - s[..., None]) * x_0_target
        x_s_pred = s[..., None] * x_1 + (1 - s[..., None]) * x_0_pred
        if loss_type == "mse":
            loss = torch.nn.functional.mse_loss(x_s_pred, x_s_target)
        elif loss_type == "mmd":
            loss = playground.imm_compute_loss(
                x_s_target, x_s_pred, playground.LaplacianKernel(sigma=kernel_radius)
            )
        else:
            raise ValueError(f"Invalid loss_type: {loss_type}")
        loss.backward()
        optimizer.step()

        sample_timer.start()
        if step % log_every == 0:
            logger.info("Compute metrics for step %d", step)
            for n_sample_steps in [1, 2, 3, 4]:
                checkerboard.compute_and_log_relative_entropy(
                    get_sample=functools.partial(
                        get_sample,
                        n_sample_steps=n_sample_steps,
                        power=2.0,
                        model=model,
                        noise_schedule=noise_schedule,
                    ),
                    n_bins_per_subblock=5,
                    n_samples=n_sample,
                    batch_size=min(n_sample, 1 << 12),
                    tag=f"{n_sample_steps}_steps",
                    step=step,
                )
            mlflow.log_metric("loss", loss.item(), step=step)
            mlflow.log_metric("step_time", step_timer.mean(), step=step)
            mlflow.log_metric("sample_time", sample_timer.mean(), step=step)
            step_timer.reset()
            sample_timer.reset()
            logger.info("Step %d, loss %f", step, loss.item())
        sample_timer.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument(
        "--output-root",
        type=str,
        default=playground.DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument("--n-steps", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--n-sample", type=int, default=100_000)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--n-particles", type=int, default=16)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--mlflow-local-path", type=str, default=None)
    parser.add_argument("--match-at", type=str, default="s")
    parser.add_argument("--loss-type", type=str, default="mmd")
    parser.add_argument("--kernel-radius", type=float, default=4.0)
    parser.add_argument("--condition-on-s", action="store_true")
    args = vars(parser.parse_args())
    main(**args, args=args)
