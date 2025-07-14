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
    x_t = torch.randn(batch_size, 2, device=generator.device, generator=generator)
    # t1_value < t2_value
    for t1_value, t2_value in zip(t_steps[1:], t_steps[:-1]):
        t1 = torch.full((batch_size,), t1_value, device=generator.device)
        t2 = torch.full((batch_size,), t2_value, device=generator.device)
        x_t = model(x_t, t1, t2, noise_schedule)
    return x_t


def main(
    tag: str,
    output_root: str,
    n_steps: int,
    batch_size: int,
    log_every: int,
    n_sample: int,
    learning_rate: float,
    n_particles: int,
    kernel_size: float,
    dt: float,
    mlflow_local_path: str | None,
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
    model = playground.ImmModel(n_channels=1024, n_layers=4).cuda()

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
        r = (1 - dt) * torch.rand(
            (group_size, 1), device=generator.device, generator=generator
        ).expand(group_size, n_particles)
        t = r + dt
        s = r * torch.rand(
            (group_size, 1), device=generator.device, generator=generator
        ).expand(group_size, n_particles)
        x_t = playground.sample_from_diffusion_process(
            noise_schedule, x_0, t, generator
        )
        x_r = playground.ddim_interpolate(x_t, x_0, r, t, noise_schedule)
        with torch.no_grad():
            x_sa = model(x_r, s, r, noise_schedule)
        x_sb = model(x_t, s, t, noise_schedule)
        loss = playground.imm_compute_loss(
            x_sa, x_sb, playground.LaplacianKernel(sigma=kernel_size)
        )
        loss.backward()
        optimizer.step()

        sample_timer.start()
        if step % log_every == 0:
            logger.info("Compute metrics for step %d", step)
            for n_sample_steps in [1, 2, 3]:
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
    parser.add_argument("--n-particles", type=int, default=4)
    parser.add_argument("--kernel-size", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--mlflow-local-path", type=str, default=None)
    args = vars(parser.parse_args())
    main(**args, args=args)
