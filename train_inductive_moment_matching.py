from typing import Any
import os
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
    constant_noise: bool,
    model: playground.ImmModel,
    noise_schedule: playground.NoiseSchedule,
):
    t_steps = np.power(1 - np.arange(n_sample_steps + 1) / n_sample_steps, power)
    x_t = torch.randn(batch_size, 2, device=generator.device, generator=generator)
    # t1_value < t2_value
    for t1_value, t2_value in zip(t_steps[1:], t_steps[:-1]):
        t1 = torch.full((batch_size,), t1_value, device=generator.device)
        t2 = torch.full((batch_size,), t2_value, device=generator.device)
        x_0 = model(x_t, t1, t2)
        if constant_noise:
            x_t = playground.ddim_interpolate(x_t, x_0, t1, t2, noise_schedule)
        else:
            x_1 = torch.randn(
                batch_size, 2, device=generator.device, generator=generator
            )
            x_t = playground.ddim_interpolate(
                x_1, x_0, t1, torch.ones_like(t1), noise_schedule
            )
    return x_t


def main(
    tag: str,
    output_root: str,
    start_from_checkpoint: str | None,
    n_steps: int,
    batch_size: int,
    log_every: int,
    save_every: int,
    n_sample: int,
    n_sample_steps: list[int],
    sample_step_progression_power: float,
    learning_rate: float,
    seed: int,
    n_particles: int,
    kernel_radius: float,
    dt: float,
    condition_on_s: bool,
    enforce_bc: bool,
    match_at_zero: bool,
    mse_loss: bool,
    sample_with_constant_noise: bool,
    use_diffusion_interpolant: bool,
    n_checkerboard_blocks: int,
    checkerboard_range: float,
    mlflow_local_path: str | None,
    args: dict[str, Any],
):
    if batch_size % n_particles != 0:
        raise ValueError(
            f"batch_size ({batch_size}) must be divisible by n_particles ({n_particles})"
        )
    group_size = batch_size // n_particles
    torch.manual_seed(seed)
    run_name, output_path = playground.init_run(
        tag, output_root, logger, mlflow_local_path
    )
    mlflow.log_params(args)

    noise_schedule = playground.NoiseSchedule.flow_matching()
    checkerboard = playground.CheckerboardDistribution(
        num_blocks=n_checkerboard_blocks, range_=checkerboard_range
    )
    if start_from_checkpoint is not None:
        starting_step, model, optimizer, generator = playground.load_checkpoint(
            start_from_checkpoint,
            playground.ImmModel.from_save_dict,
            device=torch.device("cuda"),
        )
    else:
        starting_step = 0
        model = playground.ImmModel(
            n_channels=1024,
            n_layers=4,
            condition_on_s=condition_on_s,
            enforce_bc=enforce_bc,
        )
        model = model.cuda()
        generator = torch.Generator("cuda")
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    step_timer = Timer()
    step_timer.start()
    for step in range(starting_step + 1, n_steps + 1):
        optimizer.zero_grad()
        r = (1 - dt) * torch.rand(
            (group_size, 1), device=generator.device, generator=generator
        ).expand(group_size, n_particles)
        r = torch.maximum(r, torch.full_like(r, 0.01 * dt))
        t = r + dt
        if match_at_zero:
            s = torch.zeros_like(r)
        else:
            s = r * torch.rand(
                (group_size, 1), device=generator.device, generator=generator
            ).expand(group_size, n_particles)
        x_0 = checkerboard.sample(group_size * n_particles, generator).reshape(
            group_size, n_particles, 2
        )
        x_1 = torch.randn(
            (group_size, n_particles, 2), device=generator.device, generator=generator
        )
        x_r = playground.ddim_interpolate(
            x_1, x_0, r, torch.ones_like(r), noise_schedule
        )
        x_t = playground.ddim_interpolate(
            x_1, x_0, t, torch.ones_like(t), noise_schedule
        )
        with torch.no_grad():
            x_0a = model(x_r, s, r)
        x_0b = model(x_t, s, t)
        if use_diffusion_interpolant:
            x_sa = playground.diffusion_interpolate(
                x_r, x_0a, s, r, noise_schedule, generator
            )
            x_sb = playground.diffusion_interpolate(
                x_t, x_0b, s, t, noise_schedule, generator
            )
        else:
            x_sa = playground.ddim_interpolate(x_r, x_0a, s, r, noise_schedule)
            x_sb = playground.ddim_interpolate(x_t, x_0b, s, t, noise_schedule)

        if mse_loss:
            loss = torch.nn.functional.mse_loss(x_sa, x_sb)
        else:
            loss = playground.imm_compute_loss(
                x_sa, x_sb, playground.LaplacianKernel(sigma=kernel_radius)
            )
        loss.backward()
        optimizer.step()

        if step % log_every == 0:
            step_timer.pause()
            sample_timer = Timer()
            sample_timer.start()
            logger.info("Compute metrics for step %d", step)
            for n_sample_steps_i in n_sample_steps:
                checkerboard.compute_and_log_relative_entropy(
                    get_sample=functools.partial(
                        get_sample,
                        n_sample_steps=n_sample_steps_i,
                        power=sample_step_progression_power,
                        constant_noise=sample_with_constant_noise,
                        model=model,
                        noise_schedule=noise_schedule,
                    ),
                    n_bins_per_subblock=5,
                    n_samples=n_sample,
                    batch_size=min(n_sample, 1 << 12),
                    tag=f"{n_sample_steps_i}_steps",
                    step=step,
                )
            sample_timer.stop()
            mlflow.log_metric("loss", loss.item(), step=step)
            mlflow.log_metric("step_time", step_timer.mean(), step=step)
            mlflow.log_metric("sample_time", sample_timer.mean(), step=step)
            mlflow.log_metric(
                "sample_time_amortized", sample_timer.mean() / log_every, step=step
            )
            step_timer.reset()
            step_timer.start()
            logger.info("Step %d, loss %f", step, loss.item())
        if step % save_every == 0:
            logger.info("Saving checkpoint at step %d", step)
            playground.save_checkpoint(
                os.path.join(output_path, f"checkpoint_step_{step}.pth"),
                step,
                model,
                optimizer,
                generator,
            )
        step_timer.split()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument(
        "--output-root",
        type=str,
        default=playground.DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument("--start-from-checkpoint", type=str, default=None)
    parser.add_argument("--n-steps", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--log-every", type=int, default=10_000)
    parser.add_argument("--save-every", type=int, default=20_000)
    parser.add_argument("--n-sample", type=int, default=100_000)
    parser.add_argument("--n-sample-steps", type=str, default="1,2,3,5")
    parser.add_argument("--sample-step-progression-power", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-particles", type=int, default=4)
    parser.add_argument("--kernel-radius", type=float, default=4.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--n-checkerboard-blocks", type=int, default=2)
    parser.add_argument("--checkerboard-range", type=float, default=4.0)
    parser.add_argument(
        "--condition-on-s", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--enforce-bc", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--match-at-zero", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--mse-loss", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--sample-with-constant-noise",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--use-diffusion-interpolant",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--mlflow-local-path", type=str, default=None)
    args = vars(parser.parse_args())
    args["n_sample_steps"] = [int(x) for x in args["n_sample_steps"].split(",")]
    main(**args, args=args)
