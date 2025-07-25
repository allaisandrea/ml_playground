from typing import Any
import os
import functools
import copy
import argparse
import logging
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
    model: playground.SimpleModel,
):
    t_steps = np.power(1 - np.arange(n_sample_steps) / n_sample_steps, power)
    x_0 = torch.zeros((batch_size, 2), device=generator.device)
    for t_value in t_steps:
        t = torch.full((batch_size,), t_value, device=generator.device)
        x_1 = torch.randn(batch_size, 2, device=generator.device, generator=generator)
        x_t = t[:, None] * x_1 + (1 - t[:, None]) * x_0
        x_0 = model(t, x_t)
    return x_0


class WithConsistencyBoundaryCondition(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.wrapped = model

    def forward(self, t: torch.Tensor, x: torch.Tensor):
        return t[:, None] * self.wrapped(t, x) + (1 - t[:, None]) * x

    def get_save_dict(self):
        return self.wrapped.get_save_dict()

    @staticmethod
    def from_save_dict(save_dict: dict):
        return WithConsistencyBoundaryCondition(
            playground.SimpleModel.from_save_dict(save_dict)
        )


def main(
    tag: str,
    source_model_path: str | None,
    output_root: str,
    n_steps: int,
    batch_size: int,
    log_every: int,
    save_every: int,
    n_sample: int,
    n_sample_steps: list[int],
    sample_step_progression_power: float,
    dt: float,
    learning_rate: float,
    ema_update_weight: float,
    seed: int,
    n_checkerboard_blocks: int,
    checkerboard_range: float,
    args: dict[str, Any],
):
    run_name, output_path = playground.init_run(tag, output_root, logger)
    torch.manual_seed(seed)
    mlflow.log_params(args)
    if source_model_path is not None:
        source_model = playground.SimpleModel.load(source_model_path).cuda()
    else:
        source_model = None
    checkerboard = playground.CheckerboardDistribution(
        num_blocks=n_checkerboard_blocks, range_=checkerboard_range
    )
    generator = torch.Generator("cuda")
    model = WithConsistencyBoundaryCondition(
        playground.SimpleModel(n_channels=1024, n_layers=4)
    ).cuda()
    model_ema = copy.deepcopy(model)
    model_ema.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    step_timer = Timer()
    step_timer.start()
    for step in range(1, n_steps + 1):
        optimizer.zero_grad()
        x_0 = checkerboard.sample(batch_size, generator)
        x_1 = torch.randn(batch_size, 2, device=generator.device, generator=generator)
        t1 = (1 - dt) * torch.rand(
            batch_size, device=generator.device, generator=generator
        )
        t2 = torch.minimum(t1 + dt, torch.ones_like(t1))
        x_t2 = t2[:, None] * x_1 + (1 - t2[:, None]) * x_0
        if source_model is None:
            x_t1 = t1[:, None] * x_1 + (1 - t1[:, None]) * x_0
        else:
            v_t2 = source_model(t2, x_t2)
            x_t1 = x_t2 - v_t2 * dt
        x_0_pred = model(t2, x_t2)
        x_0_ema = model_ema(t1, x_t1)
        loss = torch.nn.functional.mse_loss(x_0_pred, x_0_ema)
        loss.backward()
        optimizer.step()
        playground.update_ema(model_ema, model, ema_update_weight)
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
                        model=model,
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
        if step % save_every == 0 or step == n_steps:
            logger.info("Saving checkpoint at step %d", step)
            playground.save_checkpoint(
                os.path.join(output_path, f"checkpoint_step_{step}.pth"),
                step,
                model,
                optimizer,
                generator,
            )
        step_timer.split()

    logger.info("Saving final checkpoint")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--source-model", type=str, dest="source_model_path")
    parser.add_argument(
        "--output-root",
        type=str,
        default=playground.DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument("--n-steps", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--save-every", type=int, default=20_000)
    parser.add_argument("--log-every", type=int, default=10_000)
    parser.add_argument("--n-sample", type=int, default=100_000)
    parser.add_argument("--n-sample-steps", type=str, default="1,2,3,5")
    parser.add_argument("--sample-step-progression-power", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--ema-update-weight", type=float, default=1.0e-2)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-checkerboard-blocks", type=int, default=4)
    parser.add_argument("--checkerboard-range", type=float, default=4.0)
    args = vars(parser.parse_args())
    args["n_sample_steps"] = [
        int(n_sample_steps_i) for n_sample_steps_i in args["n_sample_steps"].split(",")
    ]
    main(**args, args=args)
