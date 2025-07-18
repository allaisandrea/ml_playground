import os
import functools
from typing import Any
import torch
import argparse
import logging
import mlflow
import playground
from playground import Timer

logger = logging.getLogger(__name__)


def get_sample(
    batch_size: int,
    generator: torch.Generator,
    n_integration_steps: int,
    model: playground.SimpleModel,
):
    return playground.flow_matching_sample(
        model=model,
        x_1=torch.randn(batch_size, 2, device=generator.device, generator=generator),
        n_steps=n_integration_steps,
    )


def compute_log_likelihood(
    model: torch.nn.Module,
    checkerboard: playground.CheckerboardDistribution,
    n_integration_steps: int,
    n_sample: int,
    batch_size: int,
    step: int,
) -> None:
    logger.info("Compute log likelihood: %d steps", n_integration_steps)
    generator = torch.Generator("cuda")
    n_batches = (n_sample + batch_size - 1) // batch_size
    log_p_0 = torch.zeros(tuple(), device=generator.device)
    integration_error = torch.zeros(tuple(), device=generator.device)
    for _ in range(n_batches):
        x_0 = checkerboard.sample(batch_size, generator)
        x_0_b, log_p_0_b, _, _ = playground.flow_matching_compute_log_likelihood(
            model=model,
            x_0=x_0,
            n_steps=n_integration_steps,
            generator=generator,
            num_divergence_samples=16,
        )
        log_p_0 += log_p_0_b.mean()
        integration_error += torch.norm(x_0 - x_0_b, dim=1).mean()
    log_p_0 = log_p_0 / n_batches
    integration_error = integration_error / n_batches
    mlflow.log_metric(
        f"nll_{n_integration_steps}_steps",
        -log_p_0.item(),
        step=step,
    )
    mlflow.log_metric(
        f"integration_error_{n_integration_steps}_steps",
        integration_error.item(),
        step=step,
    )


def main(
    tag: str,
    output_root: str,
    n_steps: int,
    batch_size: int,
    log_every: int,
    save_every: int,
    n_sample: int,
    compute_log_likelihood: bool,
    n_checkerboard_blocks: int,
    checkerboard_range: float,
    n_integration_steps: list[int],
    learning_rate: float,
    seed: int,
    args: dict[str, Any],
):
    run_name, output_path = playground.init_run(tag, output_root, logger)
    torch.manual_seed(seed)
    mlflow.log_params(args)
    checkerboard = playground.CheckerboardDistribution(
        num_blocks=n_checkerboard_blocks, range_=checkerboard_range
    )
    generator = torch.Generator("cuda")
    model = playground.SimpleModel(n_channels=1024, n_layers=4).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    step_timer = Timer()
    step_timer.start()
    for step in range(1, n_steps + 1):
        optimizer.zero_grad()
        x0 = checkerboard.sample(batch_size, generator)
        loss = playground.flow_matching_compute_loss(model, x0, generator)
        loss.backward()
        optimizer.step()
        if step % log_every == 0:
            step_timer.pause()
            sample_timer = Timer()
            sample_timer.start()
            logger.info("Compute metrics for step %d", step)
            for n_integration_steps_i in n_integration_steps:
                checkerboard.compute_and_log_relative_entropy(
                    get_sample=functools.partial(
                        get_sample,
                        n_integration_steps=n_integration_steps_i,
                        model=model,
                    ),
                    n_bins_per_subblock=5,
                    n_samples=n_sample,
                    batch_size=min(n_sample, 1 << 12),
                    tag=f"{n_integration_steps_i}_steps",
                    step=step,
                )
                if compute_log_likelihood:
                    compute_log_likelihood(
                        model,
                        checkerboard,
                        n_integration_steps=n_integration_steps_i,
                        n_sample=1 << 10,
                        batch_size=1 << 10,
                        step=step,
                    )
                    mlflow.log_metric("true_nll", checkerboard.get_nll(), step=step)
            sample_timer.stop()
            mlflow.log_metric("loss", loss.item(), step=step)
            mlflow.log_metric("step_time", step_timer.mean(), step=step)
            mlflow.log_metric("sample_time", sample_timer.mean(), step=step)
            mlflow.log_metric(
                "sample_time_amortized", sample_timer.mean() / log_every, step=step
            )
            logger.info("Step %d, loss %f", step, loss.item())
            step_timer.reset()
            step_timer.start()
        if step % save_every == 0:
            logger.info("Saving checkpoint at step %d", step)
            model.save(os.path.join(output_path, f"checkpoint_step_{step}"))
        step_timer.split()

    logger.info("Saving final checkpoint")
    model.save(os.path.join(output_path, "final_model"))
    mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument(
        "--output-root",
        type=str,
        default=playground.DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument("--n-steps", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--log-every", type=int, default=10_000)
    parser.add_argument("--save-every", type=int, default=20_000)
    parser.add_argument("--n-sample", type=int, default=100_000)
    parser.add_argument(
        "--compute-log-likelihood", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--n-checkerboard-blocks", type=int, default=4)
    parser.add_argument("--checkerboard-range", type=float, default=4.0)
    parser.add_argument("--n-integration-steps", type=str, default="50")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = vars(parser.parse_args())
    args["n_integration_steps"] = [
        int(x) for x in args["n_integration_steps"].split(",")
    ]
    main(**args, args=args)
