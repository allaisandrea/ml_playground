import os
import tempfile
import torch
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
import playground


logger = logging.getLogger("diffusion_playground_test")


def plot_checkerboard_velocity_field(output_path: str):
    """Estimate and plot the velocity field that maps the uniform
    distribution to the playground checkerboard distribution.
    """
    logger.info("Plotting velocity field")
    device = torch.device("cuda")
    n_batches = 10
    batch_size = 1 << 20
    checkerboard = playground.CheckerboardDistribution(num_blocks=3, range_=1.0)

    # Sample from the checkerboard distribution, and aggregate the
    # contributions to the velocity field in a 3D histogram, with
    # dimensions (diffusion_time, x, y). Each cell of the histogram
    # contains (count, sum(vx), sum(vy)), and can be used to compute
    # the mean velocity vector in that bin.
    generator = torch.Generator(device=device)
    hist_range = torch.tensor([[0.0, -1.0, -1.0], [1.0, 1.0, 1.0]], device=device)
    n_bins = torch.tensor([100, 20, 100], device=device)
    count = torch.zeros(torch.prod(n_bins), device=generator.device)
    sum = torch.zeros((torch.prod(n_bins), 2), device=generator.device)
    for i in range(n_batches):
        logger.info(f"Batch {i + 1} of {n_batches}")
        t = torch.rand((batch_size,), generator=generator, device=device)
        z = 0.25 * torch.randn((batch_size, 2), generator=generator, device=device)
        x0 = checkerboard.sample(batch_size, generator)
        x_t = (1 - t[:, None]) * x0 + t[:, None] * z
        u = z - x0
        p = torch.concatenate([t[:, None], x_t], dim=1)
        hist_index = torch.floor(
            (p - hist_range[0]) / (hist_range[1] - hist_range[0]) * n_bins
        ).to(torch.int64)
        index_is_valid = torch.all((hist_index >= 0) & (hist_index < n_bins), dim=1)
        hist_index = torch.where(
            index_is_valid[:, None], hist_index, torch.zeros_like(hist_index)
        )
        hist_index = (hist_index[:, 0] * n_bins[1] + hist_index[:, 1]) * n_bins[
            2
        ] + hist_index[:, 2]
        count = count.scatter_add(0, hist_index, index_is_valid.to(torch.float32))
        sum = sum.scatter_add(
            0,
            hist_index[:, None].expand(-1, 2),
            torch.where(index_is_valid[:, None], u, torch.zeros_like(u)),
        )

    mean = torch.where(count[:, None] > 1, sum / count[:, None], 0)
    mean = mean.view(n_bins[0], n_bins[1], n_bins[2], 2)
    count = count.view(n_bins[0], n_bins[1], n_bins[2])
    logger.info(f"Mean count per bin: {torch.mean(count).item()}")

    count = count.cpu().numpy()
    mean = mean.cpu().numpy()
    hist_range = hist_range.cpu().numpy()
    n_bins = n_bins.cpu().numpy()

    def linspace(start, end, n):
        return np.array((np.arange(n) + 0.5) / n * (end - start) + start)

    x_plot = linspace(hist_range[0][1], hist_range[1][1], n_bins[1])
    for i in range(n_bins[1]):
        logger.info(f"Plotting velocity field slice {i + 1} of {n_bins[1]}")
        figure, axes = plt.subplots(1, 1, figsize=(12, 6), dpi=150)
        t_plot = linspace(hist_range[0][0], hist_range[1][0], n_bins[0])
        y_plot = linspace(hist_range[0][2], hist_range[1][2], n_bins[2])
        axes.imshow(
            np.transpose(count[:, i]),
            extent=(
                hist_range[0][0],
                hist_range[1][0],
                hist_range[0][2],
                hist_range[1][2],
            ),
            origin="lower",
        )
        axes.streamplot(
            t_plot,
            y_plot,
            -np.full((n_bins[2], n_bins[0]), 1),
            -np.transpose(mean[:, i, :, 1]),
            color="white",
        )
        axes.set_aspect(0.25)
        axes.set_xlabel("Diffusion time")
        axes.set_ylabel("Coordinate y")
        axes.set_title(f"Coordinate x = {x_plot[i]:.2f}")
        figure.tight_layout()
        figure.savefig(f"{output_path}/velocity_field_x_{i:03d}.png")
        plt.close(figure)


def plot_test_histogram(output_path: str):
    """A simple test to check the histogram implementation."""
    logger.info("Plotting test histogram")
    n_samples = 1 << 20
    histogram = playground.HistogramNd(
        range_min=(-1.0, -2.0),
        range_max=(1.0, 2.0),
        n_bins=(10, 20),
        device=torch.device("cpu"),
        initial_value=0.0,
    )
    rng = torch.Generator()
    rng.manual_seed(0)
    values = torch.randn(n_samples, 2, generator=rng, device=histogram.device)
    weights = torch.ones(n_samples, device=histogram.device)
    histogram(values, weights=weights)
    figure, axes = plt.subplots(1, 1)
    axes.imshow(histogram.result().cpu().numpy())
    figure.savefig(f"{output_path}/test_histogram.png")
    plt.close(figure)


def plot_checkerboard_pdf(output_path: str):
    """Compare the exact checkerboard pdf with an histogram estimate."""

    logger.info("Plotting checkerboard pdf")
    device = torch.device("cuda")
    checkerboard = playground.CheckerboardDistribution(num_blocks=4, range_=1.0)
    pdf = checkerboard.get_pdf(num_bins_per_subblock=5, device=device)
    range_min = checkerboard.range_min()
    range_max = checkerboard.range_max()
    logger.info(f"Pdf mean: {torch.mean(pdf)}")

    sample_size = 1 << 20
    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    sample = checkerboard.sample(sample_size, generator)
    histogram = playground.HistogramNd(
        range_min=range_min,
        range_max=range_max,
        n_bins=pdf.shape,
        device=device,
        initial_value=1.0,
    )
    histogram(sample)
    pdf_sample = histogram.result()

    logger.info(
        f"Relative entropy: {playground.compute_relative_entropy(pdf, pdf_sample).item()}"
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = axes[0].imshow(
        pdf.cpu().numpy(),
        extent=(range_min[0], range_max[0], range_min[1], range_max[1]),
        origin="lower",
    )
    figure.colorbar(colors)
    colors = axes[1].imshow(
        histogram.result().cpu().numpy(),
        extent=(range_min[0], range_max[0], range_min[1], range_max[1]),
        origin="lower",
    )
    figure.colorbar(colors)
    figure.tight_layout()
    figure.savefig(f"{output_path}/checkerboard_pdf.png")
    plt.close(figure)


def test_compute_divergence(output_path: str):
    """Test the compute_divergence function."""
    logger.info("Testing compute_divergence")
    n_dim = 3
    n_samples = 1000
    device = torch.device("cuda")
    for n_dim in [2, 3]:
        x = torch.randn((10, n_dim), device=device, requires_grad=True)
        rng = torch.Generator(device=device)
        f = lambda x: x / torch.norm(x, dim=-1, keepdim=True)
        divergence = playground.compute_divergence_mc(f, x, n_samples, rng)
        divergence_exact = (n_dim - 1) / torch.norm(x, dim=-1)
        context = (
            f"n_dim={n_dim}\nn_samples={n_samples}\ndivergence={divergence}\n"
            f"divergence_exact={divergence_exact}\nratio={divergence / divergence_exact}"
        )
        assert torch.all(
            abs(divergence / divergence_exact - 1) < 10 / np.sqrt(n_samples)
        ), context


def test_integrate_flow(output_path: str):
    """Test the integrate_flow function."""
    logger.info("Testing integrate_flow")

    device = torch.device("cuda")
    generator = torch.Generator(device=device)
    x_start = torch.linspace(-3, 3, 16, device=device).unsqueeze(1)
    log_p_start = playground.standard_normal_log_pdf(x_start)

    def velocity_function(t, x):
        # x(t) = (2 - t) * x_1 = (1 - t / 2) * x0
        # x'(t) = - x1 = x(t) / (t - 2)
        return x / (t.unsqueeze(1) - 2)

    x_end, log_p_end = playground.integrate_flow(
        velocity_function=velocity_function,
        t_start=torch.ones(x_start.shape[0], device=device),
        t_end=torch.zeros(x_start.shape[0], device=device),
        x_start=x_start,
        n_steps=100,
        log_p_start=log_p_start,
        generator=generator,
    )
    x_end_expected = 2 * x_start
    log_p_end_expected = log_p_start - np.log(2)

    context = (
        f"x_start={x_start}\nlog_p_start={log_p_start}\n"
        f"x_end={x_end}\nlog_p_end={log_p_end}\n"
        f"x_end_expected={x_end_expected}\nlog_p_end_expected={log_p_end_expected}\n"
        f"x_diff={torch.norm(x_end - x_end_expected)}\n"
        f"log_p_diff={torch.norm(log_p_end - log_p_end_expected)}\n"
    )
    assert torch.allclose(x_end, x_end_expected), context
    assert torch.allclose(log_p_end, log_p_end_expected, rtol=1e-2), context


def test_flow_matching_compute_log_likelihood(output_path: str):
    logger.info("Testing flow_matching_compute_log_likelihood")

    device = torch.device("cuda")
    generator = torch.Generator(device=device)
    x_0 = torch.linspace(-3, 3, 16, device=device).unsqueeze(1)
    x_1_expected = x_0 / 2
    log_p_1_expected = playground.standard_normal_log_pdf(x_1_expected)
    log_p_0_expected = log_p_1_expected - np.log(2)

    def model(t, x):
        # x(t) = (2 - t) * x_1 = (1 - t / 2) * x0
        # x'(t) = - x1 = x(t) / (t - 2)
        return x / (t.unsqueeze(1) - 2)

    x_0_backward, log_p_0, x_1, log_p_1 = (
        playground.flow_matching_compute_log_likelihood(
            model=model,
            x_0=x_0,
            n_steps=100,
            generator=generator,
        )
    )

    if not torch.allclose(x_0, x_0_backward):
        logger.error(
            f"\nx_0: {x_0.unsqueeze(1).cpu().numpy()}\n"
            f"x_0_backward: {x_0_backward.unsqueeze(1).cpu().numpy()}\n"
            f"x0 - x_0_backward: {(x_0 - x_0_backward).unsqueeze(1).cpu().numpy()}\n"
        )
        raise ValueError()
    if not torch.allclose(x_1, x_1_expected):
        logger.error(
            f"\nx_1: {x_1.unsqueeze(1).cpu().numpy()}\n"
            f"x_1_expected: {x_1_expected.unsqueeze(1).cpu().numpy()}\n"
            f"x_1 - x_1_expected: {(x_1 - x_1_expected).unsqueeze(1).cpu().numpy()}\n"
        )
        raise ValueError()
    if not torch.allclose(log_p_1, log_p_1_expected):
        logger.error(
            f"\nlog_p_1: {log_p_1.cpu().numpy()}\n"
            f"log_p_1_expected: {log_p_1_expected.cpu().numpy()}\n"
            f"log_p_1 - log_p_1_expected: {(log_p_1 - log_p_1_expected).cpu().numpy()}\n"
        )
        raise ValueError()
    if not torch.allclose(log_p_0, log_p_0_expected, rtol=1e-2):
        logger.error(
            f"\nlog_p_0: {log_p_0.cpu().numpy()}\n"
            f"log_p_0_expected: {log_p_0_expected.cpu().numpy()}\n"
            f"log_p_0 - log_p_0_expected: {(log_p_0 - log_p_0_expected).cpu().numpy()}\n"
        )
        raise ValueError()


def test_simple_model_io(output_path: str):
    logger.info("Testing simple model io")
    model = playground.SimpleModel(n_channels=16, n_layers=4)
    with tempfile.TemporaryDirectory() as temp_dir:
        model.save(temp_dir)
        model_loaded = playground.SimpleModel.load(temp_dir)
    assert model.state_dict().keys() == model_loaded.state_dict().keys()
    for key in model.state_dict().keys():
        assert torch.all(model.state_dict()[key] == model_loaded.state_dict()[key])


def test_all(output_path: str):
    for task in DISPATCH.keys() - {"all"}:
        DISPATCH[task](output_path)


DISPATCH = {
    "all": test_all,
    "velocity_field": plot_checkerboard_velocity_field,
    "test_histogram": plot_test_histogram,
    "checkerboard_pdf": plot_checkerboard_pdf,
    "compute_divergence": test_compute_divergence,
    "integrate_flow": test_integrate_flow,
    "flow_matching_compute_log_likelihood": test_flow_matching_compute_log_likelihood,
    "simple_model_io": test_simple_model_io,
}


def main(task: str, output_path: str):
    logging.basicConfig(level=logging.INFO)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    DISPATCH[task](output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "task",
        type=str,
        choices=list(DISPATCH.keys()),
    )
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()
    main(**vars(args))
