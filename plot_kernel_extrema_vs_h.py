import argparse

import matplotlib.pyplot as plt
import numpy as np

from plot_kernel import wp_adj_cubic_kernel_derivative, wp_eval_cubic_kernel


def evaluate_extrema_for_h(h, sample_count):
    x_vals = np.linspace(0.0, h, sample_count, dtype=np.float32)

    kernel_vals = wp_eval_cubic_kernel(x_vals, h)
    kernel_max_idx = int(np.argmax(kernel_vals))

    deriv_vals, _ = wp_adj_cubic_kernel_derivative(x_vals, h)
    deriv_min_idx = int(np.argmin(deriv_vals))

    return {
        "h": float(h),
        "kernel_max": float(kernel_vals[kernel_max_idx]),
        "kernel_x_at_max": float(x_vals[kernel_max_idx]),
        "kernel_x_over_h": float(x_vals[kernel_max_idx] / h),
        "deriv_min": float(deriv_vals[deriv_min_idx]),
        "deriv_x_at_min": float(x_vals[deriv_min_idx]),
        "deriv_x_over_h": float(x_vals[deriv_min_idx] / h),
    }


def sweep_h_values(h_min, h_max, h_samples, sample_count):
    h_values = np.linspace(h_min, h_max, h_samples, dtype=np.float32)
    return [evaluate_extrema_for_h(h, sample_count) for h in h_values]


def plot_extrema(results, output_path=None):
    h_values = np.array([item["h"] for item in results], dtype=np.float32)
    kernel_max_values = np.array([item["kernel_max"] for item in results], dtype=np.float32)
    deriv_min_values = np.array([item["deriv_min"] for item in results], dtype=np.float32)
    deriv_min_abs_values = np.abs(deriv_min_values)

    fig = plt.figure(figsize=(11, 6))
    ax1 = plt.gca()

    line_kernel_max, = ax1.plot(
        h_values,
        kernel_max_values,
        linewidth=2,
        label="Kernel max value",
        color="tab:blue",
    )
    line_deriv_min, = ax1.plot(
        h_values,
        deriv_min_abs_values,
        linewidth=2,
        label="|1st derivative min value|",
        color="tab:red",
    )
    ax1.set_xlabel("h")
    ax1.set_ylabel("Value")
    ax1.set_yscale("log", base=10)
    ax1.grid(True, alpha=0.3)

    lines = [line_kernel_max, line_deriv_min]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best")

    fig.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure to {output_path}")

    plt.show()


def print_summary(results):
    kernel_x_over_h = np.array([item["kernel_x_over_h"] for item in results], dtype=np.float32)
    deriv_x_over_h = np.array([item["deriv_x_over_h"] for item in results], dtype=np.float32)
    deriv_min_values = np.array([item["deriv_min"] for item in results], dtype=np.float32)

    print("Kernel max location x/h:")
    print(f"  min = {kernel_x_over_h.min():.6f}, max = {kernel_x_over_h.max():.6f}")
    print("Derivative min location x/h:")
    print(f"  min = {deriv_x_over_h.min():.6f}, max = {deriv_x_over_h.max():.6f}")
    print("Derivative min value:")
    print(f"  min = {deriv_min_values.min():.6f}, max = {deriv_min_values.max():.6f}")
    print("Reference values:")
    print("  kernel max should occur at x/h = 0")
    print("  derivative minimum should occur near x/h = 1/3")


def main():
    parser = argparse.ArgumentParser(description="Plot cubic kernel extrema as h varies.")
    parser.add_argument("--h_min", type=float, default=0.001, help="Minimum h to sweep.")
    parser.add_argument("--h_max", type=float, default=0.1, help="Maximum h to sweep.")
    parser.add_argument("--h_samples", type=int, default=100, help="Number of h samples.")
    parser.add_argument("--sample_count", type=int, default=2000, help="Number of x samples per h.")
    parser.add_argument("--output", type=str, default=None, help="Optional image output path.")
    args = parser.parse_args()

    results = sweep_h_values(args.h_min, args.h_max, args.h_samples, args.sample_count)
    print_summary(results)
    plot_extrema(results, args.output)


if __name__ == "__main__":
    main()