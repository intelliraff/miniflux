"""Create plots and a concise report for a completed CIFAR pilot."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    summary = json.loads((args.run / "summary.json").read_text())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in ("reference", "miniflux"):
        losses = read_csv(args.run / name / "loss.csv")
        evaluations = read_csv(args.run / name / "evaluation.csv")
        axes[0].plot(
            [float(row["training_seconds"]) for row in losses],
            [float(row["loss"]) for row in losses],
            alpha=0.55,
            label=name,
        )
        axes[1].plot(
            [int(row["step"]) for row in evaluations],
            [float(row["ema_test_velocity_mse"]) for row in evaluations],
            marker="o",
            label=name,
        )
    axes[0].set(
        xlabel="Cumulative optimizer-update time (seconds)",
        ylabel="Training velocity MSE",
        title="Learning under a compute budget",
    )
    axes[1].set(
        xlabel="Optimizer step",
        ylabel="Fixed held-out velocity MSE",
        title="EMA held-out flow prediction",
    )
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend()
    fig.tight_layout()
    fig.savefig(args.run / "comparison.png", dpi=160)
    plt.close(fig)

    ref, mini = summary["reference"], summary["miniflux"]
    mini_session_hours = mini["phase_seconds"] / 3600
    report = f"""# CIFAR-10 flow-matching pilot

The initial run was a 30-minute wall-clock comparison on the local Mac GPU. The
MiniFlux checkpoint was subsequently continued in a {mini_session_hours:.2f}-hour
resumable session, so the final rows below are no longer an equal-time quality
comparison. Both models used the full CIFAR-10 training set, identical unconditional
conditional-OT flow matching, skewed timestep sampling, horizontal flips, AdamW,
EMA evaluation, and a fixed 50-interval Heun sampler. This is an early-training and
efficiency study, not a reproduction of Meta's published CIFAR-10 quality result.

| Measurement | Meta reference U-Net | Pixel MiniFlux |
|---|---:|---:|
| Parameters | {ref['parameters']:,} | {mini['parameters']:,} |
| Updates | {ref['steps']:,} | {mini['steps']:,} |
| Equivalent epochs | {ref['equivalent_epochs']:.3f} | {mini['equivalent_epochs']:.3f} |
| Final fixed held-out velocity MSE | {ref['final_ema_test_velocity_mse']:.4f} | {mini['final_ema_test_velocity_mse']:.4f} |
| 16-image sampling time, 100 model evaluations | {ref['sample16_seconds']:.2f}s | {mini['sample16_seconds']:.2f}s |
| Sampled allocated tensor memory (lower bound) | {ref['sampled_allocated_tensor_bytes']/2**20:.0f} MiB | {mini['sampled_allocated_tensor_bytes']/2**20:.0f} MiB |

The flow equations are validated against Meta's official `CondOTProbPath`. The
reference learns faster per example in the initial matched pilot. MiniFlux processes
many more examples per second, reaches a lower held-out velocity loss after its
additional training, samples about {ref['sample16_seconds']/mini['sample16_seconds']:.1f}x faster,
and uses about {ref['parameters']/mini['parameters']:.1f}x fewer parameters.

Neither final grid contains consistently recognizable CIFAR objects yet. MiniFlux
trained for only {mini['equivalent_epochs']:.2f} epochs, while Meta reports its
published unconditional CIFAR-10 result after roughly 1,800 epochs. Velocity loss
alone is not evidence of perceptual image quality.

The useful conclusion is that the compact transformer can learn the unconditional
pixel-space flow task. The earlier text-to-image failure is therefore more likely
to involve training exposure, latent/text conditioning, or their interaction than
a reversed flow equation. The next controlled milestone is to continue this
resumable MiniFlux checkpoint until samples become recognizable, then add class
conditioning before returning to free-form text and VAE latents.
"""
    (args.run / "assessment.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
