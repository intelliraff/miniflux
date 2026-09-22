"""Create plots and a concise report for a completed CIFAR pilot."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    summary = json.loads((args.run / "summary.json").read_text())
    config = json.loads((args.run / "config.json").read_text())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in summary:
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

    guidance_rows=[]
    if config.get("conditioning")=="class-conditioned" and "conditioned_hybrid" in summary:
        name="conditioned_hybrid"; step=summary[name]["steps"]; model_dir=args.run/name
        baseline=None
        for scale in config["guidance_scales"]:
            artifact=model_dir/f"samples_step{step:07d}_guidance{scale:g}.pt"
            saved=torch.load(artifact,map_location="cpu",weights_only=False)
            images=saved["images"].float(); labels=saved["labels"]
            if baseline is None: baseline=images
            edge=((images[:,:,:,1:]-images[:,:,:,:-1]).abs().mean()+(images[:,:,1:,:]-images[:,:,:-1,:]).abs().mean()).item()/2
            diversity=[]
            for class_id in range(10):
                flat=images[labels==class_id].flatten(1)
                diversity.append(torch.pdist(flat).mean().item()/(flat.shape[1]**0.5))
            metadata=json.loads(artifact.with_suffix(".json").read_text())
            guidance_rows.append({"guidance_scale":scale,"edge_energy":edge,"class_diversity":sum(diversity)/len(diversity),"mean_absolute_delta_from_scale_0":(images-baseline).abs().mean().item(),"sampling_seconds":metadata["sampling_seconds"]})
        with (args.run/"guidance_metrics.csv").open("w",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=guidance_rows[0].keys()); writer.writeheader(); writer.writerows(guidance_rows)
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        scales=[row["guidance_scale"] for row in guidance_rows]
        axes[0].plot(scales,[row["edge_energy"] for row in guidance_rows],marker="o",label="edge energy")
        axes[0].plot(scales,[row["class_diversity"] for row in guidance_rows],marker="o",label="class diversity")
        axes[1].plot(scales,[row["mean_absolute_delta_from_scale_0"] for row in guidance_rows],marker="o")
        axes[0].set(xlabel="Guidance scale",title="Sharpness and diversity"); axes[0].legend()
        axes[1].set(xlabel="Guidance scale",ylabel="Mean absolute pixel delta",title="Change from null conditioning")
        for axis in axes: axis.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(args.run/"guidance_metrics.png",dpi=160); plt.close(fig)

    rows = "\n".join(
        f"| {name} | {value['parameters']:,} | {value['steps']:,} | "
        f"{value['equivalent_epochs']:.3f} | {value['final_ema_test_velocity_mse']:.5f} | "
        f"{value['sample16_seconds']:.2f}s | {value['sampled_allocated_tensor_bytes']/2**20:.0f} MiB |"
        for name, value in summary.items()
    )
    conditioning=config.get("conditioning","unconditional")
    report = f"""# CIFAR-10 flow-matching benchmark

All listed models used the full CIFAR-10 training set, {conditioning} conditional-OT
flow matching, skewed timestep sampling, horizontal flips, AdamW, EMA evaluation,
and the fixed 50-interval Heun sampler.

| Model | Parameters | Updates | Epochs | Final EMA test MSE | Sample 16 | Peak allocated |
|---|---:|---:|---:|---:|---:|---:|
{rows}

The flow equations are validated against Meta's official `CondOTProbPath`. Velocity
loss is a training diagnostic and is not by itself evidence of perceptual image
quality; inspect the fixed-noise grids alongside this report.
"""
    (args.run / "assessment.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
