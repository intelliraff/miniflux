# Small-data overfit diagnostic

Run from the repository root:

```sh
.venv/bin/python -m experiments.overfit --steps 2000 --sample-every 250
```

Uses 16 reproducibly selected examples from `../data/coco`, center-cropped to
128×128. VAE posterior means and CLIP features are cached in memory, and the
examples are shuffled each pass. The transformer starts from scratch.

`outputs/overfit/` contains captions, configuration, per-step losses, a checkpoint,
and fixed-seed sample grids. Each row shows **original | VAE reconstruction |
generated**. Compare step 0 with later grids. Recognizable training subjects are
the first milestone; this does not test generalization to unseen captions.

The default experimental timestep scale is 1000. Existing MiniFlux callers retain
scale 1, so existing checkpoints are unaffected. To compare the old conditioning
with the same data and seed, run:

```sh
.venv/bin/python -m experiments.overfit --time-scale 1 --output outputs/overfit-scale1
```

Use a fresh output directory for each run. The diagnostic checkpoint is a dictionary
containing `model`, `optimizer`, `step`, and `config`; it is not a plain state dict
for `sampling/generate.py`. Its model must be constructed with the saved
`config['time_scale']`. The script does not currently resume checkpoints.

If reconstructions look good but generated images remain incoherent, the generator
needs further investigation. If this small set becomes recognizable, increase data
and evaluate held-out prompts. A lower training loss alone is not a success criterion.

# Train/held-out experiment

```sh
.venv/bin/python -m experiments.generalization --device mps
```

Starts a fresh model for 10,000 steps at 128×128, batch size 4, learning rate
1e-4, and timestep scale 1000. The seeded split uses 900 training images and 100
held-out images, records source indices and captions, and rejects exact image
matches crossing the split. This does not detect near-duplicates. Training order
is shuffled each pass. VAE posterior means and CLIP embeddings are cached on CPU
in small batches; held-out examples never enter gradient updates.

Every 500 steps, `outputs/generalization/` receives separate grids for eight fixed
training captions and eight fixed held-out captions. Columns show reference image,
VAE reconstruction, and generated image. A held-out generation should be judged
for coherence and caption relevance, not exact reconstruction of the reference.
`evaluation.csv` measures velocity MSE on 100 training and all 100 held-out examples
with fixed noise and timesteps. These are diagnostic losses, not image-quality
scores. Evaluation uses separate randomness from training.

`checkpoint.pt` contains the latest model and optimizer; `best_validation.pt`
contains the checkpoint with the lowest measured held-out velocity loss. Both
include configuration and step count. Like the overfit checkpoint, these require
loading the `model` field into `MiniFlux(time_scale=config['time_scale'])` and are
not directly compatible with the old sampling script. Checkpoint resume is not
implemented. Use a new `--output` directory to repeat an experiment.

After training, generate a loss plot and machine-readable summary with:

```sh
.venv/bin/python -m experiments.report_generalization outputs/generalization
```

# Matched MiniFlux v1/v2 comparison

```sh
.venv/bin/python -m experiments.compare_v2 --device mps
```

Trains each architecture from scratch for 5,000 steps on the same 128 examples,
with 32 held out. The split is a subset of the previous 900/100 split. Features
are encoded once and saved in `outputs/compare_v2/features.pt`. Both runs use the
same CPU-generated training noise, timesteps, shuffled batch sequence, seed,
learning rate, batch size, timestep scale, and 50-step Euler sampler. A shared
seed does not imply identical initial weights across different architectures.

V2 uses five joint blocks, 2D sinusoidal image positions, text positions,
per-block adaptive normalization and residual gates, Q/K RMS normalization,
pooled projected text plus time conditioning, and a zero-initialized output
projection. It has 6,714,384 parameters. V1 has 7,769,360 total parameters,
including 1,052,672 disconnected parameters. Their active parameter counts are
therefore nearly equal. This is an architecture-package comparison, not an
ablation identifying one causal change, and is not a faithful FLUX reproduction.

Grids and fixed-noise velocity losses are saved every 1,000 steps. Final grids
include correct, shuffled, and empty captions. In shuffled grids the reference
caption under each row stays unchanged but conditioning comes from the next row
(wrapping around); in empty grids conditioning is CLIP's empty-string embedding.
For dataset-wide shuffled loss, captions rotate within their own train/held-out
partition. Neither model is trained on empty captions. Empty-caption results
are out-of-training-distribution diagnostics, not calibrated unconditional
predictions or classifier-free guidance. Latent changes alone do not establish
semantic correctness; inspect caption relevance visually.

`summary.json` reports total and gradient-connected generator parameters,
full-pipeline parameter counts (including frozen CLIP and VAE), generator weight
bytes, synchronized training-update time, and median generation latency over
three warmed-up repetitions. Latency covers four images, 50 Euler steps, VAE
decoding and CPU output with cached text; it excludes CLIP encoding. Weight bytes
are not peak runtime memory. This is one seed and a small held-out set.

Checkpoints contain architecture name, configuration, model and optimizer states.
Construct `MiniFlux` or `MiniFluxV2` with `time_scale=1000` and load the `model`
field. The old sampling script does not load these checkpoint dictionaries.

After the matched comparison completes:

```sh
MPLCONFIGDIR=/private/tmp/miniflux-matplotlib .venv/bin/python -m experiments.report_v2 outputs/compare_v2
```

This writes the learning-curve/caption-intervention chart. The completed first-run
assessment is in `outputs/compare_v2/assessment.md`.
