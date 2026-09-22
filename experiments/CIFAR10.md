# CIFAR-10 flow benchmark

The benchmark uses Meta's official image example pinned at commit
`11568d37f8d5a080e12aa7b5305d9c35ae07d136`. The vendored checkout retains its
upstream license and source. The local loader changes only the import namespace so
it can coexist with this repository's `models` package.

Run a fresh 30-minute comparison:

```sh
.venv/bin/python -m experiments.cifar_benchmark \
  --device mps --output outputs/cifar_benchmark --budget-seconds 1800
```

Continue both saved checkpoints for another 30-minute budget:

```sh
.venv/bin/python -m experiments.cifar_benchmark \
  --device mps --output outputs/cifar_benchmark --budget-seconds 1800 --resume
```

Continue only MiniFlux until a cumulative epoch target, with a wall-clock safety
limit:

```sh
.venv/bin/python -m experiments.cifar_benchmark \
  --device mps --output outputs/cifar_benchmark --budget-seconds 9000 \
  --models miniflux --resume --target-epochs 25
```

The budget includes initialization, training, evaluation, sampling, and checkpoint
I/O. It is divided fairly over the requested model phases. `--resume` restores model,
EMA, optimizer, batch order, and random-generator states. It requires the same seed,
batch size, microbatch size, and pinned reference revision.

## Hierarchical MiniFlux pilot

The hierarchical model uses transformer blocks at 16x16 and 8x8, with convolutional
resolution changes and a skip connection. A fresh equal-time MPS pilot used this
command:

```sh
PYTHONPATH=. .venv/bin/python -u experiments/cifar_benchmark.py \
  --models miniflux hierarchical \
  --output outputs/cifar_hierarchical_pilot_mps \
  --budget-seconds 1800 --device mps
```

| Model | Parameters | Updates | Epochs | EMA test velocity MSE | 16-image sample time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flat MiniFlux | 6,581,004 | 1,183 | 1.514 | 0.21788 | 6.61 s |
| Hierarchical MiniFlux | 7,374,979 | 1,512 | 1.935 | 0.22276 | 3.79 s |

The hierarchy processed 28% more updates and sampled 43% faster, but its final MSE
was 2.2% higher. Both sample grids remained immature after fewer than two epochs.
This pilot supports retaining the model for a longer checkpointed experiment, but
does not establish a quality improvement over the flat transformer.

## Local-global hybrid at five epochs

The hybrid keeps five modulated transformer blocks at the 8x8 global bottleneck and
uses small conditioned residual convolution blocks at 16x16 and 32x32. It has
8,449,347 parameters. Training was resumed until exactly 5.00096 equivalent epochs
(3,907 updates) using `--models hybrid --target-epochs 5 --resume`.

At the matched 3,900-update evaluation, the flat MiniFlux had 0.19589 EMA test
velocity MSE. The hybrid finished at 0.19021, a 2.9% reduction. Its final 16-image
sample took 3.71 seconds versus 6.02 seconds for the flat model at step 3,900. The
fixed-noise grids remain visually similar and only partly recognizable, so this is
evidence of better compute efficiency and optimization rather than proof of a large
perceptual-quality improvement.

Create the comparison plot and written assessment after a completed run:

```sh
MPLCONFIGDIR=/private/tmp/miniflux-matplotlib \
  .venv/bin/python -m experiments.report_cifar outputs/cifar_benchmark
```

The default sampler uses the reference EDM time schedule with 50 intervals and
Heun integration, which costs 100 model evaluations. Reported MPS memory is sampled
allocated tensor memory, excludes allocator cache, and is a lower bound on peak.
The run does not compute FID and should not be compared directly with Meta's
published 1,800-epoch result.

## Class-conditioned local-global hybrid

`ConditionedCifarHybridMiniFlux` is a separate 8.45M-parameter architecture. It
uses CIFAR-10 class labels plus a learned null label; training replaces labels
with null at probability 0.15. The shared timestep/class vector conditions the
same residual convolution and 8x8 modulated transformer blocks as the existing
hybrid. Existing `hybrid` checkpoints and sampling remain unchanged.

Run a 200-update smoke test first, using a fresh output directory:

```sh
PYTHONPATH=. python -m experiments.cifar_benchmark \
  --device cuda --models conditioned_hybrid \
  --output outputs/cuda_conditioned_hybrid \
  --budget-seconds 1800 --max-updates 200
```

After verifying finite samples and falling loss, resume to five epochs with:

```sh
PYTHONPATH=. python -m experiments.cifar_benchmark \
  --device cuda --models conditioned_hybrid \
  --output outputs/cuda_conditioned_hybrid --resume \
  --budget-seconds 1800 --target-epochs 5 --exact-target
```

The experiment checkpoints model, EMA, optimizer, scaler,
data order, and CPU/CUDA random states. CUDA uses BF16 when available and FP16
with GradScaler otherwise. The benchmark generates the same balanced class labels
and fixed initial noise for each guidance scale (0, 1, 1.5, 2, 3, 4), with Heun
guidance on both predictor and corrector evaluations. Each evaluation writes a
class-label manifest and per-scale image grids under
`outputs/cuda_conditioned_hybrid/conditioned_hybrid/`.

### Five-epoch CUDA result

The clean primary run is in `outputs/cuda_conditioned_hybrid_5ep_pty`. It used an
RTX A4000 with BF16 autocasting, trained from scratch for exactly 250,000 examples
(3,907 updates and 5.000 epochs), and reached 0.19027 EMA test velocity MSE. The
8,452,163-parameter model used 321 MiB peak allocated CUDA tensor memory. Optimizer
update time was 1,073 seconds and the full phase, including six-scale evaluations
at every epoch, took 1,313 seconds.

| Guidance | Edge energy | Within-class diversity | Sampling time (40 images) |
| ---: | ---: | ---: | ---: |
| 0.0 | 0.13473 | 0.68268 | 3.78 s |
| 1.0 | 0.13417 | 0.67952 | 7.40 s |
| 1.5 | 0.13395 | 0.67800 | 7.32 s |
| 2.0 | 0.13380 | 0.67654 | 7.29 s |
| 3.0 | 0.13372 | 0.67379 | 7.26 s |
| 4.0 | 0.13385 | 0.67110 | 7.26 s |

The balanced grids show some class organization, most clearly for automobiles,
ships, horses, and broad animal silhouettes. They remain close to the original
hybrid grids and guidance produces only subtle changes. Higher scales steadily
reduce diversity without a visible gain in recognizability or sharpness. Scale
1.0 is the conservative recommendation, but none of the tested scales establishes
a convincing perceptual improvement over the unconditional model.

Do not continue this run automatically to ten epochs. The five-epoch result does
not satisfy the decision rule requiring clearer class structure or improved
perceptual metrics. The next revision should strengthen how class information
enters the network and add a reliable CIFAR-10 classifier/FID evaluator before
more training; if direct-pixel samples remain blurry after that revision, move the
conditioned hybrid into a suitable frozen pretrained latent space. FID and
pretrained-classifier accuracy were unavailable for this run and are recorded as
missing rather than inferred from velocity MSE.

## Strong per-block class modulation

The follow-up `strong_conditioned_hybrid` keeps time modulation separate and adds
low-rank class modulation directly to every convolutional residual block and every
transformer AdaLN shift, scale, and residual gate. It has 8,875,875 parameters.
A 200-update CUDA smoke test reached 0.35587 EMA velocity MSE with a 0.01051
conditional/null velocity MAE, so it proceeded to the five-epoch gate.

At exactly five epochs, EMA MSE was 0.18938 and conditional/null velocity MAE was
0.02204, over twice the earlier conditioned model's 0.00951 measured checkpoint
separation. Guidance 3.0 produced substantially clearer class rows, particularly
airplanes, automobiles, birds, horses, ships, and trucks. Because this satisfied
the perceptual decision rule, the preserved five-epoch checkpoint was continued
to exactly ten epochs. The five-epoch checkpoint is saved as `checkpoint_5ep.pt`.

The ten-epoch endpoint used 7,814 updates and exactly 500,000 examples. EMA MSE
finished at 0.18234 and conditional/null velocity MAE at 0.02269. Guidance-scale
metrics at ten epochs were:

| Guidance | Edge energy | Within-class diversity | Sampling time (40 images) |
| ---: | ---: | ---: | ---: |
| 0.0 | 0.13171 | 0.68052 | 3.95 s |
| 1.0 | 0.13134 | 0.67041 | 7.81 s |
| 1.5 | 0.13132 | 0.66587 | 7.83 s |
| 2.0 | 0.13069 | 0.66146 | 7.80 s |
| 3.0 | 0.12980 | 0.65653 | 7.79 s |
| 4.0 | 0.13066 | 0.65553 | 7.83 s |

Guidance 3.0 remains the recommended balance. Scale 4 produces the strongest class
templates but repeats layouts and loses additional diversity. Ten epochs refine
silhouettes and backgrounds compared with five epochs, but the 32x32 images remain
soft and lack fine detail. Stop at ten epochs; do not continue to 25. The next
experiment should use the successful strong conditioning in a suitable pretrained
latent space and include a reliable classifier/FID evaluator.
