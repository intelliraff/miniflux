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
