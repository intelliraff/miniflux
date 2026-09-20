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

The budget includes initialization, training, evaluation, sampling, and checkpoint
I/O. It is divided fairly over the requested model phases. `--resume` restores model,
EMA, optimizer, batch order, and random-generator states. It requires the same seed,
batch size, microbatch size, and pinned reference revision.

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
