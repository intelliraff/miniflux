"""Summarize a finished generalization run without loading models."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    with (args.run / 'evaluation.csv').open() as f:
        evaluations = list(csv.DictReader(f))
    with (args.run / 'loss.csv').open() as f:
        losses = list(csv.DictReader(f))
    config = json.loads((args.run / 'config.json').read_text())
    steps = [int(r['step']) for r in evaluations]
    train = [float(r['train_fixed_loss']) for r in evaluations]
    validation = [float(r['validation_fixed_loss']) for r in evaluations]
    assert len(losses) == config['steps'], 'Training has not finished'
    assert steps[-1] == config['steps'], 'Final evaluation missing'
    best = min(range(1, len(validation)), key=lambda i: validation[i])
    summary = {
        'steps': steps[-1], 'train_count': config['train_count'],
        'validation_count': config['validation_count'],
        'final_train_fixed_loss': train[-1],
        'final_validation_fixed_loss': validation[-1],
        'best_validation_step': steps[best],
        'best_validation_fixed_loss': validation[best],
        'elapsed_seconds_before_final_evaluation': float(losses[-1]['elapsed_seconds']),
        'note': 'Velocity MSE is not an image-quality score. Inspect both sample grids.'
    }
    (args.run / 'summary.json').write_text(json.dumps(summary, indent=2))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    # Exclude untrained initialization to make changes during training visible.
    ax.plot(steps[1:], train[1:], label='Training subset (100 images)')
    ax.plot(steps[1:], validation[1:], label='Held-out (100 images)')
    ax.set(xlabel='Optimizer step', ylabel='Fixed-noise velocity MSE', title='900-image training / 100-image held-out evaluation')
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(args.run / 'loss_comparison.png', dpi=150)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
