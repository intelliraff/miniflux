"""Create a comparison chart from a completed matched v1/v2 run."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    args=parser.parse_args()
    summary=json.loads((args.run/'summary.json').read_text())
    assert set(summary)=={'v1','v2'}, 'Both models must finish first'
    config=json.loads((args.run/'config.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name in ['v1','v2']:
        with (args.run/name/'evaluation.csv').open() as f: rows=list(csv.DictReader(f))[1:]
        assert int(rows[-1]['step'])==config['steps']
        x=[int(r['step']) for r in rows]
        axes[0].plot(x,[float(r['train_fixed_loss']) for r in rows],label=f'{name} train')
        axes[0].plot(x,[float(r['validation_fixed_loss']) for r in rows],linestyle='--',label=f'{name} held-out')
    axes[0].set(xlabel='Optimizer step',ylabel='Fixed-noise velocity MSE',title='Learning curves (lower is better)')
    axes[0].legend(); axes[0].grid(alpha=.2)
    names=['Correct','Shuffled','Empty']
    for offset,name in [(-.18,'v1'),(.18,'v2')]:
        checks=summary[name]['final_caption_checks']
        axes[1].bar([i+offset for i in range(3)],[checks[k]['validation'] for k in ['correct','shuffled','empty']],width=.36,label=name)
    axes[1].set(xticks=range(3),xticklabels=names,ylabel='Held-out velocity MSE',title='Caption interventions (not image quality)')
    axes[1].legend()
    fig.tight_layout(); fig.savefig(args.run/'comparison.png',dpi=160); plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
