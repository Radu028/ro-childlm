"""Rebuild both scaling panels of Figure 1."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()

args.output.mkdir(parents=True, exist_ok=True)
data = Path(__file__).resolve().parent
plt.rcParams.update({
    'font.size': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

untrained = []
with (data.parent / 'summary.jsonl').open() as file:
    for line in file:
        row = json.loads(line)
        if row['condition'] == 'random-nopunct':
            untrained.append(100 * row['accuracy'])

for name, title, output, floor in [
    ('scaling_points.csv', 'a. GPT-2 small', 'scaling-curve.png', untrained),
    ('gptbert_scaling_points.csv', 'b. GPT-BERT base', 'scaling-gbert.png', None),
]:
    scaling = defaultdict(lambda: defaultdict(list))
    with (data / name).open() as file:
        for row in csv.DictReader(file):
            scaling[row['condition']][int(row['budget_words'])].append(float(row['accuracy']))

    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    for condition, label, color in [
        ('ages04', 'Ages 0-4', '#0072B2'),
        ('ages58', 'Ages 5-8', '#E69F00'),
        ('combined', 'Combined', '#999999'),
    ]:
        budgets = sorted(scaling[condition])
        means = [statistics.mean(scaling[condition][b]) for b in budgets]
        slope, intercept = np.polyfit(np.log10(budgets), means, 1)
        for budget in budgets:
            values = scaling[condition][budget]
            ax.scatter([budget] * len(values), values, color=color, alpha=.5, s=14)
        x = np.geomspace(budgets[0] * .95, budgets[-1] * 1.05, 100)
        ax.plot(
            x, intercept + slope * np.log10(x), color=color, linewidth=2,
            label=f'{label} ({slope:.1f} pts/decade)',
        )

    if floor:
        level = statistics.mean(floor)
        ax.axhline(level, color='#bbbbbb', lw=1, ls='--')
        ax.annotate(
            f'untrained {level:.1f}', (10.5e6, level), ha='right', va='bottom',
            fontsize=10, color='#999999', xytext=(0, 3), textcoords='offset points',
        )

    ax.set_xscale('log')
    ax.set_xticks([1e6, 2e6, 5e6, 1e7], ['1M', '2M', '5M', '10M'])
    ax.set_ylim(50, 80)
    ax.set_xlabel('Training words (log scale)')
    ax.set_ylabel('MultiBLiMP accuracy (%)')
    ax.set_title(title, loc='left')
    ax.grid(axis='y', alpha=.3)
    ax.legend(frameon=False, fontsize=10, loc='upper left')
    fig.tight_layout()
    fig.savefig(args.output / output, dpi=200)
    plt.close(fig)
