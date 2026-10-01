"""Rebuild the GPT-BERT scaling and age-profile figures."""
import argparse
import csv
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
plt.rcParams.update({'font.size': 12, 'axes.spines.top': False,
                     'axes.spines.right': False})

scaling = defaultdict(lambda: defaultdict(list))
with (data / 'gptbert_scaling_points.csv').open() as file:
    for row in csv.DictReader(file):
        scaling[row['condition']][int(row['budget_words'])].append(float(row['accuracy']))
fig, ax = plt.subplots(figsize=(5.8, 3.4))
for condition, label, color in [('ages04', 'Ages 0-4', '#0072B2'),
                               ('ages58', 'Ages 5-8', '#E69F00'),
                               ('combined', 'Combined', '#999999')]:
    budgets = sorted(scaling[condition])
    means = [statistics.mean(scaling[condition][b]) for b in budgets]
    slope, intercept = np.polyfit(np.log10(budgets), means, 1)
    for budget in budgets:
        values = scaling[condition][budget]
        ax.scatter([budget] * len(values), values, color=color, alpha=.5, s=14)
    x = np.geomspace(budgets[0] * .95, budgets[-1] * 1.05, 100)
    ax.plot(x, intercept + slope * np.log10(x), color=color, linewidth=2,
            label=f'{label} ({slope:.1f} pts/decade)')
ax.set_xscale('log')
ax.set_xticks([1e6, 2e6, 5e6, 1e7], ['1M', '2M', '5M', '10M'])
ax.set_ylim(50, 80)
ax.set_xlabel('Training words (log scale)')
ax.set_ylabel('MultiBLiMP accuracy (%)')
ax.set_title('b. GPT-BERT base', loc='left')
ax.grid(axis='y', alpha=.3)
ax.legend(frameon=False, fontsize=10, loc='upper left')
fig.tight_layout()
fig.savefig(args.output / 'scaling-gbert.png', dpi=200)
plt.close(fig)

age = defaultdict(lambda: defaultdict(list))
with (data / 'age_effect.csv').open() as file:
    for row in csv.DictReader(file):
        age[row['protocol']][row['age_profile']].append(100 * float(row['accuracy']))
plt.rcParams.update({'font.size': 15})
fig, ax = plt.subplots(figsize=(10.24, 4.385))
protocols = ['gpt2_124m_20ep', 'gpt2_124m_10ep', 'gpt2_17m_10ep', 'gptbert_20ep']
labels = ['GPT-2 124M\n20 ep.', 'GPT-2 124M\n10 ep.', 'GPT-2 17M\n10 ep.', 'GPT-BERT\n20 ep.']
for index, protocol in enumerate(protocols):
    means = [statistics.mean(age[protocol][band]) for band in ['0-4', '5-8']]
    deviations = [statistics.stdev(age[protocol][band]) for band in ['0-4', '5-8']]
    positions = [index - .045, index + .045]
    ax.plot(positions, means, color='#888888', linewidth=1.5)
    for j, (band, color, marker) in enumerate([('0-4', '#0072B2', 'o'), ('5-8', '#E69F00', 's')]):
        ax.errorbar(positions[j], means[j], yerr=deviations[j], color=color,
                    marker=marker, markersize=7, capsize=4, linestyle='none',
                    label=f'Ages {band}' if index == 0 else None)
    ax.text(index, max(m + s for m, s in zip(means, deviations)) + .3,
            f'Δ = {means[0] - means[1]:+.2f}', ha='center')
ax.set_xticks(range(4), labels)
ax.set_ylim(68.3, 75.8)
ax.set_xlim(-.2, 3.2)
ax.set_ylabel('Romanian MultiBLiMP accuracy (%)')
ax.grid(axis='y', alpha=.4)
ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(.5, 1.18), ncol=2)
fig.tight_layout()
fig.savefig(args.output / 'age-effect.png', dpi=200)
plt.close(fig)
