"""
Plot a grid of all results of training on correlated multi-digit MNIST,
using different noise levels.

Example
-------
python plot_multi_mnist.py
"""
import os
import csv
import pdb
import argparse
import numpy as np
import pickle as pkl
from collections import defaultdict

# YAML setup
from ruamel.yaml import YAML
yaml = YAML()
yaml.preserve_quotes = True
yaml.boolean_representation = ['False', 'True']

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
matplotlib.rc('text', usetex=True)  # Activate latex text rendering


paths = [
    ('cls', 'saves/multi_mnist_cls'),
    ('uncond', 'saves/multi_mnist_uncond'),
    ('cond', 'saves/multi_mnist_cond'),
]

figsize = (10, 6.5)
use_correlation_list = [0, 0.6, 0.9]

colors = ["#E00072", "#00830B", "#2B1A7F", "#E06111", "#4F4C4B", "#02D4F9"]
color_train = colors[0]
color_test_uc = colors[1]
marker = 'o'


metric_to_plot = 'avg_acc'
agg_dict = defaultdict(lambda: defaultdict(dict))

for (method_name, base_dir) in paths:
  for subdir in os.listdir(base_dir):
    try:
      expdir = os.path.join(base_dir, subdir)

      with open(os.path.join(expdir, 'test_corr_range.pkl'), 'rb') as f:
        data = pkl.load(f)

      arg_dict = yaml.load(open(os.path.join(base_dir, subdir, 'args.yaml'), 'r'))

      test_uc_perf = data['avg_acc_list'][100]  # Corresponds to correlation 0
      train_perf = data['train_avg_acc']

      agg_dict[method_name][arg_dict['train_corr']][arg_dict['noise']] = (
          train_perf, test_uc_perf,
          np.min(data['avg_acc_list']), np.max(data['avg_acc_list'])
      )
    except:
      print('Issue with {}'.format(subdir))
      pass


# Plot results
# ------------
ref_noise_levels_sorted = list(sorted(agg_dict['cls'][0.0].keys()))
ref_accs = [agg_dict['cls'][0.0][noise][0] for noise in ref_noise_levels_sorted]

num_correlations = len(use_correlation_list)
fig, axs = plt.subplots(
    nrows=num_correlations, ncols=3, figsize=figsize, sharex='all', sharey='all'
)

for (i,mi_type) in enumerate(['cls', 'uncond', 'cond']):
  result_dict = agg_dict[mi_type]
  for (j,correlation) in enumerate(use_correlation_list):
    ax = axs[j,i]

    noise_levels_sorted = list(sorted(result_dict[correlation].keys()))
    corr_accs = [result_dict[correlation][noise][0] for noise in noise_levels_sorted]
    uncorr_accs = [result_dict[correlation][noise][1] for noise in noise_levels_sorted]
    min_accs = [result_dict[correlation][noise][2] for noise in noise_levels_sorted]
    max_accs = [result_dict[correlation][noise][3] for noise in noise_levels_sorted]
    ax.plot(noise_levels_sorted, corr_accs, marker=marker, markersize=4, linewidth=2, color=color_train)
    ax.plot(noise_levels_sorted, uncorr_accs, marker=marker, markersize=4, linewidth=2, color=color_test_uc)
    ax.plot(ref_noise_levels_sorted, ref_accs, marker=None, linewidth=2, color='k', alpha=0.9, linestyle='--')
    ax.fill_between(noise_levels_sorted, min_accs, max_accs, color=color_test_uc, alpha=0.2)

    if 'acc' in metric_to_plot:
      ax.set_ylim([0.45, 1.05])
    ax.tick_params(axis='both', which='both', labelsize=20)

axs[0,0].set_title(r'\textbf{Base}', fontsize=26)
axs[0,1].set_title(r'\textbf{Base+MI}', fontsize=26)
axs[0,2].set_title(r'\textbf{Base+CMI}', fontsize=26)

for i in range(3):
  axs[-1,i].set_xlabel('Noise Level', fontsize=28)

extra_labels = []
for j in range(num_correlations):
  lbl = axs[j,0].set_ylabel(
      r'\textbf{' + 'Corr = {}'.format(use_correlation_list[j]) + r'}' + '\n\nAccuracy', fontsize=18
  )
  extra_labels.append(lbl)

lgd = fig.legend(
    ['Correlated Training', 'Uncorrelated Test', 'Reference'],
    bbox_to_anchor=(0.0, 0.95, 1., .102),
    loc='lower left',
    ncol=3,
    borderaxespad=0.,
    fontsize=22
)

figure_dirname = 'figures'
if not os.path.exists(figure_dirname):
  os.makedirs(figure_dirname)

fig.subplots_adjust(hspace=0, wspace=0)
plt.savefig(
    os.path.join(figure_dirname, 'multi_mnist.png'),
    bbox_extra_artists=[lgd] + extra_labels,
    bbox_inches='tight', pad_inches=0.02
)
plt.savefig(
    os.path.join(figure_dirname, 'multi_mnist.pdf'),
    bbox_extra_artists=[lgd] + extra_labels,
    bbox_inches='tight', pad_inches=0.02
)
