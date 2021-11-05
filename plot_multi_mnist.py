"""
Plot a grid of all results of training on correlated multi-digit MNIST, using different noise levels.

Example
-------
python plot_multi_mnist.py
"""
import os
import csv
import ipdb
import argparse
import numpy as np
from collections import defaultdict

# YAML setup
from ruamel.yaml import YAML
yaml = YAML()
yaml.preserve_quotes = True
yaml.boolean_representation = ['False', 'True']

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rc('text', usetex=True)  # Activate latex text rendering


def load_log(exp_dir, fname='iteration.csv'):
    result_dict = defaultdict(list)
    with open(os.path.join(exp_dir, fname), newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for key in row:
                try:
                    if key in ['global_iteration', 'iteration', 'epoch']:
                        result_dict[key].append(int(row[key]))
                    else:
                        result_dict[key].append(float(row[key]))
                except:
                    pass
    return result_dict


# paths = [('cls', 'saves/multi_mnist_cls'),
#          ('uncond', 'saves/multi_mnist_uncond'),
#          ('cond', 'saves/multi_mnist_cond'),
#         ]

paths = [('cls', 'saves_multi/multi_mnist_cls'),
         ('uncond', 'saves_multi/multi_mnist_uncond'),
         ('cond', 'saves_multi/multi_mnist_cond'),
        ]

name_mapping = { 'avg_loss': 'Cross Entropy Loss',
                 'avg_acc': 'Accuracy',
                 'f1_acc': 'Left Accuracy',
                 'f2_acc': 'Right Accuracy'
               }


figsize = (10, 6.5)
use_correlation_list = [0, 0.6, 0.9]

colors = ["#E00072", "#00830B", "#2B1A7F", "#E06111", "#4F4C4B", "#02D4F9"]
color_train = colors[0]
color_test_ac = colors[1]
color_test_uc = colors[2]
marker = 'o'


metric_to_plot = 'avg_acc'
# metric_to_plot = 'avg_loss'
agg_dict = defaultdict(lambda: defaultdict(dict))

for (method_name, base_dir) in paths:
    for subdir in os.listdir(base_dir):
        try:
            expdir = os.path.join(base_dir, subdir)
            arg_dict = yaml.load(open(os.path.join(base_dir, subdir, 'args.yaml'), 'r'))
            log = load_log(expdir)

            if metric_to_plot == 'avg_loss':
                train_perf = log['trn_loss'][-1]
                val_perf = log['val_loss'][-1]
                test_ac_perf = log['tst_ac_loss'][-1]
                test_uc_perf = log['tst_uc_loss'][-1]
            elif metric_to_plot == 'avg_acc':
                train_perf = (log['trn_f1_acc'][-1] + log['trn_f2_acc'][-1]) / 2.0
                val_perf = (log['val_f1_acc'][-1] + log['val_f2_acc'][-1]) / 2.0
                test_ac_perf = (log['tst_ac_f2_acc'][-1] + log['tst_ac_f2_acc'][-1]) / 2.0
                test_uc_perf = (log['tst_uc_f2_acc'][-1] + log['tst_uc_f2_acc'][-1]) / 2.0
            elif metric_to_plot == 'f1_acc':
                train_perf = log['trn_f1_acc'][-1]
                val_perf = log['val_f1_acc'][-1]
                test_ac_perf = log['tst_ac_f1_acc'][-1]
                test_uc_perf = log['tst_uc_f1_acc'][-1]
            elif metric_to_plot == 'f2_acc':
                train_perf = log['trn_f2_acc'][-1]
                val_perf = log['val_f2_acc'][-1]
                test_ac_perf = log['tst_ac_f2_acc'][-1]
                test_uc_perf = log['tst_uc_f2_acc'][-1]

            agg_dict[method_name][arg_dict['train_corr']][arg_dict['noise']] = (train_perf, val_perf, test_ac_perf, test_uc_perf)
        except:
            pass


# Plot results
# ------------
ref_noise_levels_sorted = list(sorted(agg_dict['cls'][0.0].keys()))
ref_accs = [agg_dict['cls'][0.0][noise][0] for noise in ref_noise_levels_sorted]

# num_correlations = len(agg_dict['cls'].keys())
num_correlations = len(use_correlation_list)
fig, axs = plt.subplots(nrows=num_correlations, ncols=3, figsize=figsize, sharex='all', sharey='all')

for (i,mi_type) in enumerate(['cls', 'uncond', 'cond']):
    result_dict = agg_dict[mi_type]
    for (j,correlation) in enumerate(use_correlation_list):
        ax = axs[j,i]

        noise_levels_sorted = list(sorted(result_dict[correlation].keys()))
        corr_accs = [result_dict[correlation][noise][0] for noise in noise_levels_sorted]
        anticorr_accs = [result_dict[correlation][noise][2] for noise in noise_levels_sorted]
        uncorr_accs = [result_dict[correlation][noise][3] for noise in noise_levels_sorted]
        ax.plot(noise_levels_sorted, corr_accs, marker=marker, markersize=4, linewidth=2, color=color_train)
        ax.plot(noise_levels_sorted, anticorr_accs, marker=marker, markersize=4, linewidth=2, color=color_test_ac)
        ax.plot(noise_levels_sorted, uncorr_accs, marker=marker, markersize=4, linewidth=2, color=color_test_uc)
        ax.plot(ref_noise_levels_sorted, ref_accs, marker=None, linewidth=2, color='k', alpha=0.9, linestyle='--')

        if 'acc' in metric_to_plot:
            ax.set_ylim([0.45, 1.05])
        ax.tick_params(axis='both', which='both', labelsize=16)

axs[0,0].set_title(r'\textbf{Base}', fontsize=20)
axs[0,1].set_title(r'\textbf{Base + MI}', fontsize=20)
axs[0,2].set_title(r'\textbf{Base + CMI}', fontsize=20)

for i in range(3):
    axs[-1,i].set_xlabel('Noise Level', fontsize=18)

extra_labels = []
for j in range(num_correlations):
    lbl = axs[j,0].set_ylabel(r'\textbf{' + 'Corr = {}'.format(use_correlation_list[j]) + r'}' + '\n\nAccuracy', fontsize=18)
    extra_labels.append(lbl)

lgd = fig.legend(['Training', 'Anti-Correlated', 'Uncorrelated', 'Reference'],
                 bbox_to_anchor=(0.2, 0.95, 1., .102),
                 loc='lower left',
                 ncol=2,
                 borderaxespad=0.,
                 fontsize=20)

# figure_dirname = 'multi-mnist-figures'
figure_dirname = 'multi-mnist-figures-rerun'
if not os.path.exists(figure_dirname):
    os.makedirs(figure_dirname)

fig.subplots_adjust(hspace=0, wspace=0)
plt.savefig(os.path.join(figure_dirname, 'multi_mnist_{}.png'.format(metric_to_plot)), bbox_extra_artists=[lgd] + extra_labels, bbox_inches='tight', pad_inches=0.02)
plt.savefig(os.path.join(figure_dirname, 'multi_mnist_{}.pdf'.format(metric_to_plot)), bbox_extra_artists=[lgd] + extra_labels, bbox_inches='tight', pad_inches=0.02)
