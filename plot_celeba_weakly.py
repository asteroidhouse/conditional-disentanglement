"""
Plot results of training on correlated CelebA with weak supervision.

Example
-------
python plot_celeba_weakly.py
"""

import os
import sys
import csv
import pdb
import argparse
import numpy as np
from collections import defaultdict
import scipy.stats as st

# YAML setup
from ruamel.yaml import YAML
yaml = YAML()
yaml.preserve_quotes = True
yaml.boolean_representation = ['False', 'True']

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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


def get_errorbars_mean(perfs):
    ''' errorbars: 95\% confidence intervals for the mean estimation
    '''
    res_mean = []
    res_error = []
    for i in range(4):
        a = np.array(perfs)[:, i]
        if len(a)!=3:
            print(len(a))
        mean = np.mean(a)
        res_mean.append(mean)
        convidence_interval95 = st.t.interval(0.95, 
                                      len(a)-1, 
                                      loc=mean, 
                                      scale=st.sem(a))
        res_error.append((convidence_interval95 - mean)[1])
        res_error.append(np.std(a))
    return res_mean, res_error


def get_errorbars(perfs):
    ''' errorbars: standardeviation
    '''
    res_mean = []
    res_error = []
    for i in range(4):
        a = np.array(perfs)[:, i]
        res_mean.append(np.mean(a))
        res_error.append(np.std(a))
    return res_mean, res_error


figure_dirname = 'figures_celeba_weakly'
if not os.path.exists(figure_dirname):
    os.makedirs(figure_dirname)

paths = [
         ('saves/celeba_weakly/celeba_cls', 'Base'),
         ('saves/celeba_weakly/celeba_uncond', 'Base+MI'),
         ('saves/celeba_weakly/celeba_cond', 'Base+CMI'),
        ]

corr_mapping = {'0': 'Val (Corr = 0.8)',
                '1': 'Test (Corr = 0.0)',
                '2': 'Test (Corr = -0.8)',
                    }

percentage_mapping = {'0': 10260,
                     '50': 5130,
                     '75': 2565,
                     '90': 1026,
                     '95': 513,
                     '98': 206
                     }


result_dict = defaultdict(lambda: defaultdict(list))
result_fname_dict = defaultdict(lambda: defaultdict(list))

for (base_dir, name) in paths:
    for subdir in os.listdir(base_dir):
        try:
            expdir = os.path.join(base_dir, subdir)        
            arg_dict = yaml.load(open(os.path.join(base_dir, subdir, 'args.yaml'), 'r'))

            log = load_log(expdir)
            best_epoch = -1

            train_perf = (log['trn_f1_acc'][best_epoch] + log['trn_f2_acc'][best_epoch]) / 2.0
            val_perf = (log['val_f1_acc'][best_epoch] + log['val_f2_acc'][best_epoch]) / 2.0
            test_ac_perf = (log['tst_ac_f1_acc'][best_epoch] + log['tst_ac_f2_acc'][best_epoch]) / 2.0
            test_uc_perf = (log['tst_uc_f1_acc'][best_epoch] + log['tst_uc_f2_acc'][best_epoch]) / 2.0

            num_labels = percentage_mapping[str(arg_dict['weak_supervision_percentage'])]
            result_dict[name][num_labels].append((train_perf, val_perf, test_ac_perf, test_uc_perf))
            result_fname_dict[name][num_labels].append(expdir)
        except:
            pass


average_result_dict = defaultdict(lambda: {})
error_result_dict = defaultdict(lambda: {})

for name in result_dict:
    for num_labels in result_dict[name]:
        perfs = result_dict[name][num_labels]
        average_result_dict[name][num_labels], error_result_dict[name][num_labels] = get_errorbars(perfs)

colors = ["#E06111", "#4F4C4B",  "#02D4F9"]
linestyles = ['-', '--', ':']

fig, axs = plt.subplots(1, 3, figsize = (10, 3), sharex='all', sharey='all')

for i, method in enumerate(result_dict.keys()):

    num_labels_list = sorted(average_result_dict[method].keys())
    train_perf_list = [average_result_dict[method][num_labels][0] for num_labels in num_labels_list]
    val_perf_list = [average_result_dict[method][num_labels][1] for num_labels in num_labels_list]
    test_ac_perf_list = [average_result_dict[method][num_labels][2] for num_labels in num_labels_list]
    test_uc_perf_list = [average_result_dict[method][num_labels][3] for num_labels in num_labels_list]

    train_err_list = [error_result_dict[method][num_labels][0] for num_labels in num_labels_list]
    val_err_list = [error_result_dict[method][num_labels][1] for num_labels in num_labels_list]
    test_ac_err_list = [error_result_dict[method][num_labels][2] for num_labels in num_labels_list]
    test_uc_err_list = [error_result_dict[method][num_labels][3] for num_labels in num_labels_list]

    axs[0].errorbar(num_labels_list, val_perf_list, yerr=val_err_list,linewidth=2, markersize=5, color=colors[i], marker='o', label='{} Val'.format(method), zorder=i+2)
    axs[2].errorbar(num_labels_list, test_ac_perf_list, yerr=test_ac_err_list,linewidth=2, markersize=5, color=colors[i], marker='o', label='{} AC'.format(method), zorder=i+2)
    axs[1].errorbar(num_labels_list, test_uc_perf_list, yerr=test_uc_err_list,linewidth=2, markersize=5, color=colors[i], marker='o', label='{} UC'.format(method), zorder=i+2)
    axs[i].set_ylim(ymin=0.7, ymax=1)
    axs[i].set_title(corr_mapping[str(i)], fontsize=15)
    axs[i].set_xscale('log')
    axs[i].grid(True, axis='y', zorder = 0)


axs[0].set_xlabel('#Labels per attribute', fontsize=15)
axs[1].set_xlabel('#Labels per attribute', fontsize=15)
axs[2].set_xlabel('#Labels per attribute', fontsize=15)
axs[0].set_ylabel('Accuracy', fontsize=15)
fig.subplots_adjust(wspace=0, hspace=0)

lgd = fig.legend(['Base', 'Base + MI', 'Base + CMI'],
            bbox_to_anchor=(0.2, 0.95, 1., .102),
            loc='lower left',
            ncol=3,
            borderaxespad=1,
            fontsize=15)
plt.savefig(os.path.join(figure_dirname, 'avg_acc_weak_supervison_errorbars.png'), bbox_inches='tight', pad_inches=0)
plt.savefig(os.path.join(figure_dirname, 'avg_acc_weak_supervison_errorbars.pdf'), bbox_inches='tight', pad_inches=0)
plt.close(fig)
