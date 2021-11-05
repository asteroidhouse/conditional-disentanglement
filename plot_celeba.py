"""
Plot results of training on correlated CelebA.

Example
-------
python plot_celeba.py
"""
import os
import sys
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

figure_dirname = 'figures_combined_after_grid'
if not os.path.exists(figure_dirname):
    os.makedirs(figure_dirname)

paths = [('celeba_cls_corr_grid_male_smiling_5', 'Cls'),
         # ('celeba_uncond_corr_grid_male_smiling_5', 'Uncond'),
         ('celeba_uncond_corr_grid_male_smiling_6', 'Uncond'),
         ('celeba_cond_corr_grid_male_smiling_5', 'Cond'),
        ]

name_mapping = { 'avg_loss': 'Cross Entropy Loss',
                 'avg_acc': 'Accuracy',
                 'f1_acc': 'Left Accuracy',
                 'f2_acc': 'Right Accuracy'
               }

for metric_to_plot in ['avg_loss', 'avg_acc', 'f1_acc', 'f2_acc']:
    result_dict = defaultdict(lambda: defaultdict(list))
    result_fname_dict = defaultdict(lambda: defaultdict(list))
    for (base_dir, name) in paths:
        for subdir in os.listdir(base_dir):
            try:
                expdir = os.path.join(base_dir, subdir)
                arg_dict = yaml.load(open(os.path.join(base_dir, subdir, 'args.yaml'), 'r'))
                log = load_log(expdir)

                best_epoch = -1
                # best_epoch = np.argmin(log['val_loss'])

                if metric_to_plot == 'avg_loss':
                    train_perf = log['trn_loss'][best_epoch]
                    val_perf = log['val_loss'][best_epoch]
                    test_ac_perf = log['tst_ac_loss'][best_epoch]
                    test_uc_perf = log['tst_uc_loss'][best_epoch]
                elif metric_to_plot == 'avg_acc':
                    train_perf = (log['trn_f1_acc'][best_epoch] + log['trn_f2_acc'][best_epoch]) / 2.0
                    val_perf = (log['val_f1_acc'][best_epoch] + log['val_f2_acc'][best_epoch]) / 2.0
                    test_ac_perf = (log['tst_ac_f1_acc'][best_epoch] + log['tst_ac_f2_acc'][best_epoch]) / 2.0
                    test_uc_perf = (log['tst_uc_f1_acc'][best_epoch] + log['tst_uc_f2_acc'][best_epoch]) / 2.0
                elif metric_to_plot == 'f1_acc':
                    train_perf = log['trn_f1_acc'][best_epoch]
                    val_perf = log['val_f1_acc'][best_epoch]
                    test_ac_perf = log['tst_ac_f1_acc'][best_epoch]
                    test_uc_perf = log['tst_uc_f1_acc'][best_epoch]
                elif metric_to_plot == 'f2_acc':
                    train_perf = log['trn_f2_acc'][best_epoch]
                    val_perf = log['val_f2_acc'][best_epoch]
                    test_ac_perf = log['tst_ac_f2_acc'][best_epoch]
                    test_uc_perf = log['tst_uc_f2_acc'][best_epoch]

                result_dict[name][arg_dict['train_corr']].append((train_perf, val_perf, test_ac_perf, test_uc_perf))
                result_fname_dict[name][arg_dict['train_corr']].append(expdir)
            except:
                pass


    best_result_dict = defaultdict(lambda: {})
    best_fname_dict = defaultdict(lambda: {})
    for name in result_dict:
        for corr in result_dict[name]:
            perfs = result_dict[name][corr]
            # val_perfs = [perf[1] for perf in perfs]
            test_perfs = [perf[3] for perf in perfs]
            if 'acc' in metric_to_plot:
                # best_perf_idx = np.argmax(val_perfs)
                best_perf_idx = np.argmax(test_perfs)
            else:
                # best_perf_idx = np.argmin(val_perfs)
                best_perf_idx = np.argmin(test_perfs)
            best_result_dict[name][corr] = perfs[best_perf_idx]
            best_fname_dict[name][corr] = result_fname_dict[name][corr][best_perf_idx]

    print('Metric: {}'.format(metric_to_plot))
    print('='*80)
    for method_name in best_result_dict:
        for corr in sorted(best_result_dict[method_name].keys()):
            perfs = best_result_dict[method_name][corr]
            if 'acc' in metric_to_plot:
                print('{} Corr={}: Val {:4.2f} | AC {:4.2f} | UC {:4.2f}'.format(method_name, corr, perfs[1]*100.0, perfs[2]*100.0, perfs[3]*100.0))
            else:
                print('{} Corr={}: Val {:4.2e} | AC {:4.2e} | UC {:4.2e}'.format(method_name, corr, perfs[1], perfs[2], perfs[3]))

    for method_name in best_fname_dict:
        for corr in sorted(best_fname_dict[method_name].keys()):
            print('{} Corr={} Path={}'.format(method_name, corr, best_fname_dict[method_name][corr]))

    # colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # colors = ["#E00072", "#00830B", "#2B1A7F", "#E06111", "#4F4C4B", "#02D4F9"]
    colors = ["#E06111", "#4F4C4B", "#02D4F9"]
    linestyles = ['-', '--', ':']

    fig = plt.figure()
    for i, method in enumerate(best_result_dict.keys()):
        correlation_list = sorted(best_result_dict[method].keys())
        train_perf_list = [best_result_dict[method][corr][0] for corr in correlation_list]
        val_perf_list = [best_result_dict[method][corr][1] for corr in correlation_list]
        test_ac_perf_list = [best_result_dict[method][corr][2] for corr in correlation_list]
        test_uc_perf_list = [best_result_dict[method][corr][3] for corr in correlation_list]
        plt.plot(correlation_list, val_perf_list, linewidth=3, color=colors[i], marker='o', label='{} Val'.format(method))
        plt.plot(correlation_list, test_ac_perf_list, linewidth=3, color=colors[i], linestyle='--', marker='o', label='{} AC'.format(method))
        plt.plot(correlation_list, test_uc_perf_list, linewidth=3, color=colors[i], linestyle=':', marker='o', label='{} UC'.format(method))
        # plt.plot(correlation_list, val_perf_list, linewidth=3, color=colors[0], linestyle=linestyles[i], marker='o', label='{} Val'.format(method))
        # plt.plot(correlation_list, test_ac_perf_list, linewidth=3, color=colors[1], linestyle=linestyles[i], marker='o', label='{} AC'.format(method))
        # plt.plot(correlation_list, test_uc_perf_list, linewidth=3, color=colors[2], linestyle=linestyles[i], marker='o', label='{} UC'.format(method))

    plt.xticks([0, 0.2, 0.4, 0.6, 0.8], fontsize=18)
    plt.yticks([0.8, 0.84, 0.88, 0.92, 0.96], fontsize=18)
    plt.xlabel('Correlation', fontsize=20)
    plt.ylabel(name_mapping[metric_to_plot], fontsize=20)
    # plt.legend(fontsize=18, fancybox=True, framealpha=0.3, loc='center left')

    lgd = fig.legend(# bbox_to_anchor=(0.0, 0.95, 1., .102),
                     # bbox_to_anchor=(0.2, 0.95, 1., .102),
                     bbox_to_anchor=(0.95, 0.5),
                     loc='center left',
                     # ncol=3,
                     ncol=1,
                     borderaxespad=0.,
                     fontsize=18)

    plt.savefig(os.path.join(figure_dirname, '{}_corr7.png'.format(metric_to_plot)), bbox_inches='tight', pad_inches=0)
    plt.savefig(os.path.join(figure_dirname, '{}_corr7.pdf'.format(metric_to_plot)), bbox_inches='tight', pad_inches=0)
    plt.close(fig)

