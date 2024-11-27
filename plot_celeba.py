"""
Plot results of training on correlated CelebA.

Example
-------
python plot_celeba.py
"""
import os
import sys
import csv
import pdb
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


figure_dirname = 'figures'
if not os.path.exists(figure_dirname):
  os.makedirs(figure_dirname)

paths = [
    ('saves/celeba_cls', 'Base'),
    ('saves/celeba_uncond', 'Base+MI'),
    ('saves/celeba_cond', 'Base+CMI'),
]

name_mapping = {
    'avg_loss': 'Cross Entropy Loss',
    'avg_acc': 'Accuracy',
    'f1_acc': 'Left Accuracy',
    'f2_acc': 'Right Accuracy'
}

seed_list = []

for use_metric in ['avg_acc']:
  result_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
  result_fname_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
  for (base_dir, name) in paths:
    for subdir in os.listdir(base_dir):
      try:
        expdir = os.path.join(base_dir, subdir)
        arg_dict = yaml.load(open(os.path.join(base_dir, subdir, 'args.yaml'), 'r'))
        seed = arg_dict['seed']
        if seed not in seed_list:
          seed_list.append(seed)

        log = load_log(expdir)
        use_epoch = -1

        if use_metric == 'avg_loss':
          train_perf = log['trn_loss'][use_epoch]
          val_perf = log['val_loss'][use_epoch]
          test_ac_perf = log['tst_ac_loss'][use_epoch]
          test_uc_perf = log['tst_uc_loss'][use_epoch]
        elif use_metric == 'avg_acc':
          train_perf = (log['trn_f1_acc'][use_epoch] + log['trn_f2_acc'][use_epoch]) / 2.0
          val_perf = (log['val_f1_acc'][use_epoch] + log['val_f2_acc'][use_epoch]) / 2.0
          test_ac_perf = (log['tst_ac_f1_acc'][use_epoch] + log['tst_ac_f2_acc'][use_epoch]) / 2.0
          test_uc_perf = (log['tst_uc_f1_acc'][use_epoch] + log['tst_uc_f2_acc'][use_epoch]) / 2.0
        elif use_metric == 'f1_acc':
          train_perf = log['trn_f1_acc'][use_epoch]
          val_perf = log['val_f1_acc'][use_epoch]
          test_ac_perf = log['tst_ac_f1_acc'][use_epoch]
          test_uc_perf = log['tst_uc_f1_acc'][use_epoch]
        elif use_metric == 'f2_acc':
          train_perf = log['trn_f2_acc'][use_epoch]
          val_perf = log['val_f2_acc'][use_epoch]
          test_ac_perf = log['tst_ac_f2_acc'][use_epoch]
          test_uc_perf = log['tst_uc_f2_acc'][use_epoch]

        result_dict[name][arg_dict['train_corr']][seed].append(
            (train_perf, val_perf, test_ac_perf, test_uc_perf)
        )
        result_fname_dict[name][arg_dict['train_corr']][seed].append(expdir)
      except:
        pass


  best_result_dict = defaultdict(lambda: defaultdict(list))
  best_fname_dict = defaultdict(lambda: defaultdict(list))
  for name in result_dict:
    for corr in result_dict[name]:
      for seed in result_dict[name][corr]:
        perfs = result_dict[name][corr][seed]
        test_perfs = [perf[3] for perf in perfs]
        if 'acc' in use_metric:
          best_perf_idx = np.argmax(test_perfs)
        else:
          best_perf_idx = np.argmin(test_perfs)
        best_result_dict[name][corr].append(perfs[best_perf_idx])
        best_fname_dict[name][corr].append(result_fname_dict[name][corr][best_perf_idx])

  print('Metric: {}'.format(use_metric))
  print('='*80)
  for method_name in best_result_dict:
    for corr in sorted(best_result_dict[method_name].keys()):
      perfs = best_result_dict[method_name][corr][0]
      if 'acc' in use_metric:
        print('{} Corr={}: Val {:4.2f} | AC {:4.2f} | UC {:4.2f}'.format(
              method_name, corr, perfs[1]*100.0, perfs[2]*100.0, perfs[3]*100.0))
      else:
        print('{} Corr={}: Val {:4.2e} | AC {:4.2e} | UC {:4.2e}'.format(
              method_name, corr, perfs[1], perfs[2], perfs[3]))

  for method_name in best_fname_dict:
    for corr in sorted(best_fname_dict[method_name].keys()):
      print('{} Corr={} Path={}'.format(method_name, corr, best_fname_dict[method_name][corr][0]))

  colors = ["#E06111", "#4F4C4B", "#02D4F9"]
  linestyles = ['-', '--', ':']
  marker = 'o'

  fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(14, 4), sharex='all', sharey='all')

  for j, idx in enumerate([1, 3, 2]):  # Val, Uncorr, Anticorr
    for i, method in enumerate(best_result_dict.keys()):
      correlation_list = sorted(best_result_dict[method].keys())
      curr_perf_list = [best_result_dict[method][corr] for corr in correlation_list]
      val_perf_list = [[item[idx] for item in sublist] for sublist in curr_perf_list]
      val_perf_list = np.array(val_perf_list).T
      val_means = np.mean(val_perf_list, axis=0)
      val_errors = np.std(val_perf_list, axis=0)
      axs[j].errorbar(
          correlation_list, val_means, yerr=val_errors,
          linewidth=3, color=colors[i], marker=marker, label=method
      )

  axs[0].set_title(r'Val (Correlated)', fontsize=22)
  axs[1].set_title(r'Test (Uncorrelated)', fontsize=22)
  axs[2].set_title(r'Test (Anticorrelated)', fontsize=22)

  axs[0].tick_params(axis='both', which='both', labelsize=16)
  axs[1].tick_params(axis='both', which='both', labelsize=16)
  axs[2].tick_params(axis='both', which='both', labelsize=16)

  axs[0].set_xlabel('Train Correlation', fontsize=20)
  axs[1].set_xlabel('Train Correlation', fontsize=20)
  axs[2].set_xlabel('Train Correlation', fontsize=20)
  axs[0].set_ylabel(name_mapping[use_metric], fontsize=20)
  axs[0].grid(axis='y')
  axs[1].grid(axis='y')
  axs[2].grid(axis='y')

  lgd = fig.legend(
      ['Base', 'Base + MI', 'Base + CMI'],
      bbox_to_anchor=(0.25, 1.02, 1, 0.2),
      loc='center left',
      ncol=3,
      borderaxespad=0.,
      fontsize=20
  )

  plt.ylim(0.7, 0.97)

  fig.subplots_adjust(hspace=0, wspace=0)
  plt.savefig(
      os.path.join(figure_dirname, '{}_corr.png'.format(use_metric)),
      bbox_inches='tight', pad_inches=0
  )
  plt.savefig(
      os.path.join(figure_dirname, '{}_corr.pdf'.format(use_metric)),
      bbox_inches='tight', pad_inches=0
  )
  plt.close(fig)
