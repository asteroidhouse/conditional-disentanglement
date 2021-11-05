# TODO: Figure out how to visualize the saliency maps for different trained models

# 1. Load a model that has been trained from a directory like saves_multi/multi_mnist_cls/...
# 2. Load the original data that it was trained on (re-create the dataset), and make sure that we
#    can run a forward pass of the model and it works
# 3. Figure out how to incorporate the saliency map on top of the image
# 4. Figure out how to evaluate the models on different test-time correlations in a range from [-1,1]
import os
import sys
import pdb
import itertools
import numpy as np
import pickle as pkl
from collections import defaultdict
from ruamel.yaml import YAML
yaml = YAML()
yaml.preserve_quotes = True
yaml.boolean_representation = ['False', 'True']

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import data_utils


def evaluate(dataloader):
  losses = []
  num_correct_f1_dict = defaultdict(int)
  num_correct_f2_dict = defaultdict(int)
  num_total_dict = defaultdict(int)

  num_correct_left = 0
  num_correct_right = 0
  num_total = 0

  all_z_np = []
  all_labels = []
  all_mapped_preds = []

  with torch.no_grad():
    for images, labels in dataloader:
      all_labels.append(labels.numpy())

      images = images.to(device)
      f1_labels = labels[:,0].to(device)
      f2_labels = labels[:,1].to(device)

      z = model(images)
      z1 = z[:,:subspace_dim]
      z2 = z[:,subspace_dim:]

      all_z_np.append(z.detach().cpu().numpy())

      f1_logits = f1_classifier(z1)
      f2_logits = f2_classifier(z2)

      f1_xentropy_loss = F.cross_entropy(f1_logits, f1_labels)
      f2_xentropy_loss = F.cross_entropy(f2_logits, f2_labels)
      loss = (f1_xentropy_loss + f2_xentropy_loss) / 2.0

      losses.append(loss.item())

      possible_labels = [c_to_i[v] for v in classes]
      f1_preds = torch.argmax(f1_logits, axis=1).detach().cpu().numpy()
      f2_preds = torch.argmax(f2_logits, axis=1).detach().cpu().numpy()
      concat_preds = np.concatenate([f1_preds.reshape(-1,1), f2_preds.reshape(-1,1)], axis=1)  # (100, 2)
      mapped_preds = np.ones(concat_preds.shape[0]) * -1  # Just initial array to hold results
      for (i, combo) in enumerate(itertools.product(possible_labels, possible_labels)):
        idxs = np.all(concat_preds == combo, axis=1)
        mapped_preds[idxs] = i
      all_mapped_preds.append(mapped_preds)

      for combo in itertools.product(possible_labels, possible_labels):
        idxs = torch.all(labels == torch.LongTensor(combo), axis=1)
        if not torch.all(idxs == False):
          num_correct_f1_dict[combo] += torch.sum(torch.argmax(f1_logits[idxs], axis=1) == f1_labels[idxs]).item()
          num_correct_f2_dict[combo] += torch.sum(torch.argmax(f2_logits[idxs], axis=1) == f2_labels[idxs]).item()
          num_total_dict[combo] += torch.sum(idxs).item()

      num_correct_left += torch.sum(torch.argmax(f1_logits, axis=1) == f1_labels).item()
      num_correct_right += torch.sum(torch.argmax(f2_logits, axis=1) == f2_labels).item()
      num_total += len(images)

  z_np = np.concatenate(all_z_np)
  z_np -= z_np.mean(axis=0)  # Subtract mean
  z_np /= z_np.std(axis=0)   # Standardize
  covariance = (np.dot(z_np.T, z_np) / (z_np.shape[0]-1))  # Compute covariance

  f1_acc = num_correct_left / float(num_total)
  f2_acc = num_correct_right / float(num_total)

  f1_acc_dict = {combo: num_correct_f1_dict[combo] / float(num_total_dict[combo]) for combo in num_total_dict.keys()}
  f2_acc_dict = {combo: num_correct_f2_dict[combo] / float(num_total_dict[combo]) for combo in num_total_dict.keys()}

  all_z_np = np.concatenate(all_z_np)
  all_labels = np.concatenate(all_labels)
  all_mapped_preds = np.concatenate(all_mapped_preds)

  all_mapped_labels = np.ones(all_labels.shape[0]) * -1
  for (i, combo) in enumerate(itertools.product(possible_labels, possible_labels)):
    idxs = np.all(all_labels == combo, axis=1)
    all_mapped_labels[idxs] = i

  return {'loss': np.mean(losses),
          'f1_acc': f1_acc,
          'f2_acc': f2_acc,
          'f1_acc_dict': f1_acc_dict,
          'f2_acc_dict': f2_acc_dict,
          'all_z_np': all_z_np,
          'all_labels': all_labels,
          'all_mapped_preds': all_mapped_preds,
          'all_mapped_labels': all_mapped_labels}


device = 'cuda:0'

ckpt_prefix = ''
# ckpt_prefix = 'bva-'
# ckpt_prefix = 'bvl-'

# base_dirs = ['saves_multi/multi_mnist_cls', 'saves_multi/multi_mnist_cond', 'saves_multi/multi_mnist_uncond']


# THE FIRST DIRECTORY HERE WAS USED JUST A SECOND AGO (OCTOBER 28?)
# base_dirs = ['saves_multi_many/multi_mnist_cls', 'saves_multi_many/multi_mnist_cond', 'saves_multi_many/multi_mnist_uncond']
base_dirs = ['saves_multi_many_rerun_uncond/multi_mnist_uncond']



# base_dirs = ['saves_multi_combined/multi_mnist_cls', 'saves_multi_combined/multi_mnist_cond', 'saves_multi_combined/multi_mnist_uncond']

for base_dir in base_dirs:
  print('Base dir: {}'.format(base_dir))
  subdirs = os.listdir(base_dir)

  for subdir in subdirs:
    try:
      # if os.path.exists(os.path.join(base_dir, subdir, 'test_corr_range.pkl')):
      #   continue

      args = yaml.load(open(os.path.join(base_dir, subdir, 'args.yaml'), 'r'))

      subspace_dim = args['z_dim'] // 2

      model = torch.load(os.path.join(base_dir, subdir, '{}model.pt'.format(ckpt_prefix)))
      f1_classifier = torch.load(os.path.join(base_dir, subdir, '{}f1_classifier.pt'.format(ckpt_prefix)))
      f2_classifier = torch.load(os.path.join(base_dir, subdir, '{}f2_classifier.pt'.format(ckpt_prefix)))

      model = model.to(device)
      f1_classifier = f1_classifier.to(device)
      f2_classifier = f2_classifier.to(device)

      agg_result_dict = defaultdict(list)

      test_corr_list = np.linspace(-1, 1, 200)
      # Create test data with a range of correlations between -1 and 1
      for test_corr in test_corr_list:
        classes = [3, 8]
        _, _, testset_correlated, c_to_i, i_to_c = data_utils.get_correlated_data('mnist',
                                                                                  classes=classes,
                                                                                  train_corr=0.0,
                                                                                  test_corr=test_corr,
                                                                                  noise=args['noise'],
                                                                                  occlusion_patch_size=args['occlusion_patch_size'])

        possible_labels = [c_to_i[v] for v in classes]
        test_loader_correlated = DataLoader(testset_correlated, batch_size=100, shuffle=True)
        result_dict = evaluate(test_loader_correlated)

        avg_acc = (result_dict['f1_acc'] + result_dict['f2_acc']) / 2.0
        print('f1 acc: {:6.4f} | f2 acc: {:6.4f} | avg acc: {:6.4f}'.format(
               result_dict['f1_acc'], result_dict['f2_acc'], avg_acc))
        sys.stdout.flush()

        agg_result_dict['f1_acc_list'].append(result_dict['f1_acc'])
        agg_result_dict['f2_acc_list'].append(result_dict['f2_acc'])
        agg_result_dict['avg_acc_list'].append(avg_acc)
        agg_result_dict['loss_list'].append(result_dict['loss'])

      agg_result_dict['test_corr_list'] = test_corr_list

      with open(os.path.join(base_dir, subdir, 'test_corr_range.pkl'), 'wb') as f:
        pkl.dump(agg_result_dict, f)
    except:
      print('Issue with {}'.format(subdir))
