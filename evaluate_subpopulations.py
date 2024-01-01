"""Evaluate subpopulation accuracies of each model trained with different
objectives, e.g., Cls, Cls+MI, Cls+CMI

Example
-------
python evaluate_subpopulations.py \
    --target_variable1=Male \
    --target_variable2=Smiling \
    --load=saves/celeba_natural_cond/ft1t2:None_Male_Smiling-trnc:0.0-tstc:0.0-m:mlp-lr:1e-05-clr:0.0001-dlr:0.0001-on:0.0-z:10-mi:conditional-dl:10.0-cls:1-s:3
"""
import os
import sys
import pdb
import argparse
import itertools
from collections import defaultdict

import numpy as np
import sklearn.metrics

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import celeba
from celeba import CELEBA_ATTRS


parser = argparse.ArgumentParser()
parser.add_argument('--target_variable1', type=str, choices=CELEBA_ATTRS,
                    help='First attribute name')
parser.add_argument('--target_variable2', type=str, choices=CELEBA_ATTRS,
                    help='Second attribute name')
parser.add_argument('--load', type=str, default=None,
                    help='Load directory')
args = parser.parse_args()


classes = [0, 1]
c_to_i = {0: 0, 1: 1}
possible_labels = [c_to_i[v] for v in classes]


datasets = celeba.get_natural_correlated_celeba(factor1=args.target_variable1,
                                                factor2=args.target_variable2,
                                                splits=['train', 'val', 'test'])
train_loader = DataLoader(datasets['train'], batch_size=100, shuffle=True)
val_loader = DataLoader(datasets['val'], batch_size=100, shuffle=True)
test_loader = DataLoader(datasets['test'], batch_size=100, shuffle=True)

# This part is just to get the uncorrelated test set
# --------------------------------------------------
datasets_uncorrelated = celeba.get_correlated_celeba(factor1=args.target_variable1,
                                                     factor2=args.target_variable2,
                                                     train_corr=0.0,
                                                     test_corr=0.0,
                                                     noise=0.0,
                                                     splits=['test'])
test_loader_uncorrelated = DataLoader(datasets_uncorrelated['test'], batch_size=100, shuffle=True)
# --------------------------------------------------
sys.stdout.flush()


def evaluate(dataloader):
    losses = []

    # We want to keep track of how many predictions are correct
    # for each of the left and right-hand sides, separately for
    # each of the four combinations of left/right digits,
    # e.g., {(3,3), (3,8), (8,3), (8,8)}
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

            images = images.to(use_device)
            f1_labels = labels[:,0].to(use_device)
            f2_labels = labels[:,1].to(use_device)

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

    left_acc = num_correct_left / float(num_total)
    right_acc = num_correct_right / float(num_total)

    f1_acc_dict = {combo: num_correct_f1_dict[combo] / float(num_total_dict[combo]) for combo in num_total_dict.keys()}
    f2_acc_dict = {combo: num_correct_f2_dict[combo] / float(num_total_dict[combo]) for combo in num_total_dict.keys()}

    all_z_np = np.concatenate(all_z_np)
    all_labels = np.concatenate(all_labels)
    all_mapped_preds = np.concatenate(all_mapped_preds)

    all_mapped_labels = np.ones(all_labels.shape[0]) * -1
    for (i, combo) in enumerate(itertools.product(possible_labels, possible_labels)):
        idxs = np.all(all_labels == combo, axis=1)
        all_mapped_labels[idxs] = i

    return np.mean(losses), left_acc, right_acc, f1_acc_dict, f2_acc_dict, all_z_np, all_labels, all_mapped_preds, all_mapped_labels


use_device = 'cuda:0'
subspace_dim = 5

f1_classifier = torch.load(os.path.join(args.load, 'bva-f1_classifier.pt'))
f2_classifier = torch.load(os.path.join(args.load, 'bva-f2_classifier.pt'))
model = torch.load(os.path.join(args.load, 'bva-model.pt'))

f1_classifier = f1_classifier.to(use_device)
f2_classifier = f2_classifier.to(use_device)
model = model.to(use_device)

tst_loss, tst_f1_acc, tst_f2_acc, tst_l_acc_dict, tst_r_acc_dict, tst_zs, tst_labels, tst_mapped_preds, tst_mapped_labels = evaluate(train_loader)

confusion_mat = sklearn.metrics.confusion_matrix(tst_mapped_preds, tst_mapped_labels, normalize='all')

for i in range(confusion_mat.shape[0]):
    row = confusion_mat[i]
    num_correct = row[i]
    num_incorrect = np.sum(row[:i]) + np.sum(row[i+1:])
    num_total = np.sum(row)
    print(num_correct / num_total)
    sys.stdout.flush()
