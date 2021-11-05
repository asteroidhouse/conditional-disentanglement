"""Script to load models and compute disentanglement metrics.

DONE: Load the CelebA data and try to get it in an appropriate representation such that we can directly use the
      disentanglement metrics on it

DONE: Try running the disentanglement metrics with a simpler toy model (like an MLP we define in this file) to make
      sure things work structurally

DONE: Get dataloading set up so that ALL the disentanglement metrics work operationally

DONE: Then add support for loading the actual model from a checkpoint and using that as the drop-in representation_function

Examples
--------
srun -p interactive --gres=gpu:1 --mem=16G python evaluate_disentanglement.py
"""
import os
import sys
import ipdb
import random
import argparse
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

# Local imports
import models
import celeba
from celeba import CELEBA_ATTRS
from metrics import dci, irs, mig, sap_score, factor_vae, beta_vae, unsupervised_metrics, fairness


parser = argparse.ArgumentParser()
parser.add_argument('--load_dir', type=str, default=None,
                    help='Experiment directory from which to load a model checkpoint')
parser.add_argument('--metrics', type=str, default='dci',
                    help='Comma-separated list of disentanglement metrics to measure (irs,sap,unsupervised,mig,fairness,factor_vae,dci,beta_vae')

parser.add_argument('--dataset_type', type=str, default='correlated1',
                    help='Dataset type to load (dummy, conditioned, correlated1, correlated2)')
parser.add_argument('--filter_variable', type=str, choices=CELEBA_ATTRS)
parser.add_argument('--target_variable1', type=str, choices=CELEBA_ATTRS, default='Male',
                    help='First attribute name')
parser.add_argument('--target_variable2', type=str, choices=CELEBA_ATTRS, default='Smiling',
                    help='Second attribute name')
parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'resnet'],
                    help='Model')
parser.add_argument('--nhid', type=int, default=50,
                    help='Number of hidden units per layer of the MLP')
parser.add_argument('--z_dim', type=int, default=10,
                    help='Dimension of z')

parser.add_argument('--train_corr', type=float, default=0.0,
                    help='Train-time correlation between the factors')
parser.add_argument('--test_corr', type=float, default=0.0,
                    help='Test-time correlation between the factors')
parser.add_argument('--noise', type=float, default=0.0,
                    help='Amount of observation noise (occlusions)')
parser.add_argument('--seed', type=int, default=3,
                    help='Random seed')
args = parser.parse_args()

use_device = 'cuda:0'

cudnn.benchmark = False
cudnn.deterministic = True

# Set random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

# Load CelebA data
classes = [0, 1]
c_to_i = {0: 0, 1: 1}
possible_labels = [c_to_i[v] for v in classes]

if args.dataset_type == 'correlated1':
    datasets = celeba.get_correlated_celeba(factor1=args.target_variable1,
                                            factor2=args.target_variable2,
                                            train_corr=args.train_corr,
                                            test_corr=args.test_corr,
                                            noise=args.noise,
                                            splits=['train', 'val', 'test'])
    train_loader = DataLoader(datasets['train'], batch_size=100, shuffle=True)
    val_loader = DataLoader(datasets['val'], batch_size=100, shuffle=True)
    test_loader_anticorrelated = DataLoader(datasets['test'], batch_size=100, shuffle=True)

    # This part is just to get the uncorrelated test set
    # --------------------------------------------------
    datasets_uncorrelated = celeba.get_correlated_celeba(factor1=args.target_variable1,
                                                         factor2=args.target_variable2,
                                                         train_corr=0.0,
                                                         test_corr=0.0,
                                                         noise=args.noise,
                                                         splits=['test'])
    test_loader_uncorrelated = DataLoader(datasets_uncorrelated['test'], batch_size=100, shuffle=True)
    # --------------------------------------------------

elif args.dataset_type == 'correlated2':
    datasets = celeba.get_correlated_celeba_sampled(factor1=args.target_variable1,
                                                    factor2=args.target_variable2,
                                                    train_corr=args.train_corr,
                                                    test_corr=args.test_corr,
                                                    noise=args.noise)
    train_loader = DataLoader(datasets['train'], batch_size=100, shuffle=True)
    val_loader = DataLoader(datasets['val'], batch_size=100, shuffle=True)
    test_loader = DataLoader(datasets['test'], batch_size=100, shuffle=True)


class DataSampler:
    """Wrapper for the CelebA dataset to support the same type of interface as GroundTruthData.

    Mimics the interface from https://github.com/google-research/disentanglement_lib/blob/86a644d4ed35c771560dc3360756363d35477357/disentanglement_lib/data/ground_truth/ground_truth_data.py
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.num_factors = 2  # Hardcoded for now
        # self.factors_num_values = ???

        unique_factors = np.unique(self.dataset.labels, axis=0)
        # Use tuple(factor) to convert the numpy array factor into a HASHABLE tuple so that we can use it as a key!
        self.factor_to_observation_dict = {tuple(factor): np.where((self.dataset.labels == factor).all(axis=1)) for factor in unique_factors}

    def sample_observations_from_factors(self, factors, random_state):
        # factors.shape == (100,2)
        # Again need to convert the numpy array factor_val to a tuple for use as a dict key!
        idxs = []
        for factor_val in factors:
            potential_idxs = self.factor_to_observation_dict[tuple(factor_val)][0]
            idx = potential_idxs[random_state.randint(len(potential_idxs))]
            idxs.append(idx)
        idxs = np.array(idxs)
        images, _ = self.dataset[idxs]
        return images.numpy()

    def sample_factors(self, num, random_state):
        return self.sample(num, random_state)[0]

    def sample_observations(self, num, random_state):
        return self.sample(num, random_state)[1]

    def sample(self, num, random_state):
        idxs = random_state.permutation(len(self.dataset))[:num]
        images, labels = self.dataset[idxs]  # images.shape == (100, 3, 64, 64), labels.shape == (100, 2)
        return labels, images.numpy()  # labels is already a numpy array, so no need to convert it


# Compute disentanglement metrics
# --------------------------------------------------------
def compute_metrics(dataset, representation_function, random_state, num_train, num_eval, num_test, batch_size):
    metric_dict = {}

    if 'irs' in args.metrics:
        metric_dict['irs'] = irs.compute_irs(dataset,
                                             representation_function,
                                             random_state,
                                             num_train=num_train,
                                             batch_size=batch_size,
                                             diff_quantile=0.99)

    if 'sap' in args.metrics:
        metric_dict['sap'] = sap_score.compute_sap(dataset,
                                                   representation_function,
                                                   random_state,
                                                   num_train=num_train,
                                                   num_test=num_test,
                                                   continuous_factors=False,  # Not sure what to set here
                                                   batch_size=batch_size)

    if 'unsupervised' in args.metrics:
        metric_dict['unsupervised'] = unsupervised_metrics.unsupervised_metrics(dataset,
                                                                                representation_function,
                                                                                random_state,
                                                                                num_train=num_train,
                                                                                batch_size=batch_size)

    if 'mig' in args.metrics:
        metric_dict['mig'] = mig.compute_mig(dataset,
                                             representation_function,
                                             random_state,
                                             num_train=num_train,
                                             batch_size=batch_size)

    if 'fairness' in args.metrics:
        metric_dict['fairness'] = fairness.compute_fairness(dataset,
                                                            representation_function,
                                                            random_state,
                                                            num_train=num_train,
                                                            num_test_points_per_class=100,
                                                            artifact_dir=None,
                                                            batch_size=batch_size)

    if 'factor_vae' in args.metrics:
        metric_dict['factor_vae'] = factor_vae.compute_factor_vae(dataset,
                                                                  representation_function,
                                                                  random_state,
                                                                  num_train=num_train,
                                                                  num_eval=num_eval,
                                                                  num_variance_estimate=1000,  # Not sure how to set this?
                                                                  batch_size=batch_size)

    if 'dci' in args.metrics:
        metric_dict['dci'] = dci.compute_dci(dataset,
                                             representation_function,
                                             random_state,
                                             num_train=num_train,
                                             num_test=num_eval,
                                             batch_size=batch_size)

    if 'beta_vae' in args.metrics:
        metric_dict['beta_vae'] = beta_vae.compute_beta_vae_sklearn(dataset,
                                                                    representation_function,
                                                                    random_state,
                                                                    num_train=num_train,
                                                                    num_eval=num_eval,
                                                                    batch_size=batch_size)

    return metric_dict


# Load a trained model
# --------------------
# CLASSIFICATION: /scratch/gobi1/pvicol/domain-adaptation/mi-branch/isa/celeba_cls_corr_grid_male_smiling_5/ft1t2:None_Male_Smiling-trnc:0.8-tstc:-0.8-m:mlp-lr:1e-05-clr:0.0001-dlr:0.0001-on:0.0-z:10-mi:none-dl:10.0-cls:1-s:3
# UNCONDITIONAL: /scratch/gobi1/pvicol/domain-adaptation/mi-branch/isa/celeba_uncond_corr_grid_male_smiling_5/ft1t2:None_Male_Smiling-trnc:0.8-tstc:-0.8-m:mlp-lr:1e-05-clr:0.0001-dlr:0.0001-on:0.0-z:10-mi:conditional-dl:10.0-cls:1-s:3
# CONDITIONAL: /scratch/gobi1/pvicol/domain-adaptation/mi-branch/isa/celeba_cond_corr_grid_male_smiling_5/ft1t2:None_Male_Smiling-trnc:0.8-tstc:-0.8-m:mlp-lr:1e-05-clr:0.0001-dlr:0.0001-on:0.0-z:10-mi:conditional-dl:10.0-cls:1-s:3

exp_paths = [('cls', '/scratch/gobi1/pvicol/domain-adaptation/mi-branch/isa/celeba_cls_corr_grid_male_smiling_5/ft1t2:None_Male_Smiling-trnc:0.8-tstc:-0.8-m:mlp-lr:1e-05-clr:0.0001-dlr:0.0001-on:0.0-z:10-mi:none-dl:10.0-cls:1-s:3'),
             ('uncond', '/scratch/gobi1/pvicol/domain-adaptation/mi-branch/isa/celeba_uncond_corr_grid_male_smiling_6/ft1t2:None_Male_Smiling-trnc:0.8-tstc:-0.8-m:mlp-lr:1e-05-clr:0.0001-dlr:0.0001-on:0.0-z:10-mi:unconditional-dl:10.0-cls:1-s:3'),
             ('cond', '/scratch/gobi1/pvicol/domain-adaptation/mi-branch/isa/celeba_cond_corr_grid_male_smiling_5/ft1t2:None_Male_Smiling-trnc:0.8-tstc:-0.8-m:mlp-lr:1e-05-clr:0.0001-dlr:0.0001-on:0.0-z:10-mi:conditional-dl:10.0-cls:1-s:3')
            ]

# model = torch.load(os.path.join(args.load_dir, 'bva-model.pt'))
# ipdb.set_trace()
# os.path.join(args.load_dir, 'bva-f1_classifier.pt')
# os.path.join(args.load_dir, 'bva-f2_classifier.pt')

# os.path.join(args.load_dir, 'bvl-model.pt')
# os.path.join(args.load_dir, 'bvl-f1_classifier.pt')
# os.path.join(args.load_dir, 'bvl-f2_classifier.pt')

# os.path.join(args.load_dir, 'model.pt')
# os.path.join(args.load_dir, 'f1_classifier.pt')
# os.path.join(args.load_dir, 'f2_classifier.pt')
# --------------------

model = models.MLP(input_dim=64*64*3, hidden_dim=args.nhid, output_dim=args.z_dim)
model = model.to(use_device)
model.eval()

for method_name, exp_dir in exp_paths:
    model = torch.load(os.path.join(exp_dir, 'bva-model.pt'))
    # model = torch.load(os.path.join(exp_dir, 'bvl-model.pt'))
    # model = torch.load(os.path.join(exp_dir, 'model.pt'))
    model = model.to(use_device)
    model.eval()

    def representation_function(x):
        """Representation function for computing disentanglement metrics"""
        x = torch.from_numpy(x).to(use_device)
        z = model(x)
        return z.detach().cpu().numpy()

    random_state = np.random.RandomState(3)
    data_sampler = DataSampler(datasets_uncorrelated['test'])
    # current_factors, current_observations = data_sampler.sample(N=100, random_state=random_state)

    metric_dict = compute_metrics(data_sampler,
                                  representation_function,
                                  random_state,
                                  # num_train=1000,
                                  # num_eval=1000,
                                  # num_test=1000,
                                  # batch_size=100)
                                  num_train=10000,
                                  num_eval=10000,
                                  num_test=10000,
                                  batch_size=500)

    print('Method: {}'.format(method_name))
    print('='*80)
    for key in metric_dict:
        print('\t{}: {}'.format(key, metric_dict[key]))
    print('\n')
    sys.stdout.flush()

# ipdb.set_trace()
