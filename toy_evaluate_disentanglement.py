"""Evaluate disentanglement metrics using pre-trained toy model.

Examples
--------
python toy_evaluate_disentanglement.py
"""
import os
import sys
import pdb
import random
import argparse
import itertools
import numpy as np
import pickle as pkl

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from metrics import *


parser = argparse.ArgumentParser()
parser.add_argument('--metrics', type=str, default='dci,irs,mig,sap_score,factor_vae,beta_vae,unsupervised',
                    help='Comma-separated list of disentanglement metrics to measure')
parser.add_argument('--train_correlation', type=float, default=0.0,
                    help='Use models trained with this correlation')
parser.add_argument('--noise', type=float, default=0.0,
                    help='Amount of observation noise (occlusions)')
parser.add_argument('--seed', type=int, default=3,
                    help='Random seed')
args = parser.parse_args()


def get_A(dim):
  A = np.eye(dim)
  A = np.block([[A, np.eye(dim)]])
  return A


def get_translation(dim):
  """Tranlate values between 0 and 2**2 in combinations of possible attribute values.
  """
  all_list = [[-1, 1]] * dim
  return np.array(list(itertools.product(*all_list)))


def sample_z_classification(random_state, num_samples, noise_level, correlation, dim):
  """Sample target for classification task with dim attributes. Thereby, each
  attribute (=dimension) is correlated with each other attribute with correlation
  strength. Only for two dimensions it is possible to reach correlations
  between -1 and 1. For multiple attributes, strong negative correlations
  are impossible.
  """
  # We obtain the desired correlation by making the combinations where all
  # attributes are the same are less common (given by s1) compared to the other
  # combinations of attribute values (given by s2). Thus, s1 and s2 depend on
  # the number of attributes and the correlation strength (for the two
  # attributes s1=c1 and s2=c2):
  c1 = 1 + correlation
  c2 = 1 - correlation
  n = 2**(dim - 2)
  s2 = c2 / n
  s1 = c1 - (n-1) * s2

  if correlation >=0 or dim==2:
    probs = np.ones(2**dim) * s2
    probs[0] = s1
    probs[-1] = s1
  else:
    print('WARNING: negative correlations not implemented for dim>2')

  # Normalize to get probabilities
  probs /= sum(probs)

  # Sample with these probabilities. There are 2**dim different combinations of attributes.
  samples = random_state.choice(np.arange(2**dim), size=num_samples, p=probs)

  # Translate the sampled values to attribute combinations. This is the target.
  translation = get_translation(dim)
  z = translation[list(samples)].T

  # Add noise
  epsilon = random_state.randn(dim, z.shape[1]) * np.sqrt(noise_level)
  z_and_noise = np.vstack((z, epsilon))
  return z, z_and_noise


class DataSampler:
  """Wrapper for the randomly-generated Gaussian dataset to support the same
  type of interface as GroundTruthData.

  Mimics the interface from https://github.com/google-research/disentanglement_lib/blob/86a644d4ed35c771560dc3360756363d35477357/disentanglement_lib/data/ground_truth/ground_truth_data.py
  """
  def __init__(self, dim=2, noise_level=0.0, correlation=0.0):
    self.dim = dim
    self.num_factors = dim
    self.noise_level = noise_level
    self.correlation = correlation

  def sample_observations_from_factors(self, z, random_state):
    epsilon = random_state.randn(*z.shape) * np.sqrt(self.noise_level)
    z_and_noise = np.concatenate((z, epsilon), axis=1)
    x = np.dot(A, z_and_noise.T).T
    return x.astype('float32')

  def sample_factors(self, num, random_state):
    return self.sample(num, random_state)[0]

  def sample_observations(self, num, random_state):
    return self.sample(num, random_state)[1]

  def sample(self, num, random_state):
    z, z_and_noise = sample_z_classification(random_state,
                                             num,
                                             noise_level=self.noise_level,
                                             correlation=self.correlation,
                                             dim=self.dim)
    x = np.dot(A, z_and_noise)

    # Take transposes to convert (2,100) --> (100,2) since
    # the batch dimension needs to come first
    z = z.T
    x = x.T
    return z.astype('float32'), x.astype('float32')


# Compute disentanglement metrics
# --------------------------------------------------------
def compute_metrics(dataset, representation_function, random_state,
                    num_train, num_eval, num_test, batch_size):
  metric_dict = {}

  if 'irs' in args.metrics:
    metric_dict['irs'] = irs.compute_irs(
        dataset,
        representation_function,
        random_state,
        num_train=num_train,
        batch_size=batch_size,
        diff_quantile=0.99
    )

  if 'sap' in args.metrics:
    metric_dict['sap'] = sap_score.compute_sap(
        dataset,
        representation_function,
        random_state,
        num_train=num_train,
        num_test=num_test,
        continuous_factors=False,
        batch_size=batch_size
    )

  if 'unsupervised' in args.metrics:
    metric_dict['unsupervised'] = unsupervised_metrics.unsupervised_metrics(
        dataset,
        representation_function,
        random_state,
        num_train=num_train,
        batch_size=batch_size
    )

  if 'mig' in args.metrics:
    metric_dict['mig'] = mig.compute_mig(
        dataset,
        representation_function,
        random_state,
        num_train=num_train,
        batch_size=batch_size
    )

  if 'fairness' in args.metrics:
    metric_dict['fairness'] = fairness.compute_fairness(
        dataset,
        representation_function,
        random_state,
        num_train=num_train,
        num_test_points_per_class=100,
        artifact_dir=None,
        batch_size=batch_size
    )

  if 'factor_vae' in args.metrics:
    metric_dict['factor_vae'] = factor_vae.compute_factor_vae(
        dataset,
        representation_function,
        random_state,
        num_train=num_train,
        num_eval=num_eval,
        num_variance_estimate=1000,
        batch_size=batch_size
    )

  if 'dci' in args.metrics:
    metric_dict['dci'] = dci.compute_dci(
        dataset,
        representation_function,
        random_state,
        num_train=num_train,
        num_test=num_eval,
        batch_size=batch_size
    )

  if 'beta_vae' in args.metrics:
    metric_dict['beta_vae'] = beta_vae.compute_beta_vae_sklearn(
        dataset,
        representation_function,
        random_state,
        num_train=num_train,
        num_eval=num_eval,
        batch_size=batch_size
    )

  return metric_dict


exp_paths = [(2, 'saves/toy_cls/toy_cls_dim_2_noise_1.6681005372000592_corr_0.95_anticorr_0'),
             (4, 'saves/toy_cls/toy_cls_dim_4_noise_1.6681005372000592_corr_0.95_anticorr_0'),
             (10, 'saves/toy_cls/toy_cls_dim_10_noise_1.6681005372000592_corr_0.95_anticorr_0')
            ]

for correlation in [0.0]:
  print('%' * 90)
  print('TEST CORRELATION = {}'.format(correlation))
  print('%' * 90)
  sys.stdout.flush()

  for dim, exp_dir in exp_paths:
    A = get_A(dim=dim)

    for fname in ['classification.pkl', 'classification_unconditional.pkl', 'classification_conditional.pkl']:
      print('%' * 80)
      print('DIM {} FNAME {}'.format(dim, fname))
      print('%' * 80)

      with open(os.path.join(exp_dir, fname), 'rb') as f:
          weight_dict = pkl.load(f)
      W = weight_dict['W'].cpu()

      def representation_function(x):
        """Representation function for computing disentanglement metrics"""
        x = torch.from_numpy(x)
        z = torch.mm(x, W)
        return z.detach().cpu().numpy()

      random_state = np.random.RandomState(3)
      data_sampler = DataSampler(dim=dim, noise_level=0.1, correlation=correlation)

      metric_dict = compute_metrics(data_sampler,
                                    representation_function,
                                    random_state,
                                    num_train=100000,
                                    num_eval=100000,
                                    num_test=100000,
                                    batch_size=1000)

      print('='*80)
      for key in metric_dict:
        print('\t{}: {}'.format(key, metric_dict[key]))
      print('\n')
      sys.stdout.flush()
