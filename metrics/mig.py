"""Mutual Information Gap from the beta-TC-VAE paper.

Based on "Isolating Sources of Disentanglement in Variational Autoencoders"
(https://arxiv.org/pdf/1802.04942.pdf).
"""
import numpy as np

from metrics import utils


def compute_mig(ground_truth_data,
                representation_function,
                random_state,
                num_train,
                batch_size=16):
  """Computes the mutual information gap.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    random_state: Numpy random state used for randomness.
    num_train: Number of points used for training.
    batch_size: Batch size for sampling.

  Returns:
    Dict with average mutual information gap.
  """
  print("Generating training set.")
  mus_train, ys_train = utils.generate_batch_factor_code(ground_truth_data,
                                                         representation_function,
                                                         num_train,
                                                         random_state,
                                                         batch_size)
  assert mus_train.shape[1] == num_train
  return _compute_mig(mus_train, ys_train)


def _compute_mig(mus_train, ys_train):
  """Computes score based on both training and testing codes and factors."""
  score_dict = {}
  # discretized_mus = utils.make_discretizer(mus_train)
  # HARDCODED HERE BECAUSE THIS WAS PREVIOUSLY SPECIFIED BY THE GIN CONFIG!
  discretized_mus = utils.make_discretizer(mus_train, num_bins=20, discretizer_fn=utils._histogram_discretize)
  m = utils.discrete_mutual_info(discretized_mus, ys_train)
  assert m.shape[0] == mus_train.shape[0]
  assert m.shape[1] == ys_train.shape[0]
  # m is [num_latents, num_factors]
  entropy = utils.discrete_entropy(ys_train)
  sorted_m = np.sort(m, axis=0)[::-1]
  score_dict["discrete_mig"] = np.mean(np.divide(sorted_m[0, :] - sorted_m[1, :], entropy[:]))
  return score_dict
