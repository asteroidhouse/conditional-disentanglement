"""Code to reproduce results and figures for toy linear classification.

**Notation:**
- Target: s
- Data: x
- Latent space: z

Example
-------
python toy_linear_classification.py 4 0
"""
import os
import sys
import csv
import pdb
import time
import itertools
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib
matplotlib.rc('text', usetex=True)  # Activate latex text rendering


save_dir = 'saves/linear_classification/'
if not os.path.exists(save_dir):
  os.makedirs(save_dir)

use_device = 'cpu'
N = 10000
criterion_classification = nn.BCEWithLogitsLoss()

# Numbers of stacked images for PAC-GAN (https://arxiv.org/pdf/1712.04086.pdf)
pac_items = 50


def forward_generator(x, W, R, dim):
  """Compute predictions given W and R
  """
  z = torch.mm(W, x)
  a_hat = []
  for d in range(dim):
    a_hat.append(R[d:d+1] * z[d:d+1])
  a_hat = torch.cat((a_hat))
  return a_hat


def compute_x(A, a):
  """Compute data from target.
  """
  x = np.dot(A, a)
  return x


def get_translation(dim):
  """Tranlate values between 0 and 2**2 in combinations of possible attribute values.
  """
  all_list = [[-1, 1]] * dim
  return np.array(list(itertools.product(*all_list)))


def sample_a_classification(num_samples, noise_level, correlation, dim):
  """Sample target for classification task with dim attributes. Thereby, each
  attribute (=dimension) is correlated with each other attribute with correlation
  strength. Only for two dimensions it is possible to reach correlations
  between -1 and 1. For multiple attributes, strong negative correlations
  are impossible.
  """
  # We obtain the desired correlation, by making the combinations where all
  # attributes are the same are less common (given by t1) compared to the other
  # combinations of attribute values (given by t2). Thereby, we t1 and t2 depend
  # on the number of attributes and the correlation strength
  # (for 2 attributes t1=c1 and t2=c2):
  c1 = 1 + correlation
  c2 = 1 - correlation
  n = 2 ** (dim - 2)
  t2 = c2 / n
  t1 = c1 - (n-1) * t2

  if (correlation >= 0) or (dim == 2):
    probs = np.ones(2**dim) * t2
    probs[0] = t1
    probs[-1] = t1
  else:
    print('WARNING: negative correlations not implemented for dim>2')

  # normalize to get probabilities:
  probs /= sum(probs)

  # sample with these probabilities. There are 2**dim different combinations of attributes.
  samples = np.random.choice(np.arange(2**dim), size=num_samples, p=probs)

  # Translate the sampled values to attribute combinations. This is the target.
  translation = get_translation(dim)
  a = translation[list(samples)].T

  # Add noise
  epsilon = np.random.randn(dim, a.shape[1]) * np.sqrt(noise_level)
  a_and_noise = np.vstack((a, epsilon))
  return a, a_and_noise


class Net(nn.Module):
  """Architecture used for the discriminators.
  """
  def __init__(self, dim):
    super(Net, self).__init__()
    self.dim = dim
    self.fc1 = nn.Linear(self.dim * pac_items, 128)  # PacGAN
    self.fc2 = nn.Linear(128, 128)
    self.fc3 = nn.Linear(128, 1)

  def forward(self, x):
    x = x.permute([1, 0])
    x = torch.reshape(x, (-1, self.dim * pac_items))  # PacGAN
    x = nn.Softplus()(self.fc1(x))
    x = nn.Softplus()(self.fc2(x))
    x = self.fc3(x)
    return x


class Discriminator():
  """Discriminator for unconditional loss
  """
  def __init__(self, N, dim, lr):
    self.N = N
    self.dim = dim

    self.net = Net(self.dim).to(use_device)
    self.parameters = self.net.parameters()

    self.lr = lr
    self.D_optimizer = optim.Adam(list(self.parameters), lr=self.lr)
    self.criterion = nn.BCEWithLogitsLoss()

  def train(self, z, num_i_inner=1):
    running_loss_reals = 0
    running_loss_fakes = 0

    for i_inner in range(num_i_inner):
      # Shuffle z_real, such that there are different inputs stacked in each iteration
      z_real = z[:, torch.randperm(self.N)]

      # Shuffle z to get z_fake
      z_fake = torch.clone(z_real)
      for d in range(self.dim):
        z_fake[d, :] = z_fake[d, torch.randperm(self.N)]

      # Take an optimization step.
      self.D_optimizer.zero_grad()
      pred_reals = self.net(z_real).flatten()
      pred_fakes = self.net(z_fake).flatten()
      label_reals = torch.ones(self.N // pac_items).to(use_device)
      label_fakes = torch.zeros(self.N // pac_items).to(use_device)
      loss_discriminator_reals = self.criterion(pred_reals, label_reals)
      loss_discriminator_fakes = self.criterion(pred_fakes, label_fakes)

      loss_discriminator = (loss_discriminator_reals + loss_discriminator_fakes) / 2
      loss_discriminator.backward()
      self.D_optimizer.step()

      running_loss_reals += loss_discriminator_reals.item()
      running_loss_fakes += loss_discriminator_fakes.item()
      if i_inner % 1000 == 999:    # print every 1000 iterations
        print('[{:5d}] loss: {:.3f} ({:.3f})'.format(
              i_inner + 1, running_loss_reals / 1000, running_loss_fakes / 1000))
        sys.stdout.flush()
        running_loss_reals = 0.0
        running_loss_fakes = 0.0

    return ((loss_discriminator_reals + loss_discriminator_fakes) / 2).cpu().detach().numpy()

  def eval_fakes(self, z):
    """This loss measures whether _fake_ representations look like _real_ representations.
    """
    z_fake = torch.clone(z)
    for d in range(self.dim):
      z_fake[d, :] = z_fake[d, torch.randperm(self.N)]
    pred_fakes = self.net(z_fake).flatten()
    label_fakes = torch.ones(self.N // pac_items).to(use_device)
    return self.criterion(pred_fakes, label_fakes)

  def eval_reals(self, z):
    """This loss measures whether _real_ representatiosn look like _fake_ representations.
    """
    pred_reals = self.net(z).flatten()
    label_reals = torch.zeros(self.N // pac_items).to(use_device)
    return self.criterion(pred_reals, label_reals)


class ConditionalDiscriminator():
  """Discriminator for conditional loss.
  """
  def __init__(self, N, dim, lr, condition_dim, condition_value, s):
    self.dim = dim
    self.condition_dim = condition_dim
    self.condition_value = condition_value

    self.net = Net(self.dim).to(use_device)
    self.parameters = self.net.parameters()

    self.lr = lr
    self.D_optimizer = optim.Adam(list(self.parameters), lr=self.lr)
    self.criterion = nn.BCEWithLogitsLoss()

    self.s = s
    if self.condition_value == -1:
      self.mask = s[self.condition_dim] < 0.5
    elif self.condition_value == 1:
      self.mask = s[self.condition_dim] > 0.5

    self.N = self.mask.sum() // pac_items * pac_items

  def train(self, z, num_i_inner=1):
    z = z[:, self.mask]
    z = z[:, :self.N]
    running_loss_reals = 0
    running_loss_fakes = 0

    for i_inner in range(num_i_inner):
      # Shuffle z_real, such that there are different inputs stacked in each iteration:
      z_real = z[:, torch.randperm(self.N)]

      # Shuffle condition_dim to get z_fake
      z_fake = torch.clone(z_real)
      z_fake[self.condition_dim, :] = z_fake[self.condition_dim, torch.randperm(self.N)]

      # Take an optimization step
      self.D_optimizer.zero_grad()
      pred_reals = self.net(z_real).flatten()
      pred_fakes = self.net(z_fake).flatten()
      label_reals = torch.ones(self.N // pac_items).to(use_device)
      label_fakes = torch.zeros(self.N // pac_items).to(use_device)
      loss_discriminator_reals = self.criterion(pred_reals, label_reals)
      loss_discriminator_fakes = self.criterion(pred_fakes, label_fakes)

      loss_discriminator = (loss_discriminator_reals + loss_discriminator_fakes) / 2
      loss_discriminator.backward()
      self.D_optimizer.step()

      # print intermediate steps
      running_loss_reals += loss_discriminator_reals.item()
      running_loss_fakes += loss_discriminator_fakes.item()
      if i_inner % 1000 == 999:    # print every 1000 iterations
        print('[{:5d}] loss: {:.3f} ({:.3f})'.format(
               i_inner + 1, running_loss_reals / 1000, running_loss_fakes / 1000))
        sys.stdout.flush()
        running_loss_reals = 0.0
        running_loss_fakes = 0.0

    return ((loss_discriminator_reals + loss_discriminator_fakes) / 2).cpu().detach().numpy()

  def eval_fakes(self, z):
    """This loss tests if Fakes look like Reals
    """
    z = z[:, self.mask]
    z = z[:, :self.N]  # z must be dividable by pac_items
    z_fake = torch.clone(z)
    z_fake[self.condition_dim, :] = z_fake[self.condition_dim, torch.randperm(self.N)]
    pred_fakes = self.net(z_fake).flatten()
    label_fakes = torch.ones(self.N // pac_items).to(use_device)
    return self.criterion(pred_fakes, label_fakes)

  def eval_reals(self, z):
    """This loss tests if Reals look like Fakes
    """
    z = z[:, self.mask]
    z = z[:, :self.N]  # z must be dividable by pac_items
    pred_reals = self.net(z).flatten()
    label_reals = torch.zeros(self.N // pac_items).to(use_device)
    return self.criterion(pred_reals, label_reals)


def get_classification_accuracy(a_hat, a_discret):
  a_hat_discret = (a_hat > 0).float()
  correct = (a_hat_discret == a_discret).float().mean()
  return correct.item()


def minimize(x, a, loss_terms, dim):
  """Minimize loss. loss_terms defines which loss term to use
  (pure classification/ unconditional independence / conditional independence).
  """
  # Use different learning rate based on the number of attributes
  if dim == 2:
    possible_lr_discr = [5e-4, 1e-3, 5e-3]
  elif dim == 4:
    possible_lr_discr = [1e-4, 5e-4, 1e-3]
  elif dim == 10:
    possible_lr_discr = [1e-4, 2e-4, 5e-4, 1e-3]

  # loop over possible learning rates and stop if result is good enough
  for lr_discr in possible_lr_discr:
    if loss_terms == 'classification' or not initC:
      W = torch.eye(dim)
      R = torch.ones(dim) * 0.5
    else:
      W = torch.Tensor(np.load(os.path.join(save_dir, 'W_{}.npy'.format(job_id))))
      R = torch.Tensor(np.load(os.path.join(save_dir, 'R_{}.npy'.format(job_id))))

    W = W.to(use_device)
    R = R.to(use_device)
    W.requires_grad_()
    R.requires_grad_()

    if loss_terms == 'classification':
      steps = 1000
      lr_discr = 'nan'
      lr_regr = 0.01
      lr_gen = 0.01
    elif loss_terms == 'classification_unconditional':
      lr_discr = lr_discr
      lr_gen = lr_discr / 10
      lr_regr = lr_discr / 10
      steps = 8000
      discr = Discriminator(N, dim, lr_discr)
    elif loss_terms == 'classification_conditional':
      lr_discr = 0.01
      lr_gen = 0.0001
      lr_regr = 0.001
      if initC:
        steps = 8000
      else:
        steps = 1
      a_discret = (a > 0).float()
      discr_list = []
      for d in range(dim):
        discr_list.append(
            ConditionalDiscriminator(N, dim, lr_discr, d, -1, s=a_discret)
        )
        discr_list.append(
            ConditionalDiscriminator(N, dim, lr_discr, d, 1, s=a_discret)
        )

    optimizer = optim.Adam([W, R], lr=lr_gen)
    optimizer_for_R = optim.Adam([R], lr=lr_regr)

    running_loss = 0.0
    start = time.time()
    for i in range(steps):
      # lr decay
      if i == steps//2:
        for g in optimizer.param_groups:
          g['lr'] = lr_gen / 10
        for g in optimizer_for_R.param_groups:
          g['lr'] = lr_regr / 10
        if loss_terms == 'classification_unconditional':
          discr.lr = lr_discr / 10
        elif loss_terms == 'classification_conditional':
          for discr in discr_list:
            discr.lr = lr_discr / 10

      # Different ways to weight the disentanglement loss relative to the
      # classification loss
      if adaptiveweight == 'fix100':
        weight_disentanglement = 100
      elif adaptiveweight == 'fix1000':
        weight_disentanglement = 1000
      elif adaptiveweight == 'interpolate':
        weight_disentanglement = np.logspace(-2, 2, steps)[i]  # [0.01, 100]
      elif adaptiveweight == 'interpolate2':
        weight_disentanglement = np.logspace(-2, 3, steps)[i]  # [0.01, 1000]
      elif adaptiveweight == 'interpolate_linear':
        weight_disentanglement = np.linspace(0, 100, steps)[i]  # [0.01, 1000]
      elif adaptiveweight == 'schedule':
        if i < (steps//4):
          weight_disentanglement = 10
        if i > (steps//4 * 2):
          weight_disentanglement = 100
        if i > (steps//4 * 3):
          weight_disentanglement = 1000

      optimizer.zero_grad()
      a_hat = forward_generator(x, W, R, dim)

      a_discret = (a > 0).float()
      loss_classification = 0
      for d in range(dim):
        loss_classification += criterion_classification(a_hat[d], a_discret[d])
      loss = loss_classification / dim

      if loss_terms == 'classification_unconditional':
        _ = discr.train(z = torch.mm(W.detach(), x))
        loss_disentanglement = discr.eval_reals(z = torch.mm(W, x))
        loss_disentanglement += discr.eval_fakes(z = torch.mm(W, x))
        loss_disentanglement /= 2
        loss += weight_disentanglement * loss_disentanglement
      elif loss_terms == 'classification_conditional':
        loss_disentanglement = 0
        for discr in discr_list:
          loss_discriminator = discr.train(z = torch.mm(W.detach(), x))
          loss_disentanglement += discr.eval_reals(z = torch.mm(W, x))
          loss_disentanglement += discr.eval_fakes(z = torch.mm(W, x))

        loss_disentanglement /= dim * 2 * 2  # Two conditions, reals/fakes
        loss += weight_disentanglement * loss_disentanglement

      loss.backward()
      optimizer.step()

      # Take a few extra steps to adapt R
      for _ in range(10):
        optimizer_for_R.zero_grad()
        a_hat = forward_generator(x, W, R, dim)

        loss_classification_inner = 0
        a_discret = (a > 0).float()
        for d in range(dim):
          loss_classification_inner  += criterion_classification(a_hat[d], a_discret[d])
        loss_classification_inner  = loss_classification_inner  / dim
        loss_classification_inner.backward()

        optimizer_for_R.step()

      # Print statistics
      running_loss += loss.item()
      if i % 1000 == 999 and verbose:
        print('[{:5d}] loss: {:.3f}'.format(i + 1, running_loss / 1000))
        end = time.time()
        print('time for 1000 iterations: {}'.format(end - start))
        sys.stdout.flush()
        start = time.time()
        running_loss = 0.0

    # After GAN training, train only the discriminator to see its final performance
    if loss_terms == 'classification':
      loss_discriminator = 1
    elif loss_terms == 'classification_unconditional':
      discr = Discriminator(N, dim, 0.01)
      loss_discriminator = discr.train(z = torch.mm(W.detach(), x), num_i_inner=1000)
    elif loss_terms == 'classification_conditional':
      discr_list = []
      for d in range(dim):
        discr_list.append(ConditionalDiscriminator(N, dim, 0.01, d, -1, s=a_discret))
        discr_list.append(ConditionalDiscriminator(N, dim, 0.01, d, 1, s=a_discret))
      loss_discriminator = 0
      for discr in discr_list:
        loss_discriminator += discr.train(z = torch.mm(W.detach(), x), num_i_inner=1000)
      loss_discriminator /= dim * 2

    if loss_terms == 'classification':
      np.save(os.path.join(save_dir, 'W_{}.npy'.format(job_id)), W.detach().cpu().numpy())
      np.save(os.path.join(save_dir, 'R_{}.npy'.format(job_id)), R.detach().cpu().numpy())

    # If result is good enough return it. Otherwise try another learning rate
    if loss_terms == 'classification_unconditional':
      tmp =  np.corrcoef(a_hat.cpu().detach().numpy())
      corrcoeff_shat = (tmp.sum() - np.diag(tmp).sum()) / (dim**2 - dim)
      if loss_discriminator > 0.65 and np.abs(corrcoeff_shat < 0.15):
        return W, R, loss_classification_inner.detach().cpu().numpy(), loss_discriminator, lr_discr
    else:
      return W, R, loss_classification_inner.detach().cpu().numpy(), loss_discriminator, lr_discr
  return W, R, loss_classification_inner.detach().cpu().numpy(), loss_discriminator, lr_discr


def test_classification(correlation, noise_level, A, W, R, dim):
  """Compute classification accuracy averaged over all attributes.
  """
  a, a_and_noise = sample_a_classification(N, noise_level, correlation, dim=dim)
  x = compute_x(A, a_and_noise)
  x = torch.Tensor(x).to(use_device)
  a = torch.Tensor(a).to(use_device)
  a_hat = forward_generator(x, W, R, dim)

  a_discret = (a > 0).float()
  classification_accuracy = 0
  for d in range(dim):
    classification_accuracy += get_classification_accuracy(a_hat[d], a_discret[d])
  return classification_accuracy / dim


def get_results(correlation, noise_level, A, anti_correlation, loss_terms, dim, filename):
  """Get accuracy on training data, test data and min/max for a sweep over correlations
  """
  a, a_and_noise = sample_a_classification(N, noise_level, correlation, dim)
  x = compute_x(A, a_and_noise)

  x = torch.Tensor(x).to(use_device)
  a = torch.Tensor(a).to(use_device)
  A = torch.Tensor(A)

  W, R, final_cls_loss, final_discr_loss, lr_discr = minimize(x, a, loss_terms, dim)

  # Save the final W and R values resulting from the minimization, so that we can later load/evaluate them
  base_save_dir = 'toy_cls_dim_{}_noise_{}_corr_{}_anticorr_{}'.format(
                   dim, noise_level, correlation, anti_correlation)
  if not os.path.exists(base_save_dir):
    os.makedirs(base_save_dir)

  with open(os.path.join(base_save_dir, '{}.pkl'.format(loss_terms)), 'wb') as f:
    pkl.dump({ 'W': W, 'R': R }, f)

  res = test_classification(correlation, noise_level, A, W, R, dim)
  res_anti = test_classification(anti_correlation, noise_level, A, W, R, dim)

  # save res to csv
  a_hat = forward_generator(x, W, R, dim)
  tmp =  np.corrcoef(x.cpu().detach().numpy())
  corrcoeff_x = (tmp.sum() - np.diag(tmp).sum()) / (dim**2 - dim)
  tmp =  np.corrcoef(a_hat.cpu().detach().numpy())
  corrcoeff_shat = (tmp.sum() - np.diag(tmp).sum()) / (dim**2 - dim)
  optimal_W = list(W.flatten().detach().cpu().numpy())
  fields = [loss_terms,
            correlation,
            noise_level,
            lr_discr,
            np.round(res * 100, 1),
            np.round(res_anti * 100, 1),
            corrcoeff_shat,
            corrcoeff_x,
            final_cls_loss,
            final_discr_loss,
            optimal_W]

  with open(filename, 'a') as f:
    writer = csv.writer(f)
    writer.writerow(fields)

  res_tests = []
  if dim == 2:
    test_correlations = np.linspace(-1, 1, 50)
  else:
    test_correlations = np.linspace(0, 1, 25)
  for test_correlation in test_correlations:
    res_test = test_classification(test_correlation, noise_level, A, W, R, dim)
    res_tests.append(res_test)

  return res, res_anti, min(res_tests), max(res_tests)  # returns accuracy


def get_A(dim):
  A = np.eye(dim)
  A = np.block([[A, np.eye(dim)]])
  return A


def vary_A_noise_and_correlation(list_noise_level, list_correlation, dim, filename):
  """The output results_combined contains the results in the following shape:
  [3, num_trials, len(list_correlation), len(list_noise_level), 4]

  It contains the following information in these dimensions:
  [classification/unconditional/conditional,
      different A,
      correlation,
      noise_level,
      variance_explained/variance_explained_test,
  ]
  """
  results_classification = np.zeros([len(list_correlation), len(list_noise_level), 4])
  results_unconditional = np.copy(results_classification)
  results_conditional = np.copy(results_classification)

  A = get_A(dim)

  for correlation_index in range(len(list_correlation)):
    correlation = list_correlation[correlation_index]
    print(correlation)
    anti_correlation = 0#-correlation # test on correlation=0

    for noise_level_index in range(len(list_noise_level)):
      noise_level = list_noise_level[noise_level_index]
      print(noise_level)
      full_index = (correlation_index, noise_level_index)

      results_classification[full_index] = get_results(
          correlation,
          noise_level,
          A,
          anti_correlation,
          'classification',
          dim=dim,
          filename=filename
      )

      results_unconditional[full_index] = get_results(
          correlation,
          noise_level,
          A,
          anti_correlation,
          'classification_unconditional',
          dim=dim,
          filename=filename
      )

      results_conditional[full_index] = get_results(
          correlation,
          noise_level,
          A,
          anti_correlation,
          'classification_conditional',
          dim=dim,
          filename=filename
      )

  results_combined = np.array([results_classification, results_unconditional, results_conditional])
  return results_combined


def plot_noise_dependency(ax, x, results, reference=None):
  """Plot VE in dependence of noise level for a given loss_type and correlation

  results is organized as
  [different A,
      noise_level,
      variance_explained/variance_explained_test,
  ]
  """
  colors = ['#E00072', '#00830B', '#2B1A7F', '#E06111', '#4F4C4B', '#02D4F9']
  color_train = colors[0]
  color_test = colors[1]

  # Plot performance for training and test
  ax.plot(x, results[:, 0], linewidth=2, color=color_train, marker='o', markersize=3)
  ax.plot(x, results[:, 1], linewidth=2, color=color_test, marker='o', markersize=3)

  if reference is not None:
    ax.plot(x, reference[:, 1], linewidth=1, color='black', linestyle='dashed')

  # Plot max and min for test sets
  ax.fill_between(x, results[:, 2], results[:, 3], color=color_test, alpha=.1)

  ax.set_xscale('log')
  ax.set_ylim([0.45, 1.05])
  ax.set_yticks([0.5, 1])
  ax.tick_params(axis='both', which='both', labelsize=18)


def figure_noise_dependency(dim):
  """Generates figure consisting of subplots for loss type and correlation.
  Each subplot shows the VE in dependence of noise level.
  """
  subfolder = save_dir + adaptiveweight + '_initC{}_dim{}_test0/'.format(initC, dim)

  if load:
    results_combined = np.load(os.path.join(subfolder, 'results_combined.npy'))
    list_noise_level = np.load(os.path.join(subfolder, 'list_noise_level.npy'))
    list_correlation = np.load(os.path.join(subfolder, 'list_correlation.npy'))
  else:
    if not os.path.isdir(subfolder):
      os.mkdir(subfolder)

    filename = os.path.join(subfolder, 'res.csv')
    if not os.path.exists(filename):
      fields = ['loss_terms', 'correlation', 'noise_level', 'lr_discr',
                'res_train', 'res_test', 'corr_predictions', 'corr_data',
                'loss_cls', 'loss_discr', 'optimal_W']

      with open(filename, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)

    list_noise_level = np.logspace(-2, 2, 10)
    list_correlation = [0, 0.6, 0.95]

    results_combined = vary_A_noise_and_correlation(
        list_noise_level=list_noise_level,
        list_correlation=list_correlation,
        dim=dim,
        filename=filename
    )

    np.save(os.path.join(subfolder, 'results_combined.npy'), results_combined)
    np.save(os.path.join(subfolder, 'list_noise_level.npy'), list_noise_level)
    np.save(os.path.join(subfolder, 'list_correlation.npy'), list_correlation)

  num_correlations = len(list_correlation)
  fig, axs = plt.subplots(
      num_correlations, 3, figsize=(10, 6.5), sharex='all', sharey='all'
  )
  for col in range(3):
    for row in range(num_correlations):
      plot_noise_dependency(axs[row, col],
                            list_noise_level,
                            results_combined[col, row, :, :],
                            reference=results_combined[0, 0, :, :])

      ylabel = 'Correlation = {} \n \n \n'.format(list_correlation[row])
      if row == len(list_correlation) - 1:
        ylabel += "Accuracy"
      axs[row, 0].set_ylabel(ylabel)
  axs[0, 0].set_title(r'\textbf{Base}', fontsize=22)
  axs[0, 1].set_title(r'\textbf{Base + MI}', fontsize=22)
  axs[0, 2].set_title(r'\textbf{Base + CMI}', fontsize=22)

  for i in range(3):
    axs[-1, i].set_xlabel('Noise Level', fontsize=22)

  extra_labels = []
  for j in range(num_correlations):
    lbl = axs[j, 0].set_ylabel(r'\textbf{' + 'Corr = {}'.format(list_correlation[j]) + r'}' + '\n\n\huge{Accuracy}',
                               fontsize=18)
    extra_labels.append(lbl)

  lgd = fig.legend(['Correlated Training', 'Uncorrelated Test', 'Reference'],
                   bbox_to_anchor=(0.02, 0.95, 1., .102),
                   loc='lower left',
                   ncol=3,
                   borderaxespad=0.,
                   fontsize=20)

  fig.subplots_adjust(hspace=0, wspace=0)
  fig.savefig(os.path.join(subfolder, 'dim{}_Winit_I_test0.pdf').format(dim),
              bbox_inches='tight', pad_inches=0)
  fig.savefig(os.path.join(subfolder, 'dim{}_Winit_I_test0.png').format(dim),
              bbox_inches='tight', pad_inches=0)


def plot_input_data_distribution():
  """Plot the input data distribution (2 attributes) for a range of noise
     levels and correlations
  """
  colors = ["#E00072", "#00830B", "#2B1A7F", "#E06111", "#4F4C4B", "#02D4F9"]
  list_noise_level = [0.1, 0.5, 1.0]
  list_correlation = [0, 0.6, 0.95]
  num_correlations = len(list_correlation)

  fig, axs = plt.subplots(
      num_correlations, 3, figsize=(10, 8), sharex='all', sharey='all'
  )

  for noise_idx, noise_level in enumerate(list_noise_level):
    for corr_idx, correlation in enumerate(list_correlation):
      ax = axs[corr_idx, noise_idx]
      a, a_and_noise = sample_a_classification(2000, noise_level, correlation, dim=2)
      A = get_A(2)
      x = compute_x(A, a_and_noise)
      x1 = x[0]
      x2 = x[1]

      corr_a1 = a[0]
      corr_a2 = a[1]

      for i, setting in enumerate(itertools.product([-1, 1], [-1, 1])):
        idxs = (corr_a1 == setting[0]) & (corr_a2 == setting[1])

        sel_corr_x1 = x1[idxs]
        sel_corr_x2 = x2[idxs]

        if setting[0]==1:
          ax.scatter(
              sel_corr_x1, sel_corr_x2, edgecolors=None, s=2, alpha=0.3,
              color=colors[i],
              label='$a_1$ = {}, \ $a_2$ = {}'.format(setting[0], setting[1])
          )
        else:
          ax.scatter(
              sel_corr_x1, sel_corr_x2, edgecolors=None, s=2, alpha=0.3,
              color=colors[i],
              label='$a_1$ = {}, $a_2$ = {}'.format(setting[0], setting[1])
          )

      ax.set_xticks([])
      ax.set_yticks([])
      ax.set_xlabel(noise_level, fontsize=22)
      if noise_idx==0:
          ax.set_ylabel(correlation, fontsize=22)

  fig.subplots_adjust(wspace=0, hspace=0)

  fig.text(0.05, 0.44, 'Correlation', ha='center', fontsize=22, rotation=90)
  fig.text(0.51, 0.03, 'Noise', ha='center', fontsize=22)

  plt.legend(bbox_to_anchor=(1.05, 1.05, 1., .102), markerscale=4,
             loc='lower left', ncol=1, borderaxespad=0., fontsize=22)
  plt.savefig(os.path.join(save_dir, 'toy_linear_cls_data.pdf'),
              bbox_inches='tight', pad_inches=0)
  plt.savefig(os.path.join(save_dir, 'toy_linear_cls_data.png'),
              bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
  print('dim = {}'.format(sys.argv[1]))
  dim = int(sys.argv[1])

  if dim == 2:
    plot_input_data_distribution()

  job_id = 2  #int(sys.argv[2])
  verbose = False

  # If the results are already there and you want to load them for making the figure
  load = False
  adaptiveweight = 'fix100'

  # Initialize from result of optimal classification (otherwise init from identity)
  initC = False

  figure_noise_dependency(dim=dim)
