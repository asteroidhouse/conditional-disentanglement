"""Functions for toy linear regression experiments.
"""
import numpy as np
from numpy.linalg import multi_dot
from scipy.linalg import inv

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
matplotlib.rc('text', usetex=True)  # Activate latex text rendering


# -------------------------------------------------------------
# ------------------- Functions for Table 1 -------------------
# -------------------------------------------------------------
def get_cov_z_with_noise(correlation, noise_level):
  """Covariance matrix for target with noise.
  """
  cov_z = np.array([[1.0, correlation], [correlation, 1.0]])
  cov_noise = np.eye(2) * noise_level
  cov_z_with_noise = np.block([
      [cov_z, np.zeros((2, 2))],
      [np.zeros((2, 2)), cov_noise]
  ])
  return cov_z_with_noise


def covariances(cov_z_with_noise, A, W):
  """Helper function to compute covariances.
  """
  # Just adding noise variance on diagonals
  cov_x = multi_dot([A, cov_z_with_noise, A.T])
  cov_v = multi_dot([W, cov_x, W.T])
  z_noise_to_z = np.block([np.eye(2), np.zeros((2, 2))])
  cov_z = multi_dot([z_noise_to_z, cov_z_with_noise, z_noise_to_z.T])

  # Equals cov_z if A = (I, I)
  cov_z_v = multi_dot([z_noise_to_z, cov_z_with_noise, A.T, W.T])

  return cov_z, cov_x, cov_v, cov_z_v


def get_optimal_regression(cov_z_with_noise, A, W):
  """Compute optimal weights for linear regression.
  """
  _, _, cov_v, cov_z_v = covariances(cov_z_with_noise, A, W)
  optimal_regression_matrix = multi_dot([cov_z_v, np.linalg.inv(cov_v)])
  return optimal_regression_matrix


def get_optimal_split_regression(cov_z_with_noise, A, W):
  '''Find optimal linear regression given a disentanglement constraint
  '''
  _, _, cov_v, _ = covariances(cov_z_with_noise, A, W)
  z_noise_to_z = np.block([np.eye(2), np.zeros((2, 2))])

  cov_z1_v1 = multi_dot([z_noise_to_z[:1, :], cov_z_with_noise, A.T, W[:1, :].T])
  cov_z2_v2 = multi_dot([z_noise_to_z[1:2, :], cov_z_with_noise, A.T, W[1:, :].T])
  regression_v1 = multi_dot([cov_z1_v1, np.linalg.inv(cov_v[:1, :1])])
  regression_v2 = multi_dot([cov_z2_v2, np.linalg.inv(cov_v[1:, 1:])])
  optimal_split_regression = np.block([
      [regression_v1, np.zeros((1, 1))],
      [np.zeros((1, 1)), regression_v2],
  ])

  return optimal_split_regression


def test_regression(cov_z_with_noise, A, W, R):
  """Compute mse and variance explained for given A, W, and R matrices.
  """
  cov_z, _, cov_v, _ = covariances(cov_z_with_noise, A, W)
  cov_s_hat = multi_dot([R, cov_v, R.T])
  cov_s_hat_s = multi_dot([R, W, A, cov_z_with_noise[:, :2]])
  cov_s_and_s_hat = np.block([
      [cov_z, cov_s_hat_s.T],
      [cov_s_hat_s, cov_s_hat]
  ])
  subtraction_matrix = np.array([[1, 0, -1, 0], [0, 1, 0, -1]])
  cov_diff = multi_dot([subtraction_matrix, cov_s_and_s_hat, subtraction_matrix.T])

  variance = np.sum(np.linalg.eigvalsh(cov_z))
  mse = np.sum(np.linalg.eigvalsh(cov_diff))
  variance_explained = 1 - mse / variance

  return mse, variance_explained


def test_without_subspaces(cov_z_with_noise, A, W, cov_z_test_with_noise=None):
  """Helper function for linear regression
  """
  R_full = get_optimal_regression(cov_z_with_noise, A, W)
  print("Regression matrix:")
  print(multi_dot([R_full, W]))
  print('')
  mse, ve = test_regression(cov_z_with_noise, A, W, R_full)
  print(f"Train: MSE = {mse:.4f}, VE={ve:.4%}")
  if cov_z_test_with_noise is not None:
    mse, ve = test_regression(cov_z_test_with_noise, A, W, R_full)
    print(f"Test:  MSE = {mse:.4f}, VE={ve:.4%}")
  print()


def test_with_subspaces(cov_z_with_noise, A, W, cov_z_test_with_noise=None):
  """Helper function for linear regression with (conditional) disentanglement.
  """
  R = get_optimal_split_regression(cov_z_with_noise, A, W)
  mse, ve = test_regression(cov_z_with_noise, A, W, R)
  print("Regression matrix:")
  print(multi_dot([R, W]))
  print('')
  print(f"Train: MSE = {mse:.4f}, VE={ve:.4%}")
  if cov_z_test_with_noise is not None:
    mse, ve = test_regression(cov_z_test_with_noise, A, W, R)
    print(f"Test:  MSE = {mse:.4f}, VE={ve:.4%}")
  print()


# --------------------------------------------------------------------------
# Functions for Figure 2: Dependence on noise level and training correlation
# --------------------------------------------------------------------------
def get_ve(A, W, noise_level, correlation, test_correlation, subspaces, reference=False):
  """Wrapper for computing VE on training data + different test sets.
  """
  cov_z_with_noise = get_cov_z_with_noise(correlation, noise_level)

  if subspaces:
    R = get_optimal_split_regression(cov_z_with_noise, A, W)
  else:
    R = get_optimal_regression(cov_z_with_noise, A, W)
  _, ve = test_regression(cov_z_with_noise, A, W, R)

  if reference:
    return ve
  else:
      # loop over test_correlations between -1 and 1
    ve_tests = []
    for corr in np.linspace(-1, 1, 10):
      cov_z_test_with_noise = get_cov_z_with_noise(corr, noise_level)
      _, ve_test = test_regression(cov_z_test_with_noise, A, W, R)
      ve_tests.append(ve_test)

    # get results for a specific test correlation
    cov_z_test_with_noise = get_cov_z_with_noise(test_correlation, noise_level)
    _, ve_test = test_regression(cov_z_test_with_noise, A, W, R)
    return ve, ve_test, min(ve_tests), max(ve_tests)


def plot_noise_dependency(ax, correlation, loss_type, A, list_noise_level):
  """Plot VE in dependence of noise level for a given loss_type and correlation.
  """
  test_correlation = 0

  # Five values are for: ve_reference, ve, ve_test, ve_tests_min, ve_tests_max
  results = np.zeros([len(list_noise_level), 5])

  for noise_level_index, noise_level in enumerate(list_noise_level):
    if loss_type == 'regression':
      W = np.linalg.inv(A[:, :2])
      subspaces = False
    elif loss_type == 'unconditional':
      cov_z_with_noise = get_cov_z_with_noise(correlation, noise_level)
      cov_x = multi_dot([A, cov_z_with_noise, A.T])
      W_svd, eigenvalues, _ = np.linalg.svd(cov_x)
      white = np.sqrt(np.linalg.inv(np.diag(eigenvalues)))
      if correlation == 0:
        phi = 0
      else:
        phi = -np.pi / 4
      orth = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
      W = multi_dot([orth, white, W_svd])
      subspaces = True
    elif loss_type == 'conditional':
      W = np.linalg.inv(A[:, :2])
      subspaces = True

    ve, ve_test, ve_tests_min, ve_tests_max = get_ve(A, W, noise_level, correlation, test_correlation, subspaces)

    # Reference (regression for correlation=0)
    W = np.linalg.inv(A[:, :2])
    ve_reference = get_ve(A, W, noise_level, 0, 0, False, True)

    results[noise_level_index] = ve_reference, ve, ve_test, ve_tests_min, ve_tests_max

  colors = ["#E00072", "#00830B", "#2B1A7F", "#E06111", "#4F4C4B", "#02D4F9"]

  color_train = colors[0]
  color_test = colors[1]

  # Plot performance for training, test and reference
  ax.plot(list_noise_level, results[:, 1], linewidth=2, color=color_train)  # Train performance
  ax.plot(list_noise_level, results[:, 2], linewidth=2, color=color_test)   # Test performance
  ax.plot(list_noise_level, results[:, 0], linewidth=1, color='black', linestyle='dashed')  # Reference line

  # Plot shaded area showing max and min for different test correlations
  ax.fill_between(list_noise_level, results[:, 3], results[:, 4], color=color_test, alpha=.1)

  ax.set_xscale('log')
  ax.set_ylim([-0.4, 1.1])
  ax.set_yticks([0, 1])
  ax.tick_params(axis='both', which='both', labelsize=16)


def figure_noise_dependency():
  """Generates figure consisting of subplots for loss type and correlation.
  Each subplot shows the VE in dependence of noise level.
  """
  list_noise_level = np.logspace(-2, 2, 30)
  list_correlation = [0, 0.6, 0.95]
  num_correlations = len(list_correlation)
  list_loss_type = ['regression', 'unconditional', 'conditional']
  A = np.eye(2)
  A = np.block([[A, np.eye(2)]])

  f, axs = plt.subplots(num_correlations, 3, figsize = (10, 8), sharex='all', sharey='all')
  for col in range(3):
    for row in range(num_correlations):
      plot_noise_dependency(ax=axs[row, col],
                            correlation=list_correlation[row],
                            loss_type=list_loss_type[col],
                            A=A,
                            list_noise_level=list_noise_level)

  axs[0, 0].set_title(r'\textbf{Classification}', fontsize=20)
  axs[0, 1].set_title(r'\textbf{Unconditional}', fontsize=20)
  axs[0, 2].set_title(r'\textbf{Conditional}', fontsize=20)

  for i in range(3):
    axs[-1, i].set_xlabel('Noise Level', fontsize=18)

  extra_labels = []
  for j in range(num_correlations):
    lbl = axs[j, 0].set_ylabel(r'\textbf{' + 'Corr = {}'.format(list_correlation[j]) + r'}' + '\n\nVE', fontsize=18)
    extra_labels.append(lbl)

  lgd = f.legend(['Training', 'Uncorrelated', 'Reference'],
                  bbox_to_anchor=(0.15, 0.95, 1., .102),
                  loc='lower left',
                  ncol=3,
                  borderaxespad=0.,
                  fontsize=20)

  f.subplots_adjust(hspace=0, wspace=0)
  return f


# -------------------------------------------------------------------
# Functions for Figure 3: Correlation of target, data and predictions
# -------------------------------------------------------------------
def sample_z(num_samples, cov_z_with_noise):
  """Sample some data points for the target with noise
  """
  cov_z = cov_z_with_noise[:2, :2]
  mean = [0, 0]
  s1, s2 = np.random.multivariate_normal(mean, cov_z, num_samples).T
  z = np.vstack((s1, s2))
  noise_level = cov_z_with_noise[-1, -1]
  epsilon = np.random.randn(2, num_samples) * np.sqrt(noise_level)
  z_and_noise = np.vstack((z, epsilon))
  return z, z_and_noise


def compute_x(A, z):
  """Compute data from target.
  """
  x = np.dot(A, z)
  return x


def forward_generator(x, W, R):
  """Compute predictions given W and R
  """
  v = np.dot(W, x)
  s_hat = np.dot(R, v)
  return s_hat


def subplot_scatter(ax, data):
  """Make a scatterplot for given datapoints + text box showing correlations strength
  """
  ax.scatter(data[0, :1000], data[1, :1000], s=1)

  # add text box showing correlation of data
  corr = np.corrcoef(data)[0, 1]
  text = r'Corr = {:.2f}'.format(corr)
  text_box = AnchoredText(text, frameon=True, loc=4, pad=0.5, prop=dict(fontsize=14))
  plt.setp(text_box.patch, facecolor='white', alpha=0.5)
  ax.add_artist(text_box)

  # asthetics
  ax.set(aspect='equal')
  ax.set_xlim([-4, 4])
  ax.set_ylim([-4, 4])
  ax.set_xticks([])
  ax.set_yticks([])


def figure_correlations(x, z, W_regr, R_regr, W_uncond, R_uncond, W_cond, R_cond, title=True):
  '''Make a scatterplot showing how much target/ data / predictions are correlated
  '''
  f, axs = plt.subplots(1, 5, figsize = (15, 3), sharex='all', sharey='all')

  # Target
  subplot_scatter(axs[0], z)

  # Data
  subplot_scatter(axs[1], x)

  # Prediction of pure regression model
  s_hat = forward_generator(x, W_regr, R_regr)
  subplot_scatter(axs[2], s_hat)

  # Prediction of uncond. disentangled model
  s_hat = forward_generator(x, W_uncond, R_uncond)
  subplot_scatter(axs[3], s_hat)

  # Prediction of cond. disentangled model
  s_hat = forward_generator(x, W_cond, R_cond)
  subplot_scatter(axs[4], s_hat)

  # Add title
  if title:
    title_list = ['target',
                  'data',
                  r'\textbf{' + 'Prediction of pure' + r'}' + '\n' + r'\textbf{' + 'regression model' + r'}',
                  r'\textbf{' + 'Prediction of uncond.' + r'}' + '\n' + r'\textbf{' + 'disentangled model' + r'}',
                  r'\textbf{' + 'Prediction of cond.' + r'}' + '\n' + r'\textbf{' + 'disentangled  model' + r'}']
    for ax_index in range(5):
      axs[ax_index].set_title(title_list[ax_index], fontsize=14)

  # Add label for x and y axis
  axs[0].set_xlabel(r'$\textrm{s}_1$', fontsize=14)
  axs[0].set_ylabel(r'$\textrm{s}_2$', fontsize=14)
  axs[1].set_xlabel(r'$\textrm{x}_1$', fontsize=14)
  axs[1].set_ylabel(r'$\textrm{x}_2$', fontsize=14)
  for ax in axs[2:5]:
    ax.set_xlabel(r'$\hat{\textrm{s}_1}$', fontsize=14)
    ax.set_ylabel(r'$\hat{\textrm{s}_2}$', fontsize=14)

  return f


# -----------------------------------------------------------
# ----------------- Functions for Figure 8a -----------------
# -----------------------------------------------------------
def scatterplot_visualisation(ax, data, arrow_s1, arrow_s2):
  """Make a scatterplot from the data + draw arrows for s1 and s2
  """
  ax.scatter(data[0], data[1], s=2, label="Train")

  # Draw arrows
  ax.arrow(0, 0, arrow_s1[0], arrow_s1[1], head_width=0.15, color='red')
  ax.arrow(0, 0, arrow_s2[0], arrow_s2[1], head_width=0.15, color='k')

  ax.set_ylim([-3, 3])
  ax.set_xlim([-3, 3])
  ax.set_xticks([])
  ax.set_yticks([])
  ax.set(aspect='equal')


def scatterplot_visualisation_v(ax, title, W, A, x, x_s1, x_s2, cov_z_with_noise):
  """Wrapper for latent space plots
  """
  v = np.dot(W, x)

  # Transform s1 and s2 to latent space v
  v_s1 = np.dot(W, x_s1)
  v_s2 = np.dot(W, x_s2)

  scatterplot_visualisation(ax, v, v_s1, v_s2)

  # Add labels + title
  ax.set_xlabel(r'$z_1$', fontsize=22)
  ax.set_ylabel(r'$z_2$', fontsize=22)
  ax.set_title(title, fontsize=22)

  # Compute VE (analytically)
  R = get_optimal_split_regression(cov_z_with_noise, A, W)
  _, ve = test_regression(cov_z_with_noise, A, W, R)
  text = 'VE = ' + str(np.round(ve, 2)) + '%'
  text_box = AnchoredText(text, frameon=True, loc=4, pad=0.5, prop=dict(fontsize=14))
  plt.setp(text_box.patch, facecolor='white', alpha=0.5)
  ax.add_artist(text_box)


def label_arrow(x1, y1, x2, y2, ax):
  """Add a label to the arrows. Positions of the red (x1, y1) and black (x2, y2) must be hardcoded
  """
  t = ax.text(x1, y1, r'$s_1$', transform=ax.transAxes, fontsize=20, color='red')
  t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='circle, pad=-1'))
  t = ax.text(x2, y2, r'$s_2$', transform=ax.transAxes, fontsize=20, color='black')
  t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='circle, pad=-1'))


def function_unconditional_optimum(correlation, noise_level, A):
  """Figure illustrating how to obtain minimal regression under the constraint of MI=0
  """
  # Sample data for scatterplot
  N = 2000  # Number of datapoints shown in scatterplot
  cov_z_with_noise = get_cov_z_with_noise(correlation, noise_level)
  z, z_and_noise = sample_z(N, cov_z_with_noise)
  x = compute_x(A, z_and_noise)

  s1 = [1, 0]
  s2 = [0, 1]

  fig, axs = plt.subplots(1, 4, figsize = (12, 3))

  # Transform s1 and s2 to data space x
  x_s1 = np.dot(A[:2, :2], s1)
  x_s2 = np.dot(A[:2, :2], s2)

  # Figure for data
  scatterplot_visualisation(axs[0], x, x_s1, x_s2)
  axs[0].set_xlabel(r'$x_1$', fontsize=22)
  axs[0].set_ylabel(r'$x_2$', fontsize=22)

  # SVD
  cov_x = multi_dot([A, cov_z_with_noise, A.T])
  W_svd, eigenvalues, _ = np.linalg.svd(cov_x)
  scatterplot_visualisation_v(axs[1], 'svd',
                              W_svd, A, x, x_s1, x_s2, cov_z_with_noise)

  # Whitening
  white = np.sqrt(np.linalg.inv(np.diag(eigenvalues)))
  W_white = np.dot(white, W_svd)
  scatterplot_visualisation_v(axs[2], '... + whitening',
                              W_white, A, x, x_s1, x_s2, cov_z_with_noise)

  # Rotate whitened
  phi = - np.pi / 4
  orth = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
  W_rotated = multi_dot([orth, white, W_svd])
  scatterplot_visualisation_v(axs[3], '... + rotation by ' + r'$\phi_{opt}$',
                              W_rotated, A, x, x_s1, x_s2, cov_z_with_noise)

  fig.tight_layout()

  # Add labels to the red (s_1) and black (s_2) arrows
  label_arrow(0.65, 0.38, 0.35, 0.7, axs[0])
  label_arrow(0.3, 0.25, 0.3, 0.7, axs[1])
  label_arrow(0.48, 0.25, 0.45, 0.75, axs[2])
  label_arrow(0.3, 0.3, 0.65, 0.7, axs[3])
  return fig


# -----------------------------------------------------------
# ----------------- Functions for Figure 8b -----------------
# -----------------------------------------------------------
def vary_phi(correlation, noise_level, A):
  """Show VE in dependence of phi.
  """
  cov_z_with_noise = get_cov_z_with_noise(correlation, noise_level)
  cov_x = multi_dot([A, cov_z_with_noise, A.T])
  W_svd, eigenvalues, _ = np.linalg.svd(cov_x)
  white = np.sqrt(np.linalg.inv(np.diag(eigenvalues)))

  ve_list = []
  phis = np.linspace(0, 2 * np.pi, 100)
  for phi in phis:
    orth = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    W_new = multi_dot([orth, white, W_svd])

    R = get_optimal_split_regression(cov_z_with_noise, A, W_new)
    _, ve = test_regression(cov_z_with_noise, A, W_new, R)
    ve_list.append(ve)

  fig = plt.figure()
  plt.plot(phis, ve_list, linewidth=2)
  plt.xticks([0, np.pi/2, np.pi, np.pi * 3/2, np.pi*2],
  [r'$0$', r'$\nicefrac{\pi}{2}$', r'$\pi$', r'$\nicefrac{3 \pi}{2}$', r'$2 \pi$'], fontsize=22)
  plt.yticks(fontsize=22)
  plt.ylim(0, 1)
  plt.xlabel(r'$\phi$', fontsize=22)
  plt.ylabel("VE", fontsize=22)
  plt.legend()
  return fig
