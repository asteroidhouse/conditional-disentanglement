"""Code to reproduce results and figures for **toy linear regression**.

**Notation:**
- Target: s
- Data: x
- Latent space: z 

Usage
=====
python toy_linear_regression.py
"""
import os
import numpy as np
from numpy.linalg import multi_dot
from toy_linear_regression_functions import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rc('text', usetex=True)  # Activate latex text rendering


save_dir = 'saves/linear_regression'
if not os.path.exists(save_dir):
  os.makedirs(save_dir)

#### Results shown in Table 1
correlation = 0.8
test_correlation = 0
noise_level = 0.1
cov_s_with_noise = get_cov_s_with_noise(correlation, noise_level)
cov_s_test_with_noise = get_cov_s_with_noise(test_correlation, noise_level)
A = np.block([[np.eye(2), np.eye(2)]])


print('---------Regression----------')
W_regr = np.eye(2)
test_without_subspaces(cov_s_with_noise, A, W_regr,
                       cov_s_test_with_noise=cov_s_test_with_noise)


print('\n------------MI=0-------------')
# Enforce unconditional independence by choosing $W$ such that $cov(v)$ is diagonal.
# In our case this is simple, the principal components of $x$ are $x_1 + x_2$ and $x_1 - x_2$
_, cov_x, _, _ = covariances(cov_s_with_noise, A, W_regr)
W_svd, eigenvalues, _ = np.linalg.svd(cov_x)

# Whitening
white = np.sqrt(np.linalg.inv(np.diag(eigenvalues)))

# Rotation by phi (leads to minimal regression loss under the constraint that the unconditional MI=0)
phi = np.pi * 3 / 4
orth = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
W_uncond = multi_dot([orth, white, W_svd])

test_with_subspaces(cov_s_with_noise, A, W_uncond,
                    cov_s_test_with_noise=cov_s_test_with_noise)


print('\n-----------CMI=0-------------')
W_cond = np.eye(2)
test_with_subspaces(cov_s_with_noise, A, W_cond,
                    cov_s_test_with_noise=cov_s_test_with_noise)


print('\n-----------Reference-------------')
# For comparison, we test simple regression without correlation
# but with the same noise level
correlation = 0.0
cov_s_with_noise = get_cov_s_with_noise(correlation, noise_level)
test_without_subspaces(cov_s_with_noise, A, W_regr,
                       cov_s_test_with_noise=cov_s_test_with_noise)


#### Figure 2: Dependence on noise level and training correlation
fig = figure_noise_dependency()
fig.savefig(os.path.join(save_dir, 'Regression_noise_dependency.pdf'),
            bbox_inches='tight', pad_inches=0)
fig.savefig(os.path.join(save_dir, 'Regression_noise_dependency.png'),
            bbox_inches='tight', pad_inches=0)


#### Figure 3 Correlation of target, data and predictions
N = 100000
correlation = 0.8
cov_s_with_noise = get_cov_s_with_noise(correlation, noise_level)

# Prediction of pure regression model
R_regr = get_optimal_regression(cov_s_with_noise, A, W_regr)

# Prediction of uncond. disentangled model
R_uncond = get_optimal_split_regression(cov_s_with_noise, A, W_uncond)

# Prediction of cond. disentangled model
R_cond = get_optimal_split_regression(cov_s_with_noise, A, W_cond)


# -------- Figure for trainings correlation ----------
# sample datapoints
s, s_and_noise = sample_s(N, cov_s_with_noise)
x = compute_x(A, s_and_noise)

fig = figure_correlations(x, s, W_regr, R_regr, W_uncond, R_uncond, W_cond, R_cond)
fig.text(0.04, 0.5, r'\textbf{' + 'Training' + r'}'  +  '\n (Corr = 0.8)',
         va='center', fontsize=14)
fig.savefig(os.path.join(save_dir, 'correlation_of_predictions.pdf'),
            bbox_inches='tight', pad_inches=0)
fig.savefig(os.path.join(save_dir, 'correlation_of_predictions.png'),
            bbox_inches='tight', pad_inches=0)


# ----------- Figure for test correlation ------------
correlation = 0
cov_s_with_noise = get_cov_s_with_noise(correlation, noise_level)

s, s_and_noise = sample_s(N, cov_s_with_noise)
x = compute_x(A, s_and_noise)

fig = figure_correlations(x, s, W_regr, R_regr, W_uncond, R_uncond, W_cond, R_cond, title=False)
fig.text(0.04, 0.5, r'\textbf{' + 'Test' + r'}'  +  '\n (Corr = 0)',
         va='center', fontsize=14)
fig.savefig(os.path.join(save_dir, 'correlation_of_predictions_test.pdf'),
            bbox_inches='tight', pad_inches=0)
fig.savefig(os.path.join(save_dir, 'correlation_of_predictions_test.png'),
            bbox_inches='tight', pad_inches=0)


# Figure 8 (for the appendix)

# left part (Figure 8a)
correlation = 0.8
noise_level = 0.1
A = np.block([[np.eye(2), np.eye(2)]])

fig = function_unconditional_optimum(correlation, noise_level, A)
fig.savefig(os.path.join(save_dir, 'svd_whitening.pdf'),
            bbox_inches='tight', pad_inches=0)
fig.savefig(os.path.join(save_dir, 'svd_whitening.png'),
            bbox_inches='tight', pad_inches=0)

# right part (Figure 8b)
fig = vary_phi(correlation, noise_level, A)
fig.savefig(os.path.join(save_dir, 'phi_dependency_{}_{}.pdf'.format(0.8, 0.1)),
            bbox_inches='tight', pad_inches=0)
fig.savefig(os.path.join(save_dir, 'phi_dependency_{}_{}.png'.format(0.8, 0.1)),
            bbox_inches='tight', pad_inches=0)
