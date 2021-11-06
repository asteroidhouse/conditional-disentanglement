"""Code to reproduce results and figures for **toy linear regression**.

**Notation (differs from the manuscript):**
- Latent space: "v" (corresponds to "z" in the manuscript)
- Target: "z" (corresponds to "s" in the manuscript)
- Data: "x"
"""
import numpy as np
from numpy.linalg import multi_dot
from toy_linear_regression_functions import *

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('text', usetex=True)  # Activate latex text rendering


#### Results shown in Table 1
correlation = 0.8
test_correlation = 0
noise_level = 0.1
cov_z_with_noise = get_cov_z_with_noise(correlation, noise_level)
cov_z_test_with_noise = get_cov_z_with_noise(test_correlation, noise_level)
A = np.block([[np.eye(2), np.eye(2)]])


print('---------Regression----------')
W_regr = np.eye(2)
test_without_subspaces(cov_z_with_noise, A, W_regr, cov_z_test_with_noise=cov_z_test_with_noise)


print('\n------------MI=0-------------')
# Enforce unconditional independence by choosing $W$ such that $cov(v)$ is diagonal.
# In our case this is simple, the principal components of $x$ are $x_1 + x_2$ and $x_1 - x_2$
_, cov_x, _, _ = covariances(cov_z_with_noise, A, W_regr)
W_svd, eigenvalues, _ = np.linalg.svd(cov_x)

# whitening
white = np.sqrt(np.linalg.inv(np.diag(eigenvalues)))

# rotation by phi (leads to minimal regression under the constraint of MI=0)
phi = np.pi * 3 / 4
orth = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
W_uncond = multi_dot([orth, white, W_svd])

test_with_subspaces(cov_z_with_noise, A, W_uncond, cov_z_test_with_noise=cov_z_test_with_noise)


print('\n-----------CMI=0-------------')
W_cond = np.eye(2)
test_with_subspaces(cov_z_with_noise, A, W_cond, cov_z_test_with_noise=cov_z_test_with_noise)


print('\n-----------Reference-------------')
# For comparison, we test simple regression without correlation
# but with the same noise level
correlation = 0.0
cov_z_with_noise = get_cov_z_with_noise(correlation, noise_level)
test_without_subspaces(cov_z_with_noise, A, W_regr, cov_z_test_with_noise=cov_z_test_with_noise)


#### Figure 2: Dependence on noise level and training correlation
f = figure_noise_dependency()
f.savefig('saves_linear/A_is_identity.pdf', bbox_inches = "tight")


#### Figure 3 Correlation of target, data and predictions
N = 100000
correlation = 0.8
cov_z_with_noise = get_cov_z_with_noise(correlation, noise_level)

# Prediction of pure regression model
R_regr = get_optimal_regression(cov_z_with_noise, A, W_regr)

# Prediction of uncond. disentangled model
R_uncond = get_optimal_split_regression(cov_z_with_noise, A, W_uncond)

# Prediction of cond. disentangled model
R_cond = get_optimal_split_regression(cov_z_with_noise, A, W_cond)


# -------- Figure for trainings correlation ----------
# sample datapoints
z, z_and_noise = sample_z(N, cov_z_with_noise)
x = compute_x(A, z_and_noise)

f = figure_correlations(x, z, W_regr, R_regr, W_uncond, R_uncond, W_cond, R_cond)
f.text(0.04, 0.5, r'\textbf{' + 'Training' + r'}'  +  '\n (Corr = 0.8)', va='center', fontsize = 14)
f.savefig('saves_linear/correlation_of_predictions.pdf', bbox_inches = "tight")


# ----------- Figure for test correlation ------------
correlation = 0
cov_z_with_noise = get_cov_z_with_noise(correlation, noise_level)

# sample datapoints
z, z_and_noise = sample_z(N, cov_z_with_noise)
x = compute_x(A, z_and_noise)

f = figure_correlations(x, z, W_regr, R_regr, W_uncond, R_uncond, W_cond, R_cond, title=False)
f.text(0.04, 0.5, r'\textbf{' + 'Test' + r'}'  +  '\n (Corr = 0)', va='center', fontsize = 14)
f.savefig('saves_linear/correlation_of_predictions_test.pdf', bbox_inches = "tight")


#### Figure 8 (for the appendix)

# left part (Figure 8a)
correlation = 0.8
noise_level = 0.1
A = np.block([[np.eye(2), np.eye(2)]])

f = function_unconditional_optimum(correlation, noise_level, A)
f.savefig('saves_linear/svd_whitening.pdf', bbox_inches = "tight")

# right part (Figure 8b)
matplotlib.rc('text', usetex=True)  # Activate latex text rendering
matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{nicefrac}']
f = vary_phi(correlation, noise_level, A)
plt.savefig('saves_linear/phi_dependency_{}_{}.pdf'.format(0.8, 0.1), bbox_inches = "tight")
