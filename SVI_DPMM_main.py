# -*- coding: utf-8 -*-
"""
@author: Po-Kan (William) Shih
@advisor: Dr.Bahman Moraffah
Stochastic Variational inference for DPMM
Batch SVI algorithm from the paper "Stochastic Variational Inference" by
Matthew Hoffman (2013)
"""
from SVI_DPMM_func import SVI
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import sys
# print ASCII characters
sys.getdefaultencoding()

# np.random.seed(1)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10


# true number of components in mixture model
components = 3
# true means of 2-D Gaussian components
comp_mean = [(3, 3), (6, 6), (9, 9)]
# true covariance matrix (identity matrix & known a priori)
comp_var = np.eye(2)
# true mixing proportion
mix_prop = [0.3, 0.4, 0.3]

# numer of observations/samples
N = 900
# truncation level for q (max possible cluster number)
truncation = 10
# batch size for SVI resampling
batch = 100
# parameter for learning step size
kappa = 0.8


# hyper-parameters of prior p(mu_t) 
prior_mean = np.array([5, 5])
prior_var = 2 * np.eye(2)
# hyper-parameters of prior p(alpha) = Gamma(alpha| s_1, s_2)
s = [1, 1]


# generate true cluster assignment for each observation
P = np.random.multinomial(1, mix_prop, size = N)
P = np.nonzero(P)[1]
# generate observations
std = np.sqrt(comp_var)
samples = np.zeros((N, 2))
for i, m in enumerate(P):
    samples[i, :] = np.random.multivariate_normal(comp_mean[m], std, 1)


# scatter plot of observations
plt.scatter(samples[:, 0], samples[:, 1])
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.grid()
plt.show()

# create SVI object
svi = SVI(samples, truncation, s, batch, kappa)
# run SVI optimization
svi.fit()

# for each observation x_i, choose its cluster from learned q(c_i)
assignments = np.zeros(N, dtype = np.int32)
for i in range(N):
    # pick cluster by multinomial dist q(c_i)
    a = np.random.multinomial(1, svi.phi[i, :], size = 1)
    # extract cluster index
    assignments[i] = np.nonzero(a)[1]


# for who wants to see the best result, just replace "assignments"
# array in the codes below with this "debug" array
debug = np.argmax(svi.phi, axis = 1)
# indices of distinct clusters that are chosen in the sample set
cluster = list(set(debug))


# plot joint PDF of all chosen GMM components
plt.clf()

x, y = np.mgrid[0:12:200j, 0:12:200j]
coord = np.dstack((x, y))

# overlap PDFs of all GMM components
mean = svi.m[cluster[0], :]
joint_pdf = st.multivariate_normal(mean, comp_var).pdf(coord)
for i in range(1, len(cluster)):
    mean = svi.m[cluster[i], :]
    joint_pdf += st.multivariate_normal(mean, comp_var).pdf(coord)

plt.contourf(x, y, joint_pdf)
plt.set_cmap("Greys")

# scatter plot of samples, points with same color mean they are assigned to
# the same cluster by posterior categorical distributions
color = 'rgbcmykw'
for i, c in enumerate(cluster):
    group = samples[assignments == c, :]
    plt.scatter(group[:, 0], group[:, 1], s = 8, alpha = 0.5, color = color[i])
plt.xlabel("X axis")
plt.ylabel("Y axis")
# plt.title("points with same color assigned to the same cluster")
plt.grid()
plt.show()

# plot ELBO value history
plt.plot(svi.elbo_values[1:], "-o")
plt.xlabel("iteration")
plt.ylabel("ELBO value")
# plt.grid()
plt.show()

# plot the prior & approximate dist over alpha
x = np.linspace(0, 10, 200)
p = st.gamma(s[0]/s[1]).pdf(x)
q = st.gamma(svi.omega[0]/svi.omega[1]).pdf(x)
plt.plot(x, p, color = "b", label = r"prior p($\alpha$)")
plt.plot(x, q, color = "r", label = r"approximate q($\alpha$)")
plt.legend()
# plt.grid()
plt.show()

# print out some info of the simulation
print("truncation level for q:", truncation)
print("true number of clusters: %d, learned number of clusters: %d" %(components, len(cluster)))
print("true means for Gaussian components:", comp_mean)
for i, c in enumerate(cluster, start = 1):
    print("the mean of lerned q(\u03BC%d) is" %i, svi.m[c, :].round(4))
print("true covariance matrices are all", comp_var)

