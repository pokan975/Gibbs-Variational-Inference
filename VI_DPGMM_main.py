# -*- coding: utf-8 -*-
"""
@author: Po-Kan (William) Shih
@advisor: Dr.Bahman Moraffah
Variational inference for DPMM, from the paper "Variational inference for 
Dirichlet process mixtures" by David Blei and Michael I. Jordan (2006)
"""
# from CAVI_DPMM import VDPGMM
from CAVI_DPGMM import DPMM
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import sys
sys.getdefaultencoding()

# np.random.seed(1)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10


# number of components in GMM
components = 3
# true means & variances for mixing 2-D Gaussians
comp_mean = [(3, 3), (6, 6), (9, 9)]
# true cov matrices are identity matrix & known a priori
comp_var = np.eye(2)
# mixing proportion
mix_prop = [0.3, 0.4, 0.3]
# numer of observations
N = 300
# truncation level for q
truncation = 10

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

# scatter plot of samples
plt.scatter(samples[:, 0], samples[:, 1])
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.title("observations")
plt.grid()
plt.show()

# create VI object
model = DPMM(samples, truncation, s)
# run CAVI optimization
model.fit()

# for each sample x_i, choose its cluster from converged q(c_i)
assignments = np.zeros(N, dtype = np.int32)
debug = np.argmax(model.phi, axis = 1) # for debug only
for i in range(N):
    # pick cluster by multinomial dist q(c_i)
    a = np.random.multinomial(1, model.phi[i, :], size = 1)
    # extract cluster index
    assignments[i] = np.nonzero(a)[1]

# indices of distinct clusters that are chosen in the sample set
cluster = list(set(assignments))


# plot joint PDF of all chosen GMM components
plt.clf()

x, y = np.mgrid[0:12:200j, 0:12:200j]
coord = np.dstack((x, y))

# overlap PDFs of all GMM components
mean = model.m[cluster[0], :]
joint_pdf = st.multivariate_normal(mean, comp_var).pdf(coord)
for i in range(1, len(cluster)):
    mean = model.m[cluster[i], :]
    joint_pdf += st.multivariate_normal(mean, comp_var).pdf(coord)

plt.contourf(x, y, joint_pdf)
plt.set_cmap("Greys")

# scatter plot of samples, points with same color mean they are assigned to
# the same cluster
color = 'rgbcmykw'
for i, c in enumerate(cluster):
    group = samples[assignments == c, :]
    plt.scatter(group[:, 0], group[:, 1], s = 8, alpha = 0.5, color = color[i])

plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.title("points with same color assigned to the same cluster")
plt.grid()
plt.show()

# plot ELBO value history
plt.plot(model.elbo_values)
plt.xlabel("iteration")
plt.ylabel("ELBO value")
plt.grid()
plt.show()

# plot the prior & approximate dist over alpha
x = np.linspace(0, 10, 200)
p = st.gamma(s[0]/s[1]).pdf(x)
q = st.gamma(model.omega[0]/model.omega[1]).pdf(x)
plt.plot(x, p, color = "b", label = "prior p($\\alpha$)")
plt.plot(x, q, color = "r", label = "approx q($\\alpha$)")
plt.legend()
plt.grid()
plt.show()

# print out some info of the simulation
print("truncation level for q:", truncation)
print("true number of clusters: %d, converged number of clusters: %d" %(components, len(cluster)))
print("true means for Gaussian components:", comp_mean)
for i, c in enumerate(cluster, start = 1):
    print("the mean of converged q(\u03BC%d) is" %i, model.m[c, :].round(4))
print("true variances are all", comp_var)

