# -*- coding: utf-8 -*-
"""
@author: Po-Kan (William) Shih
@advisor: Dr.Bahman Moraffah
Variational inference for DPMM, from the paper "Variational inference for 
Dirichlet process mixtures" by David Blei and Michael I. Jordan (2006)
"""
from CAVI_DPMM_func import CAVI
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
N = 300
# truncation level for q (max possible cluster number)
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
plt.grid()
plt.show()

# create VI object
model = CAVI(samples, truncation, s)
# run CAVI optimization
model.fit()

# for each sample x_i, choose its cluster from learned q(c_i)
assignments = np.zeros(N, dtype = np.int32)
for i in range(N):
    # pick cluster by multinomial dist q(c_i)
    a = np.random.multinomial(1, model.phi[i, :], size = 1)
    # extract cluster index
    assignments[i] = np.nonzero(a)[1]


# for who wants to see the best result, just replace "assignments"
# array in the codes below with this "debug" array
debug = np.argmax(model.phi, axis = 1) # for debug only
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

# plot ELBO evolution history
plt.plot(model.elbo_values, "-o")
plt.xlabel("iteration")
plt.ylabel("ELBO value")
plt.show()

# plot the prior & approximate dist over alpha
x = np.linspace(0, 10, 200)
p = st.gamma(s[0]/s[1]).pdf(x)
q = st.gamma(model.omega[0]/model.omega[1]).pdf(x)
plt.plot(x, p, color = "b", label = r"prior p($\alpha$)")
plt.plot(x, q, color = "r", label = r"approx q($\alpha$)")
plt.legend()
# plt.grid()
plt.show()

# print out some info of the simulation
print("truncation level for q:", truncation)
print("true number of clusters: %d, learned number of clusters: %d" %(components, len(cluster)))
print("true means for Gaussian components:", comp_mean)
for i, c in enumerate(cluster, start = 1):
    print("the mean of learned q(\u03BC%d) is" %i, model.m[c, :].round(4))
print("true covariance matrices are all", comp_var)

