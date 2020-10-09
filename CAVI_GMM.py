# -*- coding: utf-8 -*-
"""
@author: Po-Kan (William) Shih
@advisor: Dr.Bahman Moraffah
Variational inference for univariate GMM, the update equations for phi, mu, 
and s2 are in my onenote "VI for GMM"
"""
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.getdefaultencoding()

plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10

pi = np.pi

class GMM(object):
    '''Univariate GMM with CAVI
    data: vector of observations
    K: number of mixing components
    '''
    def __init__(self, data, K):
        self.data = data
        self.K = K           # number of components
        self.N = len(data)   # number of observations
        # generate initial parameter sets for all q(c_i) from Dirichlet distribution
        self.phi = np.random.dirichlet([1.]*self.K, self.N)
        # generate initial means for all q(mu_k)
        self.m = np.random.randint(low=np.min(self.data), high=np.max(self.data), size=self.K).astype(float)
        # add some biases to avoid guessing the true means before CAVI
        self.m += np.random.random(self.K)
        # initial variances of q(mu_k) are 1 for all k = 1, ..., K
        self.s2 = np.array([1.] * self.K)

        
    def fit(self, max_iter = 100, tol = 1e-10):
        # print initial q(mu) before iterations
        for i in range(self.K):
            print("Initial q(\u03BC_%d) = N(%.4f, %.2f)" %(i+1, self.m[i], self.s2[i]))
        
        # calc initial ELBO(q)
        self.elbo_values = [self.calc_ELBO()]
        # initialize m_k & s_k^2 evolution histories
        self.m_history = [self.m]
        self.s2_history = [self.s2]
        
        # CAVI iteration
        for it in range(1, max_iter + 1):
            # CAVI update
            self._update_phi()  # update parameter set for each q(c_i)
            self._update_mu()   # update parameter set for each q(mu_k)
            self.m_history.append(self.m)
            self.s2_history.append(self.s2)
            # compute ELBO(q) at the end of each update
            self.elbo_values.append(self.calc_ELBO())
            
            # if converged, stop iteration
            if np.abs(self.elbo_values[-2] - self.elbo_values[-1]) <= tol:
                print('CAVI converged with ELBO(q) %.3f at iteration %d'%(self.elbo_values[-1], it))
                break
        
        # iteration terminates but still cannot converge
        if it == max_iter:
            print('CAVI ended with ELBO(q) %.f'%(self.elbo_values[-1]))


    def calc_ELBO(self):
        # calc ELBO(q) given current q(mu_k)'s and q(c_i)'s
        t1 = -(self.m**2 + self.s2) / (2 * prior_var) + 0.5
        t11 = (prior_mean / prior_var) * self.m
        t12 = 0.5 * np.log(2 * pi * self.s2)
        t1 = t1 + t11 + t12
        t1 = t1.sum()
        t2 = -0.5 * np.add.outer(self.data**2, self.s2 + self.m**2)
        t2 += np.outer(self.data, self.m)
        t2 -= np.log(self.phi)
        t2 += np.log(prior_pi)[np.newaxis, :]
        t2 *= self.phi
        t2 = t2.sum()
        return t1 + t2


    def _update_phi(self):
        # update the probability set for each q(c_i)
        t1 = np.outer(self.data, self.m)
        t2 = -0.5 * (self.m**2 + self.s2 + np.log(2 * pi))
        exponent = t1 + t2[np.newaxis, :]
        t3 = np.log(prior_pi)
        exponent += t3[np.newaxis, :]
        self.phi = np.exp(exponent)
        self.phi = self.phi / self.phi.sum(1)[:, np.newaxis]

    def _update_mu(self):
        # update variance of each q(mu_k)
        self.s2 = (1 / prior_var + self.phi.sum(0))**(-1)
        assert self.s2.size == self.K
        # update mean of each q(mu_k)
        self.m = (self.phi * self.data[:, np.newaxis]).sum(0)
        self.m += (prior_mean / prior_var)
        self.m *= self.s2
        assert self.m.size == self.K


###############################################################################
# =============================================================================
# Main code
# =============================================================================
# number of components in GMM
components = 2
# hyper-parameters of prior for mu 
prior_mean = 5
prior_var = 2
# prior for c_i 
prior_pi = [1/components] * components

# true means & variances for mixing Gaussians
comp_mean = [3, 6]
comp_var = 1
# mixing proportion
mix_prop = [0.4, 0.6]
# numer of observations
N = 400


# generate true cluster assignment for each observation
P = np.random.multinomial(1, mix_prop, size = N)
P = np.nonzero(P)[1]

# generate observations
std = np.sqrt(comp_var)
samples = np.zeros(N)
for i, m in enumerate(P):
    samples[i] = np.random.normal(comp_mean[m], std, 1)


# plot histogram of data & PDF of true GMM
x = np.linspace(0, 10, 500)
c1 = mix_prop[0] * st.norm(comp_mean[0], 1).pdf(x)
c2 = mix_prop[1] * st.norm(comp_mean[1], 1).pdf(x)
sup = c1 + c2
plt.hist(samples, bins = 50, density = True)
plt.plot(x, c1, 'b', label = 'component 1: N(%.1f, 1), $\pi_1$ = %.1f' %(comp_mean[0], mix_prop[0]))
plt.plot(x, c2, 'r', label = 'component 2: N(%.1f, 1), $\pi_2$ = %.1f' %(comp_mean[1], mix_prop[1]))
plt.plot(x, sup, 'k')
plt.xlabel("data points")
plt.ylabel("normalized histogram")
plt.title("total samples: %d" %N)
plt.legend()
plt.show()

# print prior distribution info
print("prior p(\u03BC) = N(%.f, %.f)"%(prior_mean, prior_var))

# create VI object
ugmm = GMM(samples, components)
# run CAVI optimization
ugmm.fit(max_iter = 150)


for i in range(components):
    print("converged q(\u03BC_%d) = N(%.4f, %.6f)" %(i+1, ugmm.m[i], ugmm.s2[i]))

# fig, ax = plt.subplots(figsize=(12, 4))
# sns.distplot(samples[:N], ax=ax, hist=True, norm_hist=True)
# sns.distplot(np.random.normal(ugmm.m[0], 1, N), color='k', hist=False, kde=True)
# sns.distplot(samples[N:N*2], ax=ax, hist=True, norm_hist=True)
# sns.distplot(np.random.normal(ugmm.m[1], 1, N), color='k', hist=False, kde=True)
# sns.distplot(samples[N*2:], ax=ax, hist=True, norm_hist=True)
# sns.distplot(np.random.normal(ugmm.m[2], 1, N), color='k', hist=False, kde=True)
# plt.show()

# plot ELBO value history
plt.plot(ugmm.elbo_values)
plt.xlabel("iteration")
plt.ylabel("ELBO value")
plt.grid()
plt.show()

# plot the comparison between true GMM & mixture of approximate dist. q(mu)'s
plt.rcParams['figure.dpi'] = 150
# plt.rcParams['axes.facecolor'] = '#dedede'
q1 = mix_prop[0] * st.norm(ugmm.m[0], np.sqrt(ugmm.s2[0])).pdf(x)
q2 = mix_prop[1] * st.norm(ugmm.m[1], np.sqrt(ugmm.s2[1])).pdf(x)
qq = q1 + q2
plt.plot(x, c1, 'm', label = 'true component 1')
plt.plot(x, c2, 'r', label = 'true component 2')
plt.plot(x, sup, 'k',label = "true GMM", linestyle = "--")
plt.plot(x, q1, 'g', label = 'approx q($\mu_1$)')
plt.plot(x, q2, 'c', label = 'approx q($\mu_2$)')
plt.plot(x, qq, 'k',label = "mixture of q($\mu$)", linestyle = ":")
plt.xlabel("support")
plt.ylabel("PDF")
plt.title("true GMM vs. mixture of converged q($\mu_k$)'s")
plt.legend()
plt.show()
