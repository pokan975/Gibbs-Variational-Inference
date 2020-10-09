# -*- coding: utf-8 -*-
"""
@author: Po-Kan (William) Shih
@advisor: Dr.Bahman Moraffah
Gibbs sampling for DPMM
this algorithms corresponds to the algo. 2 of Neal's paper 
"Markov Chain Sampling Methods for Dirichlet Process Mixture Models"
we use Gaussian for both base dist. (G0) of DP prior and mixture components
for simplification, the variance of components are constant & the same
base dist. is only over mean of all components
"""

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from scipy.integrate import quad

plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 12
np.random.seed(1)

# =============================================================================
# Parameters
# =============================================================================
pi = np.pi
comp_var = 1  # vaiances for all mixture components are constant
g0_mean = 10  # mean of base distribution
g0_var = 4  # variance of base distribution
alpha = 0.5  # concentration parameter of DP prior

N = 200  # number of samples
components = 3  # number of mixture components
iterations = 10**4  # times of Gibbs sampling iteration


# =============================================================================
# Draw samples from finite GMM
# =============================================================================
def sampleMM(n, prob, mean):
    '''
    Parameters
    ----------
    n : int
        number of total samples.
    prob : list[float]
        probability vector for mixture components
    mean : array
        mean of mixture components

    Returns
    -------
    val : array
        generate samples from a GMM.
    '''
    # pick component from multinomial choices
    pick_component = np.random.multinomial(1, prob, size = n)
    # get the indices
    mix_component = np.nonzero(pick_component)[1]

    # generate random samples
    val = np.zeros(n)
    std = np.sqrt(comp_var)
    for i, m in enumerate(mix_component):
        # given index, draw sample from corresponding component
        val[i] = np.random.normal(mean[m], std, 1)
        
    return val
    


# =============================================================================
# Gibbs sampling function
# =============================================================================
class Gibbs_sampler(object):
    def __init__(self, size, iters, data):
        '''
        Parameters
        ----------
        size : int
            number of samples.
        iters : int
            number of Gibbs sampling iterations.
        data : array
            observed samples.

        Returns
        -------
        None.
        '''
        self.size = size
        self.iters = iters
        self.data = data
        # record the number of clusters
        self.cluster_num = [1]
        # record the members of each cluster, each cluster represented as a list
        self.clusters = [list(self.data)]
        # record the history of partitions over samples
        self.partition_history = []
        # record the mean of each cluster
        self.cluster_mu = [np.random.normal(g0_mean, np.sqrt(g0_var))]
        # initialize all cluster assignment indicators as zero 
        # (all observations assigned to 1st cluster = index 0 initially)
        self.c = np.zeros(self.size, "int")
        
    def likelihood_new(self, mu, y):
        '''
        Parameters
        ----------
        mu : float
            variable of likelihood of mean given a sample.
        y : float
            value of the sample.

        Returns
        -------
        float
            likelihood value of support of G0 given a sample.
        '''
        a = 2 * pi * np.sqrt(g0_var)
        b1 = -0.5 * (y - mu)**2
        b2 = -0.5 * (mu - g0_mean)**2
        b2 = b2 / g0_var
        return np.exp(b1 + b2) / a


    def G0_posterior(self, y):
        '''
        Parameters
        ----------
        y : list[float]
            all samples assigned to certain cluster.
        
        Returns
        -------
        mu : float
            the new mean of certain cluster drawn from posterior given G0 and 
            samples assigned to this cluster.
        '''
        var = (comp_var * g0_var) / (comp_var + len(y)*g0_var)
        mean = (g0_mean/g0_var + sum(y)/comp_var) * var
        mu = np.random.normal(mean, np.sqrt(var))
        return mu


    def Gibbs_iteration(self):
        # initialize the size of each cluster (last one always represents newly created one)
        cluster_size = [self.size, alpha]
        comp_std = np.sqrt(comp_var)
        # Gibbs sampling iteration
        for t in range(self.iters):
        # =============================================================================
        #   # sample c[i] sequentially
        # =============================================================================
            # the cluster num for this sampling starts from the num of last time
            self.cluster_num.append(self.cluster_num[-1])    
            for i in range(self.size):
                   
                # remove sample[i] from current group
                # index of sample[i]'s current group is c[i]
                c_i_old = self.c[i]
                self.clusters[c_i_old].remove(self.data[i])
                cluster_size[c_i_old] -= 1
                # assert cluster_size[self.c[i]] >= 0
                
                # if sample[i]'s original cluster is empty, remove it from list
                if cluster_size[c_i_old] == 0:
                    cluster_size.pop(c_i_old)
                    self.clusters.pop(c_i_old)
                    self.cluster_mu.pop(c_i_old)
                    self.cluster_num[-1] -= 1
                    # pull indices larger than c_i_old back by 1 since group c_i_old gone
                    self.c = np.where(self.c > c_i_old, self.c - 1, self.c)
        
                # build probability vector for cluster assignment
                # last one is the prob. of new cluster
                prior_prob = np.array(cluster_size) / (self.size - 1 + alpha)
                
                # build the posterior prob. of being assigned to each existing cluster
                likelihood = st.norm(self.cluster_mu, comp_std).pdf(self.data[i])
                posterior_old = np.multiply(prior_prob[0:-1], likelihood)
                
                # build the posterior prob. of being assigned to new cluster
                # obtain likelihood by integrating over support of mu 
                likelihood, err = quad(self.likelihood_new, -np.inf, np.inf, args = (self.data[i],))
                posterior_new = np.array([prior_prob[-1] * likelihood])
                
                # combine above 2 cases to build the multinomial choices of assignment
                posterior = np.concatenate((posterior_old, posterior_new), axis = 0)
                # normalize them to make valid probabilities
                posterior = posterior / sum(posterior)
                assert 1 - sum(posterior) < 1e-10
                # select cluster based on the posterior prob. and extract its index
                c_i_new = np.random.multinomial(1, posterior).nonzero()[0][0]
                # update assignment for sample[i]
                self.c[i] = c_i_new
        
                # sample[i] to new cluster
                if c_i_new == self.cluster_num[-1]:
                    # add a 1-member cluster last position
                    cluster_size.insert(-1, 1)
                    # assign sample[i] to the cluster
                    self.clusters.append([self.data[i]])
                    # draw parameter for this new cluster from posterior given sample[i]
                    self.cluster_mu.append(self.G0_posterior([self.data[i]]))
                    # cluster num + 1
                    self.cluster_num[-1] += 1
        
                # sample[i] to existing cluster; its size + 1 & append sample[i]
                else:
                    cluster_size[c_i_new] += 1
                    self.clusters[c_i_new].append(self.data[i])
               
            # =============================================================================
            #   # update mu for each existing cluster sequentially
            # =============================================================================
            # draw new parameters for all clusters using their respetive members
            self.cluster_mu[:] = list(map(self.G0_posterior, self.clusters))
            
            # record the partition of samples at each iteration
            self.partition_history.append(cluster_size[:-1])



# =============================================================================
# Main function starts here
# =============================================================================
# use Dirichlet dist. to generate mixing proportions
# multi_prob = np.random.dirichlet([5.] * components, 1)
multi_prob = [0.4, 0.3, 0.3]
# use Gaussian dist. to generate the mean for each Gaussian component,
# take round for simplification
# comp_mean = np.round(np.random.normal(g0_mean, np.sqrt(g0_var), components), 2)
comp_mean = [7.85, 5.27, 12.27]

# generate data for fixed sample size
samples = sampleMM(N, multi_prob, comp_mean)
# do Gibbs sampling
G = Gibbs_sampler(N, iterations, samples)
G.Gibbs_iteration()


print("component mean:", comp_mean)

# plot histogram of data
plt.hist(samples, bins = 50, density = True)
plt.xlabel("data points")
plt.ylabel("normalized histogram")
plt.show()

# plot num of clusters vs. iterations
ii = np.arange(0, iterations + 1)
plt.plot(ii, G.cluster_num, "-o")
# plt.rcParams['figure.figsize'] = 20, 5
plt.xlabel("iteration")
plt.ylabel("cluster number")
plt.title("mean cluster number:{:.2f}".format(np.mean(G.cluster_num)))
plt.grid()
plt.show()



# test different sample sizes
sample_size = np.arange(100, 200 + 10, 10)
# expected number of clusters of Gibbs sampling result for each sample size
avg_cluster = []

for nn in sample_size:
    print("sample size:", nn)
    samples = sampleMM(nn, multi_prob, comp_mean)
    G = Gibbs_sampler(nn, iterations, samples)
    G.Gibbs_iteration()
    # compute expected number of clusters for each sampling result
    avg_cluster.append(np.mean(G.cluster_num))
    
# plot number of clusters vs. number of samples
plt.plot(sample_size, avg_cluster)
plt.xlabel("sample size")
plt.ylabel("mean cluster number")
plt.xscale("log")
plt.grid()
plt.show()
