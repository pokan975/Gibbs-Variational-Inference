# -*- coding: utf-8 -*-
"""
@author: Po-Kan (William) Shih
@advisor: Dr.Bahman Moraffah
CAVI module for DP-GMM
prior distributions:
alpha ~ Gamma(s_1, s_2)
v_t ~ Beta(1, alpha), for t = 1, ..., T
mu_t ~ N(mu_0, sigma_0^2)
c_i ~ SB(V), for i = 1, ..., n

likelihood:
x_i ~ N(mu_{c_i}, I_2)

variational distributions:
q(alpha) ~ Gamma(omega_1, omega_2)
q(v_t) ~ Beta(v_t| gamma_1, gamma_2)
q(mu_t) ~ N(mu_t| m_t, s2_t)
q(c_i) ~ Multi(c_i| phi_i), phi_i = (phi_i1, ..., phi_iT)

All numbered ELBO terms & parameter update functions of q refer to respective 
equations in my note "VI for DPMM"
"""

import numpy as np
from scipy.special import digamma, loggamma #gammaln


# hyper-parameters of prior for mu 
prior_mean = np.array([5, 5])
prior_var = 2 * np.eye(2)

class CAVI(object):
    '''
    CAVI module for Dirichlet process 2-D Gaussian mixture model
    '''
    def __init__(self, data, T, s):
        '''
        Parameters
        ----------
        data : array
            sample points
        T : int
            truncation level for approximate dist q
        s : list
            hyper-parameters for dist over alpha
        -------
        None.
        '''
        self.data = data  # 2-D DPMM samples
        self.N = data.shape[0]  # number of samples
        self.T = T        # truncation level for all q
        self.s = s        # hyper-parameters of Gamma(s_1, s_2)
        # prior alpha for p(v) = Beta(1, alpha)
        # self.prior_alpha = np.random.gamma(self.s[0], self.s[1])
        # prior parameter set for p(c_i) = Cat(phi_1, phi_2, ...)
        # self.prior_phi = np.random.beta(1, self.prior_alpha, self.T)
        
        
    def init_q_param(self):
        # initialize means m_t for all q(mu_t)
        m1 = np.random.randint(low=np.min(self.data[:,0]), high=np.max(self.data[:,0]), size=self.T).astype(float)
        m2 = np.random.randint(low=np.min(self.data[:,1]), high=np.max(self.data[:,1]), size=self.T).astype(float)
        self.m = np.vstack((m1, m2)).T
        # add some biases to avoid guessing the true means before CAVI
        self.m += np.random.random((self.T, 2))
        # initialize cov matrices of all q(mu_t) for t = 1, ..., T
        cov = [np.eye(2)] * self.T
        self.s2_ = np.array(cov)
        
        # since all cov matrices are diagonal, extract diag elements for computational convenience
        self.s2 = []
        for t in range(self.T):
            self.s2.append(np.diag(self.s2_[t]))
        self.s2 = np.array(self.s2)
        
        # init parameter sets for each q(v_t)
        self.gamma = np.ones((self.T, 2))
        # initialize parameter sets for all q(c_i) from Dirichlet distribution
        self.phi = np.random.dirichlet([1.]*self.T, self.N)
        # init parameters for q(alpha)
        self.omega = np.array([1, 1])
        
    
    def fit(self, max_iter = 150, tol = 1e-8):
        '''
        this function performs CAVI iteration
        '''
        # initialize variational distributions
        self.init_q_param()
        
        # calc initial ELBO(q)
        self.elbo_values = [self.calc_ELBO()]
        
        # CAVI iteration
        for it in range(1, max_iter + 1):
            # CAVI update
            self.update_c()     # update parameters for each q(c_i)
            self.update_v()     # update parameters for each q(v_t)
            self.update_mu()    # update parameters for each q(mu_t)
            self.update_alpha() # update parameters for q(alpha)
            # calc ELBO(q) after all updates
            self.elbo_values.append(self.calc_ELBO())
            
            # if ELBO change lower than tolerance, stop iteration
            if np.abs((self.elbo_values[-1] - self.elbo_values[-2])/self.elbo_values[-2]) <= tol:
                print('CAVI converged with ELBO(q) %.3f at iteration %d'%(self.elbo_values[-1], it))
                break
        
        # iteration terminates but still not meet convergence criterion
        if it == max_iter:
            print('CAVI ended with ELBO(q) %.f'%(self.elbo_values[-1]))
    
    
    def calc_ELBO(self):
        # initialize ELBO value
        lowerbound = 0
        
        # pre-compute digamma values since they are being used multi times here
        # d1d12 = E[ln(V)], d2d12 = E[ln(1-V)]
        d12 = np.sum(self.gamma, axis = 1)
        d12 = digamma(d12)
        d1d12 = digamma(self.gamma[:, 0]) - d12
        d2d12 = digamma(self.gamma[:, 1]) - d12
        
        # (1) E[lnP(v_t| alpha)]
        lb_pv1 = (self.omega[0] / self.omega[1] - 1) * d2d12
        lb_pv1 += digamma(self.omega[0])
        lb_pv1 -= np.log(self.omega[1])
        lowerbound += lb_pv1.sum()
        
        # (2) E[lnP(mu_t| mu_0, sigma_0^2)]
        lb_pmu1 = self.m * prior_mean[np.newaxis, :]
        lb_pmu1 *= np.diag(prior_var)[np.newaxis, :]
        lb_pmu2 = self.m**2
        lb_pmu2 += self.s2
        lb_pmu2 /= (2 * np.diag(prior_var)[np.newaxis, :])
        lb_pmu1 += lb_pmu2
        lowerbound += lb_pmu1.sum()
        
        # (3) E[lnp(c_i|V)]
        t1 = self.phi * d1d12[np.newaxis, :]
        t1 = t1.sum()
        t2 = np.cumsum(self.phi[:, :0:-1], axis = 1)[:, ::-1]
        t2 *= d2d12[np.newaxis, :-1]
        t2 = t2.sum()
        lb_pc = t1 + t2
        lowerbound += lb_pc.sum()
        
        # (4) E[lnp(x_i|mu_t, c_i)]
        t1 = np.outer(self.data[:, 0], self.m[:, 0])
        t1 += np.outer(self.data[:, 1], self.m[:, 1])
        t2 = self.m**2 + self.s2
        t2 = 0.5 * np.sum(t2, axis = 1)
        t1 -= t2[np.newaxis, :]
        lb_px = self.phi * t1
        lowerbound += lb_px.sum()
        
        # (5) E[q(v_t| gamma_t)]
        t1 = (self.gamma[:, 0] - 1) * d1d12
        t2 = (self.gamma[:, 1] - 1) * d2d12
        lb_qv = t1 + t2
        r12 = np.sum(self.gamma, axis = 1)
        lb_qv += loggamma(r12)
        lb_qv -= loggamma(self.gamma[:, 0])
        lb_qv -= loggamma(self.gamma[:, 1])
        lowerbound -= lb_qv.sum()
        
        # (6) E[lnq(mu_t|m_t, s2_t)]
        lb_qmu = self.s2[:, 0] * self.s2[:, 1]
        lb_qmu = -0.5 * np.log(lb_qmu)
        lowerbound -= lb_qmu.sum()
                
        # (7) E[lnq(c_i| phi_i)]
        lb_qc = self.phi * np.log(self.phi)
        lowerbound -= lb_qc.sum()
        
        # (8) E[lnp(alpha| s_1, s_2)]
        lb_palpha = (self.s[0] - 1) * (digamma(self.omega[0]) - np.log(self.omega[1]))
        lb_palpha -= (self.s[1] * self.omega[0] / self.omega[1])
        lowerbound += lb_palpha
        
        # (9) E[lnq(alpha| omega_1, omega_2)]
        lb_qalpha = (self.omega[0] - 1) * digamma(self.omega[0])
        lb_qalpha -= loggamma(self.omega[0])
        lb_qalpha -= self.omega[0]
        lb_qalpha += np.log(self.omega[1])
        lowerbound -= lb_qalpha
        
        return lowerbound
    
    
    def update_mu(self):
        # update the mean m_t & variance s2_t of all q(mu_t)
        # update means & variances of 1st dimension of q(mu_t)
        self.s2[:, 0] = (1 / prior_var[0, 0] + self.phi.sum(0))**(-1)
        means = (self.phi * self.data[:, 0, np.newaxis]).sum(0)
        means += (prior_mean[0] / prior_var[0, 0])
        self.m[:, 0] = means * self.s2[:, 0]
        
        # update means & variances of 2nd dimension of q(mu_t)
        self.s2[:, 1] = (1 / prior_var[1, 1] + self.phi.sum(0))**(-1)
        means = (self.phi * self.data[:, 1, np.newaxis]).sum(0)
        means += (prior_mean[1] / prior_var[1, 1])
        self.m[:, 1] = means * self.s2[:, 1]
        
        
    def update_c(self):
        # update the parameters (phi_1, ..., phi_T) of all q(c_i)
        self.phi = np.outer(self.data[:, 0], self.m[:, 0])
        self.phi += np.outer(self.data[:, 1], self.m[:, 1])
        
        t1 = np.sum(self.m**2, axis = 1)
        t1 += np.sum(self.s2, axis = 1)
        t1 *= 0.5
        self.phi -= t1[np.newaxis, :]
        
        d12 = np.sum(self.gamma, axis = 1)
        d12 = digamma(d12)
        t1 = digamma(self.gamma[:, 0]) - d12
        t2 = digamma(self.gamma[:, 1]) - d12
        t1[1:] += np.cumsum(t2[:-1])
        self.phi += t1[np.newaxis, :]
        self.phi = np.exp(self.phi)
        self.phi = self.phi / self.phi.sum(1)[:, np.newaxis]
        
        
    def update_v(self):
        # update the parameters (gamma_1, gamma_2) of all q(v_t)
        self.gamma[:, 0] = 1 + np.sum(self.phi, axis = 0)
        
        self.gamma[:, 1] = self.omega[0] / self.omega[1]
        t1 = np.cumsum(self.phi[:, :0:-1], axis = 1)[:, ::-1]
        self.gamma[:-1, 1] += np.sum(t1, axis = 0)
        
        
    def update_alpha(self):
        # update the parameters (omega_1, omega_2) of q(alpha)
        self.omega[0] = self.s[0] + self.T
        s1 = digamma(self.gamma[:, 1])
        ss = np.sum(self.gamma, axis = 1)
        s1 -= digamma(ss)
        self.omega[1] = self.s[1] - s1.sum()
        
    
