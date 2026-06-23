import numpy as np
from enum import Enum
from copy import deepcopy


class RegularizationType(Enum):
    NONE = None
    SHRINKAGE = 1
    DIAGONAL = 2
    DIAGONAL_SCALED = 3


class Gaussian(object):
    def __init__(self, mu=None, sigma=None):
        '''Initialize Gaussian on manifold'''

        self.mu = mu
        self.sigma = sigma

    @property
    def n_dim(self):
        return self.mu.shape[0]

    def prob(self, data):
        '''Evaluate probability of sample
        data can either be a tuple or a list of tuples
        '''

        # Regularization term
        # d = len(self.mu) # Dimensions
        reg = (2 * np.pi) ** -(self.n_dim/2) * (np.linalg.det(self.sigma) + 1e-200)**(-0.5)

        # Mahanalobis Distance:
        dist = data-self.mu
        # Correct size:
        if dist.ndim == 2:
            # Correct dimensions
            pass
        elif dist.ndim == 1 and dist.shape[0] == self.n_dim:
            # Single element
            dist = dist[None, :]
        elif dist.ndim == 1 and self.n_dim == 1:
            # Multiple elements
            dist = dist[:, None]

        dist = (dist * np.linalg.solve(self.sigma, dist.T).T).sum(axis=(dist.ndim - 1))
        probs = np.exp(-0.5 * dist) * reg

        return probs

    def margin(self, i_in):
        '''Compute the marginal distribution'''

        # Compute index:
        if type(i_in) is list:
            mu_m = [self.mu[i] for _, i in enumerate(i_in)]
        else:
            mu_m = self.mu[i_in]

        sigma_m = self.sigma[np.ix_(i_in, i_in)]

        return Gaussian(mu=mu_m, sigma=sigma_m)

    def mle(self, x, h=None, reg_lambda=1e-3, reg_type=RegularizationType.SHRINKAGE):
        '''Maximum Likelihood Estimate
        x         : input data
        h         : optional list of weights
        reg_lambda: Regularization factor
        reg_type  : Covariance RegularizationType
        '''

        self.mu = self._empirical_mean(x, h)
        self.sigma = self._empirical_covariance(x, h, reg_lambda, reg_type)

        return self

    def _empirical_mean(self, x, h=None):
        '''Compute Emperical mean
        x   : (list of) manifold element(s)
        h   : optional list of weights, if not specified weights we be taken equal
        '''

        # Determine weights
        if h is None:
            # No weights given, equal weight for all points
            if type(x) is np.ndarray:
                n_data = x.shape[0]
            elif type(x) is list:
                n_data = len(x)
            elif type(x[0]) is list:
                n_data = len(x[0])
            elif type(x[0]) is np.ndarray:
                n_data = x[0].shape[0]
            else:
                raise RuntimeError('Could not determine dimension of input')
            h = np.ones(n_data) / n_data

        mu = (x.T*h).sum(axis=1)

        return mu

    def _empirical_covariance(self, x, h=None, reg_lambda=1e-3, reg_type=RegularizationType.SHRINKAGE):
        '''Compute emperical mean
        x         : input data
        h         : optional list of weights
        reg_lambda: Regularization factor
        reg_type  : Covariance RegularizationType
        '''

        # Create weights if not supplied:
        if h is None:
            # No weights given, equal weight for all points
            # Determine dimension of input
            if type(x) is np.ndarray:
                n_data = x.shape[0]
            elif type(x) is list:
                n_data = len(x)
            elif type(x[0]) is list:
                n_data = len(x[0])
            elif type(x[0]) is np.ndarray:
                n_data = x[0].shape[0]
            else:
                raise RuntimeError('Could not determine dimension of input')
            h = np.ones(n_data) / n_data

        # Compute covariance:
        # Batch:
        tmp = x - self.mu
        # sigma = tmp.T.dot(np.diag(h).dot(tmp))  # Compute covariance
        sigma = (h * tmp.T).dot(tmp)  # Compute covariance

        # Perform Shrinkage regularization:
        if reg_type == RegularizationType.SHRINKAGE:
            return reg_lambda * np.diag(np.diag(sigma)) + (1 - reg_lambda) * sigma
        elif reg_type == RegularizationType.DIAGONAL:
            return sigma + reg_lambda * np.eye(len(sigma))
        elif reg_type == RegularizationType.DIAGONAL_SCALED:
            order_mat = 10**np.floor(np.log10(np.diag(sigma)+1e-10))
            return sigma + np.diag(reg_lambda * order_mat) + np.eye(len(sigma))*1e-20

        elif reg_type is None:
            return sigma
        else:
            raise ValueError('Unknown regularization type for covariance regularization')

    def condition(self, val, i_in=0, i_out=1):
        '''Gaussian Conditioning
        val  : (list of) elements on submanifold i_in
        i_in : index of  the input sub-manifold
        i_out: index of the sub-manifold to output
        '''

        # Convert indices to list:
        if type(i_in) is not list:
            i_in = [i_in]
        if type(i_out) is not list:
            i_out = [i_out]

        # Marginalize Gaussian to related manifolds
        # Here we create a structure [i_in, iout]
        i_total = [i for ilist in [i_in, i_out] for i in ilist]
        gtmp = self.margin(i_total)

        # Construct new indices:
        i_in = list(range(len(i_in)))
        i_out = list(range(len(i_in), len(i_in) + len(i_out)))

        # Seperate Mu:
        # Compute index:
        mu_i = gtmp.margin(i_in).mu
        mu_o = gtmp.margin(i_out).mu

        # Compute lambda and split:
        Lambda = np.linalg.inv(gtmp.sigma)

        Lambda_ii = Lambda[np.ix_(i_in, i_in)]
        Lambda_oi = Lambda[np.ix_(i_out, i_in)]
        Lambda_oo = Lambda[np.ix_(i_out, i_out)]

        sigma_oo = np.linalg.inv(Lambda_oo)
        x_o = mu_o - sigma_oo.dot(Lambda_oi.dot((val - mu_i).T)).T

        condres = []
        for x in x_o:
            condres.append(Gaussian(mu=x, sigma=sigma_oo))

        # Return result depending input value:
        if len(condres) == 1:
            return condres[-1]
        else:
            return condres

    def __mul__(self, other):
        '''Compute the product of Gaussian'''

        # Get precision matrices::
        lambda_s = np.linalg.inv(self.sigma)
        lambda_o = np.linalg.inv(other.sigma)

        # Compute new covariance:
        sigma = np.linalg.inv(lambda_s + lambda_o)
        mu = sigma.dot(lambda_o.dot(other.mu) + lambda_s.dot(self.mu))

        return Gaussian(mu, sigma)

    def copy(self):
        '''Get copy of Gaussian'''
        g_copy = Gaussian(deepcopy(self.mu), deepcopy(self.sigma))
        return g_copy

    def save(self, name):
        '''Write Gaussian parameters to files: name_mu.txt, name_sigma.txt'''
        np.savetxt('{0}_mu.txt'.format(name), self.mu)
        np.savetxt('{0}_sigma.txt'.format(name), self.sigma)

    def sample(self):
        A = np.linalg.cholesky(self.sigma)
        samp = A.dot(np.random.randn(self.n_dim))
        return self.mu+samp

    @staticmethod
    def load(name):
        '''Load Gaussian parameters from files: name_mu.txt, name_sigma.txt'''
        try:
            mu = np.loadtxt('{0}_mu.txt'.format(name))
            sigma = np.loadtxt('{0}_sigma.txt'.format(name))
            return Gaussian(mu, sigma)
        except Exception as err:
            RuntimeError('Was not able to load Gaussian {0}.txt:'.format(name, err))


class GMM:
    ''' Gaussian Mixture Model class based on sci-learn GMM.
    This child implements additional initialization algorithms

    '''

    def __init__(self, n_components):
        '''Create GMM'''

        self.gaussians = []
        for i in range(n_components):
            self.gaussians.append(Gaussian())

        self.n_components = n_components
        self.priors = np.ones(n_components) / n_components

    def expectation(self, data):
        """Expectation of data per Gaussian in GMM

        @param data:
        @return:
        """
        # Expectation:
        lik = []
        for i, gauss in enumerate(self.gaussians):
            lik.append(gauss.prob(data) * self.priors[i])
        lik = np.vstack(lik)
        return lik

    def prob(self, data):
        """Probability that data belongs to GMM (i.e. expectation sum over kernels)

        @param data:
        @return:
        """
        return self.expectation(data).sum(axis=0)

    def predict(self, data):
        '''Classify to which datapoint each kernel belongs'''
        lik = self.expectation(data)
        return np.argmax(lik, axis=0)

    def fit(self, data, convthres=1e-5, maxsteps=100, minsteps=5, reg_lambda=1e-3,
            reg_type=RegularizationType.SHRINKAGE):
        '''Initialize trajectory GMM using a time-based approach'''

        # Make sure that the data is a tuple of list:
        n_data = len(data)

        prvlik = 0
        avg_loglik = []
        for st in range(maxsteps):
            # Expectation:
            lik = self.expectation(data)
            gamma0 = (lik / (lik.sum(axis=0) + 1e-200))  # Sum over states is one
            gamma1 = (gamma0.T / gamma0.sum(axis=1)).T  # Sum over data is one

            # Maximization:
            # - Update Gaussian:
            for i, gauss in enumerate(self.gaussians):
                gauss.mle(data, gamma1[i,], reg_lambda, reg_type)
            # - Update priors:
            self.priors = gamma0.sum(axis=1)  # Sum probabilities of being in state i
            self.priors = self.priors / self.priors.sum()  # Normalize

            # Check for convergence:
            avglik = -np.log(lik.sum(0) + 1e-200).mean()
            if abs(avglik - prvlik) < convthres and st > minsteps:
                print('EM converged in %i steps' % (st))
                break
            else:
                avg_loglik.append(avglik)
                prvlik = avglik
        if (st + 1) >= maxsteps:
            print('EM did not converge in {0} steps'.format(maxsteps))

        return lik, avg_loglik

    def init_time_based(self, t, data, reg_lambda=1e-3, reg_type=RegularizationType.SHRINKAGE):

        if t.ndim == 2:
            t = t[:, 0]  # Drop last dimension

        # Timing seperation:
        timing_sep = np.linspace(t.min(), t.max(), self.n_components + 1)

        for i, g in enumerate(self.gaussians):
            # Select elements:
            idtmp = (t >= timing_sep[i]) * (t < timing_sep[i + 1])
            sl = np.ix_(idtmp, range(data.shape[1]))

            # Perform mle:
            g.mle(data, reg_lambda=reg_lambda, reg_type=reg_type)
            self.priors[i] = len(idtmp)
        self.priors = self.priors / self.priors.sum()

    def kmeans(self, data, maxsteps=100, reg_lambda=1e-3, reg_type=RegularizationType.SHRINKAGE):

        # Init means
        n_data = data.shape[0]

        id_tmp = np.random.permutation(n_data)
        for i, gauss in enumerate(self.gaussians):
            gauss.mu = data[id_tmp[i], :]

        dist = np.zeros((n_data, self.n_components))
        id_old = np.ones(n_data) + self.n_components
        for it in range(maxsteps):
            # E-step
            # batch:
            for i, gauss in enumerate(self.gaussians):
                dist[:, i] = ((data - gauss.mu) ** 2).sum(axis=1)

            id_min = np.argmin(dist, axis=1)

            # M-step
            # Batch:
            for i, gauss in enumerate(self.gaussians):
                sl = np.ix_(id_min == i, range(data.shape[1]))
                dtmp = data[sl]
                gauss.mle(dtmp, reg_lambda=reg_lambda, reg_type=reg_type)

            self.priors = self.priors / sum(self.priors)

            # Stopping criteria:
            if sum(id_min != id_old) == 0:
                # No datapoint changes:
                print('K-means converged in {0} iterations'.format(it))
                break
            else:
                id_old = id_min


if __name__ =="__main__":

    n_dim = 4
    n_data = 201

    print('Testing Gaussian')
    data = np.random.randn(n_data, n_dim)

    g1 = Gaussian(mu=np.zeros(n_dim), sigma=np.eye(n_dim))
    g1.mle(data)

    print(g1.mu)
    print(g1.sigma)

    g1.condition(data[:,0:2], i_in=[0,1], i_out=[2,3])



