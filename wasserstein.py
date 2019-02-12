# Some experiments of interest after reading "Generalizing Point Embeddings
# using the Wasserstein Space of Elliptical Distributions" by Boris Muzellec
# and Marco Cuturi.

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def bures_norm(X, Y):
    """Implementation of the Bures metric on the space of pd matrices, defined in
    eqn 3."""
    Xsqrt = sp.linalg.sqrtm(X)
    res = np.trace(X + Y - 2*sp.linalg.sqrtm(np.matmul(Xsqrt, np.matmul(Y, Xsqrt))))

    return res

def brenier_ot_map(A, B):
    asqrt = sp.linalg.sqrtm(A)
    ainvsqrt = np.linalg.inv(asqrt)

    inner = sp.linalg.sqrtm(np.matmul(asqrt, np.matmul(B, asqrt)))
    
    return np.matmul(ainvsqrt, np.matmul(inner, ainvsqrt))

class EllipticalDistribution(object):
    """An elliptical distribution parameterized by its location (a vector in R^d)
    and its scale (a d x d pd matrix). The covariance constant is called tau_h in
    the paper and is 1 for Gaussians, 1/(d + 2) for uniform ellipses."""
    def __init__(self, loc, scale, cov_constant):
        self.loc = loc
        self.scale = scale
        self.cov_constant = cov_constant

        self.d = len(self.loc)

    def distance(self, other):
        """Compute the 2-Wasserstein metric between self and other."""
        assert(self.cov_constant == other.cov_constant)

        locdist = np.linalg.norm(self.loc - other.loc)
        scaledist = self.cov_constant * bures_norm(self.scale, other.scale)

        return locdist + scale_dist

    def pseudodot(self, other):
        assert(self.cov_constant == other.cov_constant)

        asqrt = sp.linalg.sqrtm(self.scale)

        euc = np.dot(self.loc, other.loc)
        bures = np.trace(sp.linalg.sqrtm(np.matmul(asqrt, np.matmul(other.scale, asqrt))))

        return euc + bures

    def geodesic(self, other):
        """Return the geodesic from self to other, making use of the Brenier
        optimal transportation map, parameterized by a real parameter t in
        [0, 1]."""
        assert(self.cov_constant == other.cov_constant)

        Tab = brenier_ot_map(self.scale, other.scale)

        def geo_loc(t):
            return (1 - t) * self.loc + t * other.loc

        def geo_scale(t):
            s = (1 - t) * np.eye(self.d) + t * Tab
            C = np.matmul(s, np.matmul(self.scale, s))

            return C

        def geo(t)
            return EllipticalDistribution(geo_loc(t),
                                          geo_scale(t),
                                          self.cov_constant)

        return geo, geo_loc, geo_scale
                          
class EllipticalProcess(object):
    def __init__(self, loc_fn, scale_fn, cov_constant):
        self.loc_fn = loc_fn
        self.scale_fn = scale_fn
        self.cov_constant = cov_constant

    def loc(self, t):
        return self.loc_fn(t)
    
    def scale(self, t):
        return self.scale_fn(t)

    def cov(self, t):
        return self.cov_constant * self.scale_fn(t)

    def plot(self, freq=0.2):
        ts = np.arange(0, 1, 0.2)
        locs = list(map(self.loc, ts))
        precs = list(map(lambda t: np.linalg.inv(self.cov(t)), ts))
        ells = list(map(lambda t: Ellipse(xy=

