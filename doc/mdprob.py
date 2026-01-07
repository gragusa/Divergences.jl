import numpy as np
import cyipopt
import numpy as np
from scipy.sparse import coo_array, csc_array
from abc import ABC, abstractmethod

# =============================================================================
# Abstract Base Class for Moment Functions
# =============================================================================
class AbstractMomentFunction(ABC):

    @abstractmethod
    def g(self, theta):
        """
        Compute the moment matrix G(θ) for a given parameter vector θ.
        Parameters
        ----------
        theta : ndarray, shape (k,)
            Parameter vector.

        Returns
        -------
        Z : ndarray, shape (n, m)
            Moment matrix.
        """
        pass

    @abstractmethod
    def Dg(self, theta, pi):
        """
        Compute the derivative of sum(pi*G(θ)) with respect to θ.

        Parameters
        ----------
        theta : ndarray, shape (k,)
            Parameter vector.
        pi : ndarray, shape (n,)
            Weight vector.

        Returns
        -------
        dG : ndarray, shape (m, k)
            Derivative matrix.
        """
        pass

    @abstractmethod
    def Dg_lambda(self, theta, lam):
        """
        Compute the derivative of lambda'G(θ) with respect to θ.

        Parameters
        ----------
        theta : ndarray, shape (k,)
            Parameter vector.
        lam : ndarray, shape (m,)
            Lagrange multiplier vector.

        Returns
        -------
        dgl : ndarray, shape (n, k)
            The derivative matrix.
        """
        pass

    @abstractmethod
    def Dg_lambda_pi(self, theta, lam, pi):
        """
        Compute the derivative of pi*g with respect to θ,
        weighted by pi.

        Parameters
        ----------
        theta : ndarray, shape (k,)
            Parameter vector.
        lam : ndarray, shape (m,)
            Lagrange multiplier vector.
        pi : ndarray, shape (n,)
            Weight vector.

        Returns
        -------
        dgl : ndarray, shape (n, k)
            The weighted derivative matrix.
        """
        pass

    @abstractmethod
    def Hg_lambda(self, theta, lam, pi):
        """
        Compute the Hessian of lambda' g with respect to theta.
        (In this example, it returns a zero matrix as a placeholder.)

        Parameters
        ----------
        theta : ndarray, shape (k,)
            Parameter vector.
        lam : ndarray, shape (m,)
            Lagrange multiplier vector.
        pi : ndarray, shape (n,)
            Weight vector.

        Returns
        -------
        H : ndarray, shape (k, k)
            Hessian matrix.
        """
        pass

# =============================================================================
# A Concrete Implementation of the Moment Function
# Instrumental Variables Moment Function
# =============================================================================
class DefaultMomentFunction(AbstractMomentFunction):
    def __init__(self, y, x, z):
        """
        Initialize the moment function caches.

        Parameters
        ----------
        y : ndarray, shape (n,)
            Response variable.
        x : ndarray, shape (n, k)
            Regressors (first column is x, additional columns may be included).
        z : ndarray, shape (n, m)
            Instrumental variables.
        """

        self.y = np.ravel(y)
        self.n = self.y.shape[0]
        self.x = np.asarray(x)
        if self.x.ndim == 1:
            self.x = self.x.reshape(-1, 1)
        self.k = self.x.shape[1]

        # Ensure z is 2D
        self.z = np.asarray(z)
        if self.z.ndim == 1:
            self.z = self.z.reshape(-1, 1)
        self.m = self.z.shape[1]

        # Validate dimensions
        if self.x.shape[0] != self.n or self.z.shape[0] != self.n:
            raise ValueError("All inputs (y, x, z) must have the same number of observations")

        # Allocate caches (same shapes as the originals)
        self.Y = np.empty_like(self.y)      # for temporary n-vector operations
        self.X = np.empty_like(self.x)      # for temporary (n,k) operations
        self.Z = np.empty_like(self.z)      # for temporary (n,m) operations
        # Cache for the gradient matrix: shape (m,k)
        self.dG = np.empty((self.m, self.k), dtype=float)

    def g(self, theta):
        # Compute Y = x dot theta, storing the result in the cache self.Y.
        np.matmul(self.x, theta, out=self.Y)
        # Overwrite Y with (y - x dot theta) in place.
        np.subtract(self.y, self.Y, out=self.Y)
        # Compute Z: for each observation i, multiply row z[i] by Y[i].
        
        self.Z[:] = self.z * self.Y[:, np.newaxis]
        return self.Z

    def Dg(self, theta, pi):
        # Compute X = pi * x (elementwise multiplication along rows).
        self.X[:] = self.x * pi[:, np.newaxis]
        # Compute dG = - z^T dot X, storing in self.dG.
        np.dot(self.z.T, self.X, out=self.dG)
        self.dG *= -1.0
        return self.dG

    def Dg_lambda(self, theta, lam):
        # Compute Y = z  lam (vector of length n).
        np.matmul(self.z, lam, out=self.Y)
        # Compute X = - Y * x (each row scaled by -Y[i]) and divide by n.
        self.X[:] = -self.Y[:, np.newaxis] * self.x
        self.X /= self.n
        return self.X

    def Dg_lambda_pi(self, theta, lam, pi):
        # Compute the unweighted derivative first.
        dgl = self.Dg_lambda(theta, lam)
        # Multiply each row by the corresponding pi element.
        dgl[:] = dgl * pi[:, np.newaxis]
        return dgl

    def Dg_lambda_inplace(self, J, theta, lam, pi=None):
        if pi is None:
            dgl = self.Dg_lambda(theta, lam)
        else:
            dgl = self.Dg_lambda_pi(theta, lam, pi)
        # Flatten the dgl array (C-order) and copy into J.
        np.copyto(J, dgl.ravel())

    def Hg_lambda(self, theta, lam, pi):
        # Placeholder: returns a zero matrix of shape (k, k).
        return np.zeros((self.k, self.k), dtype=float)

# =============================================================================
# The MDProblem class
# =============================================================================
class MDProblem(cyipopt.Problem):
    r"""
    A Python translation of the Ipopt problem defined in Julia.
    
    The decision variable vector `u` is assumed to be partitioned as
       u = [π; θ]
    with π an n‐vector and θ a k‐vector.
    
    The constraints are defined in terms of a moment function
       g(θ) = [ z_i * (y_i - x_i' θ) ]_{i=1}^n
    and a “weighted‐sum” constraint computed as:
       c(j) = (1/n) Σ_{i=1}^{n} π[i] g(θ)[i, j]    for j=1,…, m
       c(m+1) = Σ_{i=1}^{n} π[i]
    so that the overall constraint vector is of length (m+1).
    
    The objective function is taken to be a divergence function of π.
    (Typically you’ll supply a divergence object with methods `__call__`,
    `gradient`, and `hessian`.)
    """
    def __init__(self, moment, divergence):
        """
        Parameters
        ----------
        moment : MomentFunction instance
            Holds the data and cached arrays for computing g and its derivatives.
        divergence : object
            A divergence object supporting __call__(pi), gradient(pi) and hessian(pi).
        backend : any, optional
            (Optional) backend information.
        """
        self.moment = moment
        self.divergence = divergence
        # Dimensions: n = number of observations, k = dimension of θ, m = dimension of instruments
        self.n = moment.n
        self.k = moment.k
        self.m = moment.m

    # ----------------------------
    # Objective and its gradient
    # ----------------------------
    def objective(self, u):
        r"""
        Evaluate the objective function:
             f(u) = divergence(π)
        where u = [π; θ].
        """
        pi = u[:self.n]
        return self.divergence(pi)

    def gradient(self, u):
        r"""
        Evaluate the gradient of the objective with respect to u.
        
        The derivative with respect to π is given by divergence.gradient(pi)
        and with respect to θ is zero.
        """
        pi = u[:self.n]
        grad = np.empty_like(u)
        grad[:self.n] = self.divergence.gradient(pi)
        grad[self.n:] = 0.0
        return grad

    # ----------------------------
    # Constraints and their Jacobian
    # ----------------------------
    def constraints(self, u):
        r"""
        Evaluate the constraints.
        
        Let G = g(θ) be the (n×m) moment matrix computed by the cached moment function.
        Then define
           c(j) = (1/n)*Σ_{i=1}^{n} π[i]*G[i,j]   for j = 1,..., m
           c(m+1) = Σ_{i=1}^{n} π[i]
        so that the constraint vector has length (m+1).
        """
        pi = u[:self.n]
        theta = u[self.n:]
        # Evaluate G = g(θ) (note: this call reuses cached arrays in self.moment)
        G = self.moment.g(theta)
        constr = np.empty(self.m + 1, dtype=np.float64)
        # For j = 0,..., m-1:
        constr[:self.m] = np.sum(pi[:, None] * G, axis=0) / self.n
        # Last constraint: sum of π
        constr[self.m] = np.sum(pi) - self.n
        return constr

    def jacobian(self, u):
        r"""
        Evaluate the constraint Jacobian.
        
        The Jacobian is the block matrix:
           J = [  (G/n)^T       (Dg/n)^T ]
               [  ones(1,n)       zeros(1,k) ]
        where G = g(θ) is (n×m) and Dg = Dg(θ, π) is (m×k). We return J as a
        2D array of shape ((m+1) x (n+k)). (Ipopt may require a flattened version.)
        """
        pi = u[:self.n]
        theta = u[self.n:]
        # Compute G and its derivative Dg; note that our moment function routines
        # use cached arrays to avoid allocation.
        G = self.moment.g(theta)
        G_scaled = G / self.n
        Dg = self.moment.Dg(theta, pi)
        Dg_scaled = Dg / self.n
        # Build the top m rows: for constraints 1..m.
        # With respect to π: derivative is (G_scaled)^T, with respect to θ: derivative is Dg_scaled.
        top_block = np.hstack((G_scaled.T, Dg_scaled))
        # Last row: derivative of constraint c(m+1) with respect to π is 1 and with respect to θ is 0.
        last_row = np.hstack((np.ones((1, self.n)), np.zeros((1, self.k))))
        J_full = np.vstack((top_block, last_row))
        # If a flat vector is needed (for example by Ipopt) then you might return J_full.ravel()
        return J_full

    # ----------------------------
    # Hessian of the Lagrangian
    # ----------------------------
    def hessianstructure(self):
        """Return the (row, col) indices of the lower-triangular non-zero elements of H."""
        # Diagonal elements of D: (0,0), (1,1), ..., (n-1,n-1)
        n = self.n
        k = self.k
        diag_rows = np.arange(n)
        diag_cols = np.arange(n)
    
        # Off-diagonal block Dg' (k x n block starting at row n, column 0)
        block_rows = np.repeat(np.arange(n, n + k), n)
        block_cols = np.tile(np.arange(n), k)
    
        # Combine indices
        rows = np.concatenate([diag_rows, block_rows])
        cols = np.concatenate([diag_cols, block_cols])
        return rows, cols

    def hessian(self, u, lam, sigma):
        r"""
        Evaluate the Hessian of the Lagrangian
           L(π, θ, λ) = divergence(π) + λ' g(θ)
        at the point u, with scalar multiplier sigma and Lagrange multiplier lam.
        
        The Hessian is returned as a flat vector containing:
           - The first n entries: if sigma==0, zeros; otherwise, sigma times the divergence Hessian at π.
           - The next n*k entries: the flattened version (row-major) of Dgλ (the derivative of λ'g w.r.t. θ).
           - The final k*(k+1)//2 entries: the lower triangular part of Hgλ.
        (In the Julia code Hgλ is a zero matrix; here we follow that.)
        """
        pi = u[:self.n]
        theta = u[self.n:]
        if sigma != 0:
            D_diag = self.divergence.hessian(pi)*sigma
        else:
            D_diag = np.zeros(self.n)

        # Get Dg (n x k matrix) and transpose it to k x n (Dg')
        Dg = self.moment.Dg_lambda(theta, lam[:self.m])
        Dg_T_flat = Dg.T.flatten()  # Flatten in row-major order
    
        # Combine D diagonal and Dg' block values
        H = np.concatenate([D_diag, Dg_T_flat])
        return H
    
    # def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
    #                  d_norm, regularization_size, alpha_du, alpha_pr,
    #                  ls_trials):
    #     """Prints information at every Ipopt iteration."""
    #     iterate = self.get_current_iterate()
    #     infeas = self.get_current_violations()
    #     primal = iterate["x"]
    #     jac = self.jacobian(primal)

    #     print("Iteration:", iter_count)
    #     print("Primal iterate:", primal)
    #     print("Flattened Jacobian:", jac)



exec(open('divergences.py').read())


def randiv(n=100, m=5, k=1, theta=0.0, rho=0.9, CP=20):
    """
    Simulates instrumental variables regression data
    
    Returns:
    y: outcome variable (n x 1)
    covariates: matrix [x w] (n x (1 + k))
    instruments: matrix [z w] (n x (m + k))
    theory_val: theoretical strength measure (array length m)
    """
    # Generate instrument strength vector
    tau = np.full(m, np.sqrt(CP / (m * n)))
    
    # Generate base data matrices
    z = np.random.randn(n, m)    # Instruments
    w = np.random.randn(n, k)    # Exogenous controls (corrected to k columns)
    
    # Generate correlated errors
    eta = np.random.randn(n, 1)
    u = rho * eta + np.sqrt(1 - rho**2) * np.random.randn(n, 1)
    
    # Create endogenous variable x
    x = z @ tau.reshape(-1, 1) + eta
    
    # Create outcome variable y (n,)
    y = x * theta + u
    
    # Create combined matrices
    covariates = np.hstack((x, w))
    instruments = np.hstack((z, w))
    
    
    
    return y, covariates, instruments


n = 100
n_instruments = 5
n_exo = 1
np.random.seed(42)
y, x, z = randiv(n=n,k=n_exo, m=n_instruments)    
n, m = z.shape
n, k = x.shape

np.savetxt('y.csv', y, delimiter=',')          # Shape (n,)
np.savetxt('x.csv', x, delimiter=',')          # Shape (n, n_exo)
np.savetxt('z.csv', z, delimiter=',')          # Shape (n, n_instruments)

divergence = KullbackLeibler()
momfun = DefaultMomentFunction(y,x,z)
problem = MDProblem(momfun, divergence)

pi = np.random.uniform(0,1,n)
theta = np.random.uniform(0, 1, k)


u0 = np.concatenate((pi, theta))
lb = np.concatenate((np.zeros_like(pi), -10.0 * np.ones_like(theta)))
ub = np.concatenate((np.inf*np.ones_like(pi), 10.0 * np.ones_like(theta)))

# Define constraint bounds. Our constraint vector has length m+1.
# For equality constraints, we set cl = cu.
# For instance, suppose we require c(u) == 0.
cl = np.zeros(m + 1)
cu = np.zeros(m + 1)

#prob = MDOptProblem(problem, u0, lb, ub, cl, cu)
pi = u0[:n]
theta = u0[n:]
lam = np.ones(m)

momfun.g(theta)
momfun.Dg(theta, pi)
momfun.Dg_lambda(theta, lam)
momfun.Dg_lambda_pi(theta, lam, pi)

p = cyipopt.Problem(
    n=len(u0),
    m=len(cl),
    problem_obj=problem,
    lb=lb,
    ub=ub,
    cl=cl,
    cu=cu,
)

p.add_option('derivative_test', 'second-order')
p.add_option('print_level', 5)
p.add_option('derivative_test_print_all', 'no')
p.solve(u0)
