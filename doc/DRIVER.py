import numpy as np
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
    
    # Create endogenous variable x (n x 1)
    x = z @ tau.reshape(-1, 1) + eta
    
    # Create outcome variable y (n x 1)
    y = x * theta + u
    
    # Create combined matrices
    covariates = np.hstack((x, w))
    instruments = np.hstack((z, w))
    
    # Calculate theoretical value (array length m)
    theory_val = (k * tau**2) / (1 + k * tau**2)
    
    return y, covariates, instruments, theory_val


y, x, z, tv = randiv()    

divergence = KullbackLeibler()
momfun = DefaultMomentFunction(y,x,z)
problem = MDProblem(momfun, divergence)

k = 2
n = 100
m = 6

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

theta = np.array((1,2))
momfun.g(theta)