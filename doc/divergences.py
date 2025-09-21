import numpy as np

# -----------------------------------------------------------------------------
# Utility functions (mimicking the Julia xlogx / xlogy / aloga / alogab etc.)
# -----------------------------------------------------------------------------
def xlogx(x):
    """
    Returns x * log(x) with the convention that 0*log(0)=0.
    Works for scalars or NumPy arrays.
    """
    x = np.asarray(x)
    return np.where(x == 0, 0.0, x * np.log(x))

def xlogy(x, y):
    """
    Returns x * log(y) with the convention that if x==0 then the result is 0.
    """
    x = np.asarray(x)
    return np.where(x == 0, 0.0, x * np.log(y))

def alogab(a, b):
    """
    a*log(a/b) - a + b.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    return xlogy(a, a / b) - a + b

def blogab(a, b):
    """
    -b*log(a/b) + a - b.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    return -xlogy(b, a / b) + a - b

def aloga(a):
    """
    a*log(a) - a + 1.
    """
    a = np.asarray(a)
    return xlogx(a) - a + 1.0

def loga(a):
    """
    -log(a) + a - 1.
    """
    a = np.asarray(a)
    # For a<=0, we return Inf.
    return np.where(a > 0, -np.log(a) + a - 1.0, np.inf)

# -----------------------------------------------------------------------------
# Base divergence class
# -----------------------------------------------------------------------------
class AbstractDivergence:
    def __call__(self, a, b=None):
        """
        Evaluate the divergence. If b is provided, the two‐argument version is used.
        Otherwise the one‐argument version is used.
        """
        if b is None:
            return self.eval(a)
        else:
            return self.eval(a, b)

    def eval_scalar(self, a, b=None):
        raise NotImplementedError("eval_scalar must be implemented in subclasses.")
    
    def eval(self, a, b=None):
        return np.sum(self.eval_scalar(a, b))

    def gradient(self, a, b=None):
        """
        Returns the gradient with respect to the second argument.
        For one-argument evaluation, returns the derivative with respect to a.
        """
        raise NotImplementedError("gradient must be implemented in subclasses.")

    def hessian(self, a, b=None):
        """
        Returns the (scalar or elementwise) Hessian.
        """
        raise NotImplementedError("hessian must be implemented in subclasses.")

# -----------------------------------------------------------------------------
# Kullback-Leibler Divergence
# -----------------------------------------------------------------------------
class KullbackLeibler(AbstractDivergence):
    def eval_scalar(self, a, b=None):
        a = np.asarray(a)
        if b is None:
            return aloga(a)
        else:
            return aloga(a/b)

    def gradient(self, a, b=None):
        a = np.asarray(a)
        if b is None:
            # derivative of aloga: log(a)
            return np.where(a > 0, np.log(a), -np.inf)
        else:
            a = np.asarray(a)
            b = np.asarray(b)
            cond = (a > 0) & (b > 0)
            return np.where(cond, np.log(a / b), -np.inf)

    def hessian(self, a, b=None):
        a = np.asarray(a)
        if b is None:
            # Hessian: 1/a for a>0, else Inf.
            return np.where(a > 0, 1.0 / a, np.inf)
        else:
            a = np.asarray(a)
            b = np.asarray(b)
            cond = (a > 0) & (b > 0)
            return np.where(cond, 1.0 / a, np.inf)

# -----------------------------------------------------------------------------
# Reverse Kullback-Leibler Divergence
# -----------------------------------------------------------------------------
class ReverseKullbackLeibler(AbstractDivergence):
    def eval_scalar(self, a, b=None):
        a = np.asarray(a)
        if b is None:
            return loga(a)
        else:
            return loga(a/b)*b

    def gradient(self, a, b=None):
        a = np.asarray(a)
        if b is None:
            # derivative of loga: -1/a + 1
            return np.where(a > 0, -1.0 / a + 1.0, -np.inf)
        else:
            a = np.asarray(a)
            b = np.asarray(b)
            cond = (a > 0) & (b > 0)
            # gradient: -b/a + 1.
            return np.where(cond, -b / a + 1.0, -np.inf)

    def hessian(self, a, b=None):
        a = np.asarray(a)
        if b is None:
            # Hessian: 1/a^2 for a>0.
            return np.where(a > 0, 1.0 / (a ** 2), np.inf)
        else:
            a = np.asarray(a)
            b = np.asarray(b)
            cond = (a > 0) & (b > 0)
            return np.where(cond, b / (a ** 2), np.inf)

# -----------------------------------------------------------------------------
# Hellinger Divergence
# -----------------------------------------------------------------------------
class Hellinger(AbstractDivergence):
    def eval_scalar(self, a, b=None):
        a = np.asarray(a)
        if b is None:
            # γ(a) = 2*a - 4*sqrt(a) + 2.
            return 2 * a - 4 * np.sqrt(a) + 2
        else:
            a = np.asarray(a)
            b = np.asarray(b)
            # γ(a,b) = 2*a + (2 - 4*sqrt(a/b))*b.
            return 2 * a + (2 - 4 * np.sqrt(a / b)) * b

    def gradient(self, a, b=None):
        a = np.asarray(a)
        if b is None:
            # derivative: 2 - 2/sqrt(a)
            return np.where(a > 0, 2 - 2 / np.sqrt(a), -np.inf)
        else:
            a = np.asarray(a)
            b = np.asarray(b)
            cond = (a > 0) & (b > 0)
            # derivative: 2*(1 - 1/sqrt(a/b))
            return np.where(cond, 2 * (1 - 1 / np.sqrt(a / b)), -np.inf)

    def hessian(self, a, b=None):
        a = np.asarray(a)
        if b is None:
            # Hessian: 1/sqrt(a^3)
            return np.where(a > 0, 1.0 / np.sqrt(a ** 3), np.inf)
        else:
            a = np.asarray(a)
            b = np.asarray(b)
            cond = (a > 0) & (b > 0)
            return np.where(cond, np.sqrt(b) / np.sqrt(a ** 3), np.inf)

# -----------------------------------------------------------------------------
# Chi-Squared Divergence
# -----------------------------------------------------------------------------
class ChiSquared(AbstractDivergence):
    def eval_scalar(self, a, b=None):
        a = np.asarray(a)
        if b is None:
            # γ(a) = 0.5*(a - 1)^2.
            return 0.5 * (a - 1) ** 2
        else:
            a = np.asarray(a)
            b = np.asarray(b)
            # γ(a,b) = 0.5*(a - b)^2 / b.
            return 0.5 * ((a - b) ** 2) / b

    def gradient(self, a, b=None):
        a = np.asarray(a)
        if b is None:
            # derivative: a - 1.
            return a - 1
        else:
            a = np.asarray(a)
            b = np.asarray(b)
            # derivative: a/b - 1.
            return a / b - 1

    def hessian(self, a, b=None):
        a = np.asarray(a)
        if b is None:
            return np.ones_like(a)
        else:
            b = np.asarray(b)
            return np.where(b != 0, 1.0 / b, np.inf)

# -----------------------------------------------------------------------------
# Cressie-Read Divergence (with parameter alpha)
# -----------------------------------------------------------------------------
class CressieRead(AbstractDivergence):
    def __init__(self, alpha):
        self.alpha = alpha

    def eval_scalar(self, a, b=None):
        a = np.asarray(a)
        α = self.alpha
        if b is None:
            # For one argument: if a>=0 then
            # (a^(1+α) + α - a*(1+α))/(α*(1+α))   else (if α>0 then 0 else NaN)
            cond = (a >= 0)
            val = (a ** (1 + α) + α - a * (1 + α)) / (α * (1 + α))
            # For negative a, return 0 if α>0 else NaN.
            return np.where(cond, val, 0.0 if α > 0 else np.nan)
        else:
            b = np.asarray(b)
            cond = (a > 0) & (b > 0)
            val = ((a / b) ** (1 + α) + α - (a / b) * (1 + α)) * b / (α * (1 + α))
            return np.where(cond, val, 0.0 if α > 0 else np.nan)

    def gradient(self, a, b=None):
        a = np.asarray(a)
        α = self.alpha
        if b is None:
            cond = (a >= 0)
            val = (a ** α - 1) / α
            return np.where(cond, val, 0.0 if α > 0 else np.nan)
        else:
            b = np.asarray(b)
            cond = (a >= 0) & (b > 0)
            val = ((a / b) ** α - 1) / α
            return np.where(cond, val, 0.0 if α > 0 else np.nan)

    def hessian(self, a, b=None):
        a = np.asarray(a)
        α = self.alpha
        if b is None:
            cond = (a > 0)
            val = a ** (α - 1)
            return np.where(cond, val, np.inf)
        else:
            b = np.asarray(b)
            cond = (a > 0) & (b > 0)
            val = a ** (α - 1) * b ** (-α)
            return np.where(cond, val, np.inf)

# -----------------------------------------------------------------------------
# Modified Divergences
# -----------------------------------------------------------------------------
class ModifiedDivergence(AbstractDivergence):
    """
    A modified divergence which uses an underlying divergence (self.base)
    and applies an upper modification when a > ρ * b.
    
    The parameters are passed as a dictionary with keys:
      - 'rho': the threshold parameter,
      - 'gamma0', 'gamma1', 'gamma2': parameters for the upper modification.
    """
    def __init__(self, base_divergence, params):
        self.base = base_divergence
        self.params = params

    def eval_scalar(self, a, b=None):
        if b is None:
            a = np.asarray(a)
            rho = self.params.get('rho', 1)
            cond = a > rho
            # Upper modification for one argument:
            gamma0 = self.params.get('gamma0', 0)
            gamma1 = self.params.get('gamma1', 0)
            gamma2 = self.params.get('gamma2', 0)
            val_upper = gamma0 + gamma1 * (a - rho) + 0.5 * gamma2 * (a - rho) ** 2
            val_base = self.base.eval_scalar(a)
            return np.where(cond, val_upper, val_base)
        else:
            a = np.asarray(a)
            b = np.asarray(b)
            rho = self.params.get('rho', 1)
            cond = a > rho * b
            gamma0 = self.params.get('gamma0', 0)
            gamma1 = self.params.get('gamma1', 0)
            gamma2 = self.params.get('gamma2', 0)
            # Upper modification for two arguments:
            val_upper = (gamma0 + gamma1 * ((a / b) - rho) + 0.5 * gamma2 * ((a / b) - rho) ** 2) * b
            val_base = self.base.eval_scalar(a, b)
            return np.where(cond, val_upper, val_base)

    def gradient(self, a, b=None):
        rho = self.params.get('rho', 1)
        if b is None:
            a = np.asarray(a)
            cond = a > rho
            gamma1 = self.params.get('gamma1', 0)
            gamma2 = self.params.get('gamma2', 0)
            grad_upper = gamma1 + gamma2 * (a - rho)
            grad_base = self.base.gradient(a)
            return np.where(cond, grad_upper, grad_base)
        else:
            a = np.asarray(a)
            b = np.asarray(b)
            cond = a > rho * b
            gamma1 = self.params.get('gamma1', 0)
            gamma2 = self.params.get('gamma2', 0)
            grad_upper = gamma1 + (a / b) * gamma2 - gamma2 * rho
            grad_base = self.base.gradient(a, b)
            return np.where(cond, grad_upper, grad_base)

    def hessian(self, a, b=None):
        rho = self.params.get('rho', 1)
        if b is None:
            a = np.asarray(a)
            cond = a > rho
            gamma2 = self.params.get('gamma2', 0)
            hess_upper = gamma2
            hess_base = self.base.hessian(a)
            return np.where(cond, hess_upper, hess_base)
        else:
            a = np.asarray(a)
            b = np.asarray(b)
            cond = a > rho * b
            gamma2 = self.params.get('gamma2', 0)
            hess_upper = gamma2 / b
            hess_base = self.base.hessian(a, b)
            return np.where(cond, hess_upper, hess_base)

# -----------------------------------------------------------------------------
# Fully Modified Divergence
# -----------------------------------------------------------------------------
class FullyModifiedDivergence(AbstractDivergence):
    """
    A fully modified divergence that uses both an upper and lower modification.
    
    Parameters are passed as a dictionary with keys:
      - 'rho' and 'phi': thresholds,
      - For the upper part: 'gamma0', 'gamma1', 'gamma2',
      - For the lower part: 'g0', 'g1', 'g2'.
    """
    def __init__(self, base_divergence, params):
        self.base = base_divergence
        self.params = params

    def eval_scalar(self, a, b=None):
        rho = self.params.get('rho', 1)
        phi = self.params.get('phi', 1)
        if b is None:
            a = np.asarray(a)
            cond_upper = a > rho
            cond_lower = a < phi
            gamma0 = self.params.get('gamma0', 0)
            gamma1 = self.params.get('gamma1', 0)
            gamma2 = self.params.get('gamma2', 0)
            val_upper = gamma0 + gamma1 * (a - rho) + 0.5 * gamma2 * (a - rho) ** 2
            g0 = self.params.get('g0', 0)
            g1 = self.params.get('g1', 0)
            g2 = self.params.get('g2', 0)
            val_lower = g0 + g1 * (a - phi) + 0.5 * g2 * (a - phi) ** 2
            val_base = self.base.eval_scalar(a)
            return np.where(cond_upper, val_upper, np.where(cond_lower, val_lower, val_base))
        else:
            a = np.asarray(a)
            b = np.asarray(b)
            cond_upper = a > rho * b
            cond_lower = a < phi * b
            gamma0 = self.params.get('gamma0', 0)
            gamma1 = self.params.get('gamma1', 0)
            gamma2 = self.params.get('gamma2', 0)
            val_upper = (gamma0 + gamma1 * ((a / b) - rho) + 0.5 * gamma2 * ((a / b) - rho) ** 2) * b
            g0 = self.params.get('g0', 0)
            g1 = self.params.get('g1', 0)
            g2 = self.params.get('g2', 0)
            val_lower = (g0 + g1 * ((a / b) - phi) + 0.5 * g2 * ((a / b) - phi) ** 2) * b
            val_base = self.base.eval_scalar(a, b)
            return np.where(cond_upper, val_upper, np.where(cond_lower, val_lower, val_base))

    def gradient(self, a, b=None):
        rho = self.params.get('rho', 1)
        phi = self.params.get('phi', 1)
        if b is None:
            a = np.asarray(a)
            cond_upper = a > rho
            cond_lower = a < phi
            gamma1 = self.params.get('gamma1', 0)
            gamma2 = self.params.get('gamma2', 0)
            grad_upper = gamma1 + gamma2 * (a - rho)
            g1 = self.params.get('g1', 0)
            g2 = self.params.get('g2', 0)
            grad_lower = g1 + g2 * (a - phi)
            grad_base = self.base.gradient(a)
            return np.where(cond_upper, grad_upper, np.where(cond_lower, grad_lower, grad_base))
        else:
            a = np.asarray(a)
            b = np.asarray(b)
            cond_upper = a > rho * b
            cond_lower = a < phi * b
            gamma1 = self.params.get('gamma1', 0)
            gamma2 = self.params.get('gamma2', 0)
            grad_upper = gamma1 + (a / b) * gamma2 - gamma2 * rho
            g1 = self.params.get('g1', 0)
            g2 = self.params.get('g2', 0)
            grad_lower = g1 + (a / b) * g2 - g2 * phi
            grad_base = self.base.gradient(a, b)
            return np.where(cond_upper, grad_upper, np.where(cond_lower, grad_lower, grad_base))

    def hessian(self, a, b=None):
        rho = self.params.get('rho', 1)
        phi = self.params.get('phi', 1)
        if b is None:
            a = np.asarray(a)
            cond_upper = a > rho
            cond_lower = a < phi
            gamma2 = self.params.get('gamma2', 0)
            hess_upper = gamma2
            g2 = self.params.get('g2', 0)
            hess_lower = g2
            hess_base = self.base.hessian(a)
            return np.where(cond_upper, hess_upper, np.where(cond_lower, hess_lower, hess_base))
        else:
            a = np.asarray(a)
            b = np.asarray(b)
            cond_upper = a > rho * b
            cond_lower = a < phi * b
            gamma2 = self.params.get('gamma2', 0)
            hess_upper = gamma2 / b
            g2 = self.params.get('g2', 0)
            hess_lower = g2 / b
            hess_base = self.base.hessian(a, b)
            return np.where(cond_upper, hess_upper, np.where(cond_lower, hess_lower, hess_base))

# -----------------------------------------------------------------------------
# Vectorized versions for arrays (optional)
# -----------------------------------------------------------------------------
def eval_divergence(d, a, b=None):
    """
    Evaluates the divergence d on each element of a (and b if provided)
    and returns the sum.
    """
    if b is None:
        a = np.asarray(a)
        return np.sum([d.eval_scalar(val) for val in np.nditer(a)])
    else:
        a = np.asarray(a)
        b = np.asarray(b)
        return np.sum([d.eval_scalar(a_val, b_val)
                       for a_val, b_val in zip(np.nditer(a), np.nditer(b))])

def gradient_divergence(d, a, b=None):
    """
    Returns an array with the elementwise gradient.
    """
    if b is None:
        a = np.asarray(a)
        return np.array([d.gradient(val) for val in np.nditer(a)])
    else:
        a = np.asarray(a)
        b = np.asarray(b)
        return np.array([d.gradient(a_val, b_val)
                         for a_val, b_val in zip(np.nditer(a), np.nditer(b))])

def hessian_divergence(d, a, b=None):
    """
    Returns an array with the elementwise Hessian.
    """
    if b is None:
        a = np.asarray(a)
        return np.array([d.hessian(val) for val in np.nditer(a)])
    else:
        a = np.asarray(a)
        b = np.asarray(b)
        return np.array([d.hessian(a_val, b_val)
                         for a_val, b_val in zip(np.nditer(a), np.nditer(b))])
