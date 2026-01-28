# Computation Guide

This section demonstrates how to use Divergences.jl to compute divergences, gradients, Hessians, and dual functions.

## Basic Usage

### Creating Divergence Instances

All divergence types are callable structs:

```@example computation
using Divergences

# Create divergence instances
kl = KullbackLeibler()
rkl = ReverseKullbackLeibler()
h = Hellinger()
cs = ChiSquared()
cr = CressieRead(2.0)  # Cressie-Read with α = 2
```

### Evaluating Divergences

Divergences can be evaluated using function call syntax:

```@example computation
# Single values
kl(2.0, 1.0)  # γ(2/1) * 1
```

```@example computation
# With default b = 1
kl(2.0)  # γ(2)
```

```@example computation
# Vector inputs
a = [0.2, 0.4, 0.4]
b = [0.1, 0.3, 0.6]
kl(a, b)  # Σᵢ γ(aᵢ/bᵢ) bᵢ
```

```@example computation
# Vector with default b = 1
kl([0.5, 1.0, 1.5])  # Σᵢ γ(aᵢ)
```

### Broadcasting

Divergences support broadcasting for element-wise computation:

```@example computation
a = [0.5, 1.0, 2.0]
b = [1.0, 1.0, 1.0]

# Element-wise divergence values
kl.(a, b)
```

## Gradient Computation

The gradient of a divergence ``D(a, b)`` with respect to ``a`` is:

```math
\nabla_a D(a, b) = \left( \gamma'(a_1/b_1), \ldots, \gamma'(a_n/b_n) \right)
```

### Using `gradient`

```@example computation
a = [0.5, 1.0, 2.0]
b = [1.0, 1.0, 1.0]

# Compute gradient
gradient(kl, a, b)
```

```@example computation
# Single value
gradient(kl, 2.0, 1.0)  # γ'(2) = log(2)
```

```@example computation
# With default b = 1
gradient(kl, [0.5, 1.0, 2.0])
```

### In-Place Computation

For performance-critical code, use the in-place variant:

```@example computation
a = [0.5, 1.0, 2.0]
b = [1.0, 1.0, 1.0]
out = similar(a)

gradient!(out, kl, a, b)
out
```

### Sum of Gradients

For computing ``\sum_i \gamma'(a_i)`` efficiently:

```@example computation
gradient_sum(kl, [0.5, 1.0, 2.0])
```

## Hessian Computation

The Hessian of a divergence is diagonal. This package returns the diagonal elements:

```math
\text{diag}(\nabla^2_a D) = \left( \gamma''(a_1/b_1), \ldots, \gamma''(a_n/b_n) \right)
```

### Using `hessian`

```@example computation
a = [0.5, 1.0, 2.0]
b = [1.0, 1.0, 1.0]

# Compute Hessian diagonal
hessian(kl, a, b)
```

```@example computation
# Single value
hessian(kl, 2.0)  # γ''(2) = 1/2
```

### In-Place Computation

```@example computation
a = [0.5, 1.0, 2.0]
out = similar(a)

hessian!(out, kl, a)
out
```

### Sum of Hessian Diagonal

```@example computation
hessian_sum(kl, [0.5, 1.0, 2.0])
```

## Dual (Conjugate) Functions

The dual divergence uses the Legendre-Fenchel conjugate ``\psi = \gamma^*``:

```math
D^*(v, b) = \sum_i \psi(v_i) b_i
```

### Evaluating the Dual

```@example computation
v = [0.1, 0.2, 0.3]
b = [1.0, 1.0, 1.0]

dual(kl, v, b)
```

```@example computation
# With default b = 1
dual(kl, v)
```

### Dual Gradient

The gradient of the dual function:

```@example computation
dual_gradient(kl, v)
```

### Dual Hessian

```@example computation
dual_hessian(kl, v)
```

## Primal-Dual Relationships

### Converting Between Primal and Dual

Given dual variable ``v``, recover primal ``a``:

```@example computation
v = 0.5
b = 1.0
primal_from_dual(kl, v, b)  # Returns a such that γ'(a/b) = v
```

Given primal ``a``, compute dual ``v``:

```@example computation
a = 2.0
b = 1.0
dual_from_primal(kl, a, b)  # Returns γ'(a/b)
```

### Verifying Duality

Check that the Fenchel-Young equality holds:

```@example computation
a = [0.5, 1.0, 2.0]
b = [1.0, 1.0, 1.0]
verify_duality(kl, a, b)  # Should be ≈ 0
```

## Modified Divergences

Modified divergences extend the domain with quadratic tails:

```@example computation
# Create a modified KL divergence
md = ModifiedDivergence(kl, 1.5)

# Behaves like KL for u ≤ 1.5
md(1.2)
```

```@example computation
# Uses quadratic extension for u > 1.5
md(2.0)
```

### Fully Modified Divergences

```@example computation
# Extensions at both ends
fmd = FullyModifiedDivergence(kl, 0.5, 1.5)

# Uses lower extension for u < 0.5
fmd(0.3)
```

```@example computation
# Uses upper extension for u > 1.5
fmd(2.0)
```

## Comparing Divergences

```@example computation
a = [0.2, 0.4, 0.4]
b = [0.1, 0.3, 0.6]

divergences = [
    KullbackLeibler(),
    ReverseKullbackLeibler(),
    Hellinger(),
    ChiSquared(),
    CressieRead(0.5)
]

for d in divergences
    println("$(typeof(d)): $(d(a, b))")
end
```
