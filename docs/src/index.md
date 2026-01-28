# Divergences.jl

`Divergences.jl` is a Julia package for evaluating divergence measures between two vectors. The package provides efficient computation of divergences, their gradients, and Hessians, making it particularly useful for optimization-based statistical methods.

## Quick Start

```@example quickstart
using Divergences

# Create a divergence instance
kl = KullbackLeibler()

# Evaluate divergence between two vectors
a = [0.2, 0.4, 0.4]
b = [0.1, 0.3, 0.6]
kl(a, b)
```

```@example quickstart
# Compute the gradient
gradient(kl, a, b)
```

```@example quickstart
# Compute the Hessian diagonal
hessian(kl, a, b)
```

## Related Packages

- [MomentBasedEstimators.jl](https://github.com/gragusa/MomentBasedEstimators.jl): Uses Divergences.jl for minimum divergence estimation
- [Distances.jl](https://github.com/JuliaStats/Distances.jl): General distance metrics (Divergences.jl types are subtypes of `PreMetric`)

## Contents

```@contents
Pages = ["theory.md", "divergences.md", "computation.md", "api.md"]
Depth = 2
```

## Index

```@index
```
