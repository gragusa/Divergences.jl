# Divergences.el

[![Build Status](https://travis-ci.org/gragusa/Divergences.jl.svg?branch=master)](https://travis-ci.org/gragusa/Divergences.jl) [![Coverage Status](https://coveralls.io/repos/github/gragusa/Divergences.jl/badge.svg?branch=master)](https://coveralls.io/github/gragusa/Divergences.jl?branch=master) [![codecov](https://codecov.io/gh/gragusa/Divergences.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/gragusa/Divergences.jl)


`Divergences` is a Julia package that makes it easy to evaluate divergence measures between two vectors. The package allows calculating the *gradient*  and the diagonal of the *hessian* of the divergence. These divergences are used to good effect by the  [MomentBasedEstimators](http://github.com/gragusa/MomentBasedEstimators.jl/git) package.

The package defines a `Divergence` type with the following suptypes:

## Supported divergences

* Kullback-Leibler divergence `KullbackLeibler`
* Chi-square distance `ChiSquared`
* Reverse Kullback-Leibler divergence `ReverseKullbackLeibler`
* Cressie-Read divergences

These divergence differs from the equivalent ones defined in the
`Distances` package because they are normalized. Also the package
profide methods for calculating the gradient and the (diagonal of) the
hessian of the divergence. 

The constructor for the aforementioned types are straightforward
```julia
KullbackLeibler()
ChiSqaured()
ReverseKullbackLeibler()
```
The `CressieRead` type define a family of divergences indexed by a
parameter `alpha`. The constructor for `CressieRead` is
```julia
CR(::Real)
```
The Hellinger divergence is obtained by `CR(-1/2)`. For certain value of `alpha`, `CressieRead`
correspond to a divergence that has a specific type defined. For
instance `CR(1)` is equivalent to `ChiSquared` although the underlying
code for evaluation and calculation of the gradient and hessian are
different. 

Three versions of each divergence in the above list is implemented
currently. A *vanilla* version, a modified version, and a fully modified
version. These modifications extend the domain of the divergence.

The **modified** version takes an additional argument that specifies the
point at which the divergence is modified by a convex extension. 
```julia
ModifiedKullbackLeibler(theta::Real)
ModifiedReverseKullbackLeibler(theta::Real)
ModifiedCressieRead(alpha::Real, theta::Real)
```

Similarly, the **fully modified** version takes **two** additional arguments
that specify the points at which the divergence is modified by a convex
extensions.
```julia
FullyModifiedKullbackLeibler(phi::Real, theta::Real)
FullyModifiedReverseKullbackLeibler(phi::Real, theta::Real)
FullyModifiedCressieRead(alpha::Real, phi::Real, theta::Real)
```


## Basic usage 

### Divergence between two vectors

Each divergence corresponds to a *divergence type*. You can always compute a certain divergence between two vectors using the following
syntax

```julia
d = evaluate(div, x, y)
```

Here, `div` is an instance of a divergence type. For example, the type
for Kullback Leibler divergence is ``KullbackLeibler`` (more divergence
types are described in some details in what follows), then you can
compute the Kullbacl Leibler divergence ``x`` and ``y`` as
```julia
d = evaluate(KullbackLeibler(), x, y)
```

We can also calculate the diverge between the vector ``x`` and the unit unit vector
```julia
r = evaluate(KullbackLeibler(), x)
```

The `Divergence` type is a subtype of `PreMetric` defined in the
`Distances` package. As such, the divergences can be evaluated
row-wise and column-wise for `X::Matrix` and `Y::Matrix`. 

```julia
rowise(div, X, Y)
```

```julia
colwise(div, X, Y)
```

### Gradient of the divergence

To calculate the gradient of the divergence with respect to ``x`` the
`gradient` method can be used
```julia
g = gradient(div, x, y)
```
or thourhg its in-place version

```julia
gradient!(Array(Float64, size(x)), div, x, y)
```

### Hessian of the divergence
To calculate the hessain of the divergence with respect to ``x`` the
`hessian` method can be
```julia
h = hessian(div, x, y)
```
or its in-place version 
```julia
hessian!(Array(Float64, size(x)), div, x, y)
```

Notice that the hessian of a divergence is sparse, with the only diagonal entries being different from zero. For this reason, `hessian(div, x, y)` return an `Array{Float64,1}` with the diagonal entries of the hessian.

## List of divergences


