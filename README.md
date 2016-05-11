# Divergences.el

[![Build Status](https://travis-ci.org/gragusa/Divergences.jl.svg?branch=master)](https://travis-ci.org/gragusa/Divergences.jl)

`Divergences` is a Julia package that makes it easy to evaluate divergence measures between two vectors. The package allows calculating the *gradient*  and the diagonal of the *hessian* of the divergence. These divergences are used to good effects in the package [MomentBasedEstimators](http://github.com/gragusa/MomentBasedEstimators.jl/git). 

The package defines a `Divergence` type with the following suptypes:

## Supported divergences

* Kullback-Leibler divergence
* Chi-square distance
* Reverse Kullback-Leibler divergence
* Cressie-Read divergences
* Hellinger divergence

Three versions of each divergence in the above list is implemented currently. A *vanilla* version, a modified version, and a fully modified version. These modifications extend the domain of the divergence.

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

### Gradient of the divergence

To calculate the gradient of the divergence with respect to ``x``
```julia
g = gradient(KullbackLeibler(), x, y)
```

### Hessian of the divergence
To calculate the hessain of the divergence with respect to ``x``
```julia
h = hessian(KullbackLeibler(), x, y)
```

Notice that the hessian of a divergence is sparse, with the only diagonal entries being different from zero. For this reason, `hessian(div, x, y)` return an `Array{Float64,1}` with the diagonal entries of the hessian.

## List of divergences


