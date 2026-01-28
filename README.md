# Divergences.jl

[![CI](https://github.com/gragusa/Divergences.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/gragusa/Divergences.jl/actions/workflows/ci.yml) [![codecov](https://codecov.io/gh/gragusa/Divergences.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/gragusa/Divergences.jl) [![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl) ![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826) ![lifecycle](https://img.shields.io/badge/lifecycle-stable-green.svg)

`Divergences.jl` is a Julia package that makes evaluating divergence measures between two vectors easy. The package allows for calculating the *gradient*  and the diagonal of the *Hessian* of several divergences. 


## Supported divergences

The package defines an abstract `Divergence` type with the following suptypes:

* Kullback-Leibler divergence `KullbackLeibler`
* Chi-square distance `ChiSquared`
* Reverse Kullback-Leibler divergence `ReverseKullbackLeibler`
* Cressie-Read divergences `CressieRead`

These divergences differ from the equivalent ones defined in the `Distances` package because they are **normalized**. 

Also, the package provides methods for calculating their gradient and the (diagonal elements of the) Hessian matrix.

The constructors for the types above are straightforward
```julia
KullbackLeibler()
ChiSqaured()
ReverseKullbackLeibler()
```
The `CressieRead` type define a family of divergences indexed by a parameter `alpha`. The constructor for `CressieRead` is
```julia
CR(::Real)
```
The Hellinger divergence is obtained by `CR(-1/2)`. For a certain value of `alpha`, `CressieRead` corresponds to a divergence with a defined specific type. For instance, `CR(1)` is equivalent to `ChiSquared` although the underlying code for evaluation and calculation of the gradient and Hessian are different. 

Three versions of each divergence in the above list are currently implemented: a vanilla version, a modified version, and a fully modified version. These modifications extend the domain of the divergence.

The **modified** version takes an additional argument that specifies the point at which a convex extension modifies the divergence. 
```julia
ModifiedKullbackLeibler(theta::Real)
ModifiedReverseKullbackLeibler(theta::Real)
ModifiedCressieRead(alpha::Real, theta::Real)
```

Similarly, the **fully modified** version takes two additional arguments that specify the points at which a convex extension modifies the divergence.
```julia
FullyModifiedKullbackLeibler(phi::Real, theta::Real)
FullyModifiedReverseKullbackLeibler(phi::Real, theta::Real)
FullyModifiedCressieRead(alpha::Real, phi::Real, theta::Real)
```


## Basic usage 

### Divergence between two vectors

Each divergence corresponds to a *divergence type*. You can always compute a certain divergence between two vectors using the following syntax

```julia
x = rand(100)
y = rand(100)
𝒦ℒ = KullbackLeibler()
𝒦ℒ(x, y)
```

Here, `div` is an instance of a divergence type. 

We can also calculate the divergence between the vector ``x`` and the unit vector
```julia
r = 𝒦ℒ(x)
```

The `Divergence` type is a subtype of `PreMetric` defined in the `Distances` package. As such, the divergences can be evaluated column-wise for `X::Matrix` and `Y::Matrix`. 

```julia
colwise(𝒦ℒ, X, Y)
```

The divergence function can also be broadcasted
```julia
𝒦ℒ.(x,y)
```

### Gradient of the divergence

To calculate the gradient of  `div::Divergence` with respect to ``x::AbstractArray{Float64, 1}`` the
`gradient` method can be used
```julia
g = Divergences.gradient(div, x, y)
```
or through its in-place version
```julia
u = Vector{Float64}(undef, size(x))
Divergences.gradient!(u, div, x, y)
```

### Hessian of the divergence
The `hessian` method calculates the Hessian of the divergence with respect to ``x`` 
```julia
h = Divergences.hessian(div, x, y)
```
Its in-place variant is also defined
```julia
u = Vector{Float64}(undef, size(x))
Divergences.hessian!(u, div, x, y)
```

Notice that the the divergence's Hessian is sparse, where the diagonal entries are the only ones different from zero. For this reason, `hessian(div, x, y)` returns an `Array{T,1}` with the diagonal entries of the hessian.
