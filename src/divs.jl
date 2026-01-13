function xlogx(x::Number)
    result = x * NaNMath.log(x)
    return iszero(x) ? zero(result) : result
end

function xlogy(x::Number, y::Number)
    result = x * NaNMath.log(y)
    return iszero(x) && !isnan(y) ? zero(result) : result
end

alogab(a, b) = xlogy(a, a/b) - a + b
blogab(a, b) = -xlogy(b, a ./ b) + a - b
aloga(a) = xlogx(a) - a + one(eltype(a))
loga(a) = -log(a) + a - one(eltype(a))

## -------------------------------------------------------
## Divergence functions
## -------------------------------------------------------
γ(::KullbackLeibler, a::T) where {T <: Real} = aloga(a)
γ(::ReverseKullbackLeibler, a::T) where {T <: Real} = loga(a)
γ(::Hellinger, a::T) where {T <: Real} = 2*a - 4*NaNMath.sqrt(a) + 2
γ(::ChiSquared, a::T) where {T <: Real} = abs2(a - one(T)) * half(T)

function γ(d::CressieRead{D}, a::T) where {T <: Real, D}
    α = d.α
    if a >= 0
        (a^(1 + α) + α - a*(1 + α))/(α*(1 + α))
    else
        if α > 0
            zero(eltype(a))
        else
            convert(eltype(a), NaN)
        end
    end
end

function γᵤ(d::D, a::T) where {T <: Real, D <: AbstractModifiedDivergence}
    (; γ₀, γ₁, γ₂, ρ) = d.m
    return (γ₀ + γ₁*(a-ρ) + half(T)*γ₂*(a-ρ)^2)
end

function γₗ(d::D, a::T) where {T <: Real, D <: AbstractModifiedDivergence}
    (; g₀, g₁, g₂, φ) = d.m
    return (g₀ + g₁*(a-φ) + half(T)*g₂*(a-φ)^2)
end

function γ(d::ModifiedDivergence, a::T) where {T <: Real}
    (; ρ) = d.m
    div = d.d
    return a > ρ ? γᵤ(d, a) : γ(div, a)
end

function γ(d::FullyModifiedDivergence, a::T) where {T <: Real}
    (; ρ, φ) = d.m
    div = d.d
    return a > ρ ? γᵤ(d, a) : a < φ ? γₗ(d, a) : γ(div, a)
end

function γ(d::AbstractDivergence, a::AbstractArray{T}) where {T <: Real}
    out = similar(a, divtype(T))
    γ!(out, d, a)
end

function γ!(out::AbstractArray{T}, d::AbstractDivergence, a::AbstractArray{R}) where {T, R}
    PT = divtype(R)
    @assert promote_type(PT, T) === T "Output array eltype $T cannot hold computed type $PT"
    @inbounds @simd for j in eachindex(a, out)
        out[j] = γ(d, a[j])
    end
    return out
end

## -------------------------------------------------------
## Gradient 
## -------------------------------------------------------
∇ᵧ(::KullbackLeibler, a::T) where {T} = NaNMath.log(a)
∇ᵧ(::ReverseKullbackLeibler, a::T) where {T} = a > 0 ? -1/a + one(T) : convert(T, -Inf)
function ∇ᵧ(d::CressieRead, a::T) where {T}
    return a >= 0 ? (a^d.α - one(T))/d.α : convert(T, sign(d.α)*Inf)
end
∇ᵧ(d::Hellinger, a::T) where {T} = a > 0 ? 2(one(T)-one(T)/sqrt(a)) : convert(T, -Inf)
∇ᵧ(d::ChiSquared, a::T) where {T} = a - one(T)

function ∇ᵤ(d::D, a::T) where {T, D <: AbstractModifiedDivergence}
    (; γ₀, γ₁, γ₂, ρ) = d.m
    return (γ₁ + γ₂*(a-ρ))
end

function ∇ₗ(d::D, a::T) where {T, D <: AbstractModifiedDivergence}
    (; g₀, g₁, g₂, φ) = d.m
    return (g₁ + g₂*(a-φ))
end

function ∇ᵧ(d::ModifiedDivergence, a::T) where {T <: Real}
    (; ρ) = d.m
    div = d.d
    return a > ρ ? ∇ᵤ(d, a) : ∇ᵧ(div, a)
end

function ∇ᵧ(d::FullyModifiedDivergence, a::T) where {T <: Real}
    (; ρ, φ) = d.m
    div = d.d
    return a > ρ ? ∇ᵤ(d, a) : a < φ ? ∇ₗ(d, a) : ∇ᵧ(div, a)
end

## -------------------------------------------------------
## Hessian
## -------------------------------------------------------
Hᵧ(::KullbackLeibler, a::T) where {T} = a > 0 ? one(T)/a : convert(T, Inf)
Hᵧ(::ReverseKullbackLeibler, a::T) where {T} = a > 0 ? one(T)/a^2 : convert(T, Inf)
Hᵧ(d::CressieRead, a::T) where {T} = a > 0 ? a^(d.α-1) : convert(T, Inf)
Hᵧ(d::Hellinger, a::T) where {T} = a > 0 ? one(T)/sqrt(a^(3)) : convert(T, Inf)
Hᵧ(d::ChiSquared, a::T) where {T} = one(T)

function Hᵤ(d::D, a::T) where {T, D <: AbstractModifiedDivergence}
    (; γ₀, γ₁, γ₂, ρ) = d.m
    return γ₂
end

function Hₗ(d::D, a::T) where {T, D <: AbstractModifiedDivergence}
    (; g₀, g₁, g₂, φ) = d.m
    return g₂
end

function Hᵧ(d::ModifiedDivergence, a::T) where {T <: Real}
    (; ρ) = d.m
    div = d.d
    return a > ρ ? Hᵤ(d, a) : Hᵧ(div, a)
end

function Hᵧ(d::FullyModifiedDivergence, a::T) where {T <: Real}
    (; ρ, φ) = d.m
    div = d.d
    return a > ρ ? Hᵤ(d, a) : a < φ ? Hₗ(d, a) : Hᵧ(div, a)
end

## -------------------------------------------------------
## Syntax sugar
## -------------------------------------------------------

"""
    gradient(d::AbstractDivergence, a)
    gradient(d::AbstractDivergence, a, b)

Compute the gradient of the divergence function with respect to `a`.

For a divergence `D(a,b) = Σᵢ γ(aᵢ/bᵢ) bᵢ`, the gradient is:
```math
\\nabla_a D(a,b) = (\\gamma'(a_1/b_1), \\ldots, \\gamma'(a_n/b_n))
```

# Arguments
- `d::AbstractDivergence`: The divergence type
- `a`: The numerator value(s)
- `b`: The denominator value(s) (defaults to 1 if not provided)

# Returns
- For scalar inputs: the gradient value
- For array inputs: a vector of gradient values

# Examples
```jldoctest
julia> using Divergences

julia> kl = KullbackLeibler();

julia> gradient(kl, 2.0)  # γ'(2) = log(2)
0.6931471805599453

julia> gradient(kl, [0.5, 1.0, 2.0])
3-element Vector{Float64}:
 -0.6931471805599453
  0.0
  0.6931471805599453
```
"""
gradient(d::AbstractDivergence, a::T) where {T <: Real} = ∇ᵧ(d, a)
gradient(d::AbstractDivergence, a::T, b::R) where {T <: Real, R <: Real} = ∇ᵧ(d, a/b)

"""
    gradient!(u, d::AbstractDivergence, a)
    gradient!(u, d::AbstractDivergence, a, b)

Compute the gradient in-place, storing the result in `u`.

See [`gradient`](@ref) for details on the computation.

# Arguments
- `u`: Pre-allocated output vector
- `d::AbstractDivergence`: The divergence type
- `a`: The numerator array
- `b`: The denominator array (optional)

# Returns
The modified vector `u`.
"""
function gradient!(u::AbstractVector{T},
        d::AbstractDivergence,
        a::AbstractArray{R}) where {T <: Real, R <: Real}
    PT = divtype(R)
    @assert promote_type(PT, T) === T "Output array eltype $T cannot hold computed type $PT"
    @inbounds @simd for i in eachindex(a, u)
        u[i] = ∇ᵧ(d, a[i])
    end
    return u
end

function gradient!(u::AbstractVector{T},
        d::AbstractDivergence,
        a::AbstractArray{R},
        b::AbstractArray{S}) where {T <: Real, R <: Real, S <: Real}
    PT = divtype(R, S)
    @assert promote_type(PT, T) === T "Output array eltype $T cannot hold computed type $PT"
    @inbounds @simd for i in eachindex(a, b, u)
        u[i] = ∇ᵧ(d, a[i]/b[i])
    end
    return u
end

function gradient(d::AbstractDivergence, a::AbstractArray{R}) where {R <: Real}
    u = similar(a, divtype(R))
    return gradient!(u, d, a)
end

function gradient(d::AbstractDivergence,
        a::AbstractArray{T},
        b::AbstractArray{R}) where {T <: Real, R <: Real}
    u = similar(a, divtype(T, R))
    return gradient!(u, d, a, b)
end

"""
    gradient_sum(d::AbstractDivergence, a)

Compute the sum of gradient values: `Σᵢ γ'(aᵢ)`.

This is more efficient than `sum(gradient(d, a))` as it avoids allocating
an intermediate array.

# Arguments
- `d::AbstractDivergence`: The divergence type
- `a`: The input array

# Returns
The sum of all gradient values.
"""
function gradient_sum(d::AbstractDivergence, a::AbstractArray{R}) where {R <: Real}
    r = zero(R)
    @inbounds @simd for i in eachindex(a)
        r += ∇ᵧ(d, a[i])
    end
    return r
end

"""
    hessian(d::AbstractDivergence, a)
    hessian(d::AbstractDivergence, a, b)

Compute the diagonal of the Hessian matrix of the divergence with respect to `a`.

For a divergence `D(a,b) = Σᵢ γ(aᵢ/bᵢ) bᵢ`, the Hessian diagonal is:
```math
\\text{diag}(\\nabla^2_a D(a,b)) = (\\gamma''(a_1/b_1)/b_1, \\ldots, \\gamma''(a_n/b_n)/b_n)
```

Note: This function returns `γ''(aᵢ/bᵢ)` (not divided by `bᵢ`), which are the
diagonal elements when computing ∂²D/∂aᵢ² with the chain rule applied.

# Arguments
- `d::AbstractDivergence`: The divergence type
- `a`: The numerator value(s)
- `b`: The denominator value(s) (defaults to 1 if not provided)

# Returns
- For scalar inputs: the hessian value
- For array inputs: a vector of hessian diagonal values

# Examples
```jldoctest
julia> using Divergences

julia> kl = KullbackLeibler();

julia> hessian(kl, 2.0)  # γ''(2) = 1/2
0.5

julia> hessian(kl, [0.5, 1.0, 2.0])
3-element Vector{Float64}:
 2.0
 1.0
 0.5
```
"""
hessian(d::AbstractDivergence, a::T) where {T <: Real} = Hᵧ(d, a)
hessian(d::AbstractDivergence, a::T, b::R) where {T <: Real, R <: Real} = Hᵧ(d, a/b)

"""
    hessian!(u, d::AbstractDivergence, a)
    hessian!(u, d::AbstractDivergence, a, b)

Compute the Hessian diagonal in-place, storing the result in `u`.

See [`hessian`](@ref) for details on the computation.

# Arguments
- `u`: Pre-allocated output vector
- `d::AbstractDivergence`: The divergence type
- `a`: The numerator array
- `b`: The denominator array (optional)

# Returns
The modified vector `u`.
"""
function hessian!(u::AbstractVector{T},
        d::AbstractDivergence,
        a::AbstractArray{R}) where {T <: Real, R <: Real}
    PT = divtype(R)
    @assert promote_type(PT, T) === T "Output array eltype $T cannot hold computed type $PT"
    @inbounds @simd for i in eachindex(a, u)
        u[i] = Hᵧ(d, a[i])
    end
    return u
end

function hessian!(u::AbstractVector{T},
        d::AbstractDivergence,
        a::AbstractArray{R},
        b::AbstractArray{S}) where {T <: Real, R <: Real, S <: Real}
    PT = divtype(R, S)
    @assert promote_type(PT, T) === T "Output array eltype $T cannot hold computed type $PT"
    @inbounds @simd for i in eachindex(a, b, u)
        u[i] = Hᵧ(d, a[i]/b[i])
    end
    return u
end

function hessian(d::AbstractDivergence, a::AbstractArray{R}) where {R <: Real}
    u = similar(a, divtype(R))
    return hessian!(u, d, a)
end

function hessian(d::AbstractDivergence,
        a::AbstractArray{T},
        b::AbstractArray{R}) where {T <: Real, R <: Real}
    u = similar(a, divtype(T, R))
    return hessian!(u, d, a, b)
end

"""
    hessian_sum(d::AbstractDivergence, a)

Compute the sum of Hessian diagonal values: `Σᵢ γ''(aᵢ)`.

This is more efficient than `sum(hessian(d, a))` as it avoids allocating
an intermediate array.

# Arguments
- `d::AbstractDivergence`: The divergence type
- `a`: The input array

# Returns
The sum of all Hessian diagonal values.
"""
function hessian_sum(d::AbstractDivergence, a::AbstractArray{R}) where {R <: Real}
    r = zero(R)
    @inbounds @simd for i in eachindex(a)
        r += Hᵧ(d, a[i])
    end
    return r
end

@inline half(::Type{T}) where {T <: Real} = oftype(one(T)/one(T), 0.5)

# Helper to get the result type of division (handles Int->Float64, preserves Dual types)
@inline divtype(::Type{T}, ::Type{R}) where {T, R} = typeof(one(T) / one(R))
@inline divtype(::Type{T}) where {T} = typeof(one(T) / one(T))
