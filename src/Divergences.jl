module Divergences

using NaNMath
using Distances
abstract type AbstractDivergence <: PreMetric end
abstract type Divergence <: AbstractDivergence end
abstract type AbstractModifiedDivergence <: AbstractDivergence end

"""
    CressieRead(α)

Cressie-Read family of divergences, parameterized by `α`.

The divergence function is:
```math
\\gamma(u) = \\frac{u^{1+\\alpha} - 1}{\\alpha(1+\\alpha)} - \\frac{u - 1}{\\alpha}
```

Special cases:
- `α → 0`: Kullback-Leibler divergence
- `α → -1`: Reverse Kullback-Leibler divergence
- `α = -0.5`: Hellinger divergence
- `α = 1`: Chi-squared divergence

# Arguments
- `α::Real`: The parameter (must not be -1 or 0)

# Examples
```jldoctest
julia> using Divergences

julia> cr = CressieRead(2.0)
CressieRead{Float64}(2.0)

julia> cr(2.0)  # γ(2) for α=2
0.6666666666666666
```
"""
struct CressieRead{T} <: Divergence
    α::T
    function CressieRead(α::T) where {T <: Union{Real, Int}}
        @assert (α != -1 && α != 0) "CressieRead is defined for all α != {-1,0}"
        a = float(α)
        return new{eltype(a)}(a)
    end
end

"""
    ChiSquared()

Chi-squared divergence.

The divergence function is:
```math
\\gamma(u) = \\frac{(u-1)^2}{2}
```

Equivalent to `CressieRead(1.0)`.

# Examples
```jldoctest
julia> using Divergences

julia> cs = ChiSquared()
ChiSquared()

julia> cs(2.0)  # γ(2) = (2-1)²/2 = 0.5
0.5
```
"""
struct ChiSquared <: Divergence end

"""
    KullbackLeibler()

Kullback-Leibler divergence (also known as relative entropy or I-divergence).

The divergence function is:
```math
\\gamma(u) = u \\log(u) - u + 1
```

# Examples
```jldoctest
julia> using Divergences

julia> kl = KullbackLeibler()
KullbackLeibler()

julia> kl(2.0)  # γ(2) = 2*log(2) - 2 + 1
0.3862943611198906
```
"""
struct KullbackLeibler <: Divergence end

"""
    ReverseKullbackLeibler()

Reverse Kullback-Leibler divergence (also known as Burg entropy).

The divergence function is:
```math
\\gamma(u) = -\\log(u) + u - 1
```

# Examples
```jldoctest
julia> using Divergences

julia> rkl = ReverseKullbackLeibler()
ReverseKullbackLeibler()

julia> rkl(2.0)  # γ(2) = -log(2) + 2 - 1
0.3068528194400546
```
"""
struct ReverseKullbackLeibler <: Divergence end

"""
    Hellinger()

Hellinger divergence.

The divergence function is:
```math
\\gamma(u) = 2(\\sqrt{u} - 1)^2 = 2u - 4\\sqrt{u} + 2
```

Equivalent to `CressieRead(-0.5)`.

# Examples
```jldoctest
julia> using Divergences

julia> h = Hellinger()
Hellinger()

julia> h(2.0)  # γ(2) = 2(√2 - 1)²
0.3431457505076194
```
"""
struct Hellinger <: Divergence end

"""
    ModifiedDivergence(d::Divergence, ρ::Real)

A modified divergence that extends a base divergence with a quadratic tail for `u > ρ`.

For `u ≤ ρ`, the divergence equals the base divergence. For `u > ρ`, a quadratic
extension is used that matches the value, gradient, and hessian at `ρ`:
```math
\\gamma_\\rho(u) = \\gamma(\\rho) + \\gamma'(\\rho)(u-\\rho) + \\frac{1}{2}\\gamma''(\\rho)(u-\\rho)^2
```

This modification ensures that the conjugate function has domain extending to all of ℝ.

# Arguments
- `d::Divergence`: The base divergence to modify
- `ρ::Real`: The threshold (must be > 1)

# Examples
```jldoctest
julia> using Divergences

julia> md = ModifiedDivergence(KullbackLeibler(), 1.5);

julia> md(1.2)  # Within threshold: same as KL
0.018785868152745522

julia> md(2.0)  # Above threshold: uses quadratic extension
0.3942635495496621
```
"""
struct ModifiedDivergence{D, T} <: AbstractModifiedDivergence
    d::D
    m::NamedTuple{(:γ₀, :γ₁, :γ₂, :ρ, :aθ, :bθ, :cθ, :inv_γ₂), NTuple{8, T}}
end

"""
    FullyModifiedDivergence(d::Divergence, φ::Real, ρ::Real)

A fully modified divergence with quadratic extensions at both tails.

For `u < φ`, a lower quadratic extension is used. For `u > ρ`, an upper quadratic
extension is used. For `φ ≤ u ≤ ρ`, the divergence equals the base divergence.

Both extensions match the value, gradient, and hessian at their respective thresholds.

# Arguments
- `d::Divergence`: The base divergence to modify
- `φ::Real`: The lower threshold (must be in (0, 1))
- `ρ::Real`: The upper threshold (must be > 1)

# Examples
```jldoctest
julia> using Divergences

julia> fmd = FullyModifiedDivergence(KullbackLeibler(), 0.5, 1.5);

julia> fmd(0.3)  # Below lower threshold: uses lower extension
0.3320558458320164

julia> fmd(1.0)  # Within range: same as KL
0.0

julia> fmd(2.0)  # Above upper threshold: uses upper extension
0.3942635495496621
```
"""
struct FullyModifiedDivergence{D, T} <: AbstractModifiedDivergence
    d::D
    m::NamedTuple{
        (:γ₀, :γ₁, :γ₂, :ρ, :g₀, :g₁, :g₂, :φ,
            :aθ, :bθ, :cθ, :aφ, :bφ, :cφ, :inv_γ₂, :inv_g₂),
        NTuple{16, T}}
end

# Helper to get the float type (preserves Float32, promotes Int to Float64)
_floattype(::Type{T}) where {T <: AbstractFloat} = T
_floattype(::Type{<:Integer}) = Float64
_floattype(::Type{T}) where {T <: Real} = float(T)

function ModifiedDivergence(D::Divergence, ρ::Real)
    @assert ρ > 1 "A ModifiedDivergence requires ρ > 1"
    T = _floattype(typeof(ρ))
    z = convert(T, ρ)
    γ₀ = D(z)
    γ₁ = gradient(D, z)
    γ₂ = hessian(D, z)

    # Precompute dual coefficients
    aθ = one(T) / (2 * γ₂)
    bθ = z - 2 * aθ * γ₁
    cθ = -γ₀ + aθ * γ₁^2
    inv_γ₂ = one(T) / γ₂

    return ModifiedDivergence(D, (
        γ₀ = γ₀, γ₁ = γ₁, γ₂ = γ₂, ρ = z,
        aθ = aθ, bθ = bθ, cθ = cθ, inv_γ₂ = inv_γ₂))
end

function FullyModifiedDivergence(D::Divergence, φ::Real, ρ::Real)
    @assert ρ > 1 "A ModifiedDivergence requires ρ > 1"
    @assert φ < 1 && φ > 0 "A ModifiedDivergence requires φ ∈ (0,1)"

    # Use promoted type from both parameters
    T = promote_type(_floattype(typeof(φ)), _floattype(typeof(ρ)))
    z = convert(T, ρ)
    w = convert(T, φ)

    γ₀ = D(z)
    γ₁ = gradient(D, z)
    γ₂ = hessian(D, z)
    g₀ = D(w)
    g₁ = gradient(D, w)
    g₂ = hessian(D, w)

    # Upper extension coefficients
    aθ = one(T) / (2 * γ₂)
    bθ = z - 2 * aθ * γ₁
    cθ = -γ₀ + aθ * γ₁^2
    inv_γ₂ = one(T) / γ₂

    # Lower extension coefficients
    aφ = one(T) / (2 * g₂)
    bφ = w - 2 * aφ * g₁
    cφ = -g₀ + aφ * g₁^2
    inv_g₂ = one(T) / g₂

    return FullyModifiedDivergence(D,
        (
            γ₀ = γ₀, γ₁ = γ₁, γ₂ = γ₂, ρ = z, g₀ = g₀, g₁ = g₁, g₂ = g₂, φ = w,
            aθ = aθ, bθ = bθ, cθ = cθ, aφ = aφ, bφ = bφ, cφ = cφ, inv_γ₂ = inv_γ₂, inv_g₂ = inv_g₂))
end

for div in (KullbackLeibler,
    ReverseKullbackLeibler,
    Hellinger,
    CressieRead,
    ChiSquared,
    ModifiedDivergence,
    FullyModifiedDivergence)
    @eval begin
        function (f::$div)(p, q)
            return γ(f, p/q)*q
        end
    end
end

for div in (KullbackLeibler,
    ReverseKullbackLeibler,
    Hellinger,
    CressieRead,
    ChiSquared,
    ModifiedDivergence,
    FullyModifiedDivergence)
    @eval begin
        function (f::$div)(p)
            return γ(f, p)
        end
    end
end

for div in (KullbackLeibler,
    ReverseKullbackLeibler,
    Hellinger,
    CressieRead,
    ChiSquared,
    ModifiedDivergence,
    FullyModifiedDivergence)
    @eval begin
        function (f::$div)(a::AbstractArray, b::AbstractArray)
            T = divtype(eltype(a), eltype(b))
            s = zero(T)
            @inbounds @simd for i in eachindex(a, b)
                s += γ(f, a[i] / b[i]) * b[i]
            end
            return s
        end
    end
end

for div in (KullbackLeibler,
    ReverseKullbackLeibler,
    Hellinger,
    CressieRead,
    ChiSquared,
    ModifiedDivergence,
    FullyModifiedDivergence)
    @eval begin
        function (f::$div)(a::AbstractArray)
            return sum(γ(f, a))
        end
    end
end

# Deprecated evaluate functions for backward compatibility
function evaluate(f::AbstractDivergence, a::AbstractArray)
    Base.depwarn("evaluate(div, x) is deprecated, use div(x) instead", :evaluate)
    return sum(f.(a))
end

function evaluate(f::AbstractDivergence, a::AbstractArray, b::AbstractArray)
    Base.depwarn("evaluate(div, x, y) is deprecated, use div(x, y) instead", :evaluate)
    return sum(f.(a ./ b) .* b)
end

function evaluate(f::AbstractDivergence, a::Real)
    Base.depwarn("evaluate(div, x) is deprecated, use div(x) instead", :evaluate)
    return f(a)
end

function evaluate(f::AbstractDivergence, a::Real, b::Real)
    Base.depwarn("evaluate(div, x, y) is deprecated, use div(x, y) instead", :evaluate)
    return f(a, b)
end

# Also keep the Distances.evaluate functions for compatibility
function Distances.evaluate(f::AbstractDivergence, a::AbstractArray)
    Base.depwarn("evaluate(div, x) is deprecated, use div(x) instead", :evaluate)
    return sum(f.(a))
end

function Distances.evaluate(f::AbstractDivergence, a::AbstractArray, b::AbstractArray)
    Base.depwarn("evaluate(div, x, y) is deprecated, use div(x, y) instead", :evaluate)
    return sum(f.(a ./ b) .* b)
end

include("divs.jl")
include("duals.jl")
include("plots.jl")

export
# Divergence types
      KullbackLeibler,
      ReverseKullbackLeibler,
      Hellinger,
      CressieRead,
      ChiSquared,
      ModifiedDivergence,
      FullyModifiedDivergence,
# Gradient and Hessian
#gradient,
#gradient!,
#hessian,
#hessian!,
#gradient_sum,
#hessian_sum,
# Dual (Conjugate) functions
#dual,
#dual_gradient,
#dual_gradient!,
#dual_hessian,
#dual_hessian!,
# Primal-Dual conversion
#primal_from_dual,
#dual_from_primal,
# Verification utilities
#fenchel_young,
#verify_duality,
# Deprecated
      evaluate
end
