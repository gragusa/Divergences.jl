module Divergences

using NaNMath
using Distances
abstract type AbstractDivergence <: PreMetric end
abstract type Divergence <: AbstractDivergence end
abstract type AbstractModifiedDivergence <: AbstractDivergence end

struct CressieRead{T} <: Divergence
    α::T
    function CressieRead(α::T) where {T <: Union{Real, Int}}
        @assert (α != -1 && α != 0) "CressieRead is defined for all α != {-1,0}"
        a = float(α)
        return new{eltype(a)}(a)
    end
end

struct ChiSquared <: Divergence end
struct KullbackLeibler <: Divergence end
struct ReverseKullbackLeibler <: Divergence end
struct Hellinger <: Divergence end

struct ModifiedDivergence{D, T} <: AbstractModifiedDivergence
    d::D
    m::NamedTuple{(:γ₀, :γ₁, :γ₂, :ρ, :aθ, :bθ, :cθ, :inv_γ₂), NTuple{8, T}}
end

struct FullyModifiedDivergence{D, T} <: AbstractModifiedDivergence
    d::D
    m::NamedTuple{(:γ₀, :γ₁, :γ₂, :ρ, :g₀, :g₁, :g₂, :φ,
                   :aθ, :bθ, :cθ, :aφ, :bφ, :cφ, :inv_γ₂, :inv_g₂), NTuple{16, T}}
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

    return FullyModifiedDivergence(D, (
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
# KL
      KullbackLeibler,
# RKL
      ReverseKullbackLeibler,
# HD
      Hellinger,
# CR
      CressieRead,
#
      ChiSquared,
# Modified
      ModifiedDivergence,
# FullyModified
      FullyModifiedDivergence,
# Dual (Conjugate) functions
      dual,
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
