module Divergences

using NaNMath
using Distances
abstract type AbstractDivergence <: PreMetric end
abstract type Divergence <: AbstractDivergence end
abstract type AbstractModifiedDivergence <: AbstractDivergence end

struct CressieRead{T} <: Divergence
    α::T
    function CressieRead(α::T) where T<:Union{Real, Int}
        @assert (α != -1 && α != 0) "CressieRead is defined for all α != {-1,0}"
        a = float(α)
        new{eltype(a)}(a)
    end
end

struct ChiSquared  <: Divergence end
struct KullbackLeibler  <: Divergence end
struct ReverseKullbackLeibler <: Divergence end
struct Hellinger <: Divergence end

struct ModifiedDivergence{D, T} <: AbstractModifiedDivergence
    d::D
    m::NamedTuple{(:γ₀, :γ₁, :γ₂, :ρ), Tuple{T, T, T, T}}
end

struct FullyModifiedDivergence{D, T} <: AbstractModifiedDivergence
    d::D
    m::NamedTuple{(:γ₀, :γ₁, :γ₂, :ρ, :g₀, :g₁, :g₂, :φ), Tuple{T, T, T, T, T, T, T, T}}
end

function ModifiedDivergence(D::Divergence, ρ::Union{Real, Int})
    @assert ρ > 1 "A ModifiedDivergence requires ρ > 1"
    z = float(ρ)
    γ₀ = D(z)
    γ₁ = gradient(D, z)
    γ₂ = hessian(D, z)
    ModifiedDivergence(D, (γ₀=γ₀, γ₁=γ₁, γ₂=γ₂, ρ=z))
end

function FullyModifiedDivergence(D::Divergence, φ::Union{Real,Int}, ρ::Union{Real, Int})
    @assert ρ > 1 "A ModifiedDivergence requires ρ > 1"
    @assert φ < 1 && φ > 0 "A ModifiedDivergence requires  φ ∈ (0,1)"
    z = float(ρ)
    γ₀ = D(z)
    γ₁ = gradient(D, z)
    γ₂ = hessian(D, z)
    w = float(φ)
    g₀ = D(w)
    g₁ = gradient(D, w)
    g₂ = hessian(D, w)
    FullyModifiedDivergence(D, (γ₀=γ₀, γ₁=γ₁, γ₂=γ₂, ρ=z, g₀=g₀, g₁=g₁, g₂=g₂, φ=w))
end

for div ∈ (KullbackLeibler, ReverseKullbackLeibler, Hellinger, CressieRead, ChiSquared, ModifiedDivergence, FullyModifiedDivergence)
    @eval begin
        function (f::$div)(p, q)
            return γ(f, p/q)*q
        end
    end
end

for div ∈ (KullbackLeibler, ReverseKullbackLeibler, Hellinger, CressieRead, ChiSquared, ModifiedDivergence, FullyModifiedDivergence)
    @eval begin
        function (f::$div)(p)
            return γ(f, p)
        end
    end
end

for div ∈ (KullbackLeibler, ReverseKullbackLeibler, Hellinger, CressieRead, ChiSquared, ModifiedDivergence, FullyModifiedDivergence)
    @eval begin
        function (f::$div)(a::AbstractArray, b::AbstractArray)
            return sum(γ(f, a./b).*b)
        end
    end
end

for div ∈ (KullbackLeibler, ReverseKullbackLeibler, Hellinger, CressieRead, ChiSquared, ModifiedDivergence, FullyModifiedDivergence)
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
    return sum(f.(a./b).*b)
end

function evaluate(f::AbstractDivergence, a::Real)
    Base.depwarn("evaluate(div, x) is deprecated, use div(x) instead", :evaluate)
    f(a)
end

function evaluate(f::AbstractDivergence, a::Real, b::Real)
    Base.depwarn("evaluate(div, x, y) is deprecated, use div(x, y) instead", :evaluate)
    f(a, b)
end

# Also keep the Distances.evaluate functions for compatibility
function Distances.evaluate(f::AbstractDivergence, a::AbstractArray)
    Base.depwarn("evaluate(div, x) is deprecated, use div(x) instead", :evaluate)
    return sum(f.(a))
end

function Distances.evaluate(f::AbstractDivergence, a::AbstractArray, b::AbstractArray)
    Base.depwarn("evaluate(div, x, y) is deprecated, use div(x, y) instead", :evaluate)
    return sum(f.(a./b).*b)
end

include("divs.jl")
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
    # Abbr.
    𝒦ℒ,
    ℬ𝓊𝓇ℊ,
    𝒞ℛ,
    ℋ𝒟,
    χ²,
    # Deprecated
    evaluate
end
