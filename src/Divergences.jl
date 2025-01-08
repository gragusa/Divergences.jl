module Divergences

using NaNMath
using Distances
abstract type AbstractDivergence <: PreMetric end
abstract type Divergence <: AbstractDivergence end
abstract type AbstractModifiedDivergence <: AbstractDivergence end

struct CressieRead{T} <: Divergence
    α::T
    function CressieRead(α::T) where T<:Real
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

function ModifiedDivergence(D::Divergence, ρ::Real)
    @assert ρ > 1 "A ModifiedDivergence requires ρ > 1"
    γ₀ = D(ρ)
    γ₁ = gradient(D, [ρ])[1]
    γ₂ = hessian(D, [ρ])[1]
    ModifiedDivergence(D, (γ₀=γ₀, γ₁=γ₁, γ₂=γ₂, ρ=ρ))
end

function FullyModifiedDivergence(D::Divergence, φ::Real, ρ::Real)
    @assert ρ > 1 "A ModifiedDivergence requires ρ > 1"
    @assert φ < 1 && φ > 0 "A ModifiedDivergence requires  φ ∈ (0,1)"
    γ₀ = D(ρ)
    γ₁ = gradient(D, [ρ])[1]
    γ₂ = hessian(D, [ρ])[1]
    g₀ = D(φ)
    g₁ = gradient(D, [φ])[1]
    g₂ = hessian(D, [φ])[1]
    FullyModifiedDivergence(D, (γ₀=γ₀, γ₁=γ₁, γ₂=γ₂, ρ=ρ, g₀=g₀, g₁=g₁, g₂=g₂, φ=φ))
end

for div ∈ (KullbackLeibler, ReverseKullbackLeibler, Hellinger, CressieRead, ChiSquared, ModifiedDivergence, FullyModifiedDivergence)
    @eval begin
        function (f::$div)(p::Real, q::Real)
            return γ(f, p, q)
        end
    end
end

for div ∈ (KullbackLeibler, ReverseKullbackLeibler, Hellinger, CressieRead, ChiSquared, ModifiedDivergence, FullyModifiedDivergence)
    @eval begin
        function (f::$div)(p::Real)
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

function Distances.evaluate(f::AbstractDivergence, a::AbstractArray)
    return sum(f.(a))
end

function Distances.evaluate(f::AbstractDivergence, a::AbstractArray, b::AbstractArray)
    return sum(f.(a./b).*b)
end

include("divs.jl")




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
    χ²
end
