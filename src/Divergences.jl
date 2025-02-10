module Divergences

using NaNMath

abstract type AbstractDivergence end
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
    γ₀ = evaluate(D, [ρ])[1]
    γ₁ = gradient(D, [ρ])[1]
    γ₂ = hessian(D, [ρ])[1]
    ModifiedDivergence(D, (γ₀=γ₀, γ₁=γ₁, γ₂=γ₂, ρ=ρ))
end

function FullyModifiedDivergence(D::Divergence, φ::Real, ρ::Real)
    @assert ρ > 1 "A ModifiedDivergence requires ρ > 1"
    @assert φ < 1 && φ > 0 "A ModifiedDivergence requires  φ ∈ (0,1)"
    γ₀ = evaluate(D, [ρ])[1]
    γ₁ = gradient(D, [ρ])[1]
    γ₂ = hessian(D, [ρ])[1]
    g₀ = evaluate(D, [φ])[1]
    g₁ = gradient(D, [φ])[1]
    g₂ = hessian(D, [φ])[1]
    FullyModifiedDivergence(D, (γ₀=γ₀, γ₁=γ₁, γ₂=γ₂, ρ=ρ, g₀=g₀, g₁=g₁, g₂=g₂, φ=φ))
end

const 𝒦ℒ=KullbackLeibler
const ℬ𝓊𝓇ℊ=ReverseKullbackLeibler
const 𝒞ℛ=CressieRead
const ℋ𝒟=Hellinger
const χ²=ChiSquared
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
