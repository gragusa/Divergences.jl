module Divergences

using StatsFuns
using LoopVectorization
using Parameters
import LoopVectorization: vifelse
import VectorizationBase: andmask
import Distances: PreMetric 

abstract type Divergence <: PreMetric end

struct CressieRead{T} <: Divergence
    α::T
    function CressieRead(α::T) where T<:Real
        #@assert isempty(findall((in)([-1, 0]), α)) "CressieRead is defined for all α != {-1,0}"
        new{T}(α)
    end
end

struct ChiSquared  <: Divergence end
struct KullbackLeibler  <: Divergence end
struct ReverseKullbackLeibler <: Divergence end
struct Hellinger <: Divergence end

struct ModifiedDivergence{D, T} <: Divergence
    d::D
    m::NamedTuple{(:γ₀, :γ₁, :γ₂, :ρ), Tuple{T, T, T, T}}
end

struct FullyModifiedDivergence{D, T} <: Divergence
    d::D
    m::NamedTuple{(:γ₀, :γ₁, :γ₂, :ρ, :g₀, :g₁, :g₂, :φ), Tuple{T, T, T, T, T, T, T, T}}
end

const ModDiv = Union{ModifiedDivergence, FullyModifiedDivergence}

function ModifiedDivergence(D::Divergence, ρ::Real)
    @assert ρ > 1 "A ModifiedDivergence requires ρ > 1"
    γ₀ = eval(D, [ρ])[1]
    γ₁ = gradient(D, [ρ])[1]
    γ₂ = hessian(D, [ρ])[1]
    ModifiedDivergence(D, (γ₀=γ₀, γ₁=γ₁, γ₂=γ₂, ρ=ρ))
end

function FullyModifiedDivergence(D::Divergence, φ::Real, ρ::Real)
    @assert ρ > 1 "A ModifiedDivergence requires ρ > 1"
    @assert φ < 1 && φ > 0 "A ModifiedDivergence requires  φ ∈ (0,1)"
    γ₀ = eval(D, [ρ])[1]
    γ₁ = gradient(D, [ρ])[1]
    γ₂ = hessian(D, [ρ])[1]
    g₀ = eval(D, [φ])[1]
    g₁ = gradient(D, [φ])[1]
    g₂ = hessian(D, [φ])[1]
    FullyModifiedDivergence(D, (γ₀=γ₀, γ₁=γ₁, γ₂=γ₂, ρ=ρ, g₀=g₀, g₁=g₁, g₂=g₂, φ=φ))
end

const 𝒦ℒ=KullbackLeibler
const ℬ𝓊𝓇ℊ=ReverseKullbackLeibler
const 𝒞ℛ=CressieRead
const ℋ𝒟=Hellinger
const χ²=ChiSquared
include("divergences.jl")

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
