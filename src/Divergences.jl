module Divergences

using StatsFuns
using LoopVectorization
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
    m::NTuple{4, T}
end

struct FullyModifiedDivergence{D, T} <: Divergence
    d::D
    m::NTuple{8, T}
end

const ModDiv = Union{ModifiedDivergence, FullyModifiedDivergence}

function ModifiedDivergence(D::Divergence, ρ::Real)
    @assert ρ > 0 && ρ < 1 "A ModifiedDivergence requires ρ ∈ (0, 1)"
    γ₀ = eval(D, [ρ])[1]
    γ₁ = gradient(D, [ρ])[1]
    γ₂ = hessian(D, [ρ])[1]
    ModifiedDivergence(D, (γ₀, γ₁, γ₂, ρ))
end

function FullyModifiedDivergence(D::Divergence, ρ::Real, φ::Real)
    @assert ρ > 0 && ρ < 1 "A ModifiedDivergence requires ρ ∈ (0,1)"
    @assert φ > 1 "A ModifiedDivergence requires  φ > 1"
    γ₀ = eval(D, [ρ])[1]
    γ₁ = gradient(D, [ρ])[1]
    γ₂ = hessian(D, [ρ])[1]
    g₀ = eval(D, [φ])[1]
    g₁ = gradient(D, [φ])[1]
    g₂ = hessian(D, [φ])[1]
    FullyModifiedDivergence(D, (γ₀, γ₁, γ₂, ρ, g₀, g₁, g₂, φ))
end

const KL=KullbackLeibler
const RKL=ReverseKullbackLeibler
const CR=CressieRead
const HD=Hellinger


include("divergences.jl")

#include("common.jl")
#include("cressieread.jl")
#include("modified_cressieread.jl")
#include("kl.jl")
#include("reversekl.jl")
#include("chisq.jl")

export
    Divergence,
    # KL
    KullbackLeibler,
    # RKL
    ReverseKullbackLeibler,
    # HD
    Hellinger,
    # CR
    CressieRead,
    # Modified
    ModifiedDivergence,
    # FullyModified
    FullyModifiedDivergence,
    # Abbr.
    KL,
    RKL,
    CR,
    HD,
    ChiSquared#,
end
