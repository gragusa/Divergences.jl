VERSION >= v"0.4.0-dev+6521" && __precompile__(true)
module Divergences

using StatsFuns
import Distances: evaluate, gradient, PreMetric
import Calculus: hessian

abstract type Divergence <: PreMetric end

immutable CressieRead <: Divergence
    α::Float64
    function CressieRead(α::Float64)
        @assert isempty(findin(α, [-1, 0])) "CressieRead is defined for all α != {-1,0}"
        new(α)
    end
end

CressieRead(α::Int64) = CressieRead(float(α))

immutable ChiSquared  <: Divergence end
immutable KullbackLeibler  <: Divergence end

immutable ReverseKullbackLeibler <: Divergence end

immutable ModifiedKullbackLeibler <: Divergence
    ϑ::Float64
    d::Divergence
    m::NTuple{4, Float64}
    function ModifiedKullbackLeibler(ϑ::Real)
        @assert ϑ > 0 "ModifiedKullbackLeibler is defined for ϑ > 0"
        uϑ = 1.0 + ϑ
        d  = KullbackLeibler()
        f0 = evaluate(d, [uϑ])
        f1 = gradient(d, uϑ)
        f2 = hessian(d, uϑ)
        new(float(ϑ), d, (f0, f1, f2, uϑ))
    end
end

immutable FullyModifiedKullbackLeibler <: Divergence
    φ::Float64
    ϑ::Float64
    d::Divergence
    m::NTuple{8, Float64}
    function FullyModifiedKullbackLeibler(φ::Real, ϑ::Real)
        @assert ϑ > 0 "FullyModifiedKullbackLeibler is defined for ϑ > 0"
        @assert φ > 0 && φ < 1.0 "FullyModifiedKullbackLeibler is defined for ϕ ∈ (0,1)"
        uϑ = 1.0 + ϑ
        d  = KullbackLeibler()
        f0 = evaluate(d, [uϑ])
        f1 = gradient(d, uϑ)
        f2 = hessian(d, uϑ)
        uφ  = float(φ)
        g0  = evaluate(d, [uφ])
        g1  = gradient(d, uφ)
        g2  = hessian(d, uφ)
        new(float(φ), float(ϑ), d, (f0, f1, f2, uϑ, g0, g1, g2, uφ))
    end
end


immutable ModifiedReverseKullbackLeibler <: Divergence
    ϑ::Float64
    d::Divergence
    m::NTuple{4, Float64}
    function ModifiedReverseKullbackLeibler(ϑ::Real)
        @assert ϑ > 0 "ModifiedReverseKullbackLeibler is defined for ϑ > 0"
        uϑ = 1.0 + ϑ
        d  = ReverseKullbackLeibler()
        f0 = evaluate(d, [uϑ])
        f1 = gradient(d, uϑ)
        f2 = hessian(d, uϑ)
        new(float(ϑ), d, (f0, f1, f2, uϑ))
    end
end

immutable FullyModifiedReverseKullbackLeibler <: Divergence
    φ::Float64
    ϑ::Float64
    d::Divergence
    m::NTuple{8, Float64}
    function FullyModifiedReverseKullbackLeibler(φ::Real, ϑ::Real)
        @assert ϑ > 0 "ModifiedReverseKullbackLeibler is defined for ϑ > 0"
        @assert φ > 0 && φ < 1.0 "ModifiedReverseKullbackLeibler is defined for φ ∈ (0,1)"
        d   = ReverseKullbackLeibler()
        uϑ  = 1.0 + ϑ
        f0  = evaluate(d, [uϑ])
        f1  = gradient(d, uϑ)
        f2  = hessian(d, uϑ)
        uφ  = float(φ)
        g0  = evaluate(d, [uφ])
        g1  = gradient(d, uφ)
        g2  = hessian(d, uφ)
        new(float(φ), float(ϑ), d, (f0, f1, f2, uϑ, g0, g1, g2, uφ))
    end
end

immutable ModifiedCressieRead <: Divergence
    α::Float64
    ϑ::Float64
    d::Divergence
    m::NTuple{4, Float64}
    function ModifiedCressieRead(α::Real, ϑ::Real)
        @assert isempty(findin(α, [-1, 0])) "ModifiedCressieRead is defined for all α! = {-1,0}."
        @assert ϑ > 0 "ModifiedCressieRead is defined for ϑ > 0"
        uϑ = 1.0 + ϑ
        d  = CressieRead(α)
        f0 = evaluate(d, [uϑ])
        f1 = gradient(d, uϑ)
        f2 = hessian(d, uϑ)
        new(float(α), float(ϑ), d, (f0, f1, f2, uϑ))
    end
end

immutable FullyModifiedCressieRead <: Divergence
    α::Float64
    φ::Float64
    ϑ::Float64
    d::Divergence
    m::NTuple{8, Float64}
    function FullyModifiedCressieRead(α::Real, φ::Real, ϑ::Real)
        @assert isempty(findin(α, [-1, 0])) "ModifiedCressieRead is defined for all α != {-1,0}"
        @assert ϑ > 0 "FullyModifiedCressieRead is defined for ϑ > 0"
        @assert φ > 0 && φ < 1.0 "FullyModifiedCressieRead is defined for φ ∈ (0, 1)"
        uϑ = 1.0 + ϑ
        d  = CressieRead(α)
        f0 = evaluate(d, [uϑ])
        f1 = gradient(d, uϑ)
        f2 = hessian(d, uϑ)
        uφ = float(φ)
        g0 = evaluate(d, [uφ])
        g1 = gradient(d, uφ)
        g2 = hessian(d, uφ)
        new(float(α), float(φ), float(ϑ),  d, (f0, f1, f2, uϑ, g0, g1, g2, uφ))
    end
end

const KL=KullbackLeibler
const MKL=ModifiedKullbackLeibler
const FMKL=FullyModifiedKullbackLeibler

const RKL=ReverseKullbackLeibler
const MRKL=ModifiedReverseKullbackLeibler
const FMRKL=FullyModifiedReverseKullbackLeibler

const CR=CressieRead
const MCR=ModifiedCressieRead
const FMCR=FullyModifiedCressieRead

HD()  = CressieRead(-1/2)


include("common.jl")
include("cressieread.jl")
include("modified_cressieread.jl")
include("kl.jl")
include("reversekl.jl")
include("chisq.jl")

export
    Divergence,
    # KL
    KullbackLeibler,
    ModifiedKullbackLeibler,
    FullyModifiedKullbackLeibler,
    # RKL
    ReverseKullbackLeibler,
    ModifiedReverseKullbackLeibler,
    FullyModifiedReverseKullbackLeibler,
    # CR
    CressieRead,
    ModifiedCressieRead,
    FullyModifiedCressieRead,
    # Abbr.
    KL,
    MKL,
    FMKL,
    RKL,
    MRKL,
    FMRKL,
    CR,
    MCR,
    FMCR,
    HD,
    ChiSquared,
    evaluate,
    gradient!,
    hessian!,
    gradient,
    hessian

end # module
