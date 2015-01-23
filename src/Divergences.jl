module Divergences

import Distances: gradient, PreMetric, get_common_len
import Base: gradient, evaluate

export
    Divergence,
    KullbackLeibler,
    ModifiedKullbackLeibler,
    ReverseKullbackLeibler,
    ModifiedReverseKullbackLeibler,
    CressieRead,
    ModifiedCressieRead,
    ChiSquared,
    evaluate,
    gradient!,
    hessian!,
    gradient,
    hessian

abstract Divergence <: PreMetric

immutable CressieRead <: Divergence
    α::Float64
    function CressieRead(α::Float64)
        @assert isempty(findin(α, [-1, 0])) "CressieRead is defined for all α!={-1,0}."
        new(α)
    end
end

CressieRead(α::Int64) = CressieRead(float(α))


immutable KullbackLeibler  <: Divergence end
immutable ReverseKullbackLeibler <: Divergence end

immutable ModifiedKullbackLeibler <: Divergence
	ϑ::Float64
end

immutable ModifiedReverseKullbackLeibler <: Divergence
	ϑ::Float64
end

immutable ModifiedCressieRead <: Divergence
    α::Float64
    ϑ::Float64
    function ModifiedCressieRead(α::Float64, ϑ::Float64)
        @assert isempty(findin(α, [-1, 0])) "ModifiedCressieRead is defined for all α!={-1,0}."
        @assert ϑ > 0 "ModifiedCressieRead is defined for ϑ>0."
        new(α, ϑ)
    end
end

ModifiedCressieRead(α::Real, ϑ::Real) = ModifiedCressieRead(float(α), float(ϑ))

function ModifiedReverseKullbackLeibler(ϑ::Real)
    @assert ϑ > 0 "ModifiedReverseKullbackLeibler is defined for ϑ>0."
    ModifiedReverseKullbackLeibler(float(ϑ))
end

function ModifiedReverseKullbackLeibler(ϑ::Real)
    @assert ϑ > 0 "ModifiedKullbackLeibler is defined for ϑ>0."
    ModifiedReverseKullbackLeibler(float(ϑ))
end

typealias CR CressieRead
typealias ET KullbackLeibler
typealias EL ReverseKullbackLeibler
typealias MET ModifiedKullbackLeibler
typealias MEL ModifiedReverseKullbackLeibler

include("cressieread.jl")
include("modified_cressieread.jl")
include("kl.jl")
include("reversekl.jl")
include("chisq.jl")
end # module
