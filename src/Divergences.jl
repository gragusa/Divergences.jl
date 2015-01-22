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
    evaluate,
    gradient!,
    hessian!,
    gradient,
    hessian

abstract Divergence <: PreMetric

type CressieRead <: Divergence
    α::Float64
    function CressieRead(α::Float64)
        @assert isempty(findin(α, [-1, 0])) "CressieRead is defined for all α!={-1,0}."
        new(α)
    end
end

CressieRead(α::Int64) = CressieRead(float(α))


type KullbackLeibler  <: Divergence end
type ReverseKullbackLeibler <: Divergence end

type ModifiedKullbackLeibler <: Divergence
	ϑ::Float64
end

type ModifiedReverseKullbackLeibler <: Divergence
	ϑ::Float64
end

type ModifiedCressieRead <: Divergence
    α::Float64
    ϑ::Float64
    function ModifiedCressieRead(α::Float64, ϑ::Float64)
        @assert isempty(findin(α, [-1, 0])) "ModifiedCressieRead is defined for all α!={-1,0}."
        @assert ϑ>1 "ModifiedCressieRead is defined for ϑ>1."
        new(α, ϑ)
    end
end

ModifiedCressieRead(α::Real, ϑ::Real) = ModifiedCressieRead(float(α), float(ϑ))
ModifiedReverseKullbackLeibler(ϑ::Int64) = ModifiedReverseKullbackLeibler(float(ϑ))
ModifiedKullbackLeibler(ϑ::Int64) = ModifiedKullbackLeibler(float(ϑ))




typealias CR CressieRead
typealias ET KullbackLeibler
typealias EL ReverseKullbackLeibler
typealias MET ModifiedKullbackLeibler
typealias MEL ModifiedReverseKullbackLeibler

include("cressieread.jl")
include("modified_cressieread.jl")
include("kl.jl")
include("reversekl.jl")

end # module
