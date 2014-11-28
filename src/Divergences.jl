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
    α::Real

    function CressieRead(α::Real)
        @assert isempty(findin(α, [-1, 0])) "CressieRead is defined for all α!={-1,0}."
        new(α)
    end
end

type KullbackLeibler  <: Divergence end
type ReverseKullbackLeibler <: Divergence end

type ModifiedKullbackLeibler
	ϑ::Real
end

type ModifiedReverseKullbackLeibler
	ϑ::Real
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

end # module
