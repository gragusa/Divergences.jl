module Divergences

import Distance: gradient, PreMetric

import Base.gradient

export
    Divergence,
    KullbackLeibler,
    ReverseKullbackLeibler,
    CressieRead,
    evaluate,
    gradient!,
    hessian!,
    gradient,
    hessian

include("cressieread.jl")


end # module
