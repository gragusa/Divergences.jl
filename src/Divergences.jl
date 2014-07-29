module Divergences

importall Distance


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
