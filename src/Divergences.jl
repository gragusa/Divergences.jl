module Divergences

importall Distance

import Distance: get_common_len

export
    ##
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
