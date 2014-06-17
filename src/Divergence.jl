module Divergence

using Distance

import Distance: get_common_len

export
    ##
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
