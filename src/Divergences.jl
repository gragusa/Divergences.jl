module Divergences

import Distances: gradient, PreMetric, get_common_len
import Base: gradient, evaluate


export
    Divergence,
    KullbackLeibler,
    ReverseKullbackLeibler,
    CressieRead,
    ModifiedCressieRead,
    evaluate,
    gradient!,
    hessian!,
    gradient,
    hessian

include("cressieread.jl")
include("modified_cressieread.jl")

end # module
