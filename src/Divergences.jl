module Divergences

<<<<<<< HEAD
using Distance
=======
>>>>>>> FETCH_HEAD
importall Distance

import Distance: get_common_len
#import Base: gradient

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
