cue() = CressieRead(1)
hd()  = CressieRead(-1/2)

################################################################################
## Cressie Read Divergence
################################################################################

#=---------------
Evaluate
---------------=#
function evaluate(div::CressieRead, a::T, b::T) where T <: AbstractFloat
    α = div.α
    u = a/b
    if u > 0
        u = ( (u^(1+α)-1)/(α*(α+1)) - u/α + 1/α )*b
    elseif u == 0
        u = b/(1+α)
    else
        u = oftype(a, Inf)
    end
    u
end

function evaluate(div::CressieRead, a::AbstractVector{T}) where T <: AbstractFloat
    α = div.α
    r = zero(T)
    n = length(a)::Int64
    @inbounds for i = 1:n
        u = a[i]
        r += evaluate(div, u, one(T))
    end
    return r
end

function evaluate(div::CressieRead, a::AbstractVector{T}, b::AbstractVector{T}) where T <: AbstractFloat
    if length(a) != length(b)
        throw(DimensionMismatch("first array has length $(length(a)) which does not match the length of the second, $(length(b))."))
    end
    r = zero(T)
    @inbounds for i = eachindex(a, b)
        ai = a[i]
        bi = b[i]
        r += evaluate(div, ai, bi)
    end
    return r
end

#=---------------
gradient
---------------=#
function gradient(div::CressieRead, a::T, b::T) where T <: AbstractFloat
    α = div.α
    if a >= 0 && b > 0
        u = ((a/b)^α-1.0)/α
    elseif a == 0 && b == 0
        u = zero(T)
    else
        u = oftype(a, Inf)
    end
    return u
end

function gradient(div::CressieRead, a::T) where T <: AbstractFloat
    return gradient(div, a, one(T))
end

#=---------------
hessian
---------------=#
function hessian(div::CressieRead, a::T, b::T) where T <: AbstractFloat
    α    = div.α
    if a > 0 && b > 0
        u = (a/b)^α/a
    elseif a == 0 && b > 0
        if α >= 0
            u = zero(T)
        else
            u = oftype(a, Inf)
        end
    elseif a == 0 && b == 0
        u = zero(T)
    elseif a > 0 && b == 0
        u = oftype(a, Inf)
    end
    return u
end

function hessian(div::CressieRead, a::T) where T <: AbstractFloat
    return hessian(div, a, one(T))
end
