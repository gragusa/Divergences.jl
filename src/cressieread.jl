cue() = CressieRead(1)
hd()  = CressieRead(-1/2)

################################################################################
## Cressie Read Divergence
##
## ==> \gamma(a/b)b
## ==> \gamma(u)
################################################################################

################################################################################
## evaluate
################################################################################
function evaluate{T <: AbstractFloat}(dist::CressieRead, a::T, b::T)
    α = dist.α
    u = a/b
    if u > 0
        u = (u^(1+α)-1)/(α*(α+1)) - (u-1)/α
    elseif u==0
        u = 1/(1+α)
    else
        u = oftype(a, Inf)
    end
    u
end

function evaluate{T <: AbstractFloat}(dist::CressieRead, a::AbstractVector{T})
    α = dist.α
    r = zero(T)
    n = length(a)::Int64
    @inbounds for i = 1:n
        u = a[i]
        r += evaluate(dist, u, one(T))
    end
    return r
end

function evaluate{T <: AbstractFloat}(dist::CressieRead, a::AbstractVector{T}, b::AbstractVector{T})
    if length(a) != length(b)
        throw(DimensionMismatch("first array has length $(length(a)) which does not match the length of the second, $(length(b))."))
    end
    r = zero(T)
    @inbounds for i = eachindex(a, b)
        ai = a[i]
        bi = b[i]
        r += evaluate(dist, ai, bi)
    end
    return r
end

################################################################################
## gradient
################################################################################
function gradient{T <: AbstractFloat}(dist::CressieRead, a::T, b::T)
    α = dist.α
    if a >= 0 && b > 0
        u = ((a/b)^α)/(b*α)-1/(b*α)
    elseif a == 0 && b == 0
        u = zero(T)
    else
        u = oftype(a, Inf)
    end
    return u
end

function gradient{T <: AbstractFloat}(dist::CressieRead, a::T)
    return gradient(dist, a, one(T))
end

################################################################################
## Hessian
################################################################################
function hessian{T <: AbstractFloat}(dist::CressieRead, a::T, b::T)
    α    = dist.α
    if a > 0 && b > 0
        u = (a/b)^(α-1)/b^2
    elseif a == 0 && b > 0
        if α-1>0
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

function hessian{T <: AbstractFloat}(dist::CressieRead, a::T)
    return hessian(dist, a, one(T))
end
