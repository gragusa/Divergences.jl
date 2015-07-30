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
function evaluate{T <: FloatingPoint}(dist::CressieRead, a::T, b::T)
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

function evaluate{T <: FloatingPoint}(dist::CressieRead, a::AbstractVector{T})
    α = dist.α
    r = zero(T)
    n = length(a)::Int64
    @inbounds for i = 1:n
        u = a[i]
        r += evaluate(dist, u, one(T))
    end 
    return r
end

function evaluate{T <: FloatingPoint}(dist::CressieRead, a::AbstractVector{T}, b::AbstractVector{T})
    r = zero(T)
    n = get_common_len(a, b)::Int
    @inbounds for i = 1 : n
        ai = a[i]
        bi = b[i]
        r += evaluate(dist, ai, bi)
    end
    return r
end

################################################################################
## gradient
################################################################################
function gradient{T <: FloatingPoint}(dist::CressieRead, a::T, b::T)
    ## b \left(\frac{\left(\frac{a}{b}\right)^{\alpha }}{\alpha  b}-\frac{1}{\alpha  b}\right)
    α = dist.α
    if a >= 0 && b > 0
        u = (a/b)^α/α-1/α
    elseif a == 0 && b == 0
        u = zero(T)        
    else 
        u = oftype(a, Inf)
    end 
    return u
end

function gradient{T <: FloatingPoint}(dist::CressieRead, a::T)
    return gradient(dist, a, one(T))
end

function gradient!{T <: FloatingPoint}(u::Vector{T}, dist::CressieRead, a::AbstractVector{T}, b::AbstractVector{T})
    n = get_common_len(a, b)::Int
    @inbounds for i = 1:n
        u[i] = gradient(dist, a[i], b[i])
    end 
    return u
end 

function gradient!{T <: FloatingPoint}(u::Vector{T}, dist::CressieRead, a::AbstractVector{T})    
    n = length(a)::Int
    @inbounds for i = 1:n
        u[i] = gradient(dist, a[i])
    end
    return u
end

################################################################################
## Hessian
################################################################################
function hessian{T <: FloatingPoint}(dist::CressieRead, a::T, b::T)
    α    = dist.α
    if a > 0 && b > 0
        u = (a/b)^(α-1)
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

function hessian{T <: FloatingPoint}(dist::CressieRead, a::T)
    return hessian(dist, a, one(T))
end

function hessian!{T <: FloatingPoint}(u::Vector{T}, dist::CressieRead, a::AbstractVector{T}, b::AbstractVector{T})    
    n = get_common_len(a, b)::Int
    @inbounds for i = 1:n
        u[i] = hessian(dist, a[i], b[i])
    end
end 

function hessian!{T <: FloatingPoint}(u::Vector{T}, dist::CressieRead, a::AbstractVector{T})
    n = length(a)::Int
    @inbounds for i = 1:n       
        u[i] = hessian(dist, a[i])
    end
end 

##
function gradient{T <: FloatingPoint}(dist::Divergence, a::AbstractVector{T}, b::AbstractVector{T})
    n = get_common_len(a, b)::Int
    return gradient!(Array(T, n), dist, a, b)
end

function gradient{T <: FloatingPoint}(dist::Divergence, a::T, b::T)
    return gradient!(Array(T, 1), dist, a, b)
end

function gradient{T <: FloatingPoint}(dist::Divergence, a::T)
    return gradient!(Array(T, 1), dist,  a)
end

function hessian{T <: FloatingPoint}(dist::Divergence, a::AbstractVector{T}, b::AbstractVector{T})
    n = get_common_len(a, b)::Int
    return hessian!(Array(T, n), dist, a, b)
end

function hessian{T <: FloatingPoint}(dist::Divergence, a::T, b::T)
    return hessian!(Array(T, 1), dist, a, b)
end

function hessian{T <: FloatingPoint}(dist::Divergence, a::T)
    return hessian!(Array(T, 1), dist, a)
end
