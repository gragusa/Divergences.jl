################################################################################
## Chi squared
##
## ==> \gamma(a/b)b
## ==> \gamma(u)
################################################################################

################################################################################
## evaluate
################################################################################
function evaluate{T<:FloatingPoint}(dist::ChiSquared, a::T, b::T)
    u = a/b
    return (u^2 - 1)/2.0 - u
end

function evaluate{T<:FloatingPoint}(dist::CressieRead, a::AbstractVector{T})
    return (a^2 - 1)/2.0 - a
end

function evaluate{T<:FloatingPoint}(dist::CressieRead, a::AbstractVector{T}, b::AbstractVector{T})
    n = get_common_len(a, b)::Int
    r = zero(T)
    @inbounds for i = 1 : n
        ui = ai/bi
        r += evaluate(dist, ui, 1.0)
    end
    return r
end

################################################################################
## gradient
################################################################################
function gradient{T<:FloatingPoint}(dist::CressieRead, a::T, b::T)
    ## b \left(\frac{\left(\frac{a}{b}\right)^{\alpha }}{\alpha  b}-\frac{1}{\alpha  b}\right)
    return a/b-one(T)
end

function gradient{T<:FloatingPoint}(dist::CressieRead, a::T)
    return a-one(T)
end

function gradient!{T<:FloatingPoint}(u::Vector{T}, dist::CressieRead, a::AbstractVector{T}, b::AbstractVector{T})
    n = get_common_len(a, b)::Int
    @inbounds for i = 1:n
        ai = a[i]
        bi = bi[i]
        u[i] = gardient(dist, ai, bi)
    end
end

function gradient!{T<:FloatingPoint}(u::Vector{T}, dist::CressieRead, a::AbstractVector{T})
    n = length(a)::Int
    @inbounds for i = 1:n
        u[i] = gradient(dist, ai, 1.0)
    end
end

################################################################################
## Hessian
################################################################################
function hessian{T<:FloatingPoint}(dist::CressieRead, a::T, b::T)
    return 1.0
end

function hessian{T<:FloatingPoint}(dist::CressieRead, a::T)
    return 1.0
end

function hessian!{T<:FloatingPoint}(u::Vector{T}, dist::CressieRead, a::AbstractVector{T}, b::AbstractVector{T})
    n = get_common_len(a, b)::Int
    @inbounds for i = 1:n
        u[i] = 1.0
    end
end

function hessian!{T<:FloatingPoint}(u::Vector{T}, dist::CressieRead, a::AbstractVector{T})
    n = get_common_len(a, b)::Int
    @inbounds for i = 1:n
        u[i] = 1.0
    end
end

##
function gradient{T<:FloatingPoint}(dist::Divergence, a::AbstractVector{T}, b::AbstractVector{T})
    n = get_common_len(a, b)::Int
    return gradient!(Array(T, n), dist, a, b)
end

function gradient{T<:FloatingPoint}(dist::Divergence, a::T, b::T)
    return gradient!(Array(T, 1), dist, a, b)
end

function gradient{T<:FloatingPoint}(dist::Divergence, a::T)
    return gradient!(Array(T, 1), dist,  a)
end

function hessian{T<:FloatingPoint}(dist::Divergence, a::AbstractVector{T}, b::AbstractVector{T})
    n = get_common_len(a, b)::Int
    return hessian!(Array(T, n), dist, a, b)
end

function hessian{T<:FloatingPoint}(dist::Divergence, a::T, b::T)
    return hessian!(Array(T, 1), dist, a, b)
end

function hessian{T<:FloatingPoint}(dist::Divergence, a::T)
    return hessian!(Array(T, 1), dist, a)
end
