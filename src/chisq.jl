################################################################################
## Chi squared
################################################################################

#=---------------
Evaluate
---------------=#
function evaluate{T<:AbstractFloat}(dist::ChiSquared, a::T, b::T)
    u = a/b
    return u^2/2.0 - u + 0.5
end

function evaluate{T<:AbstractFloat}(dist::ChiSquared, a::AbstractVector{T})
    r = zero(T)
    @simd for i in eachindex(a)
        @inbounds r += a[i]^2/2.0 - a[i] + 0.5
    end
    return r
end

function evaluate{T<:AbstractFloat}(dist::ChiSquared, a::AbstractVector{T}, b::AbstractVector{T})
    if length(a) != length(b)
        throw(DimensionMismatch("first array has length $(length(a)) which does not match the length of the second, $(length(b))."))
    end
    r = zero(T)
    @simd for i in eachindex(a, b)
        @inbounds ui = a[i]/b[i]
        r += ui^2/2.0 - ui + 0.5
    end
    return r
end

#=---------------
gradient
---------------=#
function gradient{T<:AbstractFloat}(dist::ChiSquared, a::T, b::T)
    ## b \left(\frac{\left(\frac{a}{b}\right)^{\alpha }}{\alpha  b}-\frac{1}{\alpha  b}\right)
    return a/b-one(T)
end

function gradient{T<:AbstractFloat}(dist::ChiSquared, a::T)
    return a-one(T)
end

function gradient!{T<:AbstractFloat}(u::Vector{T}, dist::ChiSquared, a::AbstractVector{T}, b::AbstractVector{T})
    if length(a) != length(b)
        throw(DimensionMismatch("first array has length $(length(a)) which does not match the length of the second, $(length(b))."))
    end
    onet = one(T)
    @simd for i = eachindex(a, b)
        ai = a[i]
        bi = bi[i]
        @inbounds u[i] = ai/bi - onet
    end
end

function gradient!{T<:AbstractFloat}(u::Vector{T}, dist::ChiSquared, a::AbstractVector{T})
    ι = one(T)
    @simd for i = eachindex(a)
        @inbounds u[i] = a[i]-ι
    end
end

#=---------------
hessian
---------------=#
function hessian{T<:AbstractFloat}(dist::ChiSquared, a::T, b::T)
    return one(T)
end

function hessian{T<:AbstractFloat}(dist::ChiSquared, a::T)
    return one(T)
end

function hessian!{T<:AbstractFloat}(u::Vector{T}, dist::ChiSquared, a::AbstractVector{T}, b::AbstractVector{T})
    n = get_common_len(a, b)::Int
    ι = one(T)
    @simd for i = 1:n
         @inbounds u[i] = ι
    end
end

function hessian!{T<:AbstractFloat}(u::Vector{T}, dist::ChiSquared, a::AbstractVector{T})
    onet = one(T)
    @simd for i = eachindex(a)
        @inbounds u[i] = onet
    end
end
