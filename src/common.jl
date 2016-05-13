function gradient!{T <: AbstractFloat}(u::AbstractVector{T}, dist::Divergence, a::AbstractVector{T}, b::AbstractVector{T})
    @inbounds for i = eachindex(a, b)
        u[i] = gradient(dist, a[i], b[i])
    end
    u
end

function gradient!{T <: AbstractFloat}(u::AbstractVector{T}, dist::Divergence, a::AbstractVector{T})
    @inbounds for i = eachindex(a)
        u[i] = gradient(dist, a[i])
    end
    u
end

function gradient{T <: AbstractFloat}(dist::Divergence, a::AbstractVector{T}, b::AbstractVector{T})
    if length(a) != length(b)
        throw(DimensionMismatch("first array has length $(length(a)) which does not match the length of the second, $(length(b))."))
    end
    gradient!(Array(T, length(a)), dist, a, b)
end

function gradient{T <: AbstractFloat}(dist::Divergence, a::AbstractVector{T})
    gradient!(Array(T, length(a)), dist, a)
end


function hessian!{T <: AbstractFloat}(u::Vector{T}, dist::Divergence, a::AbstractVector{T}, b::AbstractVector{T})
    @inbounds for i = eachindex(a, b)
        u[i] = hessian(dist, a[i], b[i])
    end
    u
end

function hessian!{T <: AbstractFloat}(u::Vector{T}, dist::Divergence, a::AbstractVector{T})
    @inbounds for i = eachindex(a)
        u[i] = hessian(dist, a[i])
    end
    u
end


function hessian{T <: AbstractFloat}(dist::Divergence, a::AbstractVector{T}, b::AbstractVector{T})
    if length(a) != length(b)
        throw(DimensionMismatch("first array has length $(length(a)) which does not match the length of the second, $(length(b))."))
    end
    hessian!(Array(T, length(a)), dist, a, b)
end

function hessian{T <: AbstractFloat}(dist::Divergence, a::AbstractVector{T})
    hessian!(Array(T, length(a)), dist, a)
end
