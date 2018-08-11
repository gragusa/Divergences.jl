function gradient!(u::AbstractVector{T}, dist::Divergence, a::AbstractVector{T}, b::AbstractVector{T}) where T <: AbstractFloat
    @inbounds for i = eachindex(a, b)
        u[i] = gradient(dist, a[i], b[i])
    end
    u
end

function gradient!(u::AbstractVector{T}, dist::Divergence, a::AbstractVector{T}) where T <: AbstractFloat
    @inbounds for i = eachindex(a)
        u[i] = gradient(dist, a[i])
    end
    u
end

function gradient(dist::Divergence, a::AbstractVector{T}, b::AbstractVector{T}) where T <: AbstractFloat
    if length(a) != length(b)
        throw(DimensionMismatch("First array has length $(length(a)) which does not match the length of the second, $(length(b))."))
    end
    gradient!(Array{T}(undef, length(a)), dist, a, b)
end

function gradient(dist::Divergence, a::AbstractVector{T}) where T <: AbstractFloat
    gradient!(Array{T}(undef, length(a)), dist, a)
end


function hessian!(u::Vector{T}, dist::Divergence, a::AbstractVector{T}, b::AbstractVector{T}) where T <: AbstractFloat
    @inbounds for i = eachindex(a, b)
        u[i] = hessian(dist, a[i], b[i])
    end
    u
end

function hessian!(u::Vector{T}, dist::Divergence, a::AbstractVector{T}) where T <: AbstractFloat
    @inbounds for i = eachindex(a)
        u[i] = hessian(dist, a[i])
    end
    u
end


function hessian(dist::Divergence, a::AbstractVector{T}, b::AbstractVector{T}) where T <: AbstractFloat
    if length(a) != length(b)
        throw(DimensionMismatch("first array has length $(length(a)) which does not match the length of the second, $(length(b))."))
    end
    hessian!(Array{T}(undef, length(a)), dist, a, b)
end

function hessian(dist::Divergence, a::AbstractVector{T}) where T <: AbstractFloat
    hessian!(Array{T}(undef, length(a)), dist, a)
end
