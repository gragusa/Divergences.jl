function xlogx(x::Number)
    result = x * NaNMath.log(x)
    return iszero(x) ? zero(result) : result
end

function xlogy(x::Number, y::Number)
    result = x * NaNMath.log(y)
    return iszero(x) && !isnan(y) ? zero(result) : result
end

alogab(a, b) = xlogy(a, a/b) - a + b
blogab(a, b) = -xlogy(b, a./b) + a - b
aloga(a) = xlogx(a) - a + one(eltype(a))
loga(a) = -log(a) + a - one(eltype(a))

## -------------------------------------------------------
## Divergence functions
## -------------------------------------------------------
γ(::KullbackLeibler, a::T) where T<:Real = aloga(a)
γ(::ReverseKullbackLeibler, a::T) where T<:Real = loga(a)
γ(::Hellinger, a::T) where T<:Real =  2*a - 4*NaNMath.sqrt(a) + 2
γ(::ChiSquared, a::T) where T<:Real = abs2(a - one(eltype(T)))*half(T)

function γ(d::CressieRead{D}, a::T) where {T<:Real, D} 
    α = d.α
    if a >= 0
        (a^(1 + α) + α - a*(1 + α))/(α*(1 + α))
    else
        if α > 0
            zero(eltype(a))
        else
            convert(eltype(a), NaN)
        end
    end
end

function γᵤ(d::D, a::T) where {T<:Real, D<:AbstractModifiedDivergence}
    (; γ₀, γ₁, γ₂, ρ)  = d.m
    (γ₀ + γ₁*(a-ρ) + half(T)*γ₂*(a-ρ)^2)
end

function γₗ(d::D, a::T) where {T<:Real, D<:AbstractModifiedDivergence}
    (; g₀, g₁, g₂, φ)  = d.m
    (g₀ + g₁*(a-φ) + half(T)*g₂*(a-φ)^2)
end

function γ(d::ModifiedDivergence, a::T) where T<:Real
    (; ρ ) = d.m
    div = d.d
    a > ρ ? γᵤ(d, a) : γ(div, a)
end

function γ(d::FullyModifiedDivergence, a::T) where T<:Real
    (; ρ, φ) = d.m
    div = d.d
    a > ρ ? γᵤ(d, a) : a < φ ? γₗ(d, a) : γ(div, a)
end

function γ(d::AbstractDivergence, a::AbstractArray{T}) where T<:Real
    out = similar(a)
    for j in eachindex(a)
        out[j] = γ(d, a[j])
    end
    return out
end

## -------------------------------------------------------
## Gradient 
## -------------------------------------------------------
∇ᵧ(::KullbackLeibler, a::T) where T = NaNMath.log(a)
∇ᵧ(::ReverseKullbackLeibler, a::T) where T = a > 0 ? -1/a + one(T) : convert(T, -Inf)
∇ᵧ(d::CressieRead, a::T) where T = a >= 0 ? (a^d.α - one(T))/d.α : convert(T, sign(d.α)*Inf)
∇ᵧ(d::Hellinger, a::T) where T = a > 0 ? 2(one(T)-one(T)/sqrt(a)) : convert(T, -Inf)
∇ᵧ(d::ChiSquared, a::T) where T = a - one(T)

function ∇ᵤ(d::D, a::T) where {T, D<:AbstractModifiedDivergence}
    (; γ₀, γ₁, γ₂, ρ)  = d.m
    (γ₁ + γ₂*(a-ρ))
end

function ∇ₗ(d::D, a::T) where {T, D<:AbstractModifiedDivergence}
    (; g₀, g₁, g₂, φ)  = d.m
    (g₁ + g₂*(a-φ))
end

function ∇ᵧ(d::ModifiedDivergence, a::T) where T<:Real
    (; ρ ) = d.m
    div = d.d
    a > ρ ? ∇ᵤ(d, a) : ∇ᵧ(div, a)
end

function ∇ᵧ(d::FullyModifiedDivergence, a::T) where T<:Real
    (; ρ, φ ) = d.m
    div = d.d
    a > ρ ? ∇ᵤ(d, a) : a < φ ? ∇ₗ(d, a) : ∇ᵧ(div, a)
end

## -------------------------------------------------------
## Hessian
## -------------------------------------------------------
Hᵧ(::KullbackLeibler, a::T) where T = a > 0 ? one(T)/a : convert(T, Inf)
Hᵧ(::ReverseKullbackLeibler, a::T) where T = a > 0 ? one(T)/a^2 : convert(T, Inf)
Hᵧ(d::CressieRead, a::T) where T = a > 0 ? a^(d.α-1) : convert(T, Inf)
Hᵧ(d::Hellinger, a::T) where T = a > 0 ? one(T)/sqrt(a^(3)) : convert(T, Inf)
Hᵧ(d::ChiSquared, a::T) where T = one(T)

function Hᵤ(d::D, a::T) where {T, D<:AbstractModifiedDivergence}
    (; γ₀, γ₁, γ₂, ρ)  = d.m
    γ₂
end

function Hₗ(d::D, a::T) where {T, D<:AbstractModifiedDivergence}
    (; g₀, g₁, g₂, φ)  = d.m
    g₂
end

function Hᵧ(d::ModifiedDivergence, a::T) where T<:Real
    (; ρ) = d.m
    div = d.d
    a > ρ ? Hᵤ(d, a) : Hᵧ(div, a)
end

function Hᵧ(d::FullyModifiedDivergence, a::T) where {T<:Real}
    (; ρ, φ) = d.m
    div = d.d
    a > ρ ? Hᵤ(d, a) : a < φ ? Hₗ(d, a) : Hᵧ(div, a)
end

## -------------------------------------------------------
## Syntax sugar
## -------------------------------------------------------

gradient(d::AbstractDivergence, a::T) where T<:Real = ∇ᵧ(d, a)

function gradient!(u::AbstractVector{T}, d::AbstractDivergence, a::AbstractArray{R}) where {T <: Real, R<:Real}
    @inbounds for i ∈ eachindex(a, u)
        u[i] = ∇ᵧ(d, a[i])
    end
    return u
end

function gradient(d::AbstractDivergence, a::AbstractArray{R}) where {R<:Real}
    u = similar(a)
    gradient!(u, d, a)
end

function gradient_sum(d::AbstractDivergence, a::AbstractArray{R}) where {R<:Real}
    r = zero(R)
    @inbounds for i ∈ eachindex(a)
        r += ∇ᵧ(d, a[i])
    end
    return r
end

hessian(d::AbstractDivergence, a::T) where T<:Real = Hᵧ(d, a)

function hessian!(u::AbstractVector{T}, d::AbstractDivergence, a::AbstractArray{R}) where {T <: Real, R<:Real}
    @inbounds for i ∈ eachindex(a, u)
        u[i] = Hᵧ(d, a[i])
    end
    return u
end

function hessian(d::AbstractDivergence, a::AbstractArray{R}) where {R<:Real}
    u = similar(a)
    hessian!(u, d, a)
end

function hessian_sum(d::AbstractDivergence, a::AbstractArray{R}) where {R<:Real}
    r = zero(R)
    @inbounds for i ∈ eachindex(a)
        r += Hᵧ(d, a[i])
        end
    return r
end

half(::Type{T}) where T<:Real = convert(T, 0.5)
half(::Type{T}) where T = convert(eltype(T), 0.5)
