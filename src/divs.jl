## alogab(a,b) = vifelse((a>0, b>0), a*log(a/b)-a+b, vifelse((iszero(a), iszero(b)), one(eltype(b)), convert(eltype(b), Inf)))
## blogab(a,b) = vifelse((a>0, b>0), -b*log(a/b)+a-b, convert(eltype(b), Inf))

## aloga(a) = vifelse(a>0, a*log(a)-a+one(eltype(a)), convert(eltype(a), Inf))
## loga(a) = vifelse(a>0, -log(a)+a-one(eltype(a)), convert(eltype(a), Inf))

#alogab(a,b) = ifelse((a>0 & b>0), a*log(a/b)-a+b, ifelse((iszero(a) & iszero(b)), one(eltype(b)), convert(eltype(b), Inf)))
#blogab(a,b) = ifelse((a>0 & b>0), -b*log(a/b)+a-b, convert(eltype(b), Inf))

#aloga(a) = ifelse(a>0, a*log(a)-a+one(eltype(a)), convert(eltype(a), Inf))
#loga(a) = ifelse(a>0, -log(a)+a-one(eltype(a)), convert(eltype(a), Inf))

function xlogx(x::Number)
    result = x * NaNMath.log(x)
    return iszero(x) ? zero(result) : result
end

function xlogy(x::Number, y::Number)
    result = x * NaNMath.log(y)
    return iszero(x) && !isnan(y) ? zero(result) : result
end

alogab(a, b) = xlogy(a, a/b) - a + b
blogab(a, b) = -xlogy(b, a/b) + a - b
aloga(a) = xlogx(a) - a + one(eltype(a))
loga(a) = -log(a) + a - one(eltype(a))

γ(::KullbackLeibler, a::T, b::T) where T = alogab(a,b)
γ(::ReverseKullbackLeibler, a::T, b::T) where T = blogab(a,b)
γ(::Hellinger, a::T, b::T) where T = 2*a+(2-4*NaNMath.sqrt(a/b))*b
γ(::ChiSquared, a::T, b::T) where T = half(T)*abs2(a - b)/b

function γ(d::CressieRead{D}, a::T, b::T) where {T, D} 
    α = d.α
    u = a/b
    #(NaNMath.pow(u, 1+α) + α - u*(1+α))*b/(α*(1+α)) 
    ifelse((a>0 && b>0), (u^(1+α)+α-u*(1+α))*b/(α*(1+α)), 
            α > 0 ? zero(eltype(a)) : convert(eltype(a), NaN))

end

γ(::KullbackLeibler, a::T) where T = aloga(a)
γ(::ReverseKullbackLeibler, a::T) where T = loga(a)
γ(::Hellinger, a::T) where T = 2*a - 4*NaNMath.sqrt(a) + 2
γ(::ChiSquared, a::T) where T = abs2(a - one(T))*half(T)

function γ(d::CressieRead{D}, a::T) where {T, D} 
    α = d.α
    ifelse(a >= 0, (a^(1 + α) + α - a*(1 + α))/(α*(1 + α)), 
            α > 0 ? zero(eltype(a)) : convert(eltype(a), NaN))
end

∇ᵧ(::KullbackLeibler, a::T, b::T) where T = ifelse((a>0 && b>0), log(a/b), convert(T, -Inf))
∇ᵧ(::ReverseKullbackLeibler, a::T, b::T) where T = ifelse((a>0 && b>0), -b/a + one(T), convert(T, -Inf))
∇ᵧ(d::CressieRead, a::T, b::T) where T = ifelse((a >= 0 && b>0), ((a/b)^d.α - one(T))/d.α, convert(T, sign(d.α)*Inf))
∇ᵧ(d::Hellinger, a::T, b::T) where T = ifelse((a>0 && b>0), 2(one(T)-one(T)/sqrt(a/b)), convert(T, -Inf))
∇ᵧ(d::ChiSquared, a::T, b::T) where T = a/b - one(T)

## ∇ᵧ(::KullbackLeibler, a::T, b::T) where T = NaNMath.log(a/b)
## ∇ᵧ(::ReverseKullbackLeibler, a::T, b::T) where T = ifelse((a>0 & b>0), -b/a + one(T), convert(T, -Inf))
## ∇ᵧ(d::CressieRead, a::T, b::T) where T = ifelse((a>0 & b>0), ((a/b)^d.α - one(T))/d.α, convert(T, sign(d.α)*Inf))
## ∇ᵧ(d::Hellinger, a::T, b::T) where T = ifelse((a>0 & b>0), 2(one(T)-one(T)/sqrt(a/b)), convert(T, -Inf))
## ∇ᵧ(d::ChiSquared, a::T, b::T) where T = a/b - one(T)


∇ᵧ(::KullbackLeibler, a::T) where T = NaNMath.log(a)
∇ᵧ(::ReverseKullbackLeibler, a::T) where T = ifelse(a > 0, -1/a + one(T), convert(T, -Inf))
∇ᵧ(d::CressieRead, a::T) where T = ifelse(a >= 0, (a^d.α - one(T))/d.α, convert(T, sign(d.α)*Inf))
∇ᵧ(d::Hellinger, a::T) where T = ifelse(a > 0, 2(one(T)-one(T)/sqrt(a)), convert(T, -Inf))
∇ᵧ(d::ChiSquared, a::T) where T = a - one(T)

Hᵧ(::KullbackLeibler, a::T, b::T) where T = ifelse((a > 0 && b > 0), one(T)/a, convert(T, Inf))
Hᵧ(::ReverseKullbackLeibler, a::T, b::T) where T = ifelse((a > 0 && b > 0), b/a^2, convert(T, Inf))
Hᵧ(d::CressieRead, a::T, b::T) where T = ifelse((a > 0 && b > 0), a^(d.α-1)*b^(-d.α), convert(T, Inf))
Hᵧ(d::Hellinger, a::T, b::T) where T = ifelse((a > 0 && b > 0), sqrt(b)/sqrt(a^3), convert(T, Inf))    
Hᵧ(d::ChiSquared, a::T, b::T) where T = ifelse(b >= 0, 1/b, convert(T, Inf))

Hᵧ(::KullbackLeibler, a::T) where T = ifelse(a > 0, one(T)/a, convert(T, Inf))
Hᵧ(::ReverseKullbackLeibler, a::T) where T = ifelse(a > 0, one(T)/a^2, convert(T, Inf))
Hᵧ(d::CressieRead, a::T) where T = ifelse(a > 0, a^(d.α-1), convert(T, Inf))
Hᵧ(d::Hellinger, a::T) where T = ifelse(a > 0, one(T)/sqrt(a^(3)), convert(T, Inf))    
Hᵧ(d::ChiSquared, a::T) where T = one(T)

## Modified Divergences Function
function γᵤ(d::D, a::T, b::T) where {T, D<:AbstractModifiedDivergence}
    (; γ₀, γ₁, γ₂, ρ)  = d.m
    (γ₀ + γ₁*((a/b)-ρ) + half(T)*γ₂*(a/b-ρ)^2)*b
end

function γₗ(d::D, a::T, b::T) where {T, D<:AbstractModifiedDivergence}
    (; g₀, g₁, g₂, φ)  = d.m
    (g₀ + g₁*((a/b)-φ) + half(T)*g₂*(a/b-φ)^2)*b
end

function γᵤ(d::D, a::T) where {T, D<:AbstractModifiedDivergence}
    (; γ₀, γ₁, γ₂, ρ)  = d.m
    (γ₀ + γ₁*(a-ρ) + half(T)*γ₂*(a-ρ)^2)
end

function γₗ(d::D, a::T) where {T, D<:AbstractModifiedDivergence}
    (; g₀, g₁, g₂, φ)  = d.m
    (g₀ + g₁*(a-φ) + half(T)*g₂*(a-φ)^2)
end

function ∇ᵤ(d::D, a::T, b::T) where {T, D<:AbstractModifiedDivergence}
    (; γ₀, γ₁, γ₂, ρ)  = d.m
    γ₁ + (a/b)*γ₂ - γ₂*ρ
end

function ∇ₗ(d::D, a::T, b::T) where {T, D<:AbstractModifiedDivergence}
    (; g₀, g₁, g₂, φ)  = d.m
    g₁ + (a/b)*g₂ - g₂*φ
end

function ∇ᵤ(d::D, a::T) where {T, D<:AbstractModifiedDivergence}
    (; γ₀, γ₁, γ₂, ρ)  = d.m
    (γ₁ + γ₂*(a-ρ))
end

function ∇ₗ(d::D, a::T) where {T, D<:AbstractModifiedDivergence}
    (; g₀, g₁, g₂, φ)  = d.m
    (g₁ + g₂*(a-φ))
end

function Hᵤ(d::D, a::T, b::T) where {T, D<:AbstractModifiedDivergence}
    (; γ₀, γ₁, γ₂, ρ)  = d.m
    γ₂/b
end

function Hₗ(d::D, a::T, b::T) where {T, D<:AbstractModifiedDivergence}
    (; g₀, g₁, g₂, φ)  = d.m
     g₂/b
end

function Hᵤ(d::D, a::T) where {T, D<:AbstractModifiedDivergence}
    (; γ₀, γ₁, γ₂, ρ)  = d.m
    γ₂
end

function Hₗ(d::D, a::T) where {T, D<:AbstractModifiedDivergence}
    (; g₀, g₁, g₂, φ)  = d.m
    g₂
end

##
function γ(d::ModifiedDivergence, a::T, b::T) where T
    (; ρ ) = d.m
    div = d.d
    ifelse(a>ρ*b, γᵤ(d, a, b), γ(div, a, b))
end

function γ(d::ModifiedDivergence, a::T) where T
    (; ρ ) = d.m
    div = d.d
    ifelse(a>ρ, γᵤ(d, a), γ(div, a))
end

function γ(d::FullyModifiedDivergence, a::T, b::T) where T
    (; ρ, φ ) = d.m
    div = d.d
    ifelse(a>ρ*b, γᵤ(d, a, b), ifelse(a<φ*b, γₗ(d, a, b),  γ(div, a, b)))
end

function γ(d::FullyModifiedDivergence, a::T) where T
    (; ρ, φ) = d.m
    div = d.d
    ifelse(a>ρ, γᵤ(d, a), ifelse(a<φ, γₗ(d, a), γ(div, a)))
end

function ∇ᵧ(d::ModifiedDivergence, a::T, b::T) where T
    (; ρ ) = d.m
    div = d.d
    ifelse(a>ρ*b, ∇ᵤ(d, a, b), ∇ᵧ(div, a, b))
end

function ∇ᵧ(d::ModifiedDivergence, a::T) where T
    (; ρ ) = d.m
    div = d.d
    ifelse(a>ρ, ∇ᵤ(d, a), ∇ᵧ(div, a))
end

function ∇ᵧ(d::FullyModifiedDivergence, a::T, b::T) where T
    (; ρ, φ ) = d.m
    div = d.d
    ifelse(a>ρ*b, ∇ᵤ(d, a, b),  ifelse(a<φ*b, ∇ₗ(d, a, b),  ∇ᵧ(div, a, b)))
end

function ∇ᵧ(d::FullyModifiedDivergence, a::T) where T
    (; ρ, φ ) = d.m
    div = d.d
    ifelse(a>ρ, ∇ᵤ(d, a),  ifelse(a<φ, ∇ₗ(d, a),  ∇ᵧ(div, a)))
end

function Hᵧ(d::ModifiedDivergence, a::T, b::T) where T 
    (; ρ ) = d.m
    div = d.d
    ifelse(a>ρ*b, Hᵤ(d, a, b), Hᵧ(div, a, b))
end

function Hᵧ(d::ModifiedDivergence, a::T) where T 
    (; ρ ) = d.m
    div = d.d
    ifelse(a>ρ, Hᵤ(d, a), Hᵧ(div, a))
end

function Hᵧ(d::FullyModifiedDivergence, a::T, b::T) where T 
    (; ρ, φ) = d.m
    div = d.d
    ifelse(a>ρ*b, Hᵤ(d, a, b), ifelse(a<φ*b, Hₗ(d, a, b),  Hᵧ(div, a, b)))
end

function Hᵧ(d::FullyModifiedDivergence, a::T) where T 
    (; ρ, φ) = d.m
    div = d.d
    ifelse(a>ρ, Hᵤ(d, a), ifelse(a<φ, Hₗ(d, a),  Hᵧ(div, a)))
end

eval(d::AbstractDivergence, a::T, b::T) where T<:Real = γ(d, a, b)
gradient(d::AbstractDivergence, a::T, b::T) where T<:Real = ∇ᵧ(d, a, b)
hessian(d::AbstractDivergence, a::T, b::T) where T<:Real = Hᵧ(d, a, b)

eval(d::AbstractDivergence, a::T) where T<:Real = γ(d, a)
gradient(d::AbstractDivergence, a::T) where T<:Real = ∇ᵧ(d, a)
hessian(d::AbstractDivergence, a::T) where T<:Real = Hᵧ(d, a)

## eval
function eval(d::AbstractDivergence, a::AbstractArray{T}, b::AbstractArray{T}) where T <: Real
    r = zero(T)
    @inbounds for i ∈ eachindex(a,b)
        r += γ(d, a[i], b[i])
    end
    return r
end

function eval(d::AbstractDivergence, a::AbstractArray{T}) where T <: Real
    r = zero(T)
    @inbounds for i ∈ eachindex(a)
        r += γ(d, a[i])
    end
    return r
end

## gradient
function gradient!(u::AbstractVector{T}, d::AbstractDivergence, a::AbstractArray{T}, b::AbstractArray{T}) where T <: Real
    @inbounds for i ∈ eachindex(a, b, u)
        u[i] = ∇ᵧ(d, a[i], b[i])
    end
    return u
end

function gradient!(u::AbstractVector{T}, d::AbstractDivergence, a::AbstractArray{T}) where T <: Real
    @inbounds for i ∈ eachindex(a, u)
        u[i] = ∇ᵧ(d, a[i])
    end
    return u
end

function gradient(d::AbstractDivergence, a::AbstractArray{T}, b::AbstractArray{T}) where T <: Real 
    u = similar(a)
    gradient!(u, d, a, b)
end

function gradient(d::AbstractDivergence, a::AbstractArray{T}) where T <: Real 
    u = similar(a)
    gradient!(u, d, a)
end

function gradient_sum(d::AbstractDivergence, a::AbstractArray{T}, b::AbstractArray{T}) where T <: Real
    r = zero(T)
    @inbounds for i ∈ eachindex(a,b)
        r += ∇ᵧ(d, a[i], b[i])
    end
    return r
end

function gradient_sum(d::AbstractDivergence, a::AbstractArray{T}) where T <: Real
    r = zero(T)
    @inbounds for i ∈ eachindex(a)
        r += ∇ᵧ(d, a[i])
    end
    return r
end

## hessian
function hessian!(u::AbstractVector{T}, d::AbstractDivergence, a::AbstractArray{T}, b::AbstractArray{T}) where T <: Real
    @inbounds for i ∈ eachindex(a, b, u)
        u[i] = Hᵧ(d, a[i], b[i])
    end
    return u
end

function hessian!(u::AbstractVector{T}, d::AbstractDivergence, a::AbstractArray{T}) where T <: Real
    @inbounds for i ∈ eachindex(a, u)
        u[i] = Hᵧ(d, a[i])
    end
    return u
end

function hessian(d::AbstractDivergence, a::AbstractArray{T}, b::AbstractArray{T}) where T <: Real 
    u = similar(a)
    hessian!(u, d, a, b)
end

function hessian(d::AbstractDivergence, a::AbstractArray{T}) where T <: Real 
    u = similar(a)
    hessian!(u, d, a)
end

function hessian_sum(d::AbstractDivergence, a::AbstractArray{T}, b::AbstractArray{T}) where T <: Real
    r = zero(T)
    @inbounds for i ∈ eachindex(a,b)
        r += Hᵧ(d, a[i], b[i])
    end
    return r
end

function hessian_sum(d::AbstractDivergence, a::AbstractArray{T}) where T <: Real
    r = zero(T)
    @inbounds for i ∈ eachindex(a)
        r += Hᵧ(d, a[i])
    end
    return r
end

half(::Type{T}) where T<:Real = convert(T, 0.5)
