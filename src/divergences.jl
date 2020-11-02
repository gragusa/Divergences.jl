alogab(a,b) = vifelse(andmask(a>0, b>0), a*log(a/b)-a+b, vifelse(andmask(iszero(a), iszero(b)), one(eltype(b)), convert(eltype(b), Inf)))
blogab(a,b) = vifelse(andmask(a>0, b>0), -b*log(a/b)+a-b, convert(eltype(b), Inf))

aloga(a) = vifelse(a>0, a*log(a)-a+one(eltype(a)), vifelse(iszero(a), one(eltype(a)), convert(eltype(a), Inf)))
loga(a) = vifelse(a>0, -log(a)+a-one(eltype(a)), convert(eltype(a), Inf))


γ(::KullbackLeibler, a::T, b::T) where T = alogab(a,b)
γ(::ReverseKullbackLeibler, a::T, b::T) where T = blogab(a,b)
γ(::Hellinger, a::T, b::T) where T = vifelse(addmask(a>=0, b>=0), 2*a+(2-4*sqrt(a/b))*b, convert(T, Inf))
function γ(d::CressieRead{D}, a::T, b::T) where {T, D} 
    α = d.α
    u = a/b
    vifelse(andmask(a>0, b>0), (u^(1+α)+α-u*(1+α))*b/(α*(1+α)), 
            α > 0 ? zero(eltype(a)) : convert(eltype(a), Inf))
end

γ(::KullbackLeibler, a::T) where T = aloga(a)
γ(::ReverseKullbackLeibler, a::T) where T = loga(a)
γ(::Hellinger, a::T) where T = vifelse(a>=0, 2*a - 4*sqrt(a) + 2, convert(T, Inf))

function γ(d::CressieRead{D}, a::T) where {T, D} 
    α = d.α    
    vifelse(a>=0, (a^(1+α)+α-a*(1+α))/(α*(1+α)), 
            α > 0 ? zero(eltype(a)) : convert(eltype(a), Inf))
end

∇ᵧ(::KullbackLeibler, a::T, b::T) where T = vifelse(andmask(a>0, b>0), log(a/b), convert(T, -Inf))
∇ᵧ(::ReverseKullbackLeibler, a::T, b::T) where T = vifelse(andmask(a>0, b>0), -b/a + one(T), convert(T, -Inf))
∇ᵧ(d::CressieRead, a::T, b::T) where T = vifelse(andmask(a>0, b>0), ((a/b)^α - 1)/α, convert(T, sign(d.α)*Inf))
∇ᵧ(d::Hellinger, a::T, b::T) where T = vifelse(andmask(a>0, b=0), 2(one(T)-one(T)/sqrt(a/b)), convert(T, -Inf))

∇ᵧ(::KullbackLeibler, a::T) where T = vifelse(a>0, log(a), convert(T, -Inf))
∇ᵧ(::ReverseKullbackLeibler, a::T) where T = vifelse(a>0, -1/a + one(T), convert(T, -Inf))
∇ᵧ(d::CressieRead, a::T) where T = vifelse(a>=0, (a^d.α - 1)/d.α, convert(T, sign(d.α)*Inf))
∇ᵧ(d::Hellinger, a::T) where T = vifelse(a>0, 2(one(T)-one(T)/sqrt(a)), convert(T, -Inf))

Hᵧ(::KullbackLeibler, a::T, b::T) where T = vifelse(andmask(a>0, b>0), b/a, convert(T, Inf))
Hᵧ(::ReverseKullbackLeibler, a::T, b::T) where T = vifelse(andmask(a>0, b>0), b/a^2, convert(T, Inf))
Hᵧ(d::CressieRead, a::T, b::T) where T = vifelse(andmask(a>0, b>0), a^(d.α-1)*b^(-d.α), convert(T, Inf))
Hᵧ(d::Hellinger, a::T, b::T) where T = vifelse(andmask(a>0, b>0), sqrt(b)/sqrt(a^3), convert(T, Inf))    

Hᵧ(::KullbackLeibler, a::T) where T = vifelse(a>0, one(T)/a, convert(T, Inf))
Hᵧ(::ReverseKullbackLeibler, a::T) where T = vifelse(a>0, one(T)/a^2, convert(T, Inf))
Hᵧ(d::CressieRead, a::T) where T = vifelse(a>0, a^(d.α-1), convert(T, Inf))
Hᵧ(d::Hellinger, a::T) where T = vifelse(a>0, one(T)/a^(3/2), convert(T, Inf))    

checksize(a, b) = length(a) != length(b) ? throw(DimensionMismatch()) : nothing

function eval(d::Divergence, a::AbstractArray{T}, b::AbstractArray{T}) where T <: Real
    r = zero(T)
    @avx for i ∈ eachindex(a,b)
        r += γ(d, a[i], b[i])
    end
    return r
end

function eval(d::Divergence, a::AbstractArray{T}) where T <: Real
    r = zero(T)
    @avx for i ∈ eachindex(a)
        r += γ(d, a[i])
    end
    return r
end

function gradient!(u::AbstractVector{T}, d::Divergence, a::AbstractArray{T}, b::AbstractArray{T}) where T <: Real
    @avx for i ∈ eachindex(a,b)
        u[i] = ∇ᵧ(d, a[i], b[i])
    end
    return u
end

function gradient!(u::AbstractVector{T}, d::Divergence, a::AbstractArray{T}) where T <: Real
    @avx for i ∈ eachindex(a)
        u[i] = ∇ᵧ(d, a[i])
    end
    return u
end

function gradient(d::Divergence, a::AbstractArray{T}, b::AbstractArray{T}) where T <: Real 
    u = similar(a)
    gradient!(u, d, a, b)
end

function gradient(d::Divergence, a::AbstractArray{T}) where T <: Real 
    u = similar(a)
    gradient!(u, d, a)
end

function hessian!(u::AbstractVector{T}, d::Divergence, a::AbstractArray{T}, b::AbstractArray{T}) where T <: Real
    @avx for i ∈ eachindex(a,b)
        u[i] = Hᵧ(d, a[i], b[i])
    end
    return u
end

function hessian!(u::AbstractVector{T}, d::Divergence, a::AbstractArray{T}) where T <: Real
    @avx for i ∈ eachindex(a)
        u[i] = Hᵧ(d, a[i])
    end
    return u
end

function hessian(d::Divergence, a::AbstractArray{T}, b::AbstractArray{T}) where T <: Real 
    u = similar(a)
    hessian!(u, d, a, b)
end

function hessian(d::Divergence, a::AbstractArray{T}) where T <: Real 
    u = similar(a)
    hessian!(u, d, a)
end

function gradient_sum(d::Divergence, a::AbstractArray{T}, b::AbstractArray{T}) where T <: Real
    r = zero(T)
    @avx for i ∈ eachindex(a,b)
        r += ∇ᵧ(d, a[i], b[i])
    end
    return r
end

function gradient_sum(d::Divergence, a::AbstractArray{T}) where T <: Real    
    r = zero(T)
    @avx for i ∈ eachindex(a)
        r += ∇ᵧ(d, a[i])
    end
    return r
end

function γₗ(d::D, a::T, b::T) where {T, D<:Union{FullyModifiedDivergence, ModifiedDivergence} }
    γ₀, γ₁, γ₂, ρ, ...  = d.m
    (γ₀ + γ₁*((a/b)-uₑ) + half(T)*γ₂*(a/b-uₑ)^2)*b
end

function γᵤ(d::D, a::T, b::T) where {T, D<:Union{FullyModifiedDivergence, ModifiedDivergence} }
    γ₀, γ₁, γ₂, ρ, g₀, g₁, g₂, φ  = d.m
    (g₀ + g₁*((a/b)-φ) + half(T)*g₂*(a/b-φ)^2)*b
end

function γₗ(d::FullyModifiedDivergence, a::T) where T 
    γ₀, γ₁, γ₂, ρ  = d.m
    (γ₀ + γ₁*(a-ρ) + half(T)*γ₂*(a-ρ)^2)
end

function γᵤ(d::FullyModifiedDivergence, a::T) where T 
    γ₀, γ₁, γ₂, ρ, g₀, g₁, g₂, φ  = d.m
    (g₀ + g₁*(a-φ) + half(T)*g₂*(a-φ)^2)
end

function eval(d::ModifiedDivergence, a::AbstractArray{T}, b::AbstractArray{T}) where T <: Real
    γ₀, γ₁, γ₂, ρ  = d.m
    div = d.d
    r = zero(T)
    for i ∈ eachindex(a,b)
        r += a[i]<ρ ? γₗ(d, a[i], b[i]) : γ(div, a[i], b[i])
    end
    return r
end

function eval(d::ModifiedDivergence, a::AbstractArray{T}) where T <: Real        
    γ₀, γ₁, γ₂, ρ  = d.m
    div = d.d
    r = zero(T)
    @avx for i ∈ eachindex(a)        
        r += a[i]<ρ ? γₗ(d, a[i]) : γ(div, a[i])
    end
    return r
end

function eval(d::FullyModifiedDivergence, a::AbstractArray{T}, b::AbstractArray{T}) where T <: Real
    γ₀, γ₁, γ₂, ρ, g₀, g₁, g₂, φ  = d.m
    div = d.d
    r = zero(T)
    for i ∈ eachindex(a,b)
        r += a[i]<ρ*b[i] ? γₗ(d, a[i], b[i]) : a[i]>φ*b[i] ? γᵤ(d, a[i], b[i]) : γ(div, a[i], b[i])
    end
    return r
end

function eval(d::FullyModifiedDivergence, a::AbstractArray{T}) where T <: Real        
    γ₀, γ₁, γ₂, ρ, g₀, g₁, g₂, φ  = d.m
    div = d.d
    r = zero(T)
    @avx for i ∈ eachindex(a)        
        r += a[i]<ρ ? γₗ(d, a[i]) : a[i]>φ ? γᵤ(d, a[i]) : γ(div, a[i])
    end
    return r
end

half(::Type{T}) where T = convert(T, 0.5)