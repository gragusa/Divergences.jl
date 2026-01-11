# -------------------------------------------------------
# Conjugate (Dual) Functions for Divergences
# 
# Given D(a,b) = Σᵢ γ(aᵢ/bᵢ) bᵢ, we consider the conjugate
# with respect to a (treating b as fixed).
#
# The conjugate of γ is defined via the Legendre-Fenchel transform:
#   ψ(v) = sup_u {u·v - γ(u)}
#
# For the full divergence D(a,b) with respect to a:
#   Dψ(v,b) = Σᵢ ψ(vᵢ) bᵢ
#
# where v is the dual variable corresponding to a.
# -------------------------------------------------------

## -------------------------------------------------------
## Conjugate functions ψ(v)
## -------------------------------------------------------

# Kullback-Leibler: γ(u) = u log(u) - u + 1
# γ'(u) = log(u) → u = exp(v)
# ψ(v) = exp(v) - 1
ψ(::KullbackLeibler, v::T) where {T <: Real} = exp(v) - one(T)

# Reverse Kullback-Leibler: γ(u) = -log(u) + u - 1
# γ'(u) = -1/u + 1 → u = 1/(1-v)
# ψ(v) = -log(1-v) for v < 1
function ψ(::ReverseKullbackLeibler, v::T) where {T <: Real}
    if v < one(T)
        return -NaNMath.log(one(T) - v)
    else
        return convert(T, Inf)
    end
end

# Hellinger: γ(u) = 2u - 4√u + 2 = 2(√u - 1)²
# γ'(u) = 2 - 2/√u → u = 1/(1 - v/2)²
# ψ(v) = 2v/(2-v) for v < 2
function ψ(::Hellinger, v::T) where {T <: Real}
    if v < 2*one(T)
        return 2*v / (2*one(T) - v)
    else
        return convert(T, Inf)
    end
end

# Chi-squared: γ(u) = (u-1)²/2
# γ'(u) = u - 1 → u = v + 1
# ψ(v) = u·v - γ(u) = (v+1)v - v²/2 = v²/2 + v
ψ(::ChiSquared, v::T) where {T <: Real} = v^2 * half(T) + v

# Cressie-Read: γ(u) = [u^(1+α) - 1]/[α(1+α)] - (u-1)/α
# γ'(u) = (u^α - 1)/α → u = (1 + α·v)^(1/α)
# ψ(v) = u·v - γ(u) where u = w^(1/α) and w = 1 + α·v
#      = w^(1/α)·v - [w^((1+α)/α) - 1]/[α(1+α)] + (w^(1/α) - 1)/α
function ψ(d::CressieRead{D}, v::T) where {T <: Real, D}
    α = d.α
    w = one(T) + α * v
    if w > zero(T)
        inv_α = one(T) / α
        u = w^inv_α
        # w^((1+α)/α) = w^(1/α) * w = u * w (saves one power call)
        term1 = (u * w - one(T)) / (α * (one(T) + α))
        term2 = (u - one(T)) / α
        return u * v - term1 + term2
    else
        return convert(T, Inf)
    end
end

# Modified Divergence conjugate
# For v > γ'(ρ): quadratic extension (using precomputed coefficients)
# For v ≤ γ'(ρ): original conjugate
function ψ(d::ModifiedDivergence, v::T) where {T <: Real}
    (; γ₁, aθ, bθ, cθ) = d.m
    if v > γ₁
        return aθ * v^2 + bθ * v + cθ
    else
        return ψ(d.d, v)
    end
end

# Fully Modified Divergence conjugate
function ψ(d::FullyModifiedDivergence, v::T) where {T <: Real}
    (; γ₁, g₁, aθ, bθ, cθ, aφ, bφ, cφ) = d.m
    if v > γ₁
        return aθ * v^2 + bθ * v + cθ
    elseif v < g₁
        return aφ * v^2 + bφ * v + cφ
    else
        return ψ(d.d, v)
    end
end

## -------------------------------------------------------
## Conjugate on arrays
## -------------------------------------------------------
function ψ(d::AbstractDivergence, v::AbstractArray{T}) where {T <: Real}
    out = similar(v, divtype(T))
    @inbounds @simd for j in eachindex(v)
        out[j] = ψ(d, v[j])
    end
    return out
end

## -------------------------------------------------------
## Gradient of conjugate ∇ψ(v) = (γ')⁻¹(v)
## -------------------------------------------------------

∇ψ(::KullbackLeibler, v::T) where {T} = exp(v)

function ∇ψ(::ReverseKullbackLeibler, v::T) where {T}
    return v < one(T) ? one(T) / (one(T) - v) : convert(T, Inf)
end

function ∇ψ(::Hellinger, v::T) where {T}
    return v < 2*one(T) ? 4*one(T) / (2*one(T) - v)^2 : convert(T, Inf)
end

∇ψ(::ChiSquared, v::T) where {T} = v + one(T)

function ∇ψ(d::CressieRead{D}, v::T) where {T <: Real, D}
    α = d.α
    w = one(T) + α * v
    return w > zero(T) ? w^(one(T) / α) : convert(T, Inf)
end

function ∇ψ(d::ModifiedDivergence, v::T) where {T <: Real}
    (; γ₁, aθ, bθ) = d.m
    if v > γ₁
        return 2 * aθ * v + bθ
    else
        return ∇ψ(d.d, v)
    end
end

function ∇ψ(d::FullyModifiedDivergence, v::T) where {T <: Real}
    (; γ₁, g₁, aθ, bθ, aφ, bφ) = d.m
    if v > γ₁
        return 2 * aθ * v + bθ
    elseif v < g₁
        return 2 * aφ * v + bφ
    else
        return ∇ψ(d.d, v)
    end
end

## -------------------------------------------------------
## Hessian of conjugate Hψ(v)
## -------------------------------------------------------

Hψ(::KullbackLeibler, v::T) where {T} = exp(v)

function Hψ(::ReverseKullbackLeibler, v::T) where {T}
    return v < one(T) ? one(T) / (one(T) - v)^2 : convert(T, Inf)
end

function Hψ(::Hellinger, v::T) where {T}
    return v < 2*one(T) ? 8*one(T) / (2*one(T) - v)^3 : convert(T, Inf)
end

Hψ(::ChiSquared, v::T) where {T} = one(T)

function Hψ(d::CressieRead{D}, v::T) where {T <: Real, D}
    α = d.α
    w = one(T) + α * v
    return w > zero(T) ? w^((one(T) - α) / α) : convert(T, Inf)
end

function Hψ(d::ModifiedDivergence, v::T) where {T <: Real}
    (; γ₁, inv_γ₂) = d.m
    return v > γ₁ ? oftype(v, inv_γ₂) : Hψ(d.d, v)
end

function Hψ(d::FullyModifiedDivergence, v::T) where {T <: Real}
    (; γ₁, g₁, inv_γ₂, inv_g₂) = d.m
    if v > γ₁
        return oftype(v, inv_γ₂)
    elseif v < g₁
        return oftype(v, inv_g₂)
    else
        return Hψ(d.d, v)
    end
end

## -------------------------------------------------------
## Full Dual Divergence Dψ(v, b) = Σᵢ ψ(vᵢ) bᵢ
## 
## This is the conjugate of D(a,b) with respect to a,
## treating b as fixed.
## -------------------------------------------------------

"""
    dual(d::AbstractDivergence, v, b)

Evaluate the dual (conjugate) divergence Dψ(v, b) = Σᵢ ψ(vᵢ) bᵢ.

This is the Legendre-Fenchel conjugate of D(a,b) with respect to `a`,
treating `b` as fixed. The dual variable `v` corresponds to the 
gradient of D with respect to `a`.

# Arguments
- `d`: The divergence type
- `v`: Dual variable (same dimension as `a` in the primal)
- `b`: The reference measure (same as in the primal D(a,b))

# Returns
The value of the dual divergence.
"""
function dual(d::AbstractDivergence, v::T, b::S) where {T <: Real, S <: Real}
    return ψ(d, v) * b
end

function dual(d::AbstractDivergence, v::AbstractArray{T}, b::AbstractArray{S}) where {
        T <: Real, S <: Real}
    @assert size(v) == size(b) "v and b must have the same size"
    result = zero(promote_type(T, S))
    @inbounds @simd for i in eachindex(v, b)
        result += ψ(d, v[i]) * b[i]
    end
    return result
end

"""
    dual(d::AbstractDivergence, v)

Evaluate the dual divergence when b = 1 (uniform reference).
Equivalent to Σᵢ ψ(vᵢ).
"""
function dual(d::AbstractDivergence, v::T) where {T <: Real}
    return ψ(d, v)
end

function dual(d::AbstractDivergence, v::AbstractArray{T}) where {T <: Real}
    result = zero(T)
    @inbounds @simd for i in eachindex(v)
        result += ψ(d, v[i])
    end
    return result
end

## -------------------------------------------------------
## Gradient of dual divergence with respect to v
## ∇ᵥDψ(v,b) = (ψ)'(vᵢ) bᵢ ... but we return per-element
## -------------------------------------------------------

"""
    dual_gradient(d::AbstractDivergence, v)

Evaluate the gradient of ψ(v), which equals (γ')⁻¹(v).
This gives the optimal `u = a/b` for given dual variable `v`.
"""
dual_gradient(d::AbstractDivergence, v::T) where {T <: Real} = ∇ψ(d, v)

function dual_gradient(d::AbstractDivergence, v::AbstractArray{T}) where {T <: Real}
    out = similar(v, divtype(T))
    @inbounds @simd for j in eachindex(v)
        out[j] = ∇ψ(d, v[j])
    end
    return out
end

"""
    dual_gradient(d::AbstractDivergence, v, b)

Evaluate the gradient of the dual divergence Dψ(v,b) with respect to v.
Returns ∇ᵥDψ(v,b) = ((ψ)'(v₁)·b₁, ..., (ψ)'(vₙ)·bₙ).

Note: For optimization, you often just need (ψ)'(vᵢ) to recover u = a/b.
"""
function dual_gradient(d::AbstractDivergence, v::T, b::S) where {T <: Real, S <: Real}
    return ∇ψ(d, v) * b
end

function dual_gradient(d::AbstractDivergence, v::AbstractArray{T},
        b::AbstractArray{S}) where {T <: Real, S <: Real}
    @assert size(v) == size(b) "v and b must have the same size"
    out = similar(v, divtype(T, S))
    @inbounds @simd for j in eachindex(v, b)
        out[j] = ∇ψ(d, v[j]) * b[j]
    end
    return out
end

"""
    dual_hessian(d::AbstractDivergence, v)

Evaluate the hessian (second derivative) of ψ(v).
"""
dual_hessian(d::AbstractDivergence, v::T) where {T <: Real} = Hψ(d, v)

function dual_hessian(d::AbstractDivergence, v::AbstractArray{T}) where {T <: Real}
    out = similar(v, divtype(T))
    @inbounds @simd for j in eachindex(v)
        out[j] = Hψ(d, v[j])
    end
    return out
end

"""
    dual_hessian(d::AbstractDivergence, v, b)

Evaluate the hessian of the dual divergence Dψ(v,b) with respect to v.
Returns diag((ψ)''(v₁)·b₁, ..., (ψ)''(vₙ)·bₙ).
"""
function dual_hessian(d::AbstractDivergence, v::T, b::S) where {T <: Real, S <: Real}
    return Hψ(d, v) * b
end

function dual_hessian(d::AbstractDivergence, v::AbstractArray{T},
        b::AbstractArray{S}) where {T <: Real, S <: Real}
    @assert size(v) == size(b) "v and b must have the same size"
    out = similar(v, divtype(T, S))
    @inbounds @simd for j in eachindex(v, b)
        out[j] = Hψ(d, v[j]) * b[j]
    end
    return out
end

## -------------------------------------------------------
## In-place versions
## -------------------------------------------------------

function dual_gradient!(u::AbstractVector{T}, d::AbstractDivergence,
        v::AbstractArray{R}) where {T <: Real, R <: Real}
    PT = divtype(R)
    @assert promote_type(PT, T) === T "Output array eltype $T cannot hold computed type $PT"
    @inbounds @simd for i in eachindex(v, u)
        u[i] = ∇ψ(d, v[i])
    end
    return u
end

function dual_gradient!(u::AbstractVector{T}, d::AbstractDivergence,
        v::AbstractArray{R}, b::AbstractArray{S}) where {T <: Real, R <: Real, S <: Real}
    PT = divtype(R, S)
    @assert promote_type(PT, T) === T "Output array eltype $T cannot hold computed type $PT"
    @inbounds @simd for i in eachindex(v, b, u)
        u[i] = ∇ψ(d, v[i]) * b[i]
    end
    return u
end

function dual_hessian!(u::AbstractVector{T}, d::AbstractDivergence,
        v::AbstractArray{R}) where {T <: Real, R <: Real}
    PT = divtype(R)
    @assert promote_type(PT, T) === T "Output array eltype $T cannot hold computed type $PT"
    @inbounds @simd for i in eachindex(v, u)
        u[i] = Hψ(d, v[i])
    end
    return u
end

function dual_hessian!(u::AbstractVector{T}, d::AbstractDivergence,
        v::AbstractArray{R}, b::AbstractArray{S}) where {T <: Real, R <: Real, S <: Real}
    PT = divtype(R, S)
    @assert promote_type(PT, T) === T "Output array eltype $T cannot hold computed type $PT"
    @inbounds @simd for i in eachindex(v, b, u)
        u[i] = Hψ(d, v[i]) * b[i]
    end
    return u
end

## -------------------------------------------------------
## Fenchel-Young inequality and verification
## -------------------------------------------------------

"""
    fenchel_young(d::AbstractDivergence, a, b, v)

Compute the Fenchel-Young gap for D(a,b) and its dual:
    D(a,b) + Dψ(v,b) - Σᵢ aᵢ vᵢ ≥ 0

Equality holds when v = ∇ₐD(a,b) = γ'(aᵢ/bᵢ).
"""
function fenchel_young(d::AbstractDivergence, a::T, b::S, v::R) where {
        T <: Real, S <: Real, R <: Real}
    u = a / b
    return (γ(d, u) + ψ(d, v)) * b - a * v
end

function fenchel_young(d::AbstractDivergence, a::AbstractArray, b::AbstractArray, v::AbstractArray)
    @assert size(a) == size(b) == size(v) "a, b, and v must have the same size"
    result = zero(promote_type(eltype(a), eltype(b), eltype(v)))
    @inbounds @simd for i in eachindex(a, b, v)
        u = a[i] / b[i]
        result += (γ(d, u) + ψ(d, v[i])) * b[i] - a[i] * v[i]
    end
    return result
end

"""
    verify_duality(d::AbstractDivergence, a, b)

Verify the duality relationship at point (a,b):
    D(a,b) + Dψ(v,b) = Σᵢ aᵢ vᵢ  when v = ∇ₐD(a,b)

Returns the absolute error (should be ≈ 0).
"""
function verify_duality(d::AbstractDivergence, a::T, b::S) where {T <: Real, S <: Real}
    u = a / b
    v = ∇ᵧ(d, u)  # v = γ'(u) = ∇ₐD(a,b)
    lhs = γ(d, u) * b + ψ(d, v) * b
    rhs = a * v
    return abs(lhs - rhs)
end

function verify_duality(d::AbstractDivergence, a::AbstractArray, b::AbstractArray)
    @assert size(a) == size(b) "a and b must have the same size"
    max_error = zero(promote_type(eltype(a), eltype(b)))
    for i in eachindex(a, b)
        err = verify_duality(d, a[i], b[i])
        max_error = max(max_error, err)
    end
    return max_error
end

"""
    primal_from_dual(d::AbstractDivergence, v, b)

Recover the primal variable `a` from the dual variable `v` and reference `b`.
Uses the relationship: a = b · (ψ)'(v) = b · (γ')⁻¹(v)
"""
function primal_from_dual(d::AbstractDivergence, v::T, b::S) where {T <: Real, S <: Real}
    return b * ∇ψ(d, v)
end

function primal_from_dual(d::AbstractDivergence, v::AbstractArray{T},
        b::AbstractArray{S}) where {T <: Real, S <: Real}
    @assert size(v) == size(b) "v and b must have the same size"
    out = similar(v, divtype(T, S))
    @inbounds @simd for j in eachindex(v, b)
        out[j] = b[j] * ∇ψ(d, v[j])
    end
    return out
end

"""
    dual_from_primal(d::AbstractDivergence, a, b)

Compute the dual variable `v` from the primal variables `a` and `b`.
Uses the relationship: v = γ'(a/b)
"""
function dual_from_primal(d::AbstractDivergence, a::T, b::S) where {T <: Real, S <: Real}
    return ∇ᵧ(d, a / b)
end

function dual_from_primal(d::AbstractDivergence, a::AbstractArray{T},
        b::AbstractArray{S}) where {T <: Real, S <: Real}
    @assert size(a) == size(b) "a and b must have the same size"
    return gradient(d, a ./ b)
end
