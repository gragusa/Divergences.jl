################################################################################
## ReverseKullbackLeibler
##
## ==> \gamma(a/b)b
## ==> \gamma(u)
################################################################################

################################################################################
## Evaluate
################################################################################
function evaluate{T<:AbstractFloat}(dist::ReverseKullbackLeibler,
                                    a::AbstractVector{T}, b::AbstractVector{T})
    if length(a) != length(b)
        throw(DimensionMismatch("first array has length $(length(a)) which does not match the length of the second, $(length(b))."))
    end
    r = zero(T)
    infty = convert(T, Inf)
    for i in eachindex(a, b)
        @inbounds ai = a[i]
        @inbounds bi = b[i]
        ui = ai/bi
        if ui > 0
            r += -ai*log(ui) + ai - bi
        else
            r = infty
            break
        end
    end
    r
end

function evaluate{T<:AbstractFloat}(dist::ReverseKullbackLeibler, a::AbstractVector{T})
    r = zero(T)
    onet = one(T)
    infty = convert(T, Inf)
    for i in eachindex(a)
        @inbounds ai = a[i]
        if ai > 0
            r += - logmxp1(ai)
        else
            r = infty
            break
        end
    end
    r
end

################################################################################
## Gradient
################################################################################
function gradient{T<:AbstractFloat}(dist::ReverseKullbackLeibler, a::T, b::T)
    onet = one(T)
    infty = convert(T, Inf)
    if a > 0 && b > 0
        u = - b/a + onet
    else
        u = infty
    end
    return u
end

function gradient{T<:AbstractFloat}(dist::ReverseKullbackLeibler, a::T)
    onet = one(T)
    infty = convert(T, Inf)
    if a > 0
        u = - onet / a + onet
    else
        u = infty
    end
    return u
end

################################################################################
## Hessian
################################################################################
function hessian{T<:AbstractFloat}(dist::ReverseKullbackLeibler, a::T, b::T)
    ∞ = convert(T, Inf)
    if a > 0 && b > 0
        u = b/a^2
    else
        u = ∞
    end
    u
end

function hessian{T<:AbstractFloat}(dist::ReverseKullbackLeibler, a::T)
    ι = one(T)
    ∞ = convert(T, Inf)
    if a > 0
        u = ι/a^2
    else
        u = ∞
    end
    u
end

################################################################################
## Modified Reverse Kullback-Leibler - MEL
##
## ==> \gamma(a/b)b
## ==> \gamma(u)
################################################################################

################################################################################
## evaluate
################################################################################
function evaluate{T<:AbstractFloat}(dist::MEL, a::AbstractVector{T}, b::AbstractVector{T})
    if length(a) != length(b)
        throw(DimensionMismatch("first array has length $(length(a)) which does not match the length of the second, $(length(b))."))
    end
    ∞ = convert(T, Inf)
    ϑ  = dist.ϑ
    u₀ = 1+ϑ
    rkl = ReverseKullbackLeibler()
    ϕ₀  = evaluate(rkl, [u₀])
    ϕ¹₀ = gradient(rkl, u₀)
    ϕ²₀ = hessian(rkl, u₀)
    ι = one(T)
    r = zero(T)
    for i = eachindex(a, b)
        @inbounds ai = a[i]
        @inbounds bi = b[i]
        ui = ai/bi
        if ui >= u₀
            r += (ϕ₀ + ϕ¹₀*(ui-u₀) + .5*ϕ²₀*(ui-u₀)^2)*bi
        elseif ui > 0 && ui <u₀
            r += (-log(ui) + ui - ι)*bi
        else
            r = ∞
            break
        end
    end
    r
end

function evaluate{T<:AbstractFloat}(dist::MEL, a::AbstractVector{T})
    ∞ = convert(T, Inf)
    ϑ  = dist.ϑ
    u₀ = 1+ϑ
    rkl  = ReverseKullbackLeibler()
    ϕ₀  = evaluate(rkl, [u₀])
    ϕ¹₀ = gradient(rkl, u₀)
    ϕ²₀ = hessian(rkl, u₀)
    r = zero(T)
    ι = one(T)
    for i in eachindex(a)
        @inbounds ai = a[i]
        if ai >= u₀
            r += ϕ₀ + ϕ¹₀*(ai-u₀) + .5*ϕ²₀*(ai-u₀)^2
        elseif ai>0 && ai<u₀
            r += -log(ai) + ai - ι
        else
            r = ∞
            break
        end
    end
    r
end

################################################################################
## gradient
################################################################################
function gradient{T<:AbstractFloat}(dist::MEL, a::T, b::T)
    ∞   = convert(T, Inf)
    ϑ   = dist.ϑ
    u₀  = 1+ϑ
    rkl = ReverseKullbackLeibler()
    ϕ₀  = evaluate(rkl, [u₀])
    ϕ¹₀ = gradient(rkl, u₀)
    ϕ²₀ = hessian(rkl, u₀)
    if a > 0 && b > 0
        ui = a/b
        if ui > u₀
           u = (ϕ¹₀ + ϕ²₀*(ui-u₀))*b
        elseif ui>0 && ui<=u₀
           u = gradient(rkl, a, b)
        end
    else
        u = -∞
    end
    u
end

function gradient{T<:AbstractFloat}(dist::MEL, a::T)
    ∞   = convert(T, Inf)
    ϑ   = dist.ϑ
    u₀  = 1+ϑ
    rkl = ReverseKullbackLeibler()
    ϕ¹₀ = gradient(rkl, u₀)
    ϕ²₀ = hessian(rkl, u₀)
    if a >= u₀
        u =  ϕ¹₀ + ϕ²₀*(a-u₀)
    elseif a > 0 && a < u₀
        u = gradient(rkl, a)
    else
        u = ∞
    end
    u
end

################################################################################
## hessian
################################################################################
function hessian{T<:AbstractFloat}(dist::MEL, a::T)
    ϑ   = dist.ϑ
    u₀  = 1+ϑ
    rkl = ReverseKullbackLeibler()
    ϕ²₀ = hessian(rkl, u₀)
    if a >= u₀
       u  = ϕ²₀
    else
       u = hessian(rkl, a)
    end
    u
end

function hessian{T<:AbstractFloat}(dist::MEL, a::T, b::T)
    ϑ   = dist.ϑ
    u₀  = 1+ϑ
    rkl = ReverseKullbackLeibler()
    ϕ²₀ = hessian(rkl, u₀)
    if a > 0 && b > 0
        if (a/b) >= u₀
            u  = ϕ²₀*b
        else
            u = hessian(rkl, a, b)
        end
    end
    u
end

################################################################################
## Fully Modified Reverse Kullback-Leibler - FMEL
##
################################################################################

################################################################################
## evaluate
################################################################################
function evaluate{T<:AbstractFloat}(dist::FMEL, a::AbstractVector{T}, b::AbstractVector{T})
    if length(a) != length(b)
        throw(DimensionMismatch("first array has length $(length(a)) which does not match the length of the second, $(length(b))."))
    end
    ℓ  = dist.ℓ
    υ  = dist.υ
    u₁ = 1-ℓ
    u₂ = 1+υ

    rkl = ReverseKullbackLeibler()
    ϕ₁  = evaluate(rkl, [u₁])
    ϕ¹₁ = gradient(rkl, u₁)
    ϕ²₁ = hessian(rkl, u₁)

    ϕ₂  = evaluate(rkl, [u₂])
    ϕ¹₂ = gradient(rkl, u₂)
    ϕ²₂ = hessian(rkl, u₂)

    onet = one(T)
    r = zero(T)

    @inbounds for i = eachindex(a, b)
        ai = a[i]
        bi = a[i]
        ui = ai/bi
        if ui >= u₂
            r += (ϕ₂ + ϕ¹₂*(ui-u₂) + .5*ϕ²₀*(ui-u₂)^2)*bi
        elseif ui <= u₁
            r += (ϕ₁ + ϕ¹₁*(ui-u₁) + .5*ϕ²₀*(ui-u₁)^2)*bi
        else
            r += (-log(ui) + ui - onet)*bi
        end
    end
    r
end

function evaluate{T<:AbstractFloat}(dist::FMEL, a::AbstractVector{T})

    ℓ  = dist.ℓ
    υ  = dist.υ
    u₁ = 1-ℓ
    u₂ = 1+υ

    rkl  = ReverseKullbackLeibler()
    ϕ₁  = evaluate(rkl, [u₁])
    ϕ¹₁ = gradient(rkl, u₁)
    ϕ²₁ = hessian(rkl, u₁)

    ϕ₂  = evaluate(rkl, [u₂])
    ϕ¹₂ = gradient(rkl, u₂)
    ϕ²₂ = hessian(rkl, u₂)
    r = zero(T)
    onet = one(T)
    r = zero(T)
    @inbounds for i in eachindex(a)
        ai = a[i]
        if ai >= u₂
            r += ϕ₂ + ϕ¹₂*(ai-u₂) + .5*ϕ²₂*(ai-u₂)^2
        elseif ai <= u₁
            r += ϕ₁ + ϕ¹₁*(ai-u₁) + .5*ϕ²₁*(ai-u₁)^2
        else
            r += -log(ai) + ai - onet
        end
    end
    r
end

################################################################################
## gradient
################################################################################
function gradient{T<:AbstractFloat}(dist::FMEL, a::T, b::T)
    ℓ  = dist.ℓ
    υ  = dist.υ
    u₁ = 1-ℓ
    u₂ = 1+υ

    rkl  = ReverseKullbackLeibler()
    ϕ₁  = evaluate(rkl, [u₁])
    ϕ¹₁ = gradient(rkl, u₁)
    ϕ²₁ = hessian(rkl, u₁)

    ϕ₂  = evaluate(rkl, [u₂])
    ϕ¹₂ = gradient(rkl, u₂)
    ϕ²₂ = hessian(rkl, u₂)
    r = zero(T)
    ui = a/b
    if ui >= u₂
           u = (ϕ¹₂ + ϕ²₂*(ui-u₂))*b
    elseif ui <= u₁
        u = ϕ¹₁ + ϕ²₁*(ui-u₁)
    else
        u = gradient(rkl, a, b)
    end
    u
end

function gradient{T<:AbstractFloat}(dist::FMEL, a::T)
    ℓ  = dist.ℓ
    υ  = dist.υ
    u₁ = 1-ℓ
    u₂ = 1+υ

    # 1-1/u1+(x-u1)/u1^2  x<=u1
    # 1-1/x	              x>u1 && x<u2
    # 1-1/u+(x-u2)/u2^2	  x>=u2

    if a >= u₂
        u = 1-1/u₂+(a-u₂)/(u₂*u₂)
    elseif a <= u₁
        u = 1-1/u₁+(a-u₁)/(u₁*u₁)
    else
        u = 1-1/a
    end
    u
end

################################################################################
## hessian
################################################################################
function hessian{T<:AbstractFloat}(dist::FMEL, a::T)
    ℓ  = dist.ℓ
    υ  = dist.υ
    u₁ = 1-ℓ
    u₂ = 1+υ

    if a >= u₂
        u  = 1/(u₂*u₂)
    elseif a <= u₁
        u  = 1/(u₁*u₁)
    else
        u = 1/(a*a)
    end
    u
end

function hessian{T<:AbstractFloat}(dist::FMEL, a::T, b::T)
    ℓ  = dist.ℓ
    υ  = dist.υ
    u₁ = 1-ℓ
    u₂ = 1+υ

    rkl = ReverseKullbackLeibler()
    ϕ₁  = evaluate(rkl, [u₁])
    ϕ¹₁ = gradient(rkl, u₁)
    ϕ²₁ = hessian(rkl, u₁)

    ϕ₂  = evaluate(rkl, [u₂])
    ϕ¹₂ = gradient(rkl, u₂)
    ϕ²₂ = hessian(rkl, u₂)

    ui = a/b

    if ui >= u₂
        u  = ϕ²₂*b
    elseif ui <= u₁
        u  = ϕ²₁*b
    else
        u = hessian(rkl, a, b)
    end
    u
end
