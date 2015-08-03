################################################################################
## ReverseKullbackLeibler
##
## ==> \gamma(a/b)b
## ==> \gamma(u)
################################################################################

################################################################################
## Evaluate
################################################################################
function evaluate{T<:FloatingPoint}(dist::ReverseKullbackLeibler,
                                    a::AbstractVector{T}, b::AbstractVector{T})
    r = zero(T)
    n = get_common_len(a, b)::Int
    for i = 1 : n
        @inbounds ai = a[i]
        @inbounds bi = b[i]
        @inbounds ui = ai/bi
        if ui > 0
            r += (-log(ui) + ui -1)*bi
        else
            r = oftype(a, Inf)
            break
        end
    end
    r
end

function evaluate{T<:FloatingPoint}(dist::ReverseKullbackLeibler, a::AbstractVector{T})
    r = zero(T)
    onet = one(T)
    n = length(a)::Int
    @inbounds for i = 1:n
        ai = a[i]
        if ai > 0
            r += -log(ai) + ai - onet
        else
            r = oftype(ai, Inf)
            break
        end
    end
    r
end

################################################################################
## Gradient
################################################################################
function gradient{T<:FloatingPoint}(dist::ReverseKullbackLeibler, a::T, b::T)
    onet = one(T)
    if a > 0 && b > 0
        u = - b/a + onet
    else
        u = oftype(a, Inf)
    end
    return u
end

function gradient{T<:FloatingPoint}(dist::ReverseKullbackLeibler, a::T)
    onet = one(T)
    if a > 0
        u = -onet/a + onet
    else
        u = oftype(a, Inf)
    end
    return u
end

function gradient!{T<:FloatingPoint}(u::Vector{T}, dist::ReverseKullbackLeibler,
                                     a::AbstractVector{T}, b::AbstractVector{T})
    n = get_common_len(a, b)::Int
    onet = one(T)
    @inbounds for i = 1 : n
        ai = a[i]
        bi = b[i]
        u[i] = gradient(dist, ai, bi)
    end
    u
end

function gradient!{T<:FloatingPoint}(u::Vector{T}, dist::ReverseKullbackLeibler,
                                     a::AbstractVector{T})
    n = length(a)
    onet = one(T)
    @inbounds for i = 1:n
        ai = a[i]
        u[i] = gradient(dist, ai)
    end
    u
end

################################################################################
## Hessian
################################################################################
function hessian{T<:FloatingPoint}(dist::ReverseKullbackLeibler, a::T, b::T)
    if a > 0 && b > 0
        u = b/a^2
    else
        u = oftype(a, Inf)
    end
    u
end

function hessian{T<:FloatingPoint}(dist::ReverseKullbackLeibler, a::T)
    onet = one(T)
    if a > 0
        u = onet/a^2
    else
        u = oftype(a, Inf)
    end
    u
end


function hessian!{T<:FloatingPoint}(u::Vector{T}, dist::ReverseKullbackLeibler,
                                    a::AbstractVector{T}, b::AbstractVector{T})
    onet = one(T)
    n = get_common_len(a, b)::Int
    @inbounds for i = 1 : n
        ai = a[i]
        bi = b[i]
        ui = bi/ai^2
        if ai > 0 && bi > 0
            u[i] = ui
        else
            u[i] = oftype(ai, Inf)
        end
    end
    u
end

function hessian!{T<:FloatingPoint}(u::Vector{T}, dist::ReverseKullbackLeibler,
                                    a::AbstractVector{T})
    onet = one(T)
    n = length(a)::Int
    @inbounds for i = 1:n
        ai = a[i]
        if ai > 0
            u[i] = onet/ai^2
        else
            u[i] = oftype(ai, Inf)
        end
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
function evaluate{T<:FloatingPoint}(dist::MEL, a::AbstractVector{T},
                                    b::AbstractVector{T})
    ϑ  = dist.ϑ
    u₀ = 1+ϑ
    rkl = ReverseKullbackLeibler()
    ϕ₀  = evaluate(rkl, [u₀])
    ϕ¹₀ = gradient(rkl, u₀)
    ϕ²₀ = hessian(rkl, u₀)
    onet = one(T)
    r = zero(T)
    n = get_common_len(a, b)::Int
    @inbounds for i = 1:n
        ai = a[i]
        bi = a[i]
        ui = ai/bi
        if ui >= u₀
            r += (ϕ₀ + ϕ¹₀*(ui-u₀) + .5*ϕ²₀*(ui-u₀)^2)*bi
        elseif ui > 0 && ui <u₀
            r += (-log(ui) + ui - onet)*bi
        else
            r = oftype(ai, Inf)
            break
        end
    end
    r
end

function evaluate{T<:FloatingPoint}(dist::MEL, a::AbstractVector{T})
    ϑ  = dist.ϑ
    u₀ = 1+ϑ
    rkl  = ReverseKullbackLeibler()
    ϕ₀  = evaluate(rkl, [u₀])
    ϕ¹₀ = gradient(rkl, u₀)
    ϕ²₀ = hessian(rkl, u₀)
    r = zero(T)
    onet = one(T)
    n = length(a)::Int
    @inbounds for i = 1 : n
        ai = a[i]
        if ai >= u₀
            r += ϕ₀ + ϕ¹₀*(ai-u₀) + .5*ϕ²₀*(ai-u₀)^2
        elseif ai>0 && ai<u₀
            r += -log(ai) + ai - onet
        else
            r = oftype(ai, Inf)
            break
        end
    end
    r
end

################################################################################
## gradient
################################################################################
function gradient{T<:FloatingPoint}(dist::MEL, a::T, b::T)
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
        u = oftype(a, -Inf)
    end
    u
end

function gradient{T<:FloatingPoint}(dist::MEL, a::T)
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
        u = oftype(a, Inf)
    end
    u
end

function gradient!{T<:FloatingPoint}(u::Vector{T}, dist::MEL, a::AbstractVector{T}, b::AbstractVector{T})
    n = get_common_len(a, b)::Int
    @inbounds for i = 1 : n
        ai = a[i]
        bi = b[i]
        u[i] = gradient(dist, ai, bi)
    end
    u
end

function gradient!{T<:FloatingPoint}(u::Vector{T}, dist::MEL, a::AbstractVector{T})
    n = length(a)::Int
    @inbounds for i = 1:n
        u[i] = gradient(dist, a[i])
    end
    u
end

################################################################################
## hessian
################################################################################
function hessian{T<:FloatingPoint}(dist::MEL, a::T)
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

function hessian{T<:FloatingPoint}(dist::MEL, a::T, b::T)
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

function hessian!{T<:FloatingPoint}(u::Vector{T}, dist::MEL, a::AbstractVector{T}, b::AbstractVector{T})
    n = get_common_len(a, b)::Int
    @inbounds for i = 1 : n
        ai = a[i]
        bi = b[i]
        u[i] = hessian(dist, ai, bi)
    end
    u
end

function hessian!{T<:FloatingPoint}(u::Vector{T}, dist::MEL, a::AbstractVector{T})
    n = length(a)::Int
    @inbounds for i = 1:n
        ai   = a[i]
        u[i] = hessian(dist, ai)
    end
    u
end
