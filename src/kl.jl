################################################################################
## Kullback-Leibler - ET
##
## ==> \gamma(a/b)b
## ==> \gamma(u)
################################################################################
function evaluate{T <: FloatingPoint}(dist::ET, a::AbstractVector{T}, b::AbstractVector{T})
    onet = one(T)
    r = zero(T)
    n = get_common_len(a, b)::Int
    @inbounds for i = 1 : n
        ai = a[i]
        bi = a[i]
        ui = ai/bi

        if ui > 0
            r += (ui*log(ui) - ui + onet)*bi
        else
            r = +Inf
            break
        end
    end
    r
end

function evaluate{T <: FloatingPoint}(dist::ET, a::AbstractVector{T})
    r = zero(T)
    onet = one(T)
    n = length(a)::Int
    for i = 1 : n
        @inbounds ai = a[i]
        if ai > 0
            r += ai*log(ai) - ai + onet
        else
            r = +Inf
            break
        end
    end
    r
end

################################################################################
## evaluate
################################################################################
function gradient{T <: FloatingPoint}(dist::ET, a::T, b::T)
    ## This is the derivative of
    ## \gamma(a/b) with respect to a
    if b<=0
        u = Inf
    end
    if a > 0 && b > 0
        u = log(a/b)
    else
        u = -Inf
    end
    u
end

function gradient{T <: FloatingPoint}(dist::ET, a::T)
    if a > 0
        u = log(a)
    else
        u = -Inf
    end
    u
end

function gradient!{T <: FloatingPoint}(u::Vector{T}, dist::ET, a::AbstractVector{T}, b::AbstractVector{T})
    n = get_common_len(a, b)::Int
    @inbounds for i = 1:n
        ai = a[i]
        bi = b[i]
        ui = ai/bi
        u[i] = gradient(dist, ai, bi)
    end
    u
end


function gradient!{T <: FloatingPoint}(u::Vector{T}, dist::ET, a::AbstractVector{T})
    n = length(a)::Int
    @inbounds for i = 1:n
       ai = a[i]
       u[i] = gradient(dist, ai)
    end
    u
end

################################################################################
## hessian
################################################################################
function hessian{T <: FloatingPoint}(dist::ET, a::T, b::T)
    onet = one(T)
    r    = zero(T)
    if b==0
        u = +Inf
    elseif a > 0 && b > 0
        u = onet/(b*a)
    else
        u = +Inf
    end
    u
end

function hessian{T <: FloatingPoint}(dist::ET, a::T)
    onet = one(T)
    r    = zero(T)
    if a > 0
        u = onet/a
    elseif a==0
        u = +Inf
    end
    u
end

function hessian!{T <: FloatingPoint}(u::Vector{T}, dist::ET, a::AbstractVector{T}, b::AbstractVector{T})
    n = get_common_len(a, b)::Int
    @inbounds for i = 1:n
        ai = a[i]
        bi = b[i]
        u[i] = hessian(dist, ai, bi)
    end
    u
end

function hessian!{T <: FloatingPoint}(u::Vector{T}, dist::ET, a::AbstractVector{T})
    n = length(a)::Int
    onet = one(T)
    r    = zero(T)

    @inbounds for i = 1 : n
        ai = a[i]
        u[i] = hessian(dist, ai)
    end
    u
end

################################################################################
## Modified Kullback-Leibler - MET
##
## ==> \gamma(a/b)b
## ==> \gamma(u)
################################################################################


################################################################################
## evaluate
################################################################################
function evaluate{T <: FloatingPoint}(dist::MET, a::AbstractVector{T}, b::AbstractVector{T})
    ϑ  = dist.ϑ
    u₀ = 1+ϑ
    kl  = KullbackLeibler()
    ϕ₀  = evaluate(kl, [u₀])
    ϕ¹₀ = gradient(kl, u₀)
    ϕ²₀ = hessian(kl, u₀)
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
            r += (ui*log(ui) - ui + onet)*bi
        else
            r = +Inf
        end
    end
    r
end

function evaluate{T <: FloatingPoint}(dist::MET, a::AbstractVector{T})
    ϑ  = dist.ϑ
    u₀ = 1+ϑ
    kl  = KullbackLeibler()
    ϕ₀  = evaluate(kl, [u₀])
    ϕ¹₀ = gradient(kl, u₀)
    ϕ²₀ = hessian(kl, u₀)
    r = zero(T)
    onet = one(T)
    n = length(a)::Int
    @inbounds for i = 1:n
        ui = a[i]
        if ui >= u₀
            r += ϕ₀ + ϕ¹₀*(ui-u₀) + .5*ϕ²₀*(ui-u₀)^2
        elseif ui>0 && ui<u₀
            r += ui*log(ui) - ui + onet
        else
            r = +Inf
            break
        end
    end
    r
end

################################################################################
## gradient
################################################################################
function gradient{T <: FloatingPoint}(dist::MET, a::T, b::T)
    ϑ  = dist.ϑ
    u₀ = 1+ϑ
    kl  = KullbackLeibler()
    ϕ₀  = evaluate(kl, [u₀])
    ϕ¹₀ = gradient(kl, u₀)
    ϕ²₀ = hessian(kl, u₀)
    if a > 0 && b > 0
        ui = a/b
        if ui > u₀
           u = (ϕ¹₀ + ϕ²₀*(ui-u₀))*b
        elseif ui>0 && ui<=u₀
           u = log(ui)
        end
    else
        u = -Inf
    end
    u
end

function gradient{T <: FloatingPoint}(dist::MET, a::T)
    ϑ  = dist.ϑ
    u₀ = 1+ϑ
    kl  = KullbackLeibler()
    ϕ¹₀ = gradient(kl, u₀)
    ϕ²₀ = hessian(kl, u₀)
    if a >= u₀
        u =  ϕ¹₀ + ϕ²₀*(a-u₀)
    elseif a>0 && a<u₀
        u = log(a)
    else
        u = +Inf
    end
    u
end

function gradient!{T <: FloatingPoint}(u::Vector{T}, dist::MET, a::AbstractVector{T}, b::AbstractVector{T})
    n = get_common_len(a, b)::Int
    @inbounds for i = 1 : n
        ai = a[i]
        bi = b[i]
        u[i] = gradient(dist, ai, bi)
    end
    u
end

function gradient!{T <: FloatingPoint}(u::Vector{T}, dist::MET, a::AbstractVector{T})
    n = length(a)::Int
    @inbounds for i = 1:n
        ai   = a[i]
        u[i] = gradient(dist, ai)
    end
    u
end

################################################################################
## hessian
################################################################################
function hessian{T <: FloatingPoint}(dist::MET, a::T)
    ϑ  = dist.ϑ
    u₀ = 1+ϑ
    kl  = KullbackLeibler()
    ϕ²₀ = hessian(kl, u₀)
    onet = one(T)
    r    = zero(T)
    if a >= u₀
       u  = ϕ²₀
    elseif a>0 && a<u₀
        u = onet/a
    else
        u = +Inf
    end
    u
end

function hessian{T <: FloatingPoint}(dist::MET, a::T, b::T)
    ϑ  = dist.ϑ
    u₀ = 1+ϑ
    kl  = KullbackLeibler()
    ϕ²₀ = hessian(kl, u₀)
    onet = one(T)
    r    = zero(T)
    if a > 0 && b > 0
        ui = a/b
        if ui >= u₀
            u  = ϕ²₀*b
        elseif ui>0 && ui<u₀
            u = onet/a
        end
    else
        u = +Inf
    end
    u
end

function hessian!{T <: FloatingPoint}(u::Vector{T}, dist::MET, a::AbstractVector{T}, b::AbstractVector{T})
    n = get_common_len(a, b)::Int
    onet = one(T)
    r    = zero(T)
    @inbounds for i = 1 : n
        ai = a[i]
        bi = b[i]
        u[i] = hessian(dist, ai, bi)
    end
    u
end

function hessian!{T <: FloatingPoint}(u::Vector{T}, dist::MET, a::AbstractVector{T})
    n = length(a)::Int
    onet = one(T)
    r    = zero(T)

    @inbounds for i = 1:n
        ai   = a[i]
        u[i] = hessian(dist, ai)
    end
    u
end
