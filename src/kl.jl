################################################################################
## Kullback-Leibler - ET
##
## ==> γ(a/b)b
## ==> γ(u)
################################################################################
function evaluate{T <: AbstractFloat}(dist::ET, a::AbstractVector{T}, b::AbstractVector{T})
    if length(a) != length(b)
        throw(DimensionMismatch("first array has length $(length(a)) which does not match the length of the second, $(length(b))."))
    end
    r = zero(T)
    for i = eachindex(a, b)
        @inbounds ai = a[i]
        @inbounds bi = b[i]
        ui = ai / bi
        r += ai * log(ui) - ai + bi
    end
    r
end

function evaluate{T <: AbstractFloat}(dist::ET, a::AbstractVector{T})
    r = zero(T)
    onet = one(T)
    for i = eachindex(a)
        @inbounds ai = a[i]
        r += xlogx(ai) - ai + onet
    end
    r
end

################################################################################
## gradient
################################################################################
function gradient{T <: AbstractFloat}(dist::ET, a::T, b::T)
    ## This is the derivative of
    ## γ(a/b) with respect to a
    if b<=0
        u = convert(T, Inf)
    elseif a > 0 && b > 0
        u = log(a/b)
    else
        u = convert(T, -Inf)
    end
    u
end

function gradient{T <: AbstractFloat}(dist::ET, a::T)
    if a > 0
        u = log(a)
    else
        u = convert(T, -Inf)
    end
    u
end

function gradient!{T <: AbstractFloat}(u::Vector{T}, dist::ET, a::AbstractVector{T}, b::AbstractVector{T})
    if length(a) != length(b)
        throw(DimensionMismatch("first array has length $(length(a)) which does not match the length of the second, $(length(b))."))
    end

    for i = eachindex(a, b)
        @inbounds u[i] = gradient(dist, a[i], b[i])
    end
    u
end

function gradient!{T <: AbstractFloat}(u::Vector{T}, dist::ET, a::AbstractVector{T})
    for i = eachindex(a)
        @inbounds u[i] = gradient(dist, a[i])
    end
    u
end

################################################################################
## hessian
################################################################################
function hessian{T <: AbstractFloat}(dist::ET, a::T, b::T)
    ι = one(T)
    r    = zero(T)
    if b==0
        u = convert(T, Inf)
    elseif a > 0 && b > 0
        u = ι/(b*a)
    else
        u = convert(T, Inf)
    end
    u
end

function hessian{T <: AbstractFloat}(dist::ET, a::T)
    onet = one(T)
    r    = zero(T)
    infty = convert(T, Inf)
    if a > 0
        u = onet/a
    elseif a==0
        u = infty
    end
    u
end

function hessian!{T <: AbstractFloat}(u::Vector{T}, dist::ET, a::AbstractVector{T}, b::AbstractVector{T})
    if length(a) != length(b)
        throw(DimensionMismatch("first array has length $(length(a)) which does not match the length of the second, $(length(b))."))
    end
    @inbounds for i = eachindex(a,b)
        ai = a[i]
        bi = b[i]
        u[i] = hessian(dist, ai, bi)
    end
    u
end

function hessian!{T <: AbstractFloat}(u::Vector{T}, dist::ET, a::AbstractVector{T})
    @inbounds for i = 1:eachindex(a)
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
function evaluate{T <: AbstractFloat}(dist::MET, a::AbstractVector{T}, b::AbstractVector{T})
    if length(a) != length(b)
        throw(DimensionMismatch("first array has length $(length(a)) which does not match the length of the second, $(length(b))."))
    end

    ϑ  = dist.ϑ
    u₀ = 1+ϑ
    kl  = KullbackLeibler()
    ϕ₀  = evaluate(kl, [u₀])
    ϕ¹₀ = gradient(kl, u₀)
    ϕ²₀ = hessian(kl, u₀)
    ι = one(T)
    r = zero(T)
    @inbounds for i = eachindex(a,b)
        ai = a[i]
        bi = a[i]
        ui = ai/bi
        if ui >= u₀
            r += (ϕ₀ + ϕ¹₀*(ui-u₀) + .5*ϕ²₀*(ui-u₀)^2)*bi
        elseif ui > 0 && ui <u₀
            r += (ui*log(ui) - ui + ι)*bi
        else
            r = oftype(a, Inf)
        end
    end
    r
end

function evaluate{T <: AbstractFloat}(dist::MET, a::AbstractVector{T})
    ϑ  = dist.ϑ
    u₀ = 1+ϑ
    kl  = KullbackLeibler()
    ϕ₀  = evaluate(kl, [u₀])
    ϕ¹₀ = gradient(kl, u₀)
    ϕ²₀ = hessian(kl, u₀)
    r = zero(T)
    ι = one(T)
    @inbounds for i = eachindex(a)
        ui = a[i]
        if ui >= u₀
            r += ϕ₀ + ϕ¹₀*(ui-u₀) + .5*ϕ²₀*(ui-u₀)^2
        elseif ui>0 && ui<u₀
            r += ui*log(ui) - ui + ι
        else
            r = oftype(ui, Inf)
            break
        end
    end
    r
end

################################################################################
## gradient
################################################################################
function gradient{T <: AbstractFloat}(dist::MET, a::T, b::T)
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
        u = oftype(a, -Inf)
    end
    u
end

function gradient{T <: AbstractFloat}(dist::MET, a::T)
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
        u = oftype(a, Inf)
    end
    u
end

function gradient!{T <: AbstractFloat}(u::Vector{T}, dist::MET, a::AbstractVector{T}, b::AbstractVector{T})
    if length(a) != length(b)
        throw(DimensionMismatch("first array has length $(length(a)) which does not match the length of the second, $(length(b))."))
    end
    @inbounds for i = eachindex(a, b)
        ai = a[i]
        bi = b[i]
        u[i] = gradient(dist, ai, bi)
    end
    u
end

function gradient!{T <: AbstractFloat}(u::Vector{T}, dist::MET, a::AbstractVector{T})
    @inbounds for i = eachindex(a)
        ai   = a[i]
        u[i] = gradient(dist, ai)
    end
    u
end

################################################################################
## hessian
################################################################################
function hessian{T <: AbstractFloat}(dist::MET, a::T)
    ϑ  = dist.ϑ
    u₀ = 1+ϑ
    kl  = KullbackLeibler()
    ϕ²₀ = hessian(kl, u₀)
    ι = one(T)
    if a >= u₀
       u  = ϕ²₀
    elseif a>0 && a<u₀
        u = ι/a
    else
        u = convert(T, Inf)
    end
    u
end

function hessian{T <: AbstractFloat}(dist::MET, a::T, b::T)
    ϑ  = dist.ϑ
    u₀ = 1+ϑ
    kl  = KullbackLeibler()
    ϕ²₀ = hessian(kl, u₀)
    ι = one(T)
    if a > 0 && b > 0
        ui = a/b
        if ui >= u₀
            u  = ϕ²₀*b
        elseif ui>0 && ui<u₀
            u = ι/a
        end
    else
        u = convert(T, Inf)
    end
    u
end

function hessian!{T <: AbstractFloat}(u::Vector{T}, dist::MET, a::AbstractVector{T}, b::AbstractVector{T})
    if length(a) != length(b)
        throw(DimensionMismatch("first array has length $(length(a)) which does not match the length of the second, $(length(b))."))
    end

    for i = eachindex(a, b)
        @inbounds u[i] = hessian(dist, a[i], b[i])
    end
    u
end

function hessian!{T <: AbstractFloat}(u::Vector{T}, dist::MET, a::AbstractVector{T})
    r    = zero(T)
    for i = eachindex(a)
        @inbounds u[i] = hessian(dist, a[i])
    end
    u
end
