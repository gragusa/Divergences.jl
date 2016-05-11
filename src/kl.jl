################################################################################
## Kullback-Leibler - KL
################################################################################

#=---------------
Evaluate
---------------=#
function evaluate{T <: AbstractFloat}(div::KL, a::AbstractVector{T}, b::AbstractVector{T})
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

function evaluate{T <: AbstractFloat}(div::KL, a::AbstractVector{T})
    r = zero(T)
    onet = one(T)
    for i = eachindex(a)
        @inbounds ai = a[i]
        r += xlogx(ai) - ai + onet
    end
    r
end

#=---------------
gradient
---------------=#
function gradient{T <: AbstractFloat}(div::KL, a::T, b::T)
    if b <= 0
        u = convert(T, -Inf)
    elseif a > 0 && b > 0
        u = log(a/b)
    else
        u = convert(T, -Inf)
    end
    u
end

function gradient{T <: AbstractFloat}(div::KL, a::T)
    if a > 0
        u = log(a)
    else
        u = convert(T, -Inf)
    end
    u
end

#=---------------
hessian
---------------=#
function hessian{T <: AbstractFloat}(div::KL, a::T, b::T)
    if b==0
        u = convert(T, Inf)
    elseif a > 0 && b > 0
        u = 1.0/a
    else
        u = convert(T, Inf)
    end
    u
end

function hessian{T <: AbstractFloat}(div::KL, a::T)
    r    = zero(T)
    if a > 0
        u = 1.0/a
    else
        u = convert(T, Inf)
    end
    u
end

################################################################################
## Modified Kullback-Leibler - MKL
################################################################################

#=---------------
Evaluate
---------------=#
function evaluate{T <: AbstractFloat}(div::MKL, a::AbstractVector{T}, b::AbstractVector{T})
    if length(a) != length(b)
        throw(DimensionMismatch("first array has length $(length(a)) which does not match the length of the second, $(length(b))."))
    end
    r = zero(T)
    ϑ   = div.ϑ
    f0, f1, f2, uϑ = div.m
    @inbounds for i = eachindex(a,b)
        ai = a[i]
        bi = a[i]
        ui = ai/bi
        if ui >= uϑ
            r += (f0 + f1*(ui-uϑ) + .5*f2*(ui-uϑ)^2)*bi
        elseif ui > 0 && ui < uϑ
            r += (ai*log(ui) - ai + bi)
        else
            r = oftype(a, Inf)
        end
    end
    r
end

function evaluate{T <: AbstractFloat}(div::MKL, a::AbstractVector{T})
    r = zero(T)
    ϑ   = div.ϑ
    f0, f1, f2, uϑ = div.m
    @inbounds for i = eachindex(a)
        ui = a[i]
        if ui >= uϑ
            r += f0 + f1*(ui-uϑ) + .5*f2*(ui-uϑ)^2
        elseif ui>0 && ui<uϑ
            r += ui*log(ui) - ui + 1.0
        else
            r = oftype(ui, Inf)
            break
        end
    end
    r
end

#=---------------
gradient
---------------=#
function gradient{T <: AbstractFloat}(div::MKL, a::T, b::T)

    ϑ   = div.ϑ
    f0, f1, f2, uϑ = div.m

    ui = a/b
    if ui > uϑ
        u = (f1 + f2*(ui-uϑ))*b
    else
        u = gradient(div.d, a, b)
    end
    return u
end

function gradient{T <: AbstractFloat}(div::MKL, a::T)

    ϑ   = div.ϑ
    f0, f1, f2, uϑ = div.m

    if a >= uϑ
        u =  f1 + f2*(a-uϑ)
    else
        u = gradient(div.d, a)
    end
    u
end

#=---------------
hessian
---------------=#
function hessian{T <: AbstractFloat}(div::MKL, a::T)

    ϑ   = div.ϑ
    f0, f1, f2, uϑ = div.m

    if a >= uϑ
        u  = f2
    else
        u = hessian(div.d, a)
    end
    return u
end

function hessian{T <: AbstractFloat}(div::MKL, a::T, b::T)

    ϑ   = div.ϑ
    f0, f1, f2, uϑ = div.m

    ui = a/b
    if ui >= uϑ
        u = f2*b
    else
        u = hessian(div.d, a, b)
    end
    return u
end

################################################################################
## Fully Modified Kullback-Leibler - FMKL
################################################################################

#=---------------
Evaluate
---------------=#
function evaluate{T <: AbstractFloat}(div::FMKL, a::AbstractVector{T}, b::AbstractVector{T})
    if length(a) != length(b)
        throw(DimensionMismatch("first array has length $(length(a)) which does not match the length of the second, $(length(b))."))
    end

    r = zero(T)


    ϑ   = div.ϑ
    f0, f1, f2, uϑ, g0, g1, g2, uφ = div.m

    @inbounds for i = eachindex(a,b)
        ai = a[i]
        bi = a[i]
        ui = ai/bi

        if ui >= uϑ
            r += (f0 + f1*(ui-uϑ) + .5*f2*(ui-uϑ)^2)*bi
        elseif ui <= uφ
            r += (g0 + g1*(ui-uφ) + .5*g2*(ui-uφ)^2)*bi
        else
            r += (ai*log(ui) - ai + bi)
        end
    end
    r
end

function evaluate{T <: AbstractFloat}(div::FMKL, a::AbstractVector{T})
    r = zero(T)


    ϑ   = div.ϑ
    f0, f1, f2, uϑ, g0, g1, g2, uφ = div.m

    @inbounds for i = eachindex(a,b)
        ai = a[i]
        if ai >= uϑ
            r += (f0 + f1*(ai-uϑ) + .5*f2*(ai-uϑ)^2)*bi
        elseif ai <= uφ
            r += (g0 + g1*(ai-uφ) + .5*g2*(ai-uφ)^2)*bi
        else
            r += (ai*log(ai) - ai + 1.0)
        end
    end
    return r
end

#=---------------
gradient
---------------=#
function gradient{T <: AbstractFloat}(div::FMKL, a::T, b::T)

    ϑ   = div.ϑ
    f0, f1, f2, uϑ, g0, g1, g2, uφ = div.m

    ui = a/b
    if ui > uϑ
        u = (f1 + f2*(ui-uϑ))*b
    elseif ui <= uφ
        u = (g1 + g2*(ui-uφ))*b
    else
        u = gradient(div.d, a, b)
    end
    return u
end

function gradient{T <: AbstractFloat}(div::FMKL, a::T)

    ϑ   = div.ϑ
    f0, f1, f2, uϑ, g0, g1, g2, uφ = div.m

    if a >= uϑ
        u =  f1 + f2*(a-uϑ)
    elseif a <= uφ
        u =  g1 + g2*(a-uφ)
    else
        u = gradient(div.d, a)
    end
    return u
end

#=---------------
hessian
---------------=#
function hessian{T <: AbstractFloat}(div::FMKL, a::T)

    ϑ   = div.ϑ
    f0, f1, f2, uϑ, g0, g1, g2, uφ = div.m

    if a >= uϑ
        u  = f2
    elseif a <= uφ
        u = g2
    else
        u = hessian(div.d, a)
    end
    return u
end

function hessian{T <: AbstractFloat}(div::FMKL, a::T, b::T)

    ϑ   = div.ϑ
    f0, f1, f2, uϑ, g0, g1, g2, uφ = div.m

    ui = a/b
    if ui >= uϑ
        u = f2*b
    elseif ui <= uφ
        u = g2*b
    else
        u = hessian(div.d, a, b)
    end
    return u
end
