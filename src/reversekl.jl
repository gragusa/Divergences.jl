################################################################################
## ReverseKullbackLeibler
################################################################################

#=---------------
Evaluate
---------------=#
function evaluate(div::RKL, a::AbstractVector{T}, b::AbstractVector{T}) where T <: AbstractFloat
    if length(a) != length(b)
        throw(DimensionMismatch("first array has length $(length(a)) which does not match the length of the second, $(length(b))."))
    end

    r = zero(T)

    for i in eachindex(a, b)
        @inbounds ai = a[i]
        @inbounds bi = b[i]
        ui = ai/bi
        if ui > 0
            r += -bi*log(ui) + ai - bi
        else
            r = convert(T, Inf)
            break
        end
    end
    return r
end

function evaluate(div::RKL, a::AbstractVector{T}) where T <: AbstractFloat
    r = zero(T)
    for i in eachindex(a)
        @inbounds ai = a[i]
        if ai > 0
            r += - logmxp1(ai)
        else
            r = convert(T, Inf)
            break
        end
    end
    r
end

#=---------------
Gradient
---------------=#
function gradient(div::RKL, a::T, b::T) where T <: AbstractFloat
    if a > 0 && b > 0
        u = - b/a + 1.0
    else
        u = convert(T, Inf)
    end
    return u
end

function gradient(div::RKL, a::T) where T <: AbstractFloat
    if a > 0
        u = - 1.0 / a + 1.0
    else
        u = convert(T, Inf)
    end
    return u
end

#=---------------
hessian
---------------=#
function hessian(div::RKL, a::T, b::T) where T <: AbstractFloat
    if a > 0 && b > 0
        u = b/a^2
    else
        u = convert(T, Inf)
    end
    return u
end

function hessian(div::RKL, a::T) where T <: AbstractFloat
    if a > 0
        u = 1.0/a^2
    else
        u = convert(T, Inf)
    end
    u
end

################################################################################
## Modified Reverse Kullback-Leibler - MRKL
################################################################################

#=---------------
Evaluate
---------------=#
function evaluate(div::MRKL, a::AbstractVector{T}, b::AbstractVector{T}) where T <: AbstractFloat
    if length(a) != length(b)
        throw(DimensionMismatch("first array has length $(length(a)) which does not match the length of the second, $(length(b))."))
    end
    f0, f1, f2, uϑ = div.m
    r = zero(T)
    for i = eachindex(a, b)
        @inbounds ai = a[i]
        @inbounds bi = b[i]
        ui = ai/bi
        if ui >= uϑ
            r += (f0 + f1*(ui-uϑ) + .5*f2*(ui-uϑ)^2)*bi
        elseif ui > 0 && ui <u₀
            r += -bi*log(ui) + ai - bi
        else
            r = convert(T, Inf)
            break
        end
    end
    return r
end

function evaluate(div::MRKL, a::AbstractVector{T}) where T <: AbstractFloat
    f0, f1, f2, uϑ = div.m
    r = zero(T)
    for i in eachindex(a)
        @inbounds ai = a[i]
        if ai >= uϑ
            r += f0 + f1*(ai-uϑ) + .5*f2*(ai-uϑ)^2
        elseif ai > 0 && ai < uϑ
            r += -log(ai) + ai - 1.0
        else
            r = convert(T, Inf)
            break
        end
    end
    return r
end

#=---------------
gradient
---------------=#
function gradient(div::MRKL, a::T) where T <: AbstractFloat
    f0, f1, f2, uϑ = div.m
    ui = a
    if ui > uϑ
        u = (f1 + f2*(ui-uϑ))
    elseif ui <= uϑ
        u = gradient(div.d, a)
    end
    return u
end

function gradient(div::MRKL, a::T, b::T) where T <: AbstractFloat
    f0, f1, f2, uϑ = div.m
    ui = a/b
    if ui > u₀
        u = f1 + f2*(ui-uϑ)
    elseif ui <= uϑ
        u = gradient(div.d, a, b)
    end
    return u
end

#=---------------
hessian
---------------=#
function hessian(div::MRKL, a::T) where T <: AbstractFloat
    f0, f1, f2, uϑ = div.m
    if a >= uϑ
       u  = f2
    else
       u = hessian(div.d, a)
    end
    return u
end

function hessian(div::MRKL, a::T, b::T) where T <: AbstractFloat
    f0, f1, f2, uϑ = div.m
    ui = a/b
    if ui >= u₀
        u  = f2/b
    else
        u = hessian(rkl, a, b)
    end
    return u
end

################################################################################
## Fully Modified Reverse Kullback-Leibler - FMRKL
##
################################################################################

#=---------------
Evaluate
---------------=#
function evaluate(div::FMRKL, a::AbstractVector{T}, b::AbstractVector{T}) where T <: AbstractFloat
    if length(a) != length(b)
        throw(DimensionMismatch("first array has length $(length(a)) which does not match the length of the second, $(length(b))."))
    end
    r = zero(T)
    f0, f1, f2, uϑ, g0, g1, g2, uφ = div.m
    @inbounds for i = eachindex(a, b)
        ai = a[i]
        bi = a[i]
        ui = ai/bi
        if ui >= uϑ
            r += (f0 + f1*(ui-uϑ) + .5*f2*(ui-uϑ)^2)*bi
        elseif ui <= uφ
            r += (g0 + g1*(ui-uφ) + .5*g2*(ui-uφ)^2)*bi
        else
            r += -bi*log(ui) + ai - bi
        end
    end
    return r
end

function evaluate(div::FMRKL, a::AbstractVector{T}) where T <: AbstractFloat
    r = zero(T)
    f0, f1, f2, uϑ, g0, g1, g2, uφ = div.m
    @inbounds for i = eachindex(a)
        ui = a[i]
        if ui >= uϑ
            r += (f0 + f1*(ui-uϑ) + .5*f2*(ui-uϑ)^2)
        elseif ui <= uφ
            r += (g0 + g1*(ui-uφ) + .5*g2*(ui-uφ)^2)
        else
            r += -log(ui) + ui - 1.0
        end
    end
    return r
end

#=---------------
gradient
---------------=#
function gradient(div::FMRKL, a::T, b::T) where T <: AbstractFloat
    f0, f1, f2, uϑ, g0, g1, g2, uφ = div.m
    @inbounds for i = eachindex(a, b)
        ai = a[i]
        bi = a[i]
        ui = ai/bi
        if ui >= uϑ
            u = f1 + f2*(ui-uϑ)
        elseif ui <= uφ
            u = g1 + g2*(ui-uφ)
        else
            u = 1-bi/ai
        end
    end
    return u
end


function gradient(div::FMRKL, a::T) where T <: AbstractFloat
    f0, f1, f2, uϑ, g0, g1, g2, uφ = div.m
    if a >= uϑ
        u = (f1 + f2*(a-uϑ))
    elseif a <= uφ
        u = (g1 + g2*(a-uφ))
    else
        u = 1.0 - 1.0/a
    end
    return u
end

#=---------------
hessian
---------------=#
function hessian(div::FMRKL, a::T) where T <: AbstractFloat
    f0, f1, f2, uϑ, g0, g1, g2, uφ = div.m
    if a >= uϑ
        u  = f2
    elseif a <= uφ
        u  = g2
    else
        u = 1/(a*a)
    end
    return u
end

function hessian(div::FMRKL, a::T, b::T) where T <: AbstractFloat
    f0, f1, f2, uϑ, g0, g1, g2, uφ = div.m
    ui = a/b
    if ui >= uϑ
        u  = f2/b
    elseif ui <= uφ
        u  = g2/b
    else
        u = b/(a*a)
    end
    return u
end
