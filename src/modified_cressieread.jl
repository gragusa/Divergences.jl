################################################################################
## Modified Cressie Read
################################################################################

#=---------------
evaluate
---------------=#
function evaluate(div::MCR, a::AbstractVector{T}, b::AbstractVector{T}) where T <: AbstractFloat

    if length(a) != length(b)
        throw(DimensionMismatch("first array has length $(length(a)) which does not match the length of the second, $(length(b))."))
    end

    r = zero(T)

    α  =  div.α
    ϑ   = div.ϑ
    f0, f1, f2, uϑ = div.m

    for i = eachindex(a, b)
        @inbounds ai = a[i]
        @inbounds bi = b[i]
        @inbounds ui = ai/bi
        if ui>=uϑ
      		  r += (f0 + f1*(ui-uϑ) + .5*f2*(ui-uϑ)^2)*bi
        elseif ui > 0 && ui < u₀
            r += ( (ui^(1+α)-1)/(α*(1.0 + α)) - (ui-1)/α )*bi
        elseif ui == 0
            r += bi/(1.0 + α)
        else
            r = +Inf
            break
        end
    end
    return r
end

function evaluate(div::MCR, a::AbstractVector{T}) where T <: AbstractFloat
    r = zero(T)
    α  =  div.α
    ϑ   = div.ϑ
    f0, f1, f2, uϑ = div.m

    @inbounds for i = eachindex(a)
        ui = a[i]
        if ui>=uϑ
      		r += f0 + f1*(ui-uϑ) + .5*f2*(ui-uϑ)^2
        elseif ui > 0 && ui < uϑ
            r += ( (ui^(1+α)-1)/(α*(1.0 + α)) - (ui-1)/α )
        elseif ui==0
            r += 1/(1.0 + α)
        else
            r = +Inf
            break
        end
    end
    return r
end

#=---------------
gradient
---------------=#
function gradient(div::MCR, a::T) where T <: AbstractFloat
    α  =  div.α
    ϑ  = div.ϑ
    f0, f1, f2, uϑ = div.m
    if a>=uϑ
        u = f1 + f2*(a-uϑ)
    else
        u = gradient(div.d, a)
    end
    return u
end

function gradient(div::MCR, a::T, b::T) where T <: AbstractFloat
    α  =  div.α
    ϑ  = div.ϑ
    f0, f1, f2, uϑ = div.m
    ui   = a/b
    if ui > uϑ
        u = f1 + f2*(ui-uϑ)
    else
        u = gradient(div.d, a, b)
    end
    return u
end

#=---------------
hessian
---------------=#
function hessian(div::MCR, a::T) where T <: AbstractFloat
    f0, f1, f2, uϑ = div.m
    if a >= uϑ
        u = f2
    else
        u = hessian(div.d, a)
    end
    return u
end

function hessian(div::MCR, a::T, b::T) where T <: AbstractFloat
    f0, f1, f2, uϑ = div.m
    ui = a/b
    if ui > uϑ
        u = f2/b
    else
        u = hessian(div.d, a, b)
    end
    return u
end

################################################################################
## Fully Modified Cressie Read
################################################################################

#=---------------
evaluate
---------------=#
function evaluate(div::FMCR, a::AbstractVector{T}, b::AbstractVector{T}) where T <: AbstractFloat

    if length(a) != length(b)
        throw(DimensionMismatch("first array has length $(length(a)) which does not match the length of the second, $(length(b))."))
    end

    r = zero(T)
    f0, f1, f2, uϑ, g0, g1, g2, uφ = div.m

    for i = eachindex(a, b)
        @inbounds ai = a[i]
        @inbounds bi = b[i]
        @inbounds ui = ai/bi
        if ui >= uϑ
      		  r += (f0 + f1*(ui-uϑ) + .5*f2*(ui-uϑ)^2)*bi
        elseif ui <= uφ
            r += (g0 + g1*(ui-uφ) + .5*g2*(ui-uφ)^2)*bi
        else
            r += ( (ui^(1+α)-1)/(α*(1.0 + α)) - (ui-1)/α )*bi
        end
    end
    return r
end

function evaluate(div::FMCR, a::AbstractVector{T}) where T <: AbstractFloat
    r = zero(T)
    f0, f1, f2, uϑ, g0, g1, g2, uφ = div.m

    for i = eachindex(a)
        @inbounds ui = a[i]
        if ui >= uϑ
      		  r += (f0 + f1*(ui-uϑ) + .5*f2*(ui-uϑ)^2)
        elseif ui <= uφ
            r += (g0 + g1*(ui-uφ) + .5*g2*(ui-uφ)^2)
        else
            r += ( (ui^(1+α)-1)/(α*(1.0 + α)) - (ui-1)/α )
        end
    end
    return r
end

#=---------------
gradient
---------------=#
function gradient(div::FMCR, a::T) where T <: AbstractFloat
    f0, f1, f2, uϑ, g0, g1, g2, uφ = div.m
    if a >= uϑ
        u = f1 + f2*(a-uϑ)
    elseif a <= uφ
        u = g1 + g2*(a-uφ)
    else
        u = gradient(div.d, a)
    end
    return u
end

function gradient(div::FMCR, a::T, b::T) where T <: AbstractFloat
    f0, f1, f2, uϑ, g0, g1, g2, uφ = div.m
    ui = a/b
    if ui >= uϑ
        u = f1 + f2*(ui-uϑ)
    elseif ui <= uφ
        u = g1 + g2*(ui-uφ)
    else
        u = gradient(div.d, a, b)
    end
    return u
end

#=---------------
hessian
---------------=#
function hessian(div::FMCR, a::T) where T <: AbstractFloat
    f0, f1, f2, uϑ, g0, g1, g2, uφ = div.m
    if a >= uϑ
        u = f2
    elseif a <= uφ
        u = g2
    else
        u = hessian(div.d, a)
    end
    return u
end

function hessian(div::FMCR, a::T, b::T) where T <: AbstractFloat
    f0, f1, f2, uϑ, g0, g1, g2, uφ = div.m
    ui = a/b
    if ui >= uϑ
        u = f2/b
    elseif ui <= uφ
        u = g2/b
    else
        u = hessian(div.d, a, b)
    end
    return u
end
