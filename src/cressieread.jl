abstract Divergences <: PreMetric 

type CressieRead <: Divergences
    α::Real

    function CressieRead(a::Real)
        @assert isempty(findin(1, [-1, 0])) "CressieRead defined for all a!={-1,0}."
        new(CressieRead, a)
    end 
end


type KullbackLeibler  <: Divergences end
type ReverseKullbackLeibler <: Divergences end

typealias CR ReverseKullbackLeibler
typealias EL ReverseKullbackLeibler
typealias ET KullbackLeibler


# Cressie Read Divergence
## \int f \phi(g/f) dx

function evaluate{T<:FloatingPoint}(dist::CressieRead, a::AbstractVector{T}, b::AbstractVector{T})
    α = dist.α
    const aa = 1/(α*(α+1))
    const ua = 1/α
    const pa = 1/(1+α)
    r = zero(T)
    n = get_common_len(a, b)::Int
    for i = 1 : n
        @inbounds ai = a[i]
        @inbounds bi = b[i]
        @inbounds ui = ai/bi
        if ui > 0
            r += ( (ui^(1+α)-1)*aa-ua*ui+ua )*bi
        elseif ui==0
            r += pa
        else
            r = +Inf
            break            
        end
    end 
    r
end

function evaluate{T<:FloatingPoint}(dist::CressieRead, a::AbstractVector{T})
    α = dist.α
    onet = one(T)
    aexp = (onet+α)
    const aa = onet/(α*aexp)
    const ua = onet/α
    const pa = onet/aexp
    r = zero(T)
    n = lenght(a)::Int
    for i = 1 : n
        @inbounds ai = a[i]
        if ui > 0
            r += (ai^aexp-onet)*aa-ua*ui+ua
        elseif ui==0
            r += pa
        else
            r = +Inf
            break            
        end
    end 
    r
end


function gradient!{T<:FloatingPoint}(u::Vector{T}, dist::CressieRead, a::AbstractVector{T}, b::AbstractVector{T})
    α = dist.α
    onet = one(T)
    r    = zero(T)
    n = get_common_len(a, b)::Int
    for i = 1 : n
        @inbounds ai = a[i]
        @inbounds ui = ai/bi[i]

        if ui > 0
            u[i] = ( (ui^α)-onet )/α 
        elseif ui==0
            u[i] = r
        else
            u[i] = +Inf
        end
    end 
    u
end

function gradient{T<:FloatingPoint}(dist::CressieRead, a::T, b::T)
    α = dist.α
    onet = one(T)
    r    = zero(T)
        @inbounds ai = a[i]
        @inbounds ui = ai/bi[i]

    if a > 0 && b > 0 
        u = ( (a/b)^α-onet )/α 
    elseif a == 0 || b == 0 
        u = r
    else
        u = +Inf
    end 
    u
end



function gradient!{T<:FloatingPoint}(u::Vector{T}, dist::CressieRead, a::AbstractVector{T})
    α  = dist.α
    r  = zero(T)
    onet = one(T)
    n = length(a)::Int
    for i = 1 : n
        @inbounds ai = a[i]
        if ai > 0
            @inbounds u[i] = ( (ai^α)-onet )/α 
        elseif ai==0
            u[i] = r
        else
            u[i] = +Inf
        end
    end 
    u
end

function gradient{T<:FloatingPoint}(dist::CressieRead, a::T)
    α  = dist.α
    r  = zero(T)
    onet = one(T)
    if a > 0
        u = ( (a^α)-onet )/α 
    elseif a==0
        u = r
    else
        u = +Inf
    end
    u
end





function hessian!{T<:FloatingPoint}(u::Vector{T}, dist::CressieRead, a::AbstractVector{T}, b::AbstractVector{T})
    α = dist.α
    r  = zero(T)
    onet = one(T)

    n = get_common_len(a, b)::Int
    for i = 1 : n
        @inbounds ai = a[i]
        @inbounds bi = b[i]
        @inbounds ui = ai/bi
        if ui > 0
            u[i] = ui^α/ai 
        elseif ui==0
            u[i] = r
        else
            u[i] = +Inf
        end
    end 
    u
end

function hessian!{T<:FloatingPoint}(u::Vector{T}, dist::CressieRead, a::AbstractVector{T})
    α = dist.α
    r  = zero(T)    
    onet = one(T)
    aexp = α-onet
    n = get_common_len(a, b)::Int
    for i = 1 : n
        @inbounds ai = a[i]        
        if ai > 0
            @inbounds u[i] = ai^aexp 
        elseif ai==0
            @inbounds u[i] = r
        else
            @inbounds u[i] = +Inf
        end
    end 
    u
end


function hessian{T<:FloatingPoint}(dist::CressieRead, a::T)
    α = dist.α
    r  = zero(T)    
    onet = one(T)
    aexp = α-onet
    if a > 0
        u = a^aexp 
    elseif a==0
        u = r
    else
        u = +Inf
    end    
    return u
end




## ReverseKullbackLeibler 
function evaluate{T<:FloatingPoint}(dist::ReverseKullbackLeibler, a::AbstractVector{T}, b::AbstractVector{T})
    r = zero(T)
    n = get_common_len(a, b)::Int
    for i = 1 : n
        @inbounds ai = a[i]
        @inbounds bi = b[i]
        @inbounds ui = ai/bi
        if ui > 0
            r += (-log(ui) + ui -1)*bi
        else
            r = +Inf
            break            
        end
    end 
    r
end

function evaluate{T<:FloatingPoint}(dist::ReverseKullbackLeibler, a::AbstractVector{T})
    r = zero(T)
    onet = one(T)
    n = length(a)::Int
    for i = 1 : n
        @inbounds ai = a[i]
        if ai > 0
            r += -log(ai) + ai - onet
        else
            r = +Inf
            break            
        end
    end 
    r
end


function gradient!{T<:FloatingPoint}(u::Vector{T}, dist::ReverseKullbackLeibler, a::AbstractVector{T}, b::AbstractVector{T})
    n = get_common_len(a, b)::Int
    onet = one(T)
    for i = 1 : n
        @inbounds ai = a[i]
        @inbounds bi = b[i]
        @inbounds ui = bi/ai

        if ai > 0 && bi > 0
            u[i] = - ui + onet
        else
            u[i] = +Inf
        end
    end 
    u
end

function gradient{T<:FloatingPoint}(dist::ReverseKullbackLeibler, a::T, b::T)
    onet = one(T)

    if a > 0 && b > 0
        u = - a/b + onet
    else
        u = +Inf
    end 
    return u
end

function gradient!{T<:FloatingPoint}(u::Vector{T}, dist::ReverseKullbackLeibler, a::AbstractVector{T})
    n = lenght(a)::Int
    onet = one(T)
    for i = 1 : n
        @inbounds ai = a[i]
        if ai > 0
            @inbounds u[i] = -onet/ai + onet
        else
            @inbounds u[i] = +Inf
        end
    end 
    u
end

function gradient{T<:FloatingPoint}(dist::ReverseKullbackLeibler, a::T)
    onet = one(T)
    if a > 0
        u = -onet/a + onet
    else
        u = +Inf
    end
    return u
end

function hessian!{T<:FloatingPoint}(u::Vector{T}, dist::ReverseKullbackLeibler, a::AbstractVector{T}, b::AbstractVector{T})
    onet = one(T)
    n = get_common_len(a, b)::Int
    for i = 1 : n
        @inbounds ai = a[i]
        @inbounds bi = b[i]
        @inbounds ui = bi/ai^2
        if ai > 0 && bi > 0
           @inbounds u[i] = ui
        else
           @inbounds u[i] = +Inf
        end
    end 
    u
end

function hessian{T<:FloatingPoint}(dist::ReverseKullbackLeibler, a::T, b::T)
    onet = one(T)
    if a > 0 && b > 0
        u = bi/ai^2
    else
        u = +Inf
    end 
    u
end



function hessian!{T<:FloatingPoint}(u::Vector{T}, dist::ReverseKullbackLeibler, a::AbstractVector{T})
    onet = one(T)
    n = length(a)::Int
    for i = 1 : n
        @inbounds ai = a[i]
        if ai > 0 
            @inbounds u[i] = onet/ai^2
        else
            @inbounds u[i] = +Inf
        end
    end 
    u
end


function hessian{T<:FloatingPoint}(dist::ReverseKullbackLeibler, a::T)
    onet = one(T)
    if a > 0
        u = onet/a^2
    else
        u = +Inf
    end 
    u
end


## KullbackLeibler
function evaluate{T<:FloatingPoint}(dist::KullbackLeibler, a::AbstractVector{T}, b::AbstractVector{T})
    onet = one(T)
    r = zero(T)
    n = get_common_len(a, b)::Int
    for i = 1 : n
        @inbounds ai = a[i]
        @inbounds bi = a[i]
        @inbounds ui = ai/bi

        if ui > 0 
            r += ui*log(ui) - ui + onet
        else
            r = +Inf
            break            
        end
    end 
    r
end

function evaluate{T<:FloatingPoint}(dist::KullbackLeibler, a::AbstractVector{T})
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


function gradient!{T<:FloatingPoint}(u::Vector{T}, dist::KullbackLeibler, a::AbstractVector{T}, b::AbstractVector{T})
    n = get_common_len(a, b)::Int
    for i = 1 : n
        @inbounds ai = a[i]
        @inbounds bi = b[i]
        @inbounds ui = ai/bi

        if ui > 0
            u[i] = log(ui)
        else
            u[i] = +Inf
        end
    end 
    u
end

function gradient{T<:FloatingPoint}(dist::KullbackLeibler, a::T, b::T)
    if a > 0 && b > 0
        u = log(a/b)
    else
        u = -Inf
    end
    return u
end


function gradient!{T<:FloatingPoint}(u::Vector{T}, dist::KullbackLeibler, a::AbstractVector{T})
    n = lenght(a)::Int
    for i = 1 : n
        @inbounds ai = a[i]  
        if ai > 0
            u[i] = log(ai)
        else
            u[i] = -Inf
        end
    end 
    u
end

function gradient{T<:FloatingPoint}(dist::KullbackLeibler, a::T)
    if a > 0
        u = log(a)
    else
        u = -Inf
    end
    return u
end




function hessian!{T<:FloatingPoint}(u::Vector{T}, dist::ReverseKullbackLeibler, a::AbstractVector{T}, b::AbstractVector{T})
    n = get_common_len(a, b)::Int
    onet = one(T)
    r    = zero(T)
    for i = 1 : n
        @inbounds ai = a[i]
        @inbounds bi = b[i]

        if ai > 0 && bi > 0
            u[i] = onet/ai
        elseif ui==0
            u[i] = r
        else
            u[i] = +Inf
        end
    end 
    u
end

function hessian{T<:FloatingPoint}(dist::ReverseKullbackLeibler, a::T, b::T)
    onet = one(T)
    r    = zero(T)
    
    if a > 0 && b > 0
        u = onet/ai
    elseif ui==0
        u = r
    else
        u = +Inf
    end 
    return u
end





function hessian!{T<:FloatingPoint}(u::Vector{T}, dist::ReverseKullbackLeibler, a::AbstractVector{T})
    n = length(a)::Int
    onet = one(T)
    r    = zero(T)

    for i = 1 : n
        @inbounds ai = a[i]
        if ai > 0
            u[i] = onet/ai
        elseif ai==0
            u[i] = r
        else
            u[i] = +Inf
        end
    end 
    u
end

function hessian{T<:FloatingPoint}(dist::ReverseKullbackLeibler, a::T)
    onet = one(T)
    r    = zero(T)
    
    if a > 0 
        u = onet/a
    elseif a==0
        u = r
    else
        u = +Inf
    end 
    return u
end



##


function gradient{T<:FloatingPoint}(dist::Divergences, a::AbstractVector{T}, b::AbstractVector{T})
    n = get_common_len(a, b)::Int
    return gradient!(Array(T, n), dist, a, b)
end

function gradient{T<:FloatingPoint}(dist::Divergences, a::T, b::T)
    return gradient!(Array(T, 1), dist, a, b)
end

function gradient{T<:FloatingPoint}(dist::Divergences, a::T)
    return gradient!(Array(T, 1), dist,  a)
end



function hessian{T<:FloatingPoint}(dist::Divergences, a::AbstractVector{T}, b::AbstractVector{T})
    n = get_common_len(a, b)::Int
    return hessian!(Array(T, n), dist, a, b)
end

function hessian{T<:FloatingPoint}(dist::Divergences, a::T, b::T)
    return hessian!(Array(T, 1), dist, a, b)
end

function hessian{T<:FloatingPoint}(dist::Divergences, a::T)
    return hessian!(Array(T, 1), dist, a)
end

