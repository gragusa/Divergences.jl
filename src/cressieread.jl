abstract Divergence <: PreMetric

type CressieRead <: Divergence
    α::Real

    function CressieRead(α::Real)
        @assert isempty(findin(α, [-1, 0])) "CressieRead is defined for all α!={-1,0}."
        new(α)
    end
end

type KullbackLeibler  <: Divergence end
    type ReverseKullbackLeibler <: Divergence end

        typealias CR CressieRead
        typealias EL ReverseKullbackLeibler
        typealias ET KullbackLeibler

cue() = CressieRead(1) ## ???
hd()  = CressieRead(-1/2)

################################################################################
## Cressie Read Divergence
## 
## ==> \gamma(a/b)b
## ==> \gamma(u)
################################################################################

################################################################################
## evaluate
################################################################################
function evaluate{T<:FloatingPoint}(dist::CressieRead, a::AbstractVector{T})
    α = dist.α
    onet = one(T)
    aexp = (onet+α)
    const aa = onet/(α*aexp)
    const ua = onet/α
    const pa = onet/aexp
    r = zero(T)
    n = length(a)::Int64
    @inbounds for i = 1:n
        ai = a[i]
        if ai > 0
            r += (ai^aexp-onet)*aa-ua*ai+ua
        elseif ai==0
            r += pa
        else
            r = +Inf
            break
        end
    end
    return r
end

function evaluate{T<:FloatingPoint}(dist::CressieRead, a::AbstractVector{T}, b::AbstractVector{T})
    α = dist.α
    const aa = 1/(α*(α+1))
    const ua = 1/α
    const pa = 1/(1+α)
    r = zero(T)
    n = get_common_len(a, b)::Int
    @inbounds for i = 1 : n
        ai = a[i]
        bi = b[i]
        ui = ai/bi
        if ui > 0
            r += ( (ui^(1+α)-1)*aa-ua*ui+ua )*bi
        elseif ui==0
            r += pa
        else
            r = +Inf
            break
        end
    end
    return r
end


function evaluate{T<:FloatingPoint}(dist::CressieRead, a::T, b::T)
    α = dist.α
    const aa = 1/(α*(α+1))
    const ua = 1/α
    const pa = 1/(1+α)
    r = zero(T)
    n = get_common_len(a, b)::Int
    u = a/b
    if u > 0
        return ((u^(1+α)-1)*aa-ua*ui+ua)*b
    elseif u==0
        return pa
    else
        return +Inf            
    end
end

################################################################################
## Gradient
################################################################################
function gradient!{T<:FloatingPoint}(u::Vector{T}, dist::CressieRead, a::AbstractVector{T}, b::AbstractVector{T})
    α = dist.α
    onet = one(T)
    r    = zero(T)
    n = get_common_len(a, b)::Int
    @inbounds for i = 1:n
        ai = a[i]
        ui = ai/bi[i]
        if ui > 0
            u[i] = ( (ui^α)-onet )/α
        elseif ui==0
            if a>0
                u[i] = -1/α
            else 
                u[i] = -Inf
            end 
        else
            u[i] = -Inf
        end
    end 
end 

function gradient{T<:FloatingPoint}(dist::CressieRead, a::T, b::T)
    α = dist.α
    onet = one(T)
    r    = zero(T)    
    
    if a > 0 && b > 0
        u = ( (a/b)^α-onet )/α
    elseif a == 0 && b != 0
        if α>0
            u = -1/α
        else
            u = -Inf       
        end
    else
        u = -Inf
    end
    return u
end 

function gradient!{T<:FloatingPoint}(u::Vector{T}, dist::CressieRead, a::AbstractVector{T})
    α  = dist.α
    r  = zero(T)
    onet = one(T)
    n = length(a)::Int
    @inbounds for i = 1:n
        ai = a[i]        
        if ai>0
            u[i] = (ai^α-onet)/α
        elseif ai==0
            if α>0
                u[i] = -1/α
            else 
                u[i] = -Inf
            end 
        else ai<0
            u[i] = -Inf
        end
    end
end

function gradient{T<:FloatingPoint}(dist::CressieRead, a::T)
    α  = dist.α
    r  = zero(T)
    onet = one(T)
    if a > 0
        return ( (a^α)-onet )/α
    elseif a==0
        if α>0
            return -1/α
        else 
            return -Inf
        end 
    else
        return -Inf
    end
end 

################################################################################
## Hessian
################################################################################

function hessian!{T<:FloatingPoint}(u::Vector{T}, dist::CressieRead, a::AbstractVector{T}, b::AbstractVector{T})
    α = dist.α
    r  = zero(T)
    onet = one(T)
    aexp = α-onet
    n = get_common_len(a, b)::Int
    @inbounds for i = 1:n
        ai = a[i]
        bi = b[i]
        ui = ai/bi        
        if ui > 0
            u[i] = ui^aexp/bi
        elseif ui==0
            if α>1
                u[i] = r
            else 
                u[i] = +Inf
            end 
        else
            u[i] = +Inf
        end
    end
end



function hessian!{T<:FloatingPoint}(u::Vector{T}, dist::CressieRead, a::AbstractVector{T})
    α = dist.α
    r  = zero(T)
    onet = one(T)
    aexp = α-onet
    n = get_common_len(a, b)::Int
    @inbounds for i = 1:n
        ai = a[i]
        if ai > 0
            u[i] = ai^aexp
        elseif ai==0
            if α>1
                u[i] = r
            else 
                u[i] = +Inf
            end             
        else
            u[i] = +Inf
        end
    end
end


function hessian{T<:FloatingPoint}(dist::CressieRead, a::T)
    α    = dist.α
    r    = zero(T)
    onet = one(T)
    aexp = α-onet
    if a > 0
        u = a^aexp
    elseif a==0
        if α>1
            return r
        else 
            return +Inf
        end         
    else
        return +Inf
    end
end

################################################################################
## ReverseKullbackLeibler
## 
## ==> \gamma(a/b)b
## ==> \gamma(u)
################################################################################
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
    n = length(a)
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

################################################################################
## KullbackLeibler
## 
## ==> \gamma(a/b)b
## ==> \gamma(u)
################################################################################
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
    n = length(a)::Int
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

function hessian!{T<:FloatingPoint}(u::Vector{T}, dist::KullbackLeibler, a::AbstractVector{T}, b::AbstractVector{T})
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

function hessian{T<:FloatingPoint}(dist::KullbackLeibler, a::T, b::T)
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

function hessian!{T<:FloatingPoint}(u::Vector{T}, dist::KullbackLeibler, a::AbstractVector{T})
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

function hessian{T<:FloatingPoint}(dist::KullbackLeibler, a::T)
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
function gradient{T<:FloatingPoint}(dist::Divergence, a::AbstractVector{T}, b::AbstractVector{T})
    n = get_common_len(a, b)::Int
    return gradient!(Array(T, n), dist, a, b)
end

function gradient{T<:FloatingPoint}(dist::Divergence, a::T, b::T)
    return gradient!(Array(T, 1), dist, a, b)
end

function gradient{T<:FloatingPoint}(dist::Divergence, a::T)
    return gradient!(Array(T, 1), dist,  a)
end

function hessian{T<:FloatingPoint}(dist::Divergence, a::AbstractVector{T}, b::AbstractVector{T})
    n = get_common_len(a, b)::Int
    return hessian!(Array(T, n), dist, a, b)
end

function hessian{T<:FloatingPoint}(dist::Divergence, a::T, b::T)
    return hessian!(Array(T, 1), dist, a, b)
end

function hessian{T<:FloatingPoint}(dist::Divergence, a::T)
    return hessian!(Array(T, 1), dist, a)
end