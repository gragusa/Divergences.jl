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
        @inbounds ui = ai/b[i]
        if ui > 0
            r += (ui^(1+α)-1)*aa-ua*ui+ua
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
    const aa = 1/(α*(α+1))
    const ua = 1/α
    const pa = 1/(1+α)
    r = zero(T)
    n = lenght(a)::Int
    for i = 1 : n
        if ui > 0
        @inbounds r += (a[i]^(1+α)-1)*aa-ua*ui+ua
        elseif ui==0
            r += pa
        else
            r = +Inf
            break            
        end
    end 
    r
end





function gradient!{T<:FloatingPoint}(u, dist::CressieRead, a::AbstractVector{T}, b::AbstractVector{T})
    α = dist.α
    const aa = 1/(α*(α+1))
    const aa1a = aa*(1+α)
    n = get_common_len(a, b)::Int
    for i = 1 : n
        @inbounds ai = a[i]
        @inbounds bi = b[i]
        @inbounds ui = ai/bi

        if ui > 0
            u[i] = (ui^α)*aa1a/bi
        elseif ui==0
            u[i] = 0.0
        else
            u[i] = +Inf
        end
    end 
    u
end

function gradient!{T<:FloatingPoint}(u, dist::CressieRead, a::AbstractVector{T})
    a = dist.a
    const aa = 1/(a*(a+1))
    const aa1a = aa*(1+a)
    
    n = length(a)::Int
    for i = 1 : n
        @inbounds ai = a[i]
        if ai > 0
        @inbounds u[i] = (ai^a)*aa1a
        elseif ai==0
            u[i] = 0.0
        else
            u[i] = +Inf
        end
    end 
    u
end

function hessian!{T<:FloatingPoint}(u, dist::CressieRead, a::AbstractVector{T}, b::AbstractVector{T})
    α = dist.α
    const aa = 1/(α*(α+1))
    const aa1a = aa*(1+α)
    const aa1aa = a*aa*(1+α)
    n = get_common_len(a, b)::Int
    for i = 1 : n
        @inbounds ai = a[i]
        @inbounds bi = b[i]
        @inbounds ui = ai/bi
        if ui > 0
            u[i] = ((ui^(α-1))*aa1aa)/bi^2
        elseif ui==0
            u[i] = 0.0
        else
            u[i] = +Inf
        end
    end 
    u
end

function hessian!{T<:FloatingPoint}(u, dist::CressieRead, a::AbstractVector{T})
    α = dist.α
    const aa = 1/(α*(α+1))
    const aa1a = aa*(1+α)
    const aa1aa = a*aa*(1+α)
    n = get_common_len(a, b)::Int
    for i = 1 : n
        @inbounds ai = a[i]
        
        if ai > 0
            @inbounds u[i] = (ai^(α-1))*aa1aa
        elseif ai==0
            @inbounds u[i] = 0.0
        else
            @inbounds u[i] = +Inf
        end
    end 
    u
end




## RKLDivergence
function evaluate{T<:FloatingPoint}(dist::ReverseKullbackLeibler, a::AbstractVector{T}, b::AbstractVector{T})
    r = zero(T)
    n = get_common_len(a, b)::Int
    for i = 1 : n
        @inbounds ai = a[i]
        @inbounds ui = ai/b[i]
        if ui > 0
            r += -log(ui) + ui -1
        else
            r = +Inf
            break            
        end
    end 
    r
end

function evaluate{T<:FloatingPoint}(dist::ReverseKullbackLeibler, a::AbstractVector{T})
    r = zero(T)
    n = length(a)::Int
    for i = 1 : n
        @inbounds ai = a[i]
        if ai > 0
            r += -log(ai) + ai -1
        else
            r = +Inf
            break            
        end
    end 
    r
end


function gradient!{T<:FloatingPoint}(u, dist::ReverseKullbackLeibler, a::AbstractVector{T}, b::AbstractVector{T})
    n = get_common_len(a, b)::Int
    for i = 1 : n
        @inbounds ai = a[i]
        @inbounds bi = b[i]
        @inbounds ui = ai/bi

        if ui > 0
            u[i] = (1/ui + 1.0)/bi
        else
            u[i] = +Inf
        end
    end 
    u
end

function gradient!{T<:FloatingPoint}(u, dist::ReverseKullbackLeibler, a::AbstractVector{T})
    n = lenght(a)::Int
    for i = 1 : n
        @inbounds ai = a[i]
        if ai > 0
            @inbounds u[i] = 1/ai + 1.0
        else
            @inbounds u[i] = +Inf
        end
    end 
    u
end



function hessian!{T<:FloatingPoint}(u, dist::ReverseKullbackLeibler, a::AbstractVector{T}, b::AbstractVector{T})
    n = get_common_len(a, b)::Int
    for i = 1 : n
        @inbounds ai = a[i]
        if ai > 0
            u[i] = (-1/ui^2)/bi^2
        elseif ai==0
            u[i] = 0.0
        else
            u[i] = +Inf
        end
    end 
    u
end

## KLDivergence
function evaluate{T<:FloatingPoint}(dist::KullbackLeibler, a::AbstractVector{T}, b::AbstractVector{T})
    r = zero(T)
    n = get_common_len(a, b)::Int
    for i = 1 : n
        @inbounds ai = a[i]
        @inbounds ui = ai/b[i]

        if ui > 0
            r += ui*log(ui) - ui + 1
        else
            r = +Inf
            break            
        end
    end 
    r
end

function evaluate{T<:FloatingPoint}(dist::KullbackLeibler, a::AbstractVector{T})
    r = zero(T)
    n = length(a)::Int
    for i = 1 : n
        @inbounds ai = a[i]

        if ai > 0
            r += ai*log(ai) - ai + 1
        else
            r = +Inf
            break            
        end
    end 
    r
end


function gradient!{T<:FloatingPoint}(u, dist::ReverseKullbackLeibler, a::AbstractVector{T}, b::AbstractVector{T})
    n = get_common_len(a, b)::Int
    for i = 1 : n
        @inbounds ai = a[i]
        @inbounds bi = b[i]
        @inbounds ui = ai/bi

        if ui > 0
            u[i] = (log(ui))/bi
        else
            u[i] = +Inf
        end
    end 
    u
end

function gradient!{T<:FloatingPoint}(u, dist::ReverseKullbackLeibler, a::AbstractVector{T})
    n = lenght(a)::Int
    for i = 1 : n
        @inbounds ai = a[i]  
        if ai > 0
            u[i] = log(ai)
        else
            u[i] = +Inf
        end
    end 
    u
end


function hessian!{T<:FloatingPoint}(u, dist::ReverseKullbackLeibler, a::AbstractVector{T}, b::AbstractVector{T})
    n = get_common_len(a, b)::Int
    for i = 1 : n
        @inbounds ai = a[i]
        @inbounds bi = b[i]
        @inbounds ui = ai/bi
        if ui > 0
            u[i] = (1/ui)/bi^2
        elseif ui==0
            u[i] = 0.0
        else
            u[i] = +Inf
        end
    end 
    u
end

function hessian!{T<:FloatingPoint}(u, dist::ReverseKullbackLeibler, a::AbstractVector{T})
    n = get_common_len(a, b)::Int
    for i = 1 : n
        @inbounds ai = a[i]
        if ai > 0
            u[i] = 1/ai
        elseif ai==0
            u[i] = 0.0
        else
            u[i] = +Inf
        end
    end 
    u
end



##


function gradient{T<:FloatingPoint}(dist::Divergences, a::AbstractVector{T}, b::AbstractVector{T})
    n = get_common_len(a, b)::Int
    return gradient!(Array(T, n), a, b)
end

function hessian{T<:FloatingPoint}(dist::Divergences, a::AbstractVector{T}, b::AbstractVector{T})
    n = get_common_len(a, b)::Int
    return hessian!(Array(T, n), a, b)
end


function gradient{T<:FloatingPoint}(dist::Divergences, a::AbstractVector{T}, b::AbstractVector{T})
    n = get_common_len(a, b)::Int
    return gradient!(Array(T, n), a, b)
end

function hessian{T<:FloatingPoint}(dist::Divergences, a::AbstractVector{T}, b::AbstractVector{T})
    n = get_common_len(a, b)::Int
    return hessian!(Array(T, n), a, b)
end

