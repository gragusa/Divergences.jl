cue() = CressieRead(1) 
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
function evaluate{T<:FloatingPoint}(dist::CressieRead, a::T, b::T)
    α = dist.α
    const aa = 1/(α*(α+1))
    const ua = 1/α
    const pa = 1/(1+α)
    r = zero(T)    
    u = a/b
    if u > 0
        u = ((u^(1+α)-1)*aa-ua*ui+ua)*b
    elseif u==0
        u = pa*b
    else
        u = +Inf            
    end
    u
end

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
            r += pa*bi
        else
            r = +Inf
            break
        end
    end
    return r
end

################################################################################
## gradient
################################################################################
function gradient{T<:FloatingPoint}(dist::CressieRead, a::T, b::T)
 ## b \left(\frac{\left(\frac{a}{b}\right)^{\alpha }}{\alpha  b}-\frac{1}{\alpha  b}\right)
    α = dist.α
    onet = one(T)
    r    = zero(T)  
    u    = -Inf      
    if a > 0 && b > 0
        u = b*(a/b)^α/(b*α)-onet/α
    elseif a == 0 && b != 0
        if α>0
            u = -1/α        
        end
    end 
    u
end 

function gradient{T<:FloatingPoint}(dist::CressieRead, a::T)
    α    = dist.α
    r    = zero(T)
    onet = one(T)
    u    = -Inf
    if a > 0
        u = ((a^α)-onet)/α
    elseif a==0
        if α>0
            u = -1/α        
        end     
    end
    u
end 

function gradient!{T<:FloatingPoint}(u::Vector{T}, dist::CressieRead, a::AbstractVector{T}, b::AbstractVector{T})
    α = dist.α
    n = get_common_len(a, b)::Int
    @inbounds for i = 1:n
        ai = a[i]
        bi = bi[i]        
        u[i] = gradient(dist, ai, bi)
    end 
    u
end 

function gradient!{T<:FloatingPoint}(u::Vector{T}, dist::CressieRead, a::AbstractVector{T})    
    n = length(a)::Int
    @inbounds for i = 1:n
        ai   = a[i]        
        bi   = b[i]
        u[i] = gradient(dist, ai, bi)
    end
    u
end


################################################################################
## Hessian
################################################################################
function hessian{T<:FloatingPoint}(dist::CressieRead, a::T, b::T)
    α    = dist.α
    r    = zero(T)
    onet = one(T)
    aexp = α-onet
    ui   = a/b
    u    = +Inf
    if ui > 0
        u = ui^aexp/b
    elseif ui == 0
        if α >= 1
            u = r
        elseif α == 1
            u = one(T)
        end 
    end
    u
end

function hessian{T<:FloatingPoint}(dist::CressieRead, a::T)
    α    = dist.α
    r    = zero(T)
    onet = one(T)
    aexp = α-onet
    u    = +Inf
    if a > 0
        u = a^aexp
    elseif a==0
        if α > 1
            u = r        
        elseif α == 1
            u = one(T)
        end 
    end 
    u
end

function hessian!{T<:FloatingPoint}(u::Vector{T}, dist::CressieRead, a::AbstractVector{T}, b::AbstractVector{T})    
    n = get_common_len(a, b)::Int
    @inbounds for i = 1:n
        ai = a[i]
        bi = b[i]
        u[i] = hessian(dist, ai, bi)
    end
end 

function hessian!{T<:FloatingPoint}(u::Vector{T}, dist::CressieRead, a::AbstractVector{T})
    n = get_common_len(a, b)::Int
    @inbounds for i = 1:n
        ai   = a[i]
        u[i] = hessian(dist, ai)
    end
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
