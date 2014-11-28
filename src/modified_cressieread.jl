type ModifiedCressieRead <: Divergence
    α::Real
    ϑ::Real
    function ModifiedCressieRead(α::Real, ϑ::Real)
        @assert isempty(findin(α, [-1, 0])) "ModifiedCressieRead is defined for all α!={-1,0}."
        @assert ϑ>0 "ModifiedCressieRead is defined for ϑ>1."
        new(α, ϑ)
    end
end

function evaluate{T<:FloatingPoint}(dist::ModifiedCressieRead, a::AbstractVector{T}, b::AbstractVector{T})
    α  =  dist.α
    ϑ   = dist.ϑ
    u₀  = 1+ϑ
    cr  = CressieRead(α)
    ϕ₀  = evaluate(cr, [u₀])
    ϕ¹₀ = gradient(cr, u₀)
    ϕ²₀ = hessian(cr, u₀)
    onet = one(T)
    aexp = (onet+α)
    const aa = onet/(α*aexp)
    const ua = onet/α
    const pa = onet/aexp
    r = zero(T)
    n = get_common_len(a, b)::Int
    for i = 1 : n
        @inbounds ai = a[i]
        @inbounds bi = b[i]
        @inbounds ui = ai/bi
        if ui>=u₀
      		r += ϕ₀ + ϕ¹₀*(ui-u₀) + .5*ϕ²₀*(ui-u₀)^2  	
        elseif ui > 0 && ui < u₀
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

function evaluate{T<:FloatingPoint}(dist::ModifiedCressieRead, a::AbstractVector{T})
    α  =  dist.α
    ϑ   = dist.ϑ
    u₀  = 1+ϑ
    cr  = CressieRead(α)
    ϕ₀  = evaluate(cr, [u₀])
    ϕ¹₀ = gradient(cr, u₀)
    ϕ²₀ = hessian(cr, u₀)
    onet = one(T)
    aexp = (onet+α)
    const aa = onet/(α*aexp)
    const ua = onet/α
    const pa = onet/aexp
    r = zero(T)
    n = length(a)::Int
    @inbounds for i = 1 : n
        ui = a[i]
        if ui>=u₀
      		r += ϕ₀ + ϕ¹₀*(ui-u₀) + .5*ϕ²₀*(ui-u₀)^2  	
        elseif ui > 0 && ui < u₀
            r += (ui^(1+α)-1)*aa-ua*ui+ua
        elseif ui==0
            r += pa
        else
            r = +Inf
            break
        end
    end
    return r
end


function gradient{T<:FloatingPoint}(dist::ModifiedCressieRead, a::T)
	α    = dist.α
	ϑ    = dist.ϑ
	u₀   = 1+ϑ
	cr   = CressieRead(α)	
	ϕ¹₀  = gradient(cr, u₀)
	ϕ²₀  = hessian(cr, u₀)
	onet = one(T)
	r    = zero(T)
	if a>=u₀
		return ϕ¹₀ + ϕ²₀*(a-u₀)
	else 
		gradient(cr, a)	
	end
end

function gradient{T<:FloatingPoint}(dist::ModifiedCressieRead, a::T, b::T)
	α    = dist.α
	ϑ    = dist.ϑ
	u₀   = 1+ϑ
	cr   = CressieRead(α)
	ϕ¹₀  = gradient(cr, u₀)
	ϕ²₀  = hessian(cr, u₀)
	ui   = a/b
	if ui>u₀
		return ϕ¹₀ + ϕ²₀*(u-u₀)
	else
		return gradient(cr, ui)
	end
end

function gradient!{T<:FloatingPoint}(u::Vector{T}, dist::ModifiedCressieRead, a::AbstractVector{T}, b::AbstractVector{T})
    α    =  dist.α
    ϑ    = dist.ϑ
    u₀   = 1+ϑ
    cr   = CressieRead(α)    
    ϕ¹₀  = gradient(cr, u₀)
    ϕ²₀  = hessian(cr, u₀)
    onet = one(T)
    r    = zero(T)
    n    = get_common_len(a, b)::Int

    @inbounds for i = 1:n
        ai = a[i]
        ui = ai/bi[i]
        if ui<0
        	@inbounds u[i] = +Inf
        elseif ui==0
        	@inbounds u[i] = r
        elseif ui>=u₀
       		@inbounds u[i] = ϕ¹₀ + ϕ²₀*(u-u₀)
       	else
       		@inbounds u[i] = ( (ui^α)-onet )/α
       	end
    end
end


# function gradient!{T<:FloatingPoint}(u::Vector{T}, dist::ModifiedCressieRead, a::AbstractVector{T})
#     α    =  dist.α
#     ϑ    = dist.ϑ
#     u₀   = 1+ϑ
#     cr   = CressieRead(α)   
#     ϕ¹₀  = gradient(cr, u₀)
#     ϕ²₀  = hessian(cr, u₀)
#     onet = one(T)
#     r    = zero(T)
#     n    = get_common_len(a, b)::Int
#     for i = 1 : n
#         ui = a[i]        
#         if ui<0
#         	@inbounds u[i] = +Inf
#         elseif ui==0
#         	@inbounds u[i] = r
#         elseif ui>=u₀
#        		@inbounds u[i] = ϕ¹₀ + ϕ²₀*(u-u₀)
#        	else
#        		@inbounds u[i] = ( (ui^α)-onet )/α
#        	end
#     end
# end

function hessian{T<:FloatingPoint}(dist::ModifiedCressieRead, a::T)
	α    = dist.α
	ϑ    = dist.ϑ
	u₀   = 1+ϑ
	cr   = CressieRead(α)
	ϕ²₀  = hessian(cr, u₀)	
	if a>=u₀
		return ϕ²₀
	else
		hessian(cr, a)
	end       	
end


# function hessian!{T<:FloatingPoint}(u::Vector{T}, dist::ModifiedCressieRead, a::AbstractVector{T}, b::AbstractVector{T})
#   	α    = dist.α
# 	ϑ    = dist.ϑ
# 	u₀   = 1+ϑ
# 	cr   = CressieRead(α)		
# 	ϕ²₀  = hessian(cr, u₀)

#     r  = zero(T)
#     onet = one(T)

#     n = get_common_len(a, b)::Int
#     for i = 1 : n
#         ai = a[i]
#         bi = b[i]
#         ui = ai/bi
#         if ui<0
#         	@inbounds u[i] = +Inf
#         elseif ui==0
#         	@inbounds u[i] = r
#         elseif ui>=u₀
#        		@inbounds u[i] = ϕ²₀*u
#        	else
#        		@inbounds u[i] = ui^α/ai
#        	end
# 	end
# end 

# function hessian!{T<:FloatingPoint}(u::Vector{T}, dist::ModifiedCressieRead, a::AbstractVector{T})
#   	α    = dist.α
# 	ϑ    = dist.ϑ
# 	u₀   = 1+ϑ
# 	cr   = CressieRead(α)
# 	ϕ²₀  = hessian(cr, u₀)

#     r    = zero(T)
#     onet = one(T)

#     n = length(a)::Int
#     for i = 1 : n        
#         ui = ai
#         if ui<0
#         	@inbounds u[i] = +Inf
#         elseif ui==0
#         	@inbounds u[i] = r
#         elseif ui>=u₀
#        		@inbounds u[i] = ϕ²₀*u
#        	else
#        		@inbounds u[i] = ui^α/ai
#        	end       
#     end 	    
# end

