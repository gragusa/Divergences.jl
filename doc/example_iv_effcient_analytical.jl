using MathOptInterface, Optimization, OptimizationMOI, OptimizationOptimJL, Ipopt
using ForwardDiff, DifferentiationInterface
using Divergences
using Statistics, LinearAlgebra
using Infiltrator
using StableRNGs

## -----
## High-performant
## -----
function randiv(; n = 100, m = 5, k = 1, θ = 0.0, ρ = 0.9, CP = 20)
    ## Simulate
    ## y = xθ + w'γ + u
    ## x = zτ + w'ξ + η
    ## where z ∼ N(0, Iₘ), w ∼ N(0, Iₖ) 
    ## (η ∼ N(0, I), u ∼ N(0, I)
    τ = fill(sqrt(CP / (m * n)), m)
    z = randn(n, m)  ## Instruments
    w = randn(n, k-1)  ## Exogenous
    η = randn(n, 1)
    u = ρ * η + √(1 - ρ^2) * randn(n, 1)
    x = z * τ + η
    y = x * θ + u
    return y, [x w], [z w]
end

y, x, z = randiv(k=1, CP=5)

p = (y = y,
    x = x,
    z = z,
    Y = similar(y),
    X = similar(x),
    Z = similar(z),
    ∂G = Matrix{Float64}(undef, size(z, 2), size(x,2))
    );

function g(θ, p)
    ## Gₙ(θ)
    (y, x, z, Y, X, Z, ∂G) = p.data
    mul!(Y, x, θ)
    broadcast!(-, Y, y, Y)
    broadcast!(*, Z, z, Y)
    return Z
end

function Dg(θ, π, p)
    (y, x, z, Y, X, Z, ∂G) = p.data
    broadcast!(*, X, π, x)
    mul!(∂G, z', -X)
    return ∂G
end

function Dgλ(θ, λ, p)
    (y, x, z, Y, X, Z, ∂G) = p.data
    mul!(Y, z, λ)
    broadcast!(*, X, -Y, x)
    broadcast!(/, X, X, n)
    return X
end

function Dgλ(θ, λ, π, p)
    ## Hπθ
    (y, x, z, Y, X, Z, ∂G) = p.data
    ∂gλ = Dgλ(θ, λ, p)
    broadcast!(*, ∂gλ, ∂gλ, π)
    return ∂gλ
end

function Dgλ!(J, θ, λ, p)
    ∂gλ = Dgλ(θ, λ, p)
    copy!(J, vec(∂gλ))
end

function Dgλ!(J, θ, λ, π, p)
    Dgλ(θ, λ, π, p)
    copy!(J, p.∇)
end

function Hgλ(θ, λ, π, p)
    n, k, m = size(p)
    zeros(k, k)
end

## --------------------------------------------------------------------- ##
## Optimization Problem
## --------------------------------------------------------------------- ##

const MOI = MathOptInterface

struct MDProblem <: MOI.AbstractNLPEvaluator
    div::Divergences.AbstractDivergence
    data
    backend
end

Base.size(md::MDProblem) = (size(p.x)..., size(p.z,2))
divergence(md::MDProblem) = md.div

function MOI.initialize(md::MDProblem, rf::Vector{Symbol})
    for feat in rf
        if !(feat in [:Grad, :Jac, :Hess])
            error("Unsupported feature $feat")
        end
    end
end

MOI.features_available(md::MDProblem) = [:Grad, :Jac, :Hess]

## --------------------------------------------------------------------- ##
## Objective function
## --------------------------------------------------------------------- ##
function MOI.eval_objective(md::MDProblem, u::Vector{Float64})
    ## Objective function
    ## ∑ᵢ γ(πᵢ)
    n, k, m = size(md)
    divergence(md)(view(u, 1:n))
end

function MOI.eval_objective_gradient(md::MDProblem, res, u)
    ## Gradient of the objective function
    ## ∇π ∑ᵢ γ'(πᵢ)
    n, k, m = size(md)
    T = eltype(res)
    Divergences.gradient!(view(res, 1:n), divergence(md), view(u, 1:n))
    fill!(view(res, (n+1):(n+k)), zero(T))
end

## --------------------------------------------------------------------- ##
## Constraints
## --------------------------------------------------------------------- ##
function MOI.eval_constraint(md::MDProblem, res, u)
    ## Constraints
    ## ∑ᵢ πᵢ g(θᵢ) = 0
    ## ∑ᵢ πᵢ = n
    n, k, m = size(md)
    θ = view(u, (n+1):(n+k))
    π = view(u, 1:n)
    G = g(θ, md)
    constraint!(res, π, G)
end

function constraint!(μ::AbstractVector{T}, w::AbstractVector, x::AbstractMatrix) where T
    fill!(μ, zero(T))
    @inbounds for j in axes(x,2)
        for i in axes(x,1)
            μ[j] += w[i]*x[i,j]
        end
    end
    μ[end] = sum(w)
    return μ
end

## --------------------------------------------------------------------- ##
## Constraints Jacobian
## --------------------------------------------------------------------- ##
function MOI.jacobian_structure(md::MDProblem)
    n, k, m = size(md)
    rowcol_of_dense(n+k,m+1)
end

function MOI.eval_constraint_jacobian(md::MDProblem, J, u)
    n, k, m = size(md)
    θ = view(u, (n+1):(n+k))
    π = view(u, 1:n)
    G = g(θ, md)
    #@. G = G/n
    ∂g = Dg(θ, π, md)
    @. ∂g = ∂g
    assign_constraint_jacobian!(J, G, ∂g)
end

"""
    assign_constraint_jacobian!(J, g, ∇g)

Assigns the elements of the block matrix `X = [[G'; ones(1, n)]; [∇g ; zeros(m, k)]]`.

# Arguments
- `J::Vector{Float64}`: A preallocated array of size `m * n + m * k`, where `m`, `n`, and `k` are the dimensions of `g` and `∇g`.
- `g::AbstractMatrix{T}`: An `n × m` matrix.
- `∇g::AbstractMatrix{T}`: An `m × k` matrix.
```
"""
function assign_constraint_jacobian!(J, gg, Dg)
    n, m = size(gg)
    k = size(Dg,2)
    for j in 1:n
        # Elements from gg'
        for i in 1:m
            J[(j-1)*(m+1) + i] = gg[j,i]
        end
        # Element from ones row
        J[j*(m+1)] = 1.0
    end
    # Next k columns (from Dg and zeros row)
    offset = n*(m+1)
    for j in 1:k
        # Elements from Dg
        for i in 1:m
            J[offset + (j-1)*(m+1) + i] = Dg[i,j]
        end
        # Element from 0
        J[offset + j*(m+1)] = 0.0
    end
    return J
end

## --------------------------------------------------------------------- ##
## Hessian of the Lagrangian of L(π, θ, λ) = D(π, p) + λ'g(θ)
## --------------------------------------------------------------------- ##
function MOI.hessian_lagrangian_structure(md::MDProblem)
    n, k, m = size(md)
    hele = Int(n + n*k + k*(k+1)÷2)
    rows = Array{Int64}(undef, hele)
    cols = Array{Int64}(undef, hele)
    ## Diagonal Elements
    for j = 1:n
        rows[j] = j
        cols[j] = j
    end
    idx = n+1
    ## Off-diagonal elements
    for j = 1:k
        for s = 1:n
            rows[idx] = n + j
            cols[idx] = s
            idx += 1
        end
    end
    ## For linear problem this is not needed
    for j = 1:k
        for s = 1:j
            rows[idx] = n + j
            cols[idx] = n + s
            idx += 1
        end
    end
    [(r, c) for (r, c) in zip(rows,cols)]
end

function MOI.eval_hessian_lagrangian(md::MDProblem, hess, u, σ, λ)
    n, k, m = size(md)
    π = view(u, 1:n)
    θ = view(u, (n+1):(n+k))
    if σ==0
        @inbounds for j=1:n
            hess[j] = 0.0
        end
    else
        hv = view(hess, 1:n)
        Divergences.hessian!(hv, divergence(md), π)
        hv .= hv.*σ
    end
    λv = view(λ, 1:m)
    Dgλ!(view(hess, n+1:n+n*k), θ, λv, md)
    ## For linear problem this is not needed
    copy_lower_triangular!(view(hess, n+n*k+1:n+n*k+(k*(k+1)÷2)), Hgλ(θ, λ, π, md))
end

## --------------------------------------------------------------------- ##
## Problem with fixed theta
## --------------------------------------------------------------------- ##

struct SMDProblem <: MOI.AbstractNLPEvaluator
    div::Divergences.AbstractDivergence
    G::Matrix{Float64}
    data
    backend
end

divergence(md::SMDProblem) = md.div
momfun(md::SMDProblem) = md.G

function MOI.initialize(md::SMDProblem, rf::Vector{Symbol})
    for feat in rf
        if !(feat in [:Grad, :Jac, :Hess])
            error("Unsupported feature $feat")
        end
    end
end

MOI.features_available(md::SMDProblem) = [:Grad, :Jac, :Hess]

function MOI.eval_objective(md::SMDProblem, u::Vector{Float64})
    divergence(md)(u)
end

function MOI.eval_objective_gradient(md::SMDProblem, res, u)
    n, k, m = size(md)
    T = eltype(res)
    Divergences.gradient!(res, divergence(md), u)
end

## --------------------------------------------------------------------- ##
## Constraints
## --------------------------------------------------------------------- ##
function MOI.eval_constraint(md::SMDProblem, res, u)
    π = u
    G = md.G
    constraint!(res, π, G)
end

function constraint!(μ::AbstractVector{T}, w::AbstractVector, x::AbstractMatrix) where T
    fill!(μ, zero(T))
    @inbounds for j in axes(x,2)
        for i in axes(x,1)
            μ[j] += w[i]*x[i,j]
        end
    end
    μ[end] = sum(w)
    return μ
end

## --------------------------------------------------------------------- ##
## Constraints Jacobian
## --------------------------------------------------------------------- ##
function MOI.jacobian_structure(md::SMDProblem)
    n, k, m = size(md)
    rowcol_of_dense(n,m+1)
end

function MOI.eval_constraint_jacobian(md::MDProblem, J, u)
    π = u
    G = md.G
    #@. G = G/n
    assign_constraint_jacobian!(J, G)
end

"""
    assign_constraint_jacobian!(J, g)

Assigns the elements of the block matrix `X = G'`.

# Arguments
- `J::Vector{Float64}`: A preallocated array of size `m * n + m * k`, where `m`, `n`, and `k` are the dimensions of `g` and `∇g`.
- `g::AbstractMatrix{T}`: An `n × m` matrix.
```
"""
function assign_constraint_jacobian!(J, gg)
    n, m = size(gg)
    k = size(Dg,2)
    for j in 1:n
        # Elements from gg'
        for i in 1:m
            J[(j-1)*(m+1) + i] = gg[j,i]
        end
        # Element from ones row
        J[j*(m+1)] = 1.0
    end
    return J
end

## --------------------------------------------------------------------- ##
## Hessian of the Lagrangian of L(π, θ, λ) = D(π, p) + λ'g(θ)
## --------------------------------------------------------------------- ##
function MOI.hessian_lagrangian_structure(md::SMDProblem)
    rows = Array{Int64}(undef, n)
    cols = Array{Int64}(undef, n)
    ## Diagonal Elements
    for j = 1:n
        rows[j] = j
        cols[j] = j
    end
    [(r, c) for (r, c) in zip(rows,cols)]
end

function MOI.eval_hessian_lagrangian(md::SMDProblem, hess, u, σ, λ)
    π = u
    if σ==0
        @inbounds for j=1:n
            hess[j] = 0.0
        end
    else
        hv = view(hess, 1:n)
        Divergences.hessian!(hv, divergence(md), π)
        hv .= hv.*σ
    end
end





## --------------------------------------------------------------------- ##
## Problem
## --------------------------------------------------------------------- ##

ℳ𝒟 = FullyModifiedDivergence(ReverseKullbackLeibler(), 0.1, 1.2)
mdprob = MDProblem(ℳ𝒟, p, nothing)

model = Ipopt.Optimizer()
π = MOI.add_variables(model, n)
MOI.add_constraint.(model, π, MOI.GreaterThan(0.0))
θ = MOI.add_variables(model, k)
MOI.add_constraint.(model, θ, MOI.GreaterThan(-10.0))
MOI.add_constraint.(model, θ, MOI.LessThan(+10.0))
for i ∈ 1:n
    MOI.set(model, MOI.VariablePrimalStart(), π[i], 1.0)
end
for i ∈ 1:k
    MOI.set(model, MOI.VariablePrimalStart(), θ[i], 0.0)
end
lb = [zeros(m); n]
ub = [zeros(m); n]
MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

model_el = deepcopy(model)
model_md = deepcopy(model)

block_data = MOI.NLPBlockData(MOI.NLPBoundsPair.(lb, ub), mdprob, true)
MOI.set(model_md, MOI.NLPBlock(), block_data)
for i ∈ 1:k
    MOI.set(model_md, MOI.VariablePrimalStart(), θ[i], -0.01)
end

mdprob = MDProblem(ReverseKullbackLeibler(), p, nothing)
block_data = MOI.NLPBlockData(MOI.NLPBoundsPair.(lb, ub), mdprob, true)
MOI.set(model_el, MOI.NLPBlock(), block_data)




model.options["derivative_test"] = "none"
model.options["derivative_test_print_all"] = "no"

model.options["print_level"] = 4

MOI.optimize!(model)
MOI.get(model, MOI.TerminationStatus())
MOI.get(model, MOI.DualStatus())
MOI.get(model, MOI.PrimalStatus())

MOI.get(model, MOI.SolveTimeSec())
MOI.get(model, MOI.BarrierIterations())

xstar = MOI.get(model, MOI.VariablePrimal(), θ)

function lagrangian(md::MDProblem, u, σ, λ)
    n, k, m = size(md)
    π = u[1:n]
    θ = u[(n+1):(n+k)]
    σ.*divergence(md)(π) +mean(π.*g(θ, md)*λ)
end


using Statistics
p = [0.45793379249066035, 4.999416892014921, 9.182989399836064, 3.6958463315972025, 6.220383439227501, 0.019436036309187443, 2.063484686999562, 10.894774879314305, 8.25546846552471, 4.029010019680072, -2.975818044182361, 1.4669020891138018]

lagrangian(mdprob, p, 1.0, [1.5, 0.0])

H0 = ForwardDiff.hessian(x -> lagrangian(mdprob, x, 1.5, [1.5, 0.0]), p);
H = zeros(34)
MOI.eval_hessian_lagrangian(mdprob, H, p, 1.5, [1.5, 0.0])

H0 = ForwardDiff.hessian(x -> lagrangian(mdprob, x, 0.0, [1.5, 0]), p);
MOI.eval_hessian_lagrangian(mdprob, H, p, 0.0, [1.5, 0])

## --------------------------------------------------------------------- ##
## Simple MC
## --------------------------------------------------------------------- ##

β_el = Matrix{Float64}(undef, 1000, 3)
f_el = zeros(1000)
β_md = Matrix{Float64}(undef, 1000, 3)
f_md = zeros(1000)
for j in 1:1000
    y, x, z = randiv(k=1, CP=5)
    p.y .= y
    p.x .= x
    p.z .= z
    MOI.optimize!(model_el)
    MOI.optimize!(model_md)
    β_el[j,:] .= MOI.get(model_el, MOI.VariablePrimal(), θ)
    β_md[j,:] .= MOI.get(model_md, MOI.VariablePrimal(), θ)
    f_el[j] = model_el.inner.status
    f_md[j] = model_md.inner.status
end

using StatsPlots

StatsPlots.density(β)
StatsPlots.histogram(β, nbins = 80)



## --------------------------------------------------------------------- ##
## Utilities
## --------------------------------------------------------------------- ##


# function assign_matrix(J, gg, Dg)
#     n, m = size(gg)
#     k = size(Dg,2)
#     R = [ [gg'; ones(1, n)] [Dg; zeros(1,k)]]
#     J .= vec(R)
# end


using SparseArrays

function rowcol_of_sparse(g::SparseMatrixCSC; offset_row = 0, offset_col = 0)
    rows = rowvals(g)
    vals = nonzeros(g)
    m, n = size(g)
    tup = Tuple{Int64, Int64}[]
    for j ∈ 1:n
        for i ∈ nzrange(g, j)
            push!(tup, (rows[i]+offset_row, j+offset_col))
        end
    end
    return tup
end


function weighted_sum(G, w)
    n, m = size(G)
    res = zeros(eltype(G), m)
    @inbounds for j in axes(G,2)
        for i in axes(G,1)
            res[j] += w[i]*G[i,j]
        end
    end
    return res
end

function weighted_sum2(G, w)
    @inbounds vec(sum(w.*G, dims=1))
end


"""
    rowcol_of_dense(g::AbstractMatrix; offset_row = 0, offset_col = 0)

Returns a tuple of row and column indices for all elements in a dense matrix `g`, with optional offsets for rows and columns.

# Arguments
- `g::AbstractMatrix`: The input dense matrix.
- `offset_row::Int` (default: 0): An offset to be added to each row index.
- `offset_col::Int` (default: 0): An offset to be added to each column index.

# Returns
A vector of tuples `(row, col)` representing the indices of all elements in the dense matrix.

# Example
```julia
g = [1 2; 3 4]
rowcol_of_dense(g)  # [(1, 1), (2, 1), (1, 2), (2, 2)]
```
"""
function rowcol_of_dense(n ,m; offset_row = 0, offset_col = 0)
    tup = Tuple{Int64, Int64}[]  # Initialize an empty vector of tuples
    @inbounds for j ∈ 1:n
        for i ∈ 1:m
            push!(tup, (i + offset_row, j + offset_col))
        end
    end
    return tup
end




function copy_lower_triangular!(x::AbstractVector{T}, A::Matrix{T}) where T
    @assert issquare(A)
    n = size(A, 1)
    len = (n * (n + 1)) ÷ 2  # Length of output vector
    @assert len == (n * (n + 1)) ÷ 2
    idx = 1
    @inbounds for j in 1:n
        for i in j:n
            x[idx] = A[i, j]
            idx += 1
        end
    end
    return x
end

function copy_lower_triangular!(x::AbstractVector{T}, A::Vector{T}) where T
    n = length(A)
    @assert n == 1 "`copy_lower_triangular!` for vector make sense only for singleton vector"
    @assert 1 == (n * (n + 1)) ÷ 2 "The dimension of the dest vector is wrong as it should be $(n*(n+1))//2"
    x .= A
    return x
end



abstract type SmootherType end

struct Truncated <: SmootherType end
struct Bartlett <: SmootherType end

@inline weight(::Truncated, s, St) = 1.0
@inline weight(::Bartlett, s, St) = 1.0 - s/St

# Base version
function smooter_base(tt::T, G::Matrix, ξ::Integer) where {T<:SmootherType}
    N, M = size(G)
    nG = zeros(N, M)
    St = (2.0 * ξ + 1.0) / 2.0
    for m = 1:M
        for t = 1:N
            low = max((t - N), -ξ)
            high = min(t - 1, ξ)
            for s = low:high
                κ = weight(tt, s, St)
                @inbounds nG[t, m] += κ * G[t-s, m]
            end
        end
    end
    return (nG ./ (2 * ξ + 1))
end

function smoother(tt::Truncated, G::Matrix{T}, ξ::Integer) where {T}
    N, M = size(G)
    nG   = Matrix{T}(undef, N, M)
    smoother!(tt, nG, G, ξ)
end

function smoother!(tt::Truncated, dest, G::Matrix{T}, ξ::Integer) where {T}
    N, M = size(G)
    denom = 2ξ + 1  # normalization
    Threads.@threads for m in 1:M
        for t in 1:N
            low  = max(t - N, -ξ)
            high = min(t - 1,  ξ)
            acc  = zero(T)
            @inbounds for s in low:high
                κ = weight(tt, s, ξ)
                acc += G[t - s, m]
            end
            dest[t, m] = acc / denom
        end
    end
    return dest
end

# optprob = OptimizationFunction(divergence, Optimization.AutoForwardDiff(), cons = cons)
# prob = OptimizationProblem(optprob, x0, _p,
#                            lcons = repeat([0.], 2),
#                            ucons = repeat([0.], 2),
#                            lb = [repeat([0], 100); -Inf],
#                            ub = [repeat([+Inf], 100); +Inf])

# solver = OptimizationMOI.MOI.OptimizerWithAttributes(Ipopt.Optimizer, "print_level" => 0)

# solve(prob, solver)
