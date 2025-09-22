using MathOptInterface, Optimization, OptimizationMOI, OptimizationOptimJL, Ipopt
using ForwardDiff, DifferentiationInterface
using Divergences
using Statistics, LinearAlgebra
using Infiltrator


## --------------------------------------------------------------------- ##
## Moment Conditions & Jacobian
## --------------------------------------------------------------------- ##

## This function if the moment matrix of the estimation problem. 
## This function should always be defined by the user.
function g(x, θ)
    d = x .- θ
    [d d.^2 .- 1]
end

## This function is the gradient of the mean moment matrix
## which is a (m, k), where m is the number of moments and k 
## is the number of parameters.
## \frac{\partial}{\partial\theta}\left[\sum_{i=1}^{n}\pi_{i}g(x_{i},\theta)/n\right]
## It should be written in a way that it can be used with ForwardDiff or Zygote.
function ∇g(x, θ, π)
    d = x .- θ
    res = Matrix{promote_type(eltype(θ), eltype(π))}(undef, 2, 1)
    res[1] = -sum(π)  ## mean here
    res[2] = -2.0*sum(π.*d) ## mean here
    return res
end

function λ∇g(θ, π, λ, x)
    first(λ'∇g(x, θ, π)./n)
end

## This must return a (k,n)
function ∇gᵢλ!(dest::AbstractMatrix, x, θ, λ)
    d = x .- θ
    n = length(d)
    for j in axes(dest, 1)
        for i in axes(dest, 2)
            dest[j,i] = (-λ[1] .- 2.0.*d[i].*λ[2])/n
        end
    end
    return dest
end

Base.@propagate_inbounds function ∇gᵢλ(x, θ, λ)
    n = length(x)
    k = length(θ)
    res = Matrix{promote_type(eltype(θ), eltype(λ), eltype(x))}(undef, k, n)
    ∇gᵢλ!(res, x, θ, λ)
end

## --------------------------------------------------------------------- ##
## Optimization Problem
## --------------------------------------------------------------------- ##

const MOI = MathOptInterface

struct MDProblem{D} <: MOI.AbstractNLPEvaluator
    div::Divergences.AbstractDivergence
    data::D
    size::Tuple{Int, Int, Int}
    backend::DifferentiationInterface.AbstractADType
end

Base.size(md::MDProblem) = md.size
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
    n, k, m = size(md)
    divergence(md)(view(u, 1:n))
end

function MOI.eval_objective_gradient(md::MDProblem, res, u)
    n, k, m = size(md)
    T = eltype(res)
    Divergences.gradient!(view(res, 1:n), divergence(md), view(u, 1:n))
    fill!(view(res, (n+1):(n+k)), zero(T))
end

## --------------------------------------------------------------------- ##
## Constraints
## --------------------------------------------------------------------- ##
function MOI.eval_constraint(md::MDProblem, res, u)
    n, k, m = size(md)
    θ = view(u, (n+1):(n+k))
    π = view(u, 1:n)
    G = g(md.data, θ)
    weighted_mean!(res, π, G)
end

function MOI.jacobian_structure(md::MDProblem)
    n, k, m = size(md)
    rowcol_of_dense(n+1,m+1)
end

## --------------------------------------------------------------------- ##
## Constraints Jacobian
## --------------------------------------------------------------------- ##
function MOI.eval_constraint_jacobian(md::MDProblem, J, u)
    n, k, m = size(md)
    θ = view(u, (n+1):(n+k))
    π = u[1:n]
    G = g(md.data, θ)
    G .= G./n
    ∇gₙ = ∇g(md.data, θ, π)
    ∇gₙ .= ∇gₙ./n
    assign_matrix!(J, G, ∇gₙ)
end

## --------------------------------------------------------------------- ##
## Hessian of the Lagrangian
## --------------------------------------------------------------------- ##

## The lagrangian is given by:
##
## L(π, θ, λ) = D(π, p) + λ'g(θ)

function MOI.hessian_lagrangian_structure(md::MDProblem)  
    n, k, m = size(md)
    hele = Int(n + n*k + k*k)
    rows = Array{Int64}(undef, hele)
    cols = Array{Int64}(undef, hele)
    ## Diagonal Elements
    for j = 1:n
        rows[j] = j
        cols[j] = j
    end
    idx = n+1

    # for j = 1:k
    #     for s = 1:n
    #         rows[idx] = s
    #         cols[idx] = n + j
    #         idx += 1
    #     end
    # end

    ## Off-diagonal elements
    for j = 1:k
        for s = 1:n
            rows[idx] = n + j
            cols[idx] = s
            idx += 1
        end
    end

    ## Last Block 
    for j = 1:k
        for s = 1:k
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
    #v = ∇gᵢ(md.data, θ)*λv./n
    ## As this matrix is symmetric, Ipopt expects that only the lower diagonal entries are specified.!!!!
    ## hess[n+1:n+n*k] .= vec(v')
    ∇gᵢλ!(reshape(view(hess, n+1:n+n*k), k, n), md.data, θ, λv)
    # @infiltrate
    ## If k>1, the we should only get the lower diagonal entries of
    ## the gradient of λ∇g
    ## hess[n+n*k+1:n+n*k+k^2] .= gradient(λ∇g, md.backend, θ, Constant(π), Constant(λv), Constant(md.data))
    ##vv = gradient(λ∇g, md.backend, θ, Constant(π), Constant(λv), Constant(md.data))
    ##@infiltrate
    copy_lower_triangular!(view(hess, n+n*k+1:n+n*k+(k*(k+1)÷2)), gradient(λ∇g, md.backend, θ, Constant(π), Constant(λv), Constant(md.data)))
end



## --------------------------------------------------------------------- ##
## Problem
## --------------------------------------------------------------------- ##

## Small problem to test the implementation
n = 1000
k = 1
m = 2

𝒟 = ChiSquared()
ℳ𝒟 = FullyModifiedDivergence(𝒟, 0.7, 1.2)

mdprob = MDProblem(𝒟, √0.64.*randn(n), (n, k, m), AutoForwardDiff())

n, k, m = size(mdprob)

model = Ipopt.Optimizer()
π = MOI.add_variables(model, n)
MOI.add_constraint.(model, π, MOI.GreaterThan(0.0))
θ = MOI.add_variables(model, k)
MOI.add_constraint.(model, θ, MOI.GreaterThan(-10.0))
MOI.add_constraint.(model, θ, MOI.LessThan(+10.0))

MOI.get(model, MOI.NumberOfVariables())

for i ∈ 1:n
    MOI.set(model, MOI.VariablePrimalStart(), π[i], 1.0)
end

for i ∈ 1:k
    MOI.set(model, MOI.VariablePrimalStart(), θ[i], 0.0)
end

lb = [zeros(m); n]
ub = [zeros(m); n]

MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
block_data = MOI.NLPBlockData(MOI.NLPBoundsPair.(lb, ub), mdprob, true)
MOI.set(model, MOI.NLPBlock(), block_data)

model.options["derivative_test"] = "none"
model.options["derivative_test_print_all"] = "no"

model.options["print_level"] = 3

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
    σ.*divergence(md)(π) +mean(π.*g(md.data, θ)*λ)
end


using Statistics

p = [0.45793379249066035, 4.999416892014921, 9.182989399836064, 3.6958463315972025, 6.220383439227501, -9.964661752165114]
lagrangian(mdprob, p, 1.5, [0.0, 0.0])

H0 = ForwardDiff.hessian(x -> lagrangian(mdprob, x, 1.5, [0.0, 0]), p);
H = zeros(16)
MOI.eval_hessian_lagrangian(mdprob, H, p, 1.5, [0.0, 0])

H0 = ForwardDiff.hessian(x -> lagrangian(mdprob, x, 0.0, [1.5, 0]), p);
MOI.eval_hessian_lagrangian(mdprob, H, p, 0.0, [1.5, 0])

## --------------------------------------------------------------------- ##
## Simple MC
## --------------------------------------------------------------------- ##

β = Vector{Float64}(undef, 5000)
for j in 1:5000
    x = √0.64.*randn(1000)
    mdprob.data .= x
    MOI.optimize!(model)
    β[j] = MOI.get(model, MOI.VariablePrimal(), θ)[1]
end

using StatsPlots

StatsPlots.density(β)
StatsPlots.histogram(β, nbins = 80)



## --------------------------------------------------------------------- ##
## Utilities
## --------------------------------------------------------------------- ##

"""
    assign_matrix!(J, g, ∇g)

Assigns the elements of the block matrix `X = [[g'; ones(1, n)]; [∇g; zeros(m, k)]]` into the preallocated array `J`, excluding
the `ones(1, n)` and `zeros(m, k)` blocks.

# Arguments
- `J::Vector{Float64}`: A preallocated array of size `m * n + m * k`, where `m`, `n`, and `k` are the dimensions of `g` and `∇g`.
- `g::AbstractMatrix{T}`: An `n × m` matrix.
- `∇g::AbstractMatrix{T}`: An `m × k` matrix.

# Behavior
- The function directly assigns:
  - The elements of the transpose of `g` (`g'`) in column-major order.
  - The elements of `∇g` in column-major order.
- The blocks `ones(1, n)` and `zeros(m, k)` are skipped.

# Example
```julia
A = [1 2; 3 4; 5 6]  # 3 × 2 matrix (n = 3, m = 2)
B = [7 8; 9 10]     # 2 × 2 matrix (m = 2, k = 2)

J = Vector{Float64}(undef, 2 * 3 + 2 * 2)  # Preallocate array
assign_matrix!(J, g, ∇g)

# J will be:
# [1.0, 3.0, 5.0, 2.0, 4.0, 6.0, 7.0, 9.0, 8.0, 10.0]
```
"""
Base.@propagate_inbounds function assign_matrix!(J, gg, Dg)
    n, m = size(gg)
    k = size(Dg,2)
    
    # First block: gg' (m×n matrix)
    for i in 1:n
        for j in 1:m
            J[(i-1)*(m+1) + j] = gg[i,j]  # Transpose while assigning
        end
    end
    
    # Row of ones (1×n matrix)
    for i in 1:n
        J[i*(m+1)] = 1.0
    end
    
    # Second block: Dg (m×k matrix)
    for i in 1:k
        for j in 1:m
            J[n*(m+1) + (i-1)*m + j] = Dg[j,i]
        end
    end
    
    # Final block of zeros (1×k matrix)
    baseIdx = n*(m+1) + k*m
    for i in 1:k
        J[baseIdx + i] = 0.0
    end
    
    return J
end

function assign_matrix(J, gg, Dg)
    n, m = size(gg)
    k = size(Dg,2)
    R = [ [gg'; ones(1, n)] [Dg; zeros(1,k)]]
    J .= vec(R)
end


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

function weighted_mean!(μ::AbstractVector{T}, w::AbstractVector, x::AbstractMatrix) where T
    fill!(μ, zero(T))
    @inbounds for j in axes(x,2)
        for i in axes(x,1)
            μ[j] += w[i]*x[i,j]/n
        end
    end
    μ[end] = sum(w)
    #μ[1:end-1] ./= n
    return μ
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
    @assert isquare(A)
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




# optprob = OptimizationFunction(divergence, Optimization.AutoForwardDiff(), cons = cons)
# prob = OptimizationProblem(optprob, x0, _p, 
#                            lcons = repeat([0.], 2),    
#                            ucons = repeat([0.], 2),
#                            lb = [repeat([0], 100); -Inf], 
#                            ub = [repeat([+Inf], 100); +Inf])

# solver = OptimizationMOI.MOI.OptimizerWithAttributes(Ipopt.Optimizer, "print_level" => 0)

# solve(prob, solver)










