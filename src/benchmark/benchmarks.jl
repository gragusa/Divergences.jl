using BenchmarkTools
using Distances
using Divergences

const SUITE = BenchmarkGroup()

function create_distances()
    divs = [
        KullbackLeibler(),
        ReverseKullbackLeibler(),
        Hellinger(),
        CressieRead(2.0),
        ChiSquared(),
        ModifiedDivergence(KullbackLeibler(), 2.0),
        FullyModifiedDivergence(KullbackLeibler(), 0.5, 2.0),
    ]

    return divs
end

###########
# Eval    #
###########

SUITE["evaluation"] = BenchmarkGroup()

function evaluate(dist, x, y)
    n = size(x, 1)
    T = typeof(dist(x[1, 1], y[1, 1]))
    dist(x, y)
end

SUITE["gradient"] = BenchmarkGroup()

###########
# Colwise #
###########

SUITE["colwise"] = BenchmarkGroup()

function evaluate_colwise(dist, x, y)
    n = size(x, 2)
    T = typeof(evaluate(dist, x[:, 1], y[:, 1]))
    r = Vector{T}(undef, n)
    for j = 1:n
        r[j] = @views evaluate(dist, x[:, j], y[:, j])
    end
    return r
end

function add_colwise_benchmarks!(SUITE)

    m = 200
    n = 10000

    x = rand(m, n)
    y = rand(m, n)

    p = x
    q = y
    for i = 1:n
        p[:, i] /= sum(x[:, i])
        q[:, i] /= sum(y[:, i])
    end

    divs = create_distances()

    for (dists, (a, b)) in [(divs, (p,q))]
        for dist in (dists)
            Tdist = typeof(dist)
            SUITE["colwise"][Tdist] = BenchmarkGroup()
            SUITE["colwise"][Tdist]["loop"]        = @benchmarkable evaluate_colwise($dist, $a, $b)
            SUITE["colwise"][Tdist]["specialized"] = @benchmarkable colwise($dist, $a, $b)
        end
    end
end

add_colwise_benchmarks!(SUITE)


############
# Pairwise #
############

SUITE["pairwise"] = BenchmarkGroup()

function evaluate_pairwise(dist, x, y)
    nx = size(x, 2)
    ny = size(y, 2)
    T = typeof(evaluate(dist, x[:, 1], y[:, 1]))
    r = Matrix{T}(undef, nx, ny)
    for j = 1:ny
        @inbounds for i = 1:nx
            r[i, j] = @views evaluate(dist, x[:, i], y[:, j])
        end
    end
    return r
end

function add_pairwise_benchmarks!(SUITE)
    m = 100
    nx = 200
    ny = 250

    x = rand(m, nx)
    y = rand(m, ny)

    p = x
    for i = 1:nx
        p[:, i] /= sum(x[:, i])
    end

    q = y
    for i = 1:ny
        q[:, i] /= sum(y[:, i])
    end

    divs = create_distances()

     for (dists, (a, b)) in [(divs, (p,q))]
        for dist in (dists)
            Tdist = typeof(dist)
            SUITE["pairwise"][Tdist] = BenchmarkGroup()
            SUITE["pairwise"][Tdist]["loop"]        = @benchmarkable evaluate_pairwise($dist, $a, $b)
            SUITE["pairwise"][Tdist]["specialized"] = @benchmarkable pairwise($dist, $a, $b; dims=2)
        end
    end
end

add_pairwise_benchmarks!(SUITE)
