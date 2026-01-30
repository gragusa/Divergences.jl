"""
Benchmarks for Divergences.jl

Benchmark suite measuring:
1. Divergence evaluation on vectors
2. Gradient computation (in-place and out-of-place)
3. Hessian computation (in-place and out-of-place)
"""

using BenchmarkTools
using Divergences
using StableRNGs
using Divergences: gradient, gradient!, hessian, hessian!

# ============================================================================
# Setup
# ============================================================================

const DEFAULT_SEED = 20240612
const N = 1000

# Generate test data: values in (0.1, 3.0) to avoid boundary issues
function generate_test_data(rng::AbstractRNG, n::Int)
    return 0.1 .+ 2.9 .* rand(rng, n)
end

# Pre-generate data for all benchmarks
const RNG = StableRNG(DEFAULT_SEED)
const X = generate_test_data(RNG, N)
const U = similar(X)  # Pre-allocated output buffer

# Divergence instances
const DIVS = (
    kl = KullbackLeibler(),
    rkl = ReverseKullbackLeibler(),
    hellinger = Hellinger(),
    chisq = ChiSquared(),
    cr2 = CressieRead(2.0),
    modified = ModifiedDivergence(KullbackLeibler(), 2.0),
    fullymod = FullyModifiedDivergence(KullbackLeibler(), 0.5, 2.0)
)

# ============================================================================
# Benchmark Suite
# ============================================================================

const SUITE = BenchmarkGroup()

# ----------------------------------------------------------------------------
# Evaluation Benchmarks
# ----------------------------------------------------------------------------

SUITE["eval"] = BenchmarkGroup()

for (name, div) in pairs(DIVS)
    SUITE["eval"][string(name)] = @benchmarkable $div($X)
end

# ----------------------------------------------------------------------------
# Gradient Benchmarks (out-of-place)
# ----------------------------------------------------------------------------

SUITE["gradient"] = BenchmarkGroup()

for (name, div) in pairs(DIVS)
    SUITE["gradient"][string(name)] = @benchmarkable gradient($div, $X)
end

# ----------------------------------------------------------------------------
# Gradient Benchmarks (in-place)
# ----------------------------------------------------------------------------

SUITE["gradient!"] = BenchmarkGroup()

for (name, div) in pairs(DIVS)
    u = similar(X)
    SUITE["gradient!"][string(name)] = @benchmarkable gradient!($u, $div, $X)
end

# ----------------------------------------------------------------------------
# Hessian Benchmarks (out-of-place)
# ----------------------------------------------------------------------------

SUITE["hessian"] = BenchmarkGroup()

for (name, div) in pairs(DIVS)
    SUITE["hessian"][string(name)] = @benchmarkable hessian($div, $X)
end

# ----------------------------------------------------------------------------
# Hessian Benchmarks (in-place)
# ----------------------------------------------------------------------------

SUITE["hessian!"] = BenchmarkGroup()

for (name, div) in pairs(DIVS)
    u = similar(X)
    SUITE["hessian!"][string(name)] = @benchmarkable hessian!($u, $div, $X)
end
