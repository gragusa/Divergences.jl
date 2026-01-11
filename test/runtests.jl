using Divergences
using Test
using ForwardDiff

const 𝒦ℒ=KullbackLeibler()
const ℬ𝓊𝓇ℊ=ReverseKullbackLeibler()
const 𝒞ℛ=CressieRead(2.0)
const ℋ𝒟=Hellinger()
const χ²=ChiSquared()
const ℳ𝒟 = ModifiedDivergence(𝒦ℒ, 1.2)
const ℱℳ𝒟 = FullyModifiedDivergence(𝒦ℒ, 0.9, 1.2)

function testfun(𝒟, t₀, s)
    println("Testing "*string(𝒟))
    for (f, v) in t₀
        str2 = "    "*string(f)
        print(str2)
        if f == Divergences.eval
            d = 𝒟.(s)
            @test d ≈ v rtol = 1e-05
            @test sum(d) ≈ sum(v) rtol=1e-05
        else
            d = map(u -> f(𝒟, u), s)
            @test d ≈ v rtol = 1e-05
        end
        printstyled(" "*repeat(".", 40-length(str2))*" [✓]"*"\n"; color = :green)
    end
end

#=
Check that all Divergences satisfy the normalization
1.   γ(1) ≖ 0
2.  γ'(1) ≖ 0
3.  γ(x) ⩾ 0
=#
#region

#=
Check that all Divergences satisfy the normalization
1.  γ(1) ≖ 0
2. γ'(1) ≖ 0
3.  γ(x) ⩾ 0
=#
seq = 0:0.1:3
divs = (KullbackLeibler(),
    ReverseKullbackLeibler(),
    Hellinger(),
    [CressieRead(p) for p in (-0.5, 0.5, 2.0)]...)
for d in divs
    str = "Testing normalization: "*string(d)
    println(str)
    str2 = "     γ(1) == 0"
    print(str2)
    @test d(1.0) ≈ 0
    printstyled(" "*repeat(".", 50-length(str2))*" [✓]"*"\n"; color = :green)

    str2 = "    γ'(1) == 0"
    print(str2)
    @test Divergences.gradient(d, [1.0]) == [0.0]
    printstyled(" "*repeat(".", 50-length(str2))*" [✓]"*"\n"; color = :green)

    str2 = "     γ(x) ⩾  0"
    print(str2)
    @test map(u -> d(u), seq) > [0.0]
    printstyled(" "*repeat(".", 50-length(str2))*" [✓]"*"\n"; color = :green)
end
#endregion

## ---- CressieRead ----
#=
Test Divergence.eval
=#
#region
seq = collect(0:0.1:3)
t₀ = Dict(
    2 => [0.3333333333,
        0.2835,
        0.234667,
        0.187833,
        0.144,
        0.104167,
        0.0693333,
        0.0405,
        0.0186667,
        0.00483333,
        0.0,
        0.00516667,
        0.0213333,
        0.0495,
        0.0906667,
        0.145833,
        0.216,
        0.302167,
        0.405333,
        0.5265,
        0.666667,
        0.826833,
        1.008,
        1.21117,
        1.43733,
        1.6875,
        1.96267,
        2.26383,
        2.592,
        2.94817,
        3.33333],
    0.5 => [0.6666666666,
        0.50883,
        0.385924,
        0.285756,
        0.203976,
        0.138071,
        0.086344,
        0.0475494,
        0.0207223,
        0.00508662,
        0.0,
        0.00491964,
        0.0193789,
        0.0429707,
        0.0753365,
        0.116156,
        0.165144,
        0.222038,
        0.286605,
        0.358626,
        0.437903,
        0.524252,
        0.617503,
        0.717497,
        0.824085,
        0.937129,
        1.0565,
        1.18207,
        1.31373,
        1.45136,
        1.59487],
    -0.5 => [2.0,
        0.935089,
        0.611146,
        0.40911,
        0.270178,
        0.171573,
        0.101613,
        0.0533599,
        0.0222912,
        0.00526681,
        0.0,
        0.00476461,
        0.0182195,
        0.0392983,
        0.0671362,
        0.101021,
        0.140356,
        0.184638,
        0.233437,
        0.28638,
        0.343146,
        0.403449,
        0.467041,
        0.5337,
        0.603227,
        0.675445,
        0.750194,
        0.827329,
        0.90672,
        0.988245,
        1.0718])
for (kv, val) in t₀
    cr = CressieRead(kv)
    str = "Testing "*string(cr)
    print(str)
    d = cr.(seq)
    @test d[2:end] ≈ val[2:end] rtol = 1e-05
    @test d[1] ≈ val[1]
    printstyled(" "*repeat(".", 40-length(str))*" [✓]"*"\n"; color = :green)
    @test cr(seq[2:end]) ≈ sum(val[2:end]) rtol = 1e-05
end

#=
Test gradient
=#
t₀ = Dict(
    2 => [-0.5,
        -0.495,
        -0.48,
        -0.455,
        -0.42,
        -0.375,
        -0.32,
        -0.255,
        -0.18,
        -0.095,
        0.0,
        0.105,
        0.22,
        0.345,
        0.48,
        0.625,
        0.78,
        0.945,
        1.12,
        1.305,
        1.5,
        1.705,
        1.92,
        2.145,
        2.38,
        2.625,
        2.88,
        3.145,
        3.42,
        3.705,
        4.0],
    0.5 => [-2.0,
        -1.36754,
        -1.10557,
        -0.904555,
        -0.735089,
        -0.585786,
        -0.450807,
        -0.32668,
        -0.211146,
        -0.102633,
        0.0,
        0.0976177,
        0.19089,
        0.280351,
        0.366432,
        0.44949,
        0.529822,
        0.607681,
        0.683282,
        0.75681,
        0.828427,
        0.898275,
        0.966479,
        1.03315,
        1.09839,
        1.16228,
        1.2249,
        1.28634,
        1.34664,
        1.40588,
        1.4641],
    -0.5 => [-Inf,
        -4.32456,
        -2.47214,
        -1.65148,
        -1.16228,
        -0.828427,
        -0.581989,
        -0.390457,
        -0.236068,
        -0.108185,
        0.0,
        0.0930748,
        0.174258,
        0.245884,
        0.309691,
        0.367007,
        0.418861,
        0.46607,
        0.509288,
        0.549047,
        0.585786,
        0.619869,
        0.6516,
        0.681239,
        0.709006,
        0.735089,
        0.759653,
        0.782839,
        0.804771,
        0.82556,
        0.84529])
for (kv, val) in t₀
    cr = CressieRead(kv)
    str = "Testing "*string(cr)
    print(str)
    d = map(a -> Divergences.gradient(cr, a), seq)
    @test maximum(d[2:end] .- val[2:end]) <= 1e-05
    @test d[1] ≈ val[1]
    printstyled(" "*repeat(".", 40-length(str))*" [✓]"*"\n"; color = :green)
    @test Divergences.gradient(cr, seq[2:end]) ≈ d[2:end]
end

#=
Test Divergence.hessian
=#
seq = 0:0.1:2
t₀ = Dict(
    2 => [Inf,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.1,
        1.2,
        1.3,
        1.4,
        1.5,
        1.6,
        1.7,
        1.8,
        1.9,
        2.0],
    0.5 => [Inf,
        3.16228,
        2.23607,
        1.82574,
        1.58114,
        1.41421,
        1.29099,
        1.19523,
        1.11803,
        1.05409,
        1.0,
        0.953463,
        0.912871,
        0.877058,
        0.845154,
        0.816497,
        0.790569,
        0.766965,
        0.745356,
        0.725476,
        0.707107])
for (kv, val) in t₀
    cr = CressieRead(kv)
    str = "Testing "*string(cr)
    print(str)
    d = map(a -> Divergences.hessian(cr, a), seq)
    @test maximum(d[2:end] .- val[2:end]) <= 1e-05
    @test d[1] == val[1]
    printstyled(" "*repeat(".", 40-length(str))*" [✓]"*"\n"; color = :green)
    @test Divergences.hessian(cr, seq[2:end]) ≈ d[2:end]
end
#endregion		

## ---- KullbackLeibler ----
#region
t₀ = Dict(
    Divergences.eval => [1.0,
        0.669741,
        0.478112,
        0.338808,
        0.233484,
        0.153426,
        0.0935046,
        0.0503275,
        0.0214852,
        0.00517554,
        0.0,
        0.0048412,
        0.0187859,
        0.0410735,
        0.0710611,
        0.108198,
        0.152006,
        0.202068,
        0.258016,
        0.319522,
        0.386294,
        0.458068,
        0.534606,
        0.615691,
        0.701125,
        0.790727,
        0.88433,
        0.98178,
        1.08293,
        1.18766,
        1.29584],
    Divergences.gradient => [-Inf,
        -2.30259,
        -1.60944,
        -1.20397,
        -0.916291,
        -0.693147,
        -0.510826,
        -0.356675,
        -0.223144,
        -0.105361,
        0.0,
        0.0953102,
        0.182322,
        0.262364,
        0.336472,
        0.405465,
        0.470004,
        0.530628,
        0.587787,
        0.641854,
        0.693147,
        0.741937,
        0.788457,
        0.832909,
        0.875469,
        0.916291,
        0.955511,
        0.993252,
        1.02962,
        1.06471,
        1.09861],
    Divergences.hessian => [Inf,
        10.0,
        5.0,
        3.33333,
        2.5,
        2.0,
        1.66667,
        1.42857,
        1.25,
        1.11111,
        1.0,
        0.909091,
        0.833333,
        0.769231,
        0.714286,
        0.666667,
        0.625,
        0.588235,
        0.555556,
        0.526316,
        0.5,
        0.47619,
        0.454545,
        0.434783,
        0.416667,
        0.4,
        0.384615,
        0.37037,
        0.357143,
        0.344828,
        0.333333])
testfun(𝒦ℒ, t₀, 0:0.1:3)
#endregion

## ---- ReverseKullbackLeibler ----
#region
t₀ = Dict(
    Divergences.eval => [Inf,
        1.40259,
        0.809438,
        0.503973,
        0.316291,
        0.193147,
        0.110826,
        0.0566749,
        0.0231436,
        0.00536052,
        0.0,
        0.00468982,
        0.0176784,
        0.0376357,
        0.0635278,
        0.0945349,
        0.129996,
        0.169372,
        0.212213,
        0.258146,
        0.306853,
        0.358063,
        0.411543,
        0.467091,
        0.524531,
        0.583709,
        0.644489,
        0.706748,
        0.770381,
        0.835289,
        0.901388],
    Divergences.gradient => [-Inf,
        -9.0,
        -4.0,
        -2.33333,
        -1.5,
        -1.0,
        -0.666667,
        -0.428571,
        -0.25,
        -0.111111,
        0.0,
        0.0909091,
        0.166667,
        0.230769,
        0.285714,
        0.333333,
        0.375,
        0.411765,
        0.444444,
        0.473684,
        0.5,
        0.52381,
        0.545455,
        0.565217,
        0.583333,
        0.6,
        0.615385,
        0.62963,
        0.642857,
        0.655172,
        0.666667],
    Divergences.hessian => [Inf,
        100.0,
        25.0,
        11.1111,
        6.25,
        4.0,
        2.77778,
        2.04082,
        1.5625,
        1.23457,
        1.0,
        0.826446,
        0.694444,
        0.591716,
        0.510204,
        0.444444,
        0.390625,
        0.346021,
        0.308642,
        0.277008,
        0.25,
        0.226757,
        0.206612,
        0.189036,
        0.173611,
        0.16,
        0.147929,
        0.137174,
        0.127551,
        0.118906,
        0.111111])

testfun(ℬ𝓊𝓇ℊ, t₀, 0:0.1:3)
#endregion

## ---- Hellinger ----
#region
t₀ = Dict(
    Divergences.eval => [2.0,
        0.935089,
        0.611146,
        0.40911,
        0.270178,
        0.171573,
        0.101613,
        0.0533599,
        0.0222912,
        0.00526681,
        0.0,
        0.00476461,
        0.0182195,
        0.0392983,
        0.0671362,
        0.101021,
        0.140356,
        0.184638,
        0.233437,
        0.28638,
        0.343146],
    Divergences.gradient => [-Inf,
        -4.32456,
        -2.47214,
        -1.65148,
        -1.16228,
        -0.828427,
        -0.581989,
        -0.390457,
        -0.236068,
        -0.108185,
        0.0,
        0.0930748,
        0.174258,
        0.245884,
        0.309691,
        0.367007,
        0.418861,
        0.46607,
        0.509288,
        0.549047,
        0.585786],
    Divergences.hessian => [Inf,
        31.6228,
        11.1803,
        6.08581,
        3.95285,
        2.82843,
        2.15166,
        1.70747,
        1.39754,
        1.17121,
        1.0,
        0.866784,
        0.760726,
        0.67466,
        0.603682,
        0.544331,
        0.494106,
        0.451156,
        0.414087,
        0.38183,
        0.353553])
testfun(ℋ𝒟, t₀, 0:0.1:2)
#endregion

## ---- Chi Squared ----
#region
seq = 0:0.1:2
t₀ = Dict(Divergences.eval => (seq .- 1) .^ 2/2,
    Divergences.gradient => (seq .- 1),
    Divergences.hessian => [1, seq[2:end] ./ seq[2:end]...])
testfun(χ², t₀, 0:0.1:2)
#endregion

## ---- Modified Divergence ----
#region 
#=
Given a divergence γ(x), the modified divergence is 
γᵤ(x) if x > ρ 
γ(x) if x <= ρ
where ρ > 1
=#

t₀ = Dict(
    Divergences.eval => [1.0,
        0.669741,
        0.478112,
        0.338808,
        0.233484,
        0.153426,
        0.0935046,
        0.0503275,
        0.0214852,
        0.00517554,
        0.0,
        0.0048412,
        0.0187859,
        0.0411847,
        0.0719168,
        0.110982,
        0.158381,
        0.214113,
        0.278179,
        0.350578,
        0.43131],
    Divergences.gradient => [-Inf,
        -2.30259,
        -1.60944,
        -1.20397,
        -0.916291,
        -0.693147,
        -0.510826,
        -0.356675,
        -0.223144,
        -0.105361,
        0.0,
        0.0953102,
        0.18232155679395456,
        0.265655,
        0.348988,
        0.432322,
        0.515655,
        0.598988,
        0.682322,
        0.765655,
        0.848988],
    Divergences.hessian => [Inf,
        10.0,
        5.0,
        3.33333,
        2.5,
        2.0,
        1.66667,
        1.42857,
        1.25,
        1.11111,
        1.0,
        0.909091,
        0.833333,
        0.833333,
        0.833333,
        0.833333,
        0.833333,
        0.833333,
        0.833333,
        0.833333,
        0.833333])
testfun(ModifiedDivergence(𝒦ℒ, 1.2), t₀, 0:0.1:2)
#endregion

## ---- FullyModified Divergence ----
#region
#=
Given a divergence γ(x), the modified divergence is 
γᵤ(x) if x >= ρ 
γₗ(x) if x <= φ 
γ(x) if x ∈ (φ, ρ)
where ρ > 1 && φ <1
=#
t₀ = Dict(
    Divergences.eval => [1.7039728053,
        1.14786,
        0.726195,
        0.438663,
        0.260981,
        0.147837,
        0.0755154,
        0.0313648,
        0.00783337,
        0.0000503359,
        0.00468982,
        0.0193798,
        0.0426784,
        0.0743798,
        0.114484,
        0.162991,
        0.219901,
        0.285213,
        0.358928,
        0.441046,
        0.531567,
        0.630491,
        0.737817,
        0.853546,
        0.977678,
        1.11021,
        1.25115,
        1.40049],
    Divergences.gradient => [-5.666666666,
        -4.44444,
        -3.22222,
        -2.0303,
        -1.27273,
        -0.818182,
        -0.515152,
        -0.298701,
        -0.136364,
        -0.010101,
        0.0909091,
        0.173611,
        0.25,
        0.326389,
        0.402778,
        0.479167,
        0.555556,
        0.631944,
        0.708333,
        0.784722,
        0.861111,
        0.9375,
        1.01389,
        1.09028,
        1.16667,
        1.24306,
        1.31944,
        1.3958],
    Divergences.hessian => [11.11111111,
        11.11111111,
        11.11111111,
        9.18274,
        5.16529,
        3.30579,
        2.29568,
        1.68663,
        1.29132,
        1.0203,
        0.826446,
        0.694444,
        0.694444,
        0.694444,
        0.694444,
        0.694444,
        0.694444,
        0.694444,
        0.694444,
        0.694444,
        0.694444,
        0.694444,
        0.694444,
        0.694444,
        0.694444,
        0.694444,
        0.694444,
        0.694444])
𝒟 = FullyModifiedDivergence(ℬ𝓊𝓇ℊ, 0.3, 1.2)
testfun(𝒟, t₀, 0:0.11:3)
#endregion

## Additional tests
@test_throws(DimensionMismatch, KullbackLeibler()(rand(10), rand(11)))
@test_throws(DimensionMismatch, CressieRead(2.0)(rand(10), rand(11)))
@test_throws(DimensionMismatch, ChiSquared()(rand(10), rand(11)))

divs = (𝒦ℒ, ℬ𝓊𝓇ℊ, CressieRead(1), ℋ𝒟, χ²)

# for d in divs
# 	r = rand(10)
# 	@test d.(r) ≈ d.(r, ones(length(r)))
#     @test d(r) ≈ d(r, ones(length(r)))
# 	@test Divergences.gradient(d, r) ≈ Divergences.gradient(d, r, ones(length(r)))
# 	@test Divergences.hessian(d, r) ≈ Divergences.hessian(d, r, ones(length(r)))
# end

x = rand(10)
@test sum(Divergences.gradient(ℱℳ𝒟, x)) ≈ Divergences.gradient_sum(ℱℳ𝒟, x)
@test sum(Divergences.hessian(ℱℳ𝒟, x)) ≈ Divergences.hessian_sum(ℱℳ𝒟, x)

@test Divergences.evaluate(ℱℳ𝒟, 3.2) ≈ Divergences.evaluate(ℱℳ𝒟, 3.2, 1.0)
@test Divergences.gradient(ℱℳ𝒟, 3.2) ≈ Divergences.gradient(ℱℳ𝒟, 3.2, 1.0)
@test Divergences.hessian(ℱℳ𝒟, 3.2) ≈ Divergences.hessian(ℱℳ𝒟, 3.2, 1.0)

# Divergences.hessian(ℱℳ𝒟, 3.2, 1)

# using ForwardDiff

# f(x) = Divergences.evaluate(ℱℳ𝒟, x)
# ForwardDiff.gradient(f, rand(10))

# f(x) = Divergences.evaluate(ℱℳ𝒟, x, rand(10))
# ForwardDiff.gradient(f, rand(10))

# Run Aqua.jl quality assurance tests
include("Aqua.jl")

# ξ = rand(1_000_000);
# using BenchmarkTools
# @btime Divergences.evaluate(ℱℳ𝒟, ξ)

# Test that deprecation warnings are issued for evaluate
# Use stderr capture to check for warnings
x = [1.0, 2.0, 3.0]
y = [1.0, 1.5, 2.0]
div = KullbackLeibler()

# Test single value evaluate with deprecation warning
@test_logs (:warn, r"evaluate\(div, x\) is deprecated, use div\(x\) instead") evaluate(div,
    2.0)
@test_logs (:warn, r"evaluate\(div, x, y\) is deprecated, use div\(x, y\) instead") evaluate(
    div,
    2.0,
    1.0)

# Test array evaluate with deprecation warning
@test_logs (:warn, r"evaluate\(div, x\) is deprecated, use div\(x\) instead") evaluate(div,
    x)
@test_logs (:warn, r"evaluate\(div, x, y\) is deprecated, use div\(x, y\) instead") evaluate(
    div,
    x,
    y)

# Test that results are the same
@test evaluate(div, 2.0) ≈ div(2.0)
@test evaluate(div, 2.0, 1.0) ≈ div(2.0, 1.0)
@test evaluate(div, x) ≈ div(x)
@test evaluate(div, x, y) ≈ div(x, y)

println("    Deprecation warnings... [✓]")
#endregion

## ---- Integer Array Type Promotion ----
#region
# Test that integer arrays are correctly promoted to float types
# This is important for type stability and compatibility with autodiff

@testset "Integer Array Type Promotion" begin
    divs = [KullbackLeibler(), ReverseKullbackLeibler(), Hellinger(),
            ChiSquared(), CressieRead(2.0)]

    a_int = [1, 2, 3]
    b_int = [1, 1, 1]
    a_float = [1.0, 2.0, 3.0]
    b_float = [1.0, 1.0, 1.0]

    for d in divs
        # Test divergence evaluation with Int arrays returns Float
        result = d(a_int, b_int)
        @test result isa AbstractFloat
        @test result ≈ d(a_float, b_float)

        # Test gradient with Int arrays returns Float array
        grad = Divergences.gradient(d, a_int, b_int)
        @test eltype(grad) <: AbstractFloat
        @test grad ≈ Divergences.gradient(d, a_float, b_float)

        # Test gradient with single Int array
        grad_single = Divergences.gradient(d, a_int)
        @test eltype(grad_single) <: AbstractFloat

        # Test hessian with Int arrays returns Float array
        hess = Divergences.hessian(d, a_int, b_int)
        @test eltype(hess) <: AbstractFloat
        @test hess ≈ Divergences.hessian(d, a_float, b_float)

        # Test hessian with single Int array
        hess_single = Divergences.hessian(d, a_int)
        @test eltype(hess_single) <: AbstractFloat

        # Test γ with Int array
        gamma_result = Divergences.γ(d, a_int)
        @test eltype(gamma_result) <: AbstractFloat
    end

    # Test mixed Int/Float arrays
    d = KullbackLeibler()
    result_mixed = d(a_int, b_float)
    @test result_mixed isa AbstractFloat
    @test result_mixed ≈ d(a_float, b_float)

    grad_mixed = Divergences.gradient(d, a_int, b_float)
    @test eltype(grad_mixed) <: AbstractFloat
end
#endregion

## ---- Float32 Type Preservation ----
#region
# Test that Float32 inputs produce Float32 outputs for ModifiedDivergence
# This ensures type stability when working with single-precision arrays

@testset "Float32 Type Preservation" begin
    KL = KullbackLeibler()
    RKL = ReverseKullbackLeibler()

    # Float32 input should give Float32 coefficients
    md32 = ModifiedDivergence(KL, 1.2f0)
    @test eltype(md32.m) == Float32

    # Float64 input should give Float64 coefficients
    md64 = ModifiedDivergence(KL, 1.2)
    @test eltype(md64.m) == Float64

    # Int input should promote to Float64
    md_int = ModifiedDivergence(KL, 2)
    @test eltype(md_int.m) == Float64

    # FullyModifiedDivergence with Float32 inputs
    fmd32 = FullyModifiedDivergence(KL, 0.3f0, 1.2f0)
    @test eltype(fmd32.m) == Float32

    # FullyModifiedDivergence with mixed Float32/Float64 should promote to Float64
    fmd_mixed = FullyModifiedDivergence(KL, 0.3f0, 1.2)
    @test eltype(fmd_mixed.m) == Float64

    # Test that operations on Float32 arrays produce Float32 results
    v32 = rand(Float32, 100) .* 2 .- 0.5f0

    # Test ψ (dual function) with Float32
    result_psi = Divergences.ψ(md32, v32[1])
    @test result_psi isa Float32

    # Test ∇ψ (dual gradient) with Float32
    result_grad = Divergences.∇ψ(md32, v32[1])
    @test result_grad isa Float32

    # Test Hψ (dual hessian) with Float32
    result_hess = Divergences.Hψ(md32, v32[1])
    @test result_hess isa Float32

    # Test array versions
    psi_array = Divergences.ψ(md32, v32)
    @test eltype(psi_array) == Float32

    # Test FullyModifiedDivergence dual functions with Float32
    fmd32_rkl = FullyModifiedDivergence(RKL, 0.3f0, 1.2f0)
    for v in [0.1f0, 0.5f0, 1.5f0]  # Test lower, middle, upper extension regions
        @test Divergences.ψ(fmd32_rkl, v) isa Float32
        @test Divergences.∇ψ(fmd32_rkl, v) isa Float32
        @test Divergences.Hψ(fmd32_rkl, v) isa Float32
    end

    # Verify correctness: Float32 and Float64 should give same numerical results
    v_test = 1.5f0
    @test Divergences.ψ(md32, v_test) ≈ Float32(Divergences.ψ(md64, Float64(v_test)))
    @test Divergences.∇ψ(md32, v_test) ≈ Float32(Divergences.∇ψ(md64, Float64(v_test)))
    @test Divergences.Hψ(md32, v_test) ≈ Float32(Divergences.Hψ(md64, Float64(v_test)))
end
#endregion

## ---- ForwardDiff Verification ----
#region
# Verify that analytical gradients and hessians match ForwardDiff automatic differentiation

@testset "ForwardDiff Verification" begin
    # Test points that cover different regions
    test_points = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]

    # Basic divergences
    basic_divs = [
        KullbackLeibler(),
        ReverseKullbackLeibler(),
        Hellinger(),
        ChiSquared(),
        CressieRead(2.0),
        CressieRead(0.5),
        CressieRead(-0.5)
    ]

    @testset "Primal γ derivatives - $(typeof(d))" for d in basic_divs
        for x in test_points
            # Skip invalid points for certain divergences
            if d isa ReverseKullbackLeibler && x <= 0
                continue
            end

            # Test gradient: compare analytical vs ForwardDiff
            analytical_grad = Divergences.gradient(d, x)
            forwarddiff_grad = ForwardDiff.derivative(u -> Divergences.γ(d, u), x)
            @test analytical_grad ≈ forwarddiff_grad rtol=1e-10

            # Test hessian: compare analytical vs ForwardDiff
            analytical_hess = Divergences.hessian(d, x)
            forwarddiff_hess = ForwardDiff.derivative(u -> Divergences.∇ᵧ(d, u), x)
            if isfinite(analytical_hess) && isfinite(forwarddiff_hess)
                @test analytical_hess ≈ forwarddiff_hess rtol=1e-10
            end
        end
    end

    # Modified divergences
    mod_divs = [
        ModifiedDivergence(KullbackLeibler(), 1.2),
        ModifiedDivergence(ReverseKullbackLeibler(), 1.3),
        ModifiedDivergence(Hellinger(), 1.5),
        ModifiedDivergence(ChiSquared(), 1.4),
        ModifiedDivergence(CressieRead(2.0), 1.2)
    ]

    @testset "Modified γ derivatives - $(typeof(d.d))" for d in mod_divs
        for x in test_points
            # Skip invalid points
            if d.d isa ReverseKullbackLeibler && x <= 0
                continue
            end

            analytical_grad = Divergences.gradient(d, x)
            forwarddiff_grad = ForwardDiff.derivative(u -> Divergences.γ(d, u), x)
            @test analytical_grad ≈ forwarddiff_grad rtol=1e-10

            analytical_hess = Divergences.hessian(d, x)
            forwarddiff_hess = ForwardDiff.derivative(u -> Divergences.∇ᵧ(d, u), x)
            if isfinite(analytical_hess) && isfinite(forwarddiff_hess)
                @test analytical_hess ≈ forwarddiff_hess rtol=1e-10
            end
        end
    end

    # FullyModified divergences
    fmod_divs = [
        FullyModifiedDivergence(KullbackLeibler(), 0.5, 1.5),
        FullyModifiedDivergence(ReverseKullbackLeibler(), 0.3, 1.2),
        FullyModifiedDivergence(ChiSquared(), 0.4, 1.6)
    ]

    @testset "FullyModified γ derivatives - $(typeof(d.d))" for d in fmod_divs
        for x in test_points
            analytical_grad = Divergences.gradient(d, x)
            forwarddiff_grad = ForwardDiff.derivative(u -> Divergences.γ(d, u), x)
            @test analytical_grad ≈ forwarddiff_grad rtol=1e-10

            analytical_hess = Divergences.hessian(d, x)
            forwarddiff_hess = ForwardDiff.derivative(u -> Divergences.∇ᵧ(d, u), x)
            if isfinite(analytical_hess) && isfinite(forwarddiff_hess)
                @test analytical_hess ≈ forwarddiff_hess rtol=1e-10
            end
        end
    end

    # Dual function derivatives
    dual_test_points = [-0.5, 0.0, 0.3, 0.5, 0.8, 1.0]

    @testset "Dual ψ derivatives - $(typeof(d))" for d in basic_divs
        for v in dual_test_points
            # Skip invalid points (e.g., RKL requires v < 1, Hellinger requires v < 2)
            if d isa ReverseKullbackLeibler && v >= 1
                continue
            end
            if d isa Hellinger && v >= 2
                continue
            end
            if d isa CressieRead && (1 + d.α * v) <= 0
                continue
            end

            # Test ∇ψ: compare analytical vs ForwardDiff
            analytical_grad = Divergences.∇ψ(d, v)
            forwarddiff_grad = ForwardDiff.derivative(u -> Divergences.ψ(d, u), v)
            if isfinite(analytical_grad) && isfinite(forwarddiff_grad)
                @test analytical_grad ≈ forwarddiff_grad rtol=1e-10
            end

            # Test Hψ: compare analytical vs ForwardDiff
            analytical_hess = Divergences.Hψ(d, v)
            forwarddiff_hess = ForwardDiff.derivative(u -> Divergences.∇ψ(d, u), v)
            if isfinite(analytical_hess) && isfinite(forwarddiff_hess)
                @test analytical_hess ≈ forwarddiff_hess rtol=1e-10
            end
        end
    end

    # Modified divergence dual derivatives
    @testset "Modified ψ derivatives - $(typeof(d.d))" for d in mod_divs
        for v in dual_test_points
            # Get the threshold for this modified divergence
            γ₁ = d.m.γ₁

            analytical_grad = Divergences.∇ψ(d, v)
            forwarddiff_grad = ForwardDiff.derivative(u -> Divergences.ψ(d, u), v)
            if isfinite(analytical_grad) && isfinite(forwarddiff_grad)
                @test analytical_grad ≈ forwarddiff_grad rtol=1e-10
            end

            analytical_hess = Divergences.Hψ(d, v)
            forwarddiff_hess = ForwardDiff.derivative(u -> Divergences.∇ψ(d, u), v)
            if isfinite(analytical_hess) && isfinite(forwarddiff_hess)
                @test analytical_hess ≈ forwarddiff_hess rtol=1e-10
            end
        end
    end

    # Test array divergence gradient via ForwardDiff
    @testset "Array divergence gradient" begin
        d = KullbackLeibler()
        a = [0.5, 1.0, 1.5, 2.0]
        b = [1.0, 1.0, 1.0, 1.0]

        # Gradient of D(a,b) w.r.t. a
        f(x) = d(x, b)
        forwarddiff_grad = ForwardDiff.gradient(f, a)
        # The gradient is γ'(aᵢ/bᵢ) for each i
        analytical_grad = Divergences.gradient(d, a, b)
        @test forwarddiff_grad ≈ analytical_grad rtol=1e-10

        # Test with ModifiedDivergence
        md = ModifiedDivergence(KullbackLeibler(), 1.2)
        f_mod(x) = md(x, b)
        forwarddiff_grad_mod = ForwardDiff.gradient(f_mod, a)
        analytical_grad_mod = Divergences.gradient(md, a, b)
        @test forwarddiff_grad_mod ≈ analytical_grad_mod rtol=1e-10

        # Test with FullyModifiedDivergence
        fmd = FullyModifiedDivergence(KullbackLeibler(), 0.5, 1.5)
        f_fmod(x) = fmd(x, b)
        forwarddiff_grad_fmod = ForwardDiff.gradient(f_fmod, a)
        analytical_grad_fmod = Divergences.gradient(fmd, a, b)
        @test forwarddiff_grad_fmod ≈ analytical_grad_fmod rtol=1e-10
    end

    # Test dual divergence gradient via ForwardDiff
    @testset "Dual divergence gradient" begin
        d = KullbackLeibler()
        v = [0.0, 0.2, 0.4, 0.6]
        b = [1.0, 1.0, 1.0, 1.0]

        # Gradient of Dψ(v,b) w.r.t. v
        f(x) = Divergences.dual(d, x, b)
        forwarddiff_grad = ForwardDiff.gradient(f, v)
        # The gradient is ψ'(vᵢ) * bᵢ for each i
        analytical_grad = Divergences.dual_gradient(d, v, b)
        @test forwarddiff_grad ≈ analytical_grad rtol=1e-10

        # Test with ModifiedDivergence
        md = ModifiedDivergence(KullbackLeibler(), 1.2)
        f_mod(x) = Divergences.dual(md, x, b)
        forwarddiff_grad_mod = ForwardDiff.gradient(f_mod, v)
        analytical_grad_mod = Divergences.dual_gradient(md, v, b)
        @test forwarddiff_grad_mod ≈ analytical_grad_mod rtol=1e-10
    end
end
#endregion

include("test_duals.jl")
