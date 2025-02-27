using Divergences
using Test

function testfun(𝒟, t₀, s)
    println("Testing "*string(𝒟))
    for (f, v) ∈ t₀
        str2 = "    "*string(f)
        print(str2)
        d = map(a -> f(𝒟, a), s)
        @test maximum(d[2:end] .- v[2:end]) <= 1e-04
        @test d[1] ≈ v[1]
        s₀ = f != Divergences.evaluate ? d[2:end] : sum(d[2:end])
        @test f(𝒟, s[2:end]) ≈ s₀
        printstyled(" "*repeat(".", 40-length(str2))*" [✓]"*"\n", color = :green)
    end
end

#=
Check that all Divergences satisfy the normalization
1.  γ(1) ≖ 0
2. γ'(1) ≖ 0
3.  γ(x) ⩾ 0
=#
#region

#=
Check that all Divergences satisfy the normalization
1.  γ(1) ≖ 0
2. γ'(1) ≖ 0
3.  γ(x) ⩾ 0
=#
seq = 0:.1:3
divs = (KullbackLeibler(), ReverseKullbackLeibler(), Hellinger(), [CressieRead(p) for p ∈ (-0.5, 0.5, 2. )]...)
for d ∈ divs
	str = "Testing normalization: "*string(d)
	println(str)
	str2 = "     γ(1) == 0"
	print(str2)
	@test Divergences.evaluate(d, [1.]) == 0
	printstyled(" "*repeat(".", 50-length(str2))*" [✓]"*"\n", color = :green)

	str2 = "    γ'(1) == 0"
	print(str2)
	@test Divergences.gradient(d, [1.]) == [0.0]
	printstyled(" "*repeat(".", 50-length(str2))*" [✓]"*"\n", color = :green)

	str2 = "     γ(x) ⩾  0"
	print(str2)
	@test map(u -> Divergences.evaluate(d, [u])[1], seq) > [0.0]
	printstyled(" "*repeat(".", 50-length(str2))*" [✓]"*"\n", color = :green)
end
#endregion

## ---- CressieRead ----
#=
Test Divergence.eval
=#
#region
seq = 0:.1:3
t₀ = Dict(
	2    => [0.3333333333, 0.2835, 0.234667, 0.187833, 0.144, 0.104167, 0.0693333, 0.0405, 0.0186667, 0.00483333, 0., 0.00516667, 0.0213333, 0.0495, 0.0906667, 0.145833, 0.216, 0.302167, 0.405333, 0.5265, 0.666667, 0.826833, 1.008, 1.21117, 1.43733, 1.6875, 1.96267, 2.26383, 2.592, 2.94817, 3.33333],
	0.5  => [0.6666666666, 0.50883, 0.385924, 0.285756, 0.203976, 0.138071, 0.086344, 0.0475494, 0.0207223, 0.00508662, 0., 0.00491964, 0.0193789, 0.0429707, 0.0753365, 0.116156, 0.165144, 0.222038, 0.286605, 0.358626, 0.437903, 0.524252, 0.617503, 0.717497, 0.824085, 0.937129, 1.0565, 1.18207, 1.31373, 1.45136, 1.59487],
	-0.5 => [2., 0.935089, 0.611146, 0.40911, 0.270178, 0.171573, 0.101613, 0.0533599, 0.0222912, 0.00526681, 0., 0.00476461, 0.0182195, 0.0392983, 0.0671362, 0.101021, 0.140356, 0.184638, 0.233437, 0.28638, 0.343146, 0.403449, 0.467041, 0.5337, 0.603227, 0.675445, 0.750194, 0.827329, 0.90672, 0.988245, 1.0718]
)
for (kv, val) ∈ t₀
	cr = CressieRead(kv)
	str = "Testing "*string(cr)
	print(str)
	d = map(a -> Divergences.evaluate(cr, a), seq)
	@test maximum(d[2:end] .- val[2:end]) <= 1e-05
	@test d[1] ≈ val[1]
	printstyled(" "*repeat(".", 40-length(str))*" [✓]"*"\n", color = :green)
	@test Divergences.evaluate(cr, seq[2:end]) ≈ sum(d[2:end])
end

#=
Test gradient
=#
t₀ = Dict(
	2    => [-0.5, -0.495, -0.48, -0.455, -0.42, -0.375, -0.32, -0.255, -0.18, -0.095, 0., 0.105, 0.22, 0.345, 0.48, 0.625, 0.78, 0.945, 1.12, 1.305, 1.5, 1.705, 1.92, 2.145, 2.38, 2.625, 2.88, 3.145, 3.42, 3.705, 4.],
	0.5  => [-2., -1.36754, -1.10557, -0.904555, -0.735089, -0.585786, -0.450807, -0.32668, -0.211146, -0.102633, 0., 0.0976177, 0.19089, 0.280351, 0.366432, 0.44949, 0.529822, 0.607681, 0.683282, 0.75681, 0.828427, 0.898275, 0.966479, 1.03315, 1.09839, 1.16228, 1.2249, 1.28634,  1.34664, 1.40588, 1.4641],
	-0.5 => [-Inf, -4.32456, -2.47214, -1.65148, -1.16228, -0.828427, -0.581989, -0.390457, -0.236068, -0.108185, 0., 0.0930748, 0.174258, 0.245884, 0.309691, 0.367007, 0.418861, 0.46607, 0.509288, 0.549047, 0.585786, 0.619869, 0.6516, 0.681239, 0.709006, 0.735089, 0.759653, 0.782839, 0.804771, 0.82556, 0.84529]
)
for (kv, val) ∈ t₀
	cr = CressieRead(kv)
	str = "Testing "*string(cr)
	print(str)
	d = map(a -> Divergences.gradient(cr, a), seq)
	@test maximum(d[2:end] .- val[2:end]) <= 1e-05
	@test d[1] ≈ val[1]
	printstyled(" "*repeat(".", 40-length(str))*" [✓]"*"\n", color = :green)
	@test Divergences.gradient(cr, seq[2:end]) ≈ d[2:end]
end

#=
Test Divergence.hessian
=#
seq = 0:0.1:2
t₀ = Dict(
	2    => [Inf, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.],
	0.5  => [Inf,3.16228,2.23607,1.82574,1.58114,1.41421,1.29099,1.19523,1.11803,1.05409,1.,0.953463,0.912871,0.877058,0.845154,0.816497,0.790569,0.766965,0.745356,0.725476,0.707107]
)
for (kv, val) ∈ t₀
	cr = CressieRead(kv)
	str = "Testing "*string(cr)
	print(str)
	d = map(a -> Divergences.hessian(cr, a), seq)
	@test maximum(d[2:end] .- val[2:end]) <= 1e-05
	@test d[1] == val[1]
	printstyled(" "*repeat(".", 40-length(str))*" [✓]"*"\n", color = :green)
	@test Divergences.hessian(cr, seq[2:end]) ≈ d[2:end]
end
#endregion

## ---- KullbackLeibler ----
#region
t₀ = Dict(
	Divergences.evaluate     => [1.0, 0.669741, 0.478112, 0.338808, 0.233484, 0.153426, .0935046, .0503275, .0214852, 0.00517554, 0., 0.0048412, 0.0187859, 0.0410735, 0.0710611, 0.108198, 							0.152006, 0.202068, 0.258016, 0.319522, 0.386294, 0.458068, 0.534606, 0.615691, 0.701125, 0.790727, 0.88433, 0.98178, 1.08293, 1.18766, 1.29584],
	Divergences.gradient => [-Inf, -2.30259, -1.60944, -1.20397, -0.916291, -0.693147, -0.510826, -0.356675, -0.223144, -0.105361, 0., 0.0953102, 0.182322, 0.262364, 0.336472, 0.405465, 0.470004, 0.530628, 0.587787, 0.641854, 0.693147, 0.741937, 0.788457, 0.832909, 0.875469, 0.916291, 0.955511, 0.993252, 1.02962, 1.06471, 1.09861],
	Divergences.hessian  => [Inf, 10., 5., 3.33333, 2.5, 2., 1.66667, 1.42857, 1.25, 1.11111, 1., 0.909091, 0.833333, 0.769231, 0.714286, 0.666667, 0.625, 0.588235, 0.555556, 0.526316, 0.5, 0.47619, 0.454545, 0.434783, 0.416667, 0.4, 0.384615, 0.37037, 0.357143, 0.344828, 0.333333]
)
𝒟 = 𝒦ℒ()
testfun(𝒦ℒ(), t₀, 0:0.1:3)
#endregion

## ---- ReverseKullbackLeibler ----
#region
t₀ = Dict(
	Divergences.evaluate     => [Inf ,1.40259,0.809438,0.503973,0.316291,0.193147,0.110826,0.0566749,0.0231436,0.00536052,0.,0.00468982,0.0176784,0.0376357,0.0635278,0.0945349,0.129996,0.169372,0.212213,0.258146,0.306853,0.358063,0.411543,0.467091,0.524531,0.583709,0.644489,0.706748,0.770381,0.835289,0.901388],
	Divergences.gradient => [-Inf,-9.,-4.,-2.33333,-1.5,-1.,-0.666667,-0.428571,-0.25,-0.111111,0.,0.0909091,0.166667,0.230769,0.285714,0.333333,0.375,0.411765,0.444444,0.473684,0.5,0.52381,0.545455,0.565217,0.583333,0.6,0.615385,0.62963,0.642857,0.655172,0.666667],
	Divergences.hessian  => [Inf, 100., 25., 11.1111, 6.25, 4., 2.77778, 2.04082, 1.5625, 1.23457, 1., 0.826446, 0.694444, 0.591716, 0.510204, 0.444444, 0.390625, 0.346021, 0.308642, 0.277008, 0.25, 0.226757, 0.206612, 0.189036, 0.173611, 0.16, 0.147929, 0.137174, 0.127551, 0.118906, 0.111111]
)

testfun(ℬ𝓊𝓇ℊ(), t₀, 0:0.1:3)
#endregion

## ---- Hellinger ----
#region
t₀ = Dict(
	Divergences.evaluate     => [2., 0.935089, 0.611146, 0.40911, 0.270178, 0.171573, 0.101613,  0.0533599, 0.0222912, 0.00526681, 0., 0.00476461, 0.0182195, 0.0392983, 0.0671362, 0.101021, 0.140356, 0.184638, 0.233437,  0.28638, 0.343146],
	Divergences.gradient => [-Inf, -4.32456, -2.47214, -1.65148, -1.16228, -0.828427, -0.581989, -0.390457, -0.236068, -0.108185, 0., 0.0930748, 0.174258, 0.245884, 0.309691, 0.367007, 0.418861, 0.46607, 0.509288, 0.549047, 0.585786],
	Divergences.hessian  => [Inf, 31.6228, 11.1803, 6.08581, 3.95285, 2.82843, 2.15166, 1.70747, 1.39754, 1.17121, 1., 0.866784, 0.760726, 0.67466, 0.603682, 0.544331, 0.494106, 0.451156, 0.414087, 0.38183, 0.353553]
)
testfun(ℋ𝒟(), t₀, 0:0.1:2)
#endregion

## ---- Chi Squared ----
#region
seq = 0:0.1:2
t₀ = Dict(
	Divergences.evaluate     => (seq .- 1).^2/2,
	Divergences.gradient => (seq .- 1),
	Divergences.hessian  => [1, seq[2:end]./seq[2:end]...]
)
testfun(χ²(), t₀, 0:0.1:2)
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
	Divergences.evaluate => [1.0, 0.669741, 0.478112, 0.338808, 0.233484, 0.153426, 0.0935046, 0.0503275, 0.0214852, 0.00517554, 0., 0.0048412, 0.0187859, 0.0411847, 0.0719168, 0.110982, 0.158381, 0.214113, 0.278179, 0.350578, 0.43131],
	Divergences.gradient => [-Inf,-2.30259,-1.60944,-1.20397,-0.916291,-0.693147,-0.510826,-0.356675,-0.223144,-0.105361,0.,0.0953102,0.18232155679395456,0.265655,0.348988,0.432322,0.515655,0.598988,0.682322,0.765655,0.848988],
	Divergences.hessian => [Inf, 10., 5., 3.33333, 2.5, 2., 1.66667, 1.42857, 1.25, 1.11111, 1., 0.909091, 0.833333, 0.833333, 0.833333, 0.833333, 0.833333, 0.833333, 0.833333, 0.833333, 0.833333]
)
testfun(ModifiedDivergence(𝒦ℒ(), 1.2), t₀, 0:0.1:2)
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
	Divergences.evaluate => [1.7039728053,1.14786,0.726195,0.438663,0.260981,0.147837,0.0755154,0.0313648,0.00783337,0.0000503359,0.00468982,0.0193798,0.0426784,0.0743798,0.114484,0.162991,0.219901,0.285213,0.358928,0.441046,0.531567,0.630491,0.737817,0.853546,0.977678,1.11021,1.25115,1.40049],
	Divergences.gradient => [-5.666666666,-4.44444,-3.22222,-2.0303,-1.27273,-0.818182,-0.515152,-0.298701,-0.136364,-0.010101,0.0909091,0.173611,0.25,0.326389,0.402778,0.479167,0.555556,0.631944,0.708333,0.784722,0.861111,0.9375,1.01389,1.09028,1.16667,1.24306,1.31944,1.3958],
	Divergences.hessian => [11.11111111,11.11111111,11.11111111,9.18274,5.16529,3.30579,2.29568,1.68663,1.29132,1.0203,0.826446,0.694444,0.694444,0.694444,0.694444,0.694444,0.694444,0.694444,0.694444,0.694444,0.694444,0.694444,0.694444,0.694444,0.694444,0.694444,0.694444,0.694444]
)
𝒟 = FullyModifiedDivergence(ℬ𝓊𝓇ℊ(), 0.3, 1.2)
testfun(𝒟, t₀, 0:0.11:3)
#endregion

## Additional tests
@test_throws(DimensionMismatch, Divergences.evaluate(𝒦ℒ(), rand(10), rand(11)))
@test_throws(DimensionMismatch, Divergences.evaluate(ℬ𝓊𝓇ℊ(), rand(10), rand(11)))
@test_throws(DimensionMismatch, Divergences.evaluate(𝒞ℛ(1), rand(10), rand(11)))
@test_throws(DimensionMismatch, Divergences.evaluate(χ²(), rand(10), rand(11)))

divs = (𝒦ℒ(), ℬ𝓊𝓇ℊ(), 𝒞ℛ(1), ℋ𝒟(), χ²())

for d in divs
	@show d
	r = rand(10)
	@test Divergences.evaluate(d, r) ≈ Divergences.evaluate(d, r, ones(length(r)))
	@test Divergences.gradient(d, r) ≈ Divergences.gradient(d, r, ones(length(r)))
	@test Divergences.hessian(d, r) ≈ Divergences.hessian(d, r, ones(length(r)))
end

ℳ𝒟 = ModifiedDivergence(𝒦ℒ(), 1.2)
ℱℳ𝒟 = FullyModifiedDivergence(𝒦ℒ(), 0.9, 1.2)

s = 1:0.1:2
t₀ = [0.0176784,0.00428749,0.,0.00405552,0.015811,0.0347323,0.0606034,0.0934189,0.133179,0.179883,0.233532]
d₀ = map(u -> Divergences.evaluate(ℳ𝒟, u, 1.2), s)
@test maximum(d₀ .- t₀) < 1e-06
@test Divergences.evaluate(ℳ𝒟, s, repeat([1.2], length(s))) ≈ sum(d₀)
t₀ = [-0.182322, -0.0870114, 0., 0.0800427, 0.154151, 0.223988, 0.293433, 0.362877, 0.432322, 0.501766, 0.5712]
d₀ = map(u -> Divergences.gradient(ℳ𝒟, u, 1.2), s)
@test maximum(d₀ - t₀) < 2e-05
@test Divergences.gradient(ℳ𝒟, s, repeat([1.2], length(s))) ≈ d₀
t₀ =[1.,0.909091,0.833333,0.769231,0.714286,0.694444,0.694444,0.694444,0.694444,0.694444,0.694444]
d₀ = map(u -> Divergences.hessian(ℳ𝒟, u, 1.2), s)
@test maximum(d₀ - t₀) < 2e-05
@test Divergences.hessian(ℳ𝒟, s, repeat([1.2], length(s))) ≈ d₀

s = 0.1:0.1:2
t₀ = [0.554094,0.457446,0.370059,0.29193,0.22306,0.16345,0.113099,0.0720079,0.0401755,0.0176024,0.00428749,
	  0.,0.00405552,0.015811,0.0347323,0.0606034,0.0934189,0.133179,0.179883,0.233532]
d₀ = map(u -> Divergences.evaluate(ℱℳ𝒟, u, 1.2), s)
@test maximum(d₀ .- t₀) < 1e-06
@test Divergences.evaluate(ℱℳ𝒟, s, repeat([1.2], length(s))) ≈ sum(d₀)
t₀ = [-1.01277,-0.920175,-0.827583,-0.73499,-0.642398,-0.549805,-0.457212,-0.36462,-0.272027,-0.179435,-0.0870114,
	  2.22045*10^-16,0.0800427,0.154151,0.223988,0.293433,0.362877,0.432322,0.501766,0.57121]
d₀ = map(u -> Divergences.gradient(ℱℳ𝒟, u, 1.2), s)
@test maximum(d₀ - t₀) < 2e-05
@test Divergences.gradient(ℱℳ𝒟, s, repeat([1.2], length(s))) ≈ d₀
t₀ = [0.925926,0.925926,0.925926,0.925926,0.925926,0.925926,0.925926,0.925926,0.925926,0.925926,0.909091,0.833333,
	  0.769231,0.714286,0.694444,0.694444,0.694444,0.694444,0.694444,0.694444]
d₀ = map(u -> Divergences.hessian(ℱℳ𝒟, u, 1.2), s)
@test maximum(d₀ - t₀) < 2e-05
@test Divergences.hessian(ℱℳ𝒟, s, repeat([1.2], length(s))) ≈ d₀

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

# ξ = rand(1_000_000);
# using BenchmarkTools
# @btime Divergences.evaluate(ℱℳ𝒟, ξ)

# @btime Divergences.evaluate(𝒦ℒ(), ξ)
# @btime Divergences.gradient(𝒦ℒ(), ξ);
# @btime Divergences.hessian(𝒦ℒ(), ξ);
