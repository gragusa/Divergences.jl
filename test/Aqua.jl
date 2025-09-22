using Test
using Aqua
using Divergences

@testset "Aqua.jl" begin
    Aqua.test_all(Divergences)
end