using Divergences
using Test

@testset "Dual Functions" begin
    # Define test divergences
    KL = KullbackLeibler()
    RKL = ReverseKullbackLeibler()
    CR2 = CressieRead(2.0)
    CR05 = CressieRead(0.5)
    CRneg = CressieRead(-0.5)
    HD = Hellinger()
    Chi2 = ChiSquared()
    MD = ModifiedDivergence(KL, 1.2)
    FMD = FullyModifiedDivergence(RKL, 0.3, 1.2)

    @testset "Dual values: ψ(v) analytical forms" begin
        # Kullback-Leibler: ψ(v) = exp(v) - 1
        @testset "KullbackLeibler" begin
            for v in -2:0.5:3
                @test Divergences.dual(KL, v) ≈ exp(v) - 1 rtol=1e-10
                @test Divergences.dual_gradient(KL, v) ≈ exp(v) rtol=1e-10
                @test Divergences.dual_hessian(KL, v) ≈ exp(v) rtol=1e-10
            end
        end

        # Reverse KL: ψ(v) = -log(1-v) for v < 1
        @testset "ReverseKullbackLeibler" begin
            for v in -2:0.2:0.8
                @test Divergences.dual(RKL, v) ≈ -log(1 - v) rtol=1e-10
                @test Divergences.dual_gradient(RKL, v) ≈ 1 / (1 - v) rtol=1e-10
                @test Divergences.dual_hessian(RKL, v) ≈ 1 / (1 - v)^2 rtol=1e-10
            end
        end

        # Chi-squared: ψ(v) = v²/2 + v
        @testset "ChiSquared" begin
            for v in -3:0.5:3
                @test Divergences.dual(Chi2, v) ≈ v^2/2 + v rtol=1e-10
                @test Divergences.dual_gradient(Chi2, v) ≈ v + 1 rtol=1e-10
                @test Divergences.dual_hessian(Chi2, v) ≈ 1.0 rtol=1e-10
            end
        end

        # Hellinger: ψ(v) = 2v/(2-v) for v < 2
        @testset "Hellinger" begin
            for v in -2:0.3:1.5
                @test Divergences.dual(HD, v) ≈ 2v / (2 - v) rtol=1e-10
                @test Divergences.dual_gradient(HD, v) ≈ 4 / (2 - v)^2 rtol=1e-10
                @test Divergences.dual_hessian(HD, v) ≈ 8 / (2 - v)^3 rtol=1e-10
            end
        end

        # Cressie-Read: ψ(v) = u·v - γ(u) where u = (1+αv)^(1/α)
        # γ(u) = [u^(1+α) - 1]/[α(1+α)] - (u-1)/α
        # Domain: w = 1 + αv > 0, i.e., v > -1/α
        @testset "CressieRead" begin
            for (cr, α, v_range) in [(CR2, 2.0, -0.4:0.3:2),      # boundary at v=-0.5
                (CR05, 0.5, -0.5:0.3:1.5),   # boundary at v=-2
                (CRneg, -0.5, -1.5:0.3:1.8)] # boundary at v=2
                for v in v_range
                    w = 1 + α * v
                    if w > 0  # only test inside domain
                        u = w^(1/α)
                        γ_u = (u^(1+α) - 1)/(α*(1+α)) - (u - 1)/α
                        expected = u * v - γ_u
                        @test Divergences.dual(cr, v) ≈ expected rtol=1e-10
                        @test Divergences.dual_gradient(cr, v) ≈ w^(1/α) rtol=1e-10
                        @test Divergences.dual_hessian(cr, v) ≈ w^((1-α)/α) rtol=1e-10
                    end
                end
            end
        end
    end

    @testset "Full dual divergence Dψ(v, b)" begin
        for (d, name, v_vals) in [
            (KL, "KL", [-1.0, 0.0, 0.5, 1.0]),
            (RKL, "RKL", [-0.5, 0.0, 0.3, 0.5]),
            (Chi2, "Chi²", [-1.0, 0.0, 0.5, 1.0]),
            (HD, "Hellinger", [-0.5, 0.0, 0.5, 1.0])
        ]
            b = [0.2, 0.3, 0.25, 0.25]
            expected = sum(Divergences.dual.(Ref(d), v_vals) .* b)
            @test Divergences.dual(d, v_vals, b) ≈ expected rtol=1e-10
        end
    end

    @testset "Fenchel-Young inequality" begin
        # D(a,b) + Dψ(v,b) - Σᵢ aᵢvᵢ ≥ 0
        for (d, v_range) in [
            (KL, -2:0.5:2),
            (RKL, -2:0.2:0.8),
            (Chi2, -3:0.5:3),
            (HD, -2:0.3:1.5),
            (CR2, -1:0.3:2)
        ]
            for a in 0.1:0.3:3, b in 0.5:0.5:2, v in v_range
                gap = Divergences.fenchel_young(d, a, b, v)
                @test gap >= -1e-10
            end
        end
    end

    @testset "Duality relationship" begin
        # D(a,b) + Dψ(v,b) = Σᵢ aᵢvᵢ when v = ∇ₐD(a,b)
        for d in [KL, RKL, Chi2, HD, CR2, CR05, CRneg]
            for a in 0.1:0.3:3, b in 0.5:0.5:2

                @test Divergences.verify_duality(d, a, b) < 1e-10
            end
        end
    end

    @testset "Primal-dual conversion" begin
        # a → v → a should recover a
        for d in [KL, RKL, Chi2, HD, CR2, CR05, CRneg]
            for a in 0.1:0.3:3, b in 0.5:0.5:2

                v = Divergences.dual_from_primal(d, a, b)
                a_recovered = Divergences.primal_from_dual(d, v, b)
                @test a ≈ a_recovered rtol=1e-10
            end
        end
    end

    @testset "Array operations" begin
        for (d, v_gen) in [
            (KL, () -> randn(10)),
            (RKL, () -> rand(10) .- 0.5),
            (Chi2, () -> randn(10)),
            (HD, () -> rand(10) .* 1.5 .- 0.5)
        ]
            v_arr = v_gen()
            b_arr = rand(10) .+ 0.1

            # dual(d, v, b) == sum of elementwise
            @test Divergences.dual(d, v_arr, b_arr) ≈
                  sum(Divergences.dual.(Ref(d), v_arr) .* b_arr)

            # dual_gradient with b
            grad_arr = Divergences.dual_gradient(d, v_arr, b_arr)
            grad_manual = [Divergences.dual_gradient(d, v_arr[i]) * b_arr[i]
                           for i in eachindex(v_arr)]
            @test grad_arr ≈ grad_manual

            # dual_hessian with b
            hess_arr = Divergences.dual_hessian(d, v_arr, b_arr)
            hess_manual = [Divergences.dual_hessian(d, v_arr[i]) * b_arr[i]
                           for i in eachindex(v_arr)]
            @test hess_arr ≈ hess_manual

            # primal_from_dual
            a_arr = Divergences.primal_from_dual(d, v_arr, b_arr)
            a_manual = [Divergences.primal_from_dual(d, v_arr[i], b_arr[i])
                        for i in eachindex(v_arr)]
            @test a_arr ≈ a_manual
        end
    end

    @testset "Modified divergence duals" begin
        ρ = 1.2
        md_kl = ModifiedDivergence(KL, ρ)
        γ1_kl = Divergences.gradient(KL, ρ)

        # Below threshold: matches original KL dual
        for v in -2:0.2:(γ1_kl - 0.1)
            @test Divergences.dual(md_kl, v) ≈ exp(v) - 1 rtol=1e-10
        end

        # Duality holds
        for a in 0.1:0.2:3, b in 0.5:0.5:2

            @test Divergences.verify_duality(md_kl, a, b) < 1e-10
        end
    end

    @testset "Fully modified divergence duals" begin
        φ, ρ = 0.3, 1.2
        fmd = FullyModifiedDivergence(RKL, φ, ρ)
        g1 = Divergences.gradient(RKL, φ)
        γ1 = Divergences.gradient(RKL, ρ)

        # Middle region: matches original RKL dual
        for v in (g1 + 0.05):0.1:(γ1 - 0.05)
            @test Divergences.dual(fmd, v) ≈ -log(1 - v) rtol=1e-10
        end

        # Duality holds
        for a in 0.1:0.15:3, b in 0.5:0.5:2

            @test Divergences.verify_duality(fmd, a, b) < 1e-9
        end
    end

    @testset "Domain boundaries" begin
        # RKL: v < 1
        @test Divergences.dual(RKL, 0.99) < Inf
        @test Divergences.dual(RKL, 1.0) == Inf
        @test Divergences.dual(RKL, 1.5) == Inf

        # Hellinger: v < 2
        @test Divergences.dual(HD, 1.9) < Inf
        @test Divergences.dual(HD, 2.0) == Inf
        @test Divergences.dual(HD, 2.5) == Inf

        # CressieRead(α=-0.5): v < 2
        @test Divergences.dual(CRneg, 1.9) < Inf
        @test Divergences.dual(CRneg, 2.0) == Inf
    end

    @testset "Normalization ψ(0) = 0" begin
        for d in [KL, RKL, Chi2, HD, CR05, CR2]
            @test Divergences.dual(d, 0.0) ≈ 0.0 atol=1e-12
        end
    end

    @testset "In-place operations" begin
        v = randn(10)
        b = rand(10) .+ 0.1
        out = similar(v)

        Divergences.dual_gradient!(out, KL, v, b)
        @test out ≈ Divergences.dual_gradient(KL, v, b)

        Divergences.dual_hessian!(out, KL, v, b)
        @test out ≈ Divergences.dual_hessian(KL, v, b)
    end

    @testset "Consistency: D(a,b) via primal and dual" begin
        for d in [KL, RKL, Chi2]
            a = [0.2, 0.4, 0.4]
            b = [0.1, 0.3, 0.6]

            D_primal = d(a, b)
            v = Divergences.dual_from_primal(d, a, b)
            D_via_dual = sum(a .* v) - Divergences.dual(d, v, b)

            @test D_primal ≈ D_via_dual rtol=1e-10
        end
    end
end
