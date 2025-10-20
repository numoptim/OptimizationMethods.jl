# Date: 2025/10/08
# Author: Christian Varner
# Purpose: Test cases for functions in sampling_helpers.jl

module TestSamplingHelpers

using Test, OptimizationMethods, LinearAlgebra, Random

@testset "sampling_helpers.jl -- inverse_logistic" begin

    # test definition
    @test isdefined(OptimizationMethods, :inverse_logistic)

    # test functionality
    μ = 0.0:0.01:1.0
    for p in μ
        if p == 0.0 || p == 1.0
            continue
        end
        η = OptimizationMethods.inverse_logistic(p)
        @test isapprox(p, OptimizationMethods.logistic(η), atol=1e-8)
    end
end

@testset "sampling_helpers.jl -- get_design" begin

    # test definition
    @test isdefined(OptimizationMethods, :get_design)

    # test functionality
    for i in 1:10
        a = .5 * rand()
        nobs = 100
        nvar = 10
        x, β = OptimizationMethods.get_design(a, nobs, nvar)

        # test sizes
        @test size(x) == (nobs, nvar)
        @test length(β) == nvar

        # test ranges
        ub = OptimizationMethods.inverse_logistic(1-a)/nvar
        lb = OptimizationMethods.inverse_logistic(a)/nvar
        η = x * β

        # test range on x
        @test all(x .>= 0.0)
        @test all(x .<= 1.0)

        # test range on β
        @test all(β .>= lb)
        @test all(β .<= ub)

        # test range on linear predictor
        @test all(η .>= lb * nvar)
        @test all(η .<= ub * nvar)
    end
end

@testset "sampling_helpers.jl -- get_noise" begin

    # test definition
    @test isdefined(OptimizationMethods, :get_noise)

    # test functionality
    for i in 1:10
        a = .5 * rand()
        nobs = 100
        vmax = 10.0 * rand() + 1.0
        ϵ = OptimizationMethods.get_noise(a, nobs, vmax)

        # test sizes
        @test length(ϵ) == nobs

        # test ranges
        lb = -a/sqrt(vmax)
        ub = a/sqrt(vmax)
        @test all(ϵ .>= lb)
        @test all(ϵ .<= ub)
    end
end

end # end module