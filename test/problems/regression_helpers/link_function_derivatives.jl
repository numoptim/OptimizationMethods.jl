# Date: 12/27/2024
# Author: Christian Varner
# Purpose: Test cases for the derivatives of link functions

module TestLinkFunctionDerivatives

using Test, ForwardDiff, OptimizationMethods, Random

@testset "Link Function First Derivative - Logistic" begin

    # set the seed for reproducibility
    Random.seed!(1010)

    # test definition
    @test isdefined(OptimizationMethods, :dlogistic)

    # check derivative at a series of points
    grad_forward_diff(η) = ForwardDiff.derivative(
        OptimizationMethods.logistic,
        η
    )
    let num_points = 10
        float_type = [Float16, Float32, Float64]
        tolerances = [1e-3, 1e-7, 1e-9]
        for (type, toler) in zip(float_type, tolerances)
            points = randn(type, num_points)
            for x in points
                g = OptimizationMethods.dlogistic(x)
                @test typeof(g) == type
                @test g ≈ grad_forward_diff(x) atol = toler
            end
        end
    end
end

@testset "Link Function Second Derivative - Logistic" begin

    # set the seed for reproducibility
    Random.seed!(1010)

    # test definition
    @test isdefined(OptimizationMethods, :ddlogistic)

    # check derivative at a series of points
    grad_forward_diff(η) = ForwardDiff.derivative(
        OptimizationMethods.dlogistic, 
        η
    )

    num_points = 10
    float_type = [Float16, Float32, Float64]
    tolerances = [1e-3, 1e-7, 1e-9]
    let num_points = num_points, float_type = float_type, tolerances = tolerances
        for (type, toler) in zip(float_type, tolerances)
            points = randn(type, num_points)
            for x in points
                g = OptimizationMethods.ddlogistic(x)
                @test typeof(g) == type
                @test g ≈ grad_forward_diff(x) atol = toler
            end
        end
    end

end

end