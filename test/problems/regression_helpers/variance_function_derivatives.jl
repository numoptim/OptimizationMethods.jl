# Date: 12/27/2024
# Author: Christian Varner
# Purpose: Test the derivatives for the variance functions

module TestVarianceFunctionDerivatives

using Test, ForwardDiff, OptimizationMethods, Random

@testset "Variance Function First Derivatives - Linear plus sin" begin

    # set seec for reproducibility
    Random.seed!(1010)

    # test definition
    @test isdefined(OptimizationMethods, :dlinear_plus_sin)

    # create function to test against
    grad_forward_diff(μ) = ForwardDiff.derivative(
        OptimizationMethods.linear_plus_sin,
        μ
    )

    # test at a sequence of random points
    num_points = 10
    float_type = [Float16, Float32, Float64]
    tolerance = [1e-3, 1e-6, 1e-9]
    let num_points = num_points, float_type = float_type, tolerance = tolerance
        for (type, toler) in zip(float_type, tolerance)
            points = randn(type, num_points)
            for x in points
                g = OptimizationMethods.dlinear_plus_sin(x)
                @test typeof(g) == type
                @test g ≈ grad_forward_diff(x) atol = toler
            end
        end
    end
end

end