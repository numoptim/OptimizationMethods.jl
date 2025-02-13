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

@testset "Variance Function First Derivatives -- Centered Exponential" begin

    # set seed for reproducibility
    Random.seed!(1010)

    # test definition
    @test isdefined(OptimizationMethods, :dcentered_exp)

    # test at a sequence of random points
    num_points = 1
    float_type = [Float32, Float64]
    tolerance = [1e-6, 1e-9]
    c = [-.5, -.1, 0.0, .1, .5, 1, 2, 5]
    p = [.51, .6, 1, 1.5, 2]
    let num_points =  num_points, 
        float_types = float_type, 
        tolerances = tolerance,
        c = c,
        p = p

        for constant in c
            for power in p

                # create gradient function to test against
                f(μ) = exp(-abs(μ-constant)^(2*power))
                g(μ) = ForwardDiff.derivative(f, μ)

                for (float_type, float_toler) in zip(float_types, tolerances)
                    for npoint in 1:num_points

                        # test gradient value
                        x = randn(float_type)

                        output = OptimizationMethods.dcentered_exp(x, 
                            float_type(power), float_type(constant))
                        gdiff = float_type(g(x))

                        @test typeof(output) == float_type
                        @test gdiff ≈ output atol = float_toler
                    end
                end
            end
        end

    end
end

end