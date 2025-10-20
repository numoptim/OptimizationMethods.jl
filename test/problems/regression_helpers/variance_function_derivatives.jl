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


@testset "Variance Function First Derivatives -- Monomial Plus Constant" begin

    # set seed for reproducibility
    Random.seed!(1010)

    # test definition
    @test isdefined(OptimizationMethods, :dmonomial_plus_constant)

    # test at a sequence of random points
    num_points = 1
    float_type = [Float32, Float64]
    tolerance = [1e-6, 1e-9]
    c = [.01, .1, .5, 1, 2, 5]
    p = [.51, .6, 1, 1.5, 2]
    let num_points =  num_points, 
        float_types = float_type, 
        tolerances = tolerance,
        c = c,
        p = p

        for constant in c
            for power in p

                # create gradient function to test against
                f(μ) = (μ^(2))^power + constant
                g(μ) = ForwardDiff.derivative(f, μ)

                for (float_type, float_toler) in zip(float_types, tolerances)
                    for npoint in 1:num_points

                        # test gradient value
                        x = randn(float_type)

                        output = OptimizationMethods.dmonomial_plus_constant(x, 
                            float_type(power))
                        gdiff = float_type(g(x))

                        @test typeof(output) == float_type
                        @test gdiff ≈ output atol = float_toler

                    end
                end
            end
        end

    end

end

@testset "Variance Function First Derivatives -- Centered Shifted Log" begin

    # set seed for reproducibility
    Random.seed!(1010)

    # test definition
    @test isdefined(OptimizationMethods, :dcentered_shifted_log)

    # test at a sequence of random points
    num_points = 1
    float_type = [Float32, Float64]
    tolerance = [1e-6, 1e-9]
    c = [.01, .1, .5, 1, 2, 5]
    p = [.51, .6, 1, 1.5, 2]
    let num_points =  num_points, 
        float_types = float_type, 
        tolerances = tolerance,
        c = c,
        p = p

        for constant in c
            for power in p

                # create gradient function to test against
                f(μ) = log(abs(μ-constant)^(2*power) + 1)
                g(μ) = ForwardDiff.derivative(f, μ)

                for (float_type, float_toler) in zip(float_types, tolerances)
                    for npoint in 1:num_points

                        # test gradient value
                        x = randn(float_type)

                        output = OptimizationMethods.dcentered_shifted_log(x, 
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

@testset "Variance Function First Derivatives -- Logistic" begin

    # test definition
    @test isdefined(OptimizationMethods, :dlogistic_variance)

    # define the "true" derivative
    f(μ) = OptimizationMethods.logistic_variance(μ)
    g(μ) = ForwardDiff.derivative(f, μ)

    for float_type in [Float64]
        for npoint in 1:10
            random_point = rand(float_type)
            est_true_val = g(random_point)
            returned_val = OptimizationMethods.dlogistic_variance(random_point)

            @test typeof(returned_val) == float_type
            @test est_true_val ≈ returned_val atol = 1e-9
        end
    end

end

@testset "Variance Function First Derivatives -- Logistic Squared" begin

    # test definition
    @test isdefined(OptimizationMethods, :dlogistic_variance_squared)

    # define the "true" derivative
    f(μ) = OptimizationMethods.logistic_variance_squared(μ)
    g(μ) = ForwardDiff.derivative(f, μ)

    for float_type in [Float64]
        for npoint in 1:10
            random_point = rand(float_type)
            est_true_val = g(random_point)
            returned_val = OptimizationMethods.dlogistic_variance_squared(random_point)

            @test typeof(returned_val) == float_type
            @test est_true_val ≈ returned_val atol = 1e-9
        end
    end

end

@testset "Variance Function First Derivatives -- Logistic P" begin

    # test definition
    @test isdefined(OptimizationMethods, :dlogistic_variance_p)
    for p in 0.0:.1:2.0

        # define the "true" derivative
        f(μ) = (μ * (1-μ))^p
        g(μ) = ForwardDiff.derivative(f, μ)

        for float_type in [Float64]
            for npoint in 1:10
                random_point = rand(float_type)
                est_true_val = g(random_point)
                returned_val = OptimizationMethods.dlogistic_variance_p(random_point, p)

                @test typeof(returned_val) == float_type
                @test est_true_val ≈ returned_val atol = 1e-9
            end
        end
    end

end


end