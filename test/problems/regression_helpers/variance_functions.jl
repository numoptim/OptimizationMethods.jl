# Date: 12/11/2024
# Author: Christian Varner
# Purpose: Test variance functions

module ProceduralVarianceFunctions

using Test, OptimizationMethods, Random

################################################################################
# Add a new test set for each function implemented in varance_functions.jl
################################################################################

@testset "Variance Function -- Monomial Plus Constant" begin 

    Random.seed!(1010)

    ## test to make sure the function is defined
    @assert isdefined(OptimizationMethods, :linear_plus_sin)

    float_types = [Float16, Float32, Float64]
    for float_type in float_types
        test_points = randn(float_type, 10)
        for μ in test_points
            p = rand(float_type)
            c = rand(float_type)
            v = (μ^(2))^p + c
            
            returned_v = OptimizationMethods.monomial_plus_constant(μ, p, c)
            @test v ≈ returned_v
            @test typeof(returned_v) == float_type 
        end
    end
end

@testset "Variance Function -- Linear Plus Sin" begin

    Random.seed!(1010)

    ## test to make sure the function is defined
    @assert isdefined(OptimizationMethods, :linear_plus_sin)

    float_types = [Float16, Float32, Float64]
    for float_type in float_types
        test_points = randn(float_type, 10)
        for μ in test_points
            v = float_type(1 + μ + sin(2*pi*μ))
            
            returned_v = OptimizationMethods.linear_plus_sin(μ)
            @test v ≈ returned_v
            @test typeof(returned_v) == float_type 
        end
    end
end

@testset "Variance Function -- Centered Exponential" begin

    Random.seed!(1010)

    ## test to make sure the function is defined
    @assert isdefined(OptimizationMethods, :centered_exp)

    float_types = [Float16, Float32, Float64]
    for float_type in float_types
        test_points = randn(float_type, 10)
        for μ in test_points
            p = rand(float_type)
            c = rand(float_type)
            v = exp(-abs(μ-c)^(2*p))
            
            returned_v = OptimizationMethods.centered_exp(μ, p, c)
            @test v ≈ returned_v
            @test typeof(returned_v) == float_type 
        end
    end
end

@testset "Variance Function -- Centered Shifted Log" begin

    Random.seed!(1010)

    ## test to make sure the function is defined
    @assert isdefined(OptimizationMethods, :linear_plus_sin)

    float_types = [Float16, Float32, Float64]
    for float_type in float_types
        test_points = randn(float_type, 10)
        for μ in test_points
            p = rand(float_type)
            c = rand(float_type)
            v = log(abs(μ-c)^(2*p) + 1)
            
            returned_v = OptimizationMethods.centered_shifted_log(μ, p, c)
            @test v ≈ returned_v
            @test typeof(returned_v) == float_type 
        end
    end
end

end