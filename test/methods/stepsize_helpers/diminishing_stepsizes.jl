module TestDiminshingStepSizes

using Test, OptimizationMethods, LinearAlgebra, Random

@testset "Method Helpers: Diminishing Step Sizes" begin 

    # Tesitng Parameters 
    ks = [0, 10, 100, 1000, 10000, 100000]
    types = [Float16, Float32, Float64]

    # Test for inverse_k_step_size 
    for (k, type) in Iterators.product(ks, types)
        @test OptimizationMethods.inverse_k_step_size(type, k) == 
            type(1/(k+1))
    end

    # Test for inverse_log2k_step_size 
    for (k, type) in Iterators.product(ks, types)
        @test OptimizationMethods.inverse_log2k_step_size(type, k) == 
            type(1/floor(log2(k+1)+1))
    end

    # Test for stepdown_100_step_size
    for (k, type) in Iterators.product(ks, types)
        j = floor(log2(1 + k / 100))
        @test OptimizationMethods.stepdown_100_step_size(type, k) == 
            type(1/2^j)
    end
end
end