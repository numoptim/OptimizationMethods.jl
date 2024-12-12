# Date: 12/11/2024
# Author: Christian Varner
# Purpose: Test set for link functions

module ProceduralLinkFunctions

using Test, OptimizationMethods, Random

################################################################################
# Add a new test set for each function implemented in link_functions.jl
################################################################################

@testset "Link Function -- Logistic" begin 

    Random.seed!(1010)

    ## test to make sure the function is defined
    @assert isdefined(OptimizationMethods, :logistic)

    ## test the functions output
    float_types = [Float16, Float32, Float64]
    for float_type in float_types
        test_values = rand(float_type, 10)
        for x in test_values
            y = 1 / (1 + exp(-x))
            
            returned_val = OptimizationMethods.logistic(x)
            @test isapprox(y, returned_val; atol = 1e-16)
            @test typeof(returned_val) == float_type
        end
    end
end

@testset "Link Function -- Inverse CLogLog" begin

    Random.seed!(1010)

    ## test to make sure the function is defined
    @assert isdefined(OptimizationMethods, :inverse_complimentary_log_log)

    float_types = [Float16, Float32, Float64]

    for float_type in float_types
        ## test the functions output
        test_values = rand(float_type, 10)
        for x in test_values
            y = 1 - exp(-exp(x))

            returned_val = OptimizationMethods.inverse_complimentary_log_log(x)
            @test y ≈ returned_val atol = 1e-16
            @test typeof(returned_val) == float_type
        end
    end
end

@testset "Link Function -- Inverse Probit" begin

        Random.seed!(1010)

        ##test to make sure the function is defined
        @assert isdefined(OptimizationMethods, :inverse_probit)

        ## test the function values
        float_types = [Float16, Float32, Float64]
        for float_type in float_types
            test_values = rand(float_type, 10)
            for x in test_values
                y = float_type((1/sqrt(2*pi)) * exp(-.5*(x^2)))

                returned_val = OptimizationMethods.inverse_probit(x)
                @test y ≈ returned_val atol = 1e-16
                @test typeof(returned_val) == float_type
            end
        end
end

end