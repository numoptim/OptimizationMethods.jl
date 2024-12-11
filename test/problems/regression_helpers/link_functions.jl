# Date: 12/11/2024
# Author: Christian Varner
# Purpose: Test set for link functions

module ProceduralLinkFunctions

using Test, OptimizationMethods, Random

################################################################################
# Add a new test set for each function implemented in link_functions.jl
################################################################################

@testset "Link Function -- Logistic" begin 

    ## test to make sure the function is defined
    @assert isdefined(OptimizationMethods, :logistic)

    ## test the functions output
    test_values = Vector{Float64}(collect(-5:5:1))
    for x in test_values
        y = 1 / (1 + exp(-x))
        
        returned_val = OptimizationMethods.logistic(x)
        @test isapprox(y, returned_val; atol = 1e-10)
        @test typeof(returned_val) == Float64
    end
end

end