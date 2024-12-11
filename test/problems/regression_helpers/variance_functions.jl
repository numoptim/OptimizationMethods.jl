# Date: 12/11/2024
# Author: Christian Varner
# Purpose: Test variance functions

module ProceduralLinkFunctions

using Test, OptimizationMethods, Random

################################################################################
# Add a new test set for each function implemented in varance_functions.jl
################################################################################

@testset "Variance Functions -- Monomial Plus Constant" begin 

    Random.seed!(1010)

    ## test to make sure the function is defined
    @assert isdefined(OptimizationMethods, :monomial_plus_constant)

    ## test the functions output
    test_values = Vector{Float64}(collect(-5:5:1))
    for μ in test_values
        p = rand()
        c = rand()
        y = (μ^(2))^p + c
        
        returned_val = OptimizationMethods.monomial_plus_constant(μ, p, c)
        @test isapprox(y, returned_val; atol = 1e-10)
        @test typeof(returned_val) == Float64
    end
end

end