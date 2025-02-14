# Date: 02/14/2025
# Author: Christian Varner
# Purpose: Test cases for gradient descent with non-sequential armijo
# (our method).

module TestNonsequentialArmijo

using Test, OptimizationMethods, LinearAlgebra, Random

################################################################################
# Test cases for utility
################################################################################

@testset "Utility -- Local Lipschitz Approximation" begin

    # set seed for reproducibility
    Random.seed!(1010)

    # test definition 
    @test isdefined(OptimizationMethods, :update_local_lipschitz_approximation)

    # test functionality
    cases = [(1,1,false), (1,2,false), (2,1,false),
            (1,1,true), (1,2,true), (2,1,true), 
            (2,2,true), (2,2,false)]
    let cases = cases, dim = 50
        for (j,k,past_acceptance) in cases

            # generate parameters for the function
            djk = randn(50)
            curr_grad = randn(50)
            prev_grad = randn(50)
            prev_approximation = abs(randn(1)[1])
        
            # get the output
            output = OptimizationMethods.update_local_lipschitz_approximation(j, 
                k, djk, curr_grad, prev_grad, prev_approximation, 
                past_acceptance)

            # test output
            @test typeof(output) == Float64

            # TODO - test correctness
        end
    end

end

@testset "Utility -- Novel Step Size Computation" begin
end

@testset "Utility -- Inner Loop" begin
end

@testset "Utility -- Update Algorithm Parameters" begin
end

################################################################################
# Test cases for the method
################################################################################

@testset "Method -- Gradient Descent with Nonsequential Armijo" begin
end

end