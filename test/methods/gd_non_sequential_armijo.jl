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
                k, norm(djk), curr_grad, prev_grad, prev_approximation, 
                past_acceptance)

            # test output
            @test typeof(output) == Float64

            # test correctness of output
            if j == 1 && k == 1
                @test output == 1.0
            elseif j == 1 && k > 1
                @test output == prev_approximation
            elseif j > 1 && k == 1
                @test output == norm(curr_grad - prev_grad) / norm(djk)
            elseif j > 1 && k > 1 && past_acceptance
                @test output == norm(curr_grad - prev_grad) / norm(djk)
            elseif j > 1 && k > 1 && (!past_acceptance)
                @test output == 
                    max(prev_approximation, norm(curr_grad - prev_grad) / norm(djk))
            end
        end
    end
end

@testset "Utility -- Novel Step Size Computation" begin

    # set seed for reproducibility
    Random.seed!(1010)

    # test definition
    @test isdefined(OptimizationMethods, :compute_step_size)

    # Test case 1
    let
        τ_lower = 1.0
        norm_grad = 1.0
        local_lipschitz_estimate = 1.0

        output = OptimizationMethods.compute_step_size(τ_lower, norm_grad,
            local_lipschitz_estimate)

        @test output == (1/(1 + .5 + 1e-16)) + 1e-16
    end

    # Test case 2
    let 
        τ_lower = 2.0
        norm_grad = 1.0
        local_lipschitz_estimate = 1.0

        output = OptimizationMethods.compute_step_size(τ_lower, norm_grad,
            local_lipschitz_estimate)

        @test output == (1/(1 + .5 + 1e-16)) + 1e-16
    end
    
    # Test case 3
    let 
        τ_lower = .5
        norm_grad = 1.0
        local_lipschitz_estimate = 1.0

        output = OptimizationMethods.compute_step_size(τ_lower, norm_grad,
            local_lipschitz_estimate)

        @test output == ((.5 ^ 2)/(1 + .5 + 1e-16)) + 1e-16
    end

    # Test case 4
    let 
        τ_lower = 1.0
        norm_grad = 10.
        local_lipschitz_estimate = 1.

        output = OptimizationMethods.compute_step_size(τ_lower, norm_grad,
            local_lipschitz_estimate)
        
        @test output == (1 / (10 ^ 3 + .5 * (10 ^ 2) + 1e-16)) + 1e-16
    end

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