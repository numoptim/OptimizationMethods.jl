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

    # Test case 1 -- both are equal
    let
        τ_lower = 1.0
        norm_grad = 1.0
        local_lipschitz_estimate = 1.0

        output = OptimizationMethods.compute_step_size(τ_lower, norm_grad,
            local_lipschitz_estimate)

        @test output == (1/(1 + .5 + 1e-16)) + 1e-16
    end

    # Test case 2 -- 2nd element in minimum
    let 
        τ_lower = 2.0
        norm_grad = 1.0
        local_lipschitz_estimate = 1.0

        output = OptimizationMethods.compute_step_size(τ_lower, norm_grad,
            local_lipschitz_estimate)

        @test output == (1/(1 + .5 + 1e-16)) + 1e-16
    end
    
    # Test case 3 -- 1st element in minimum
    let 
        τ_lower = .5
        norm_grad = 1.0
        local_lipschitz_estimate = 1.0

        output = OptimizationMethods.compute_step_size(τ_lower, norm_grad,
            local_lipschitz_estimate)

        @test output == ((.5 ^ 2)/(1 + .5 + 1e-16)) + 1e-16
    end

    # Test case 4 -- 1st element in minimum
    let 
        τ_lower = 1.0
        norm_grad = 10.
        local_lipschitz_estimate = 1.

        output = OptimizationMethods.compute_step_size(τ_lower, norm_grad,
            local_lipschitz_estimate)
        
        @test output == (1 / (10 ^ 3 + .5 * (10 ^ 2) + 1e-16)) + 1e-16
    end

    # Test case 5 -- 2nd element in minimum
    let
        τ_lower = 1.0
        norm_grad = .5
        local_lipschitz_estimate = 1.

        output = OptimizationMethods.compute_step_size(τ_lower, norm_grad,
            local_lipschitz_estimate)
        
        @test output == 1.0
    end

    # test case 6 == -- transition at norm_grad = tau
    let 
        τ_lower = abs(randn(1)[1]) + 1
        norm_grad = collect(0.1:0.1:(2 * τ_lower))
        local_lipschitz_estimate = abs(randn(1)[1])

        for ng in norm_grad
            output = OptimizationMethods.compute_step_size(
                τ_lower, ng, local_lipschitz_estimate
            )

            if ng <= τ_lower
                @test output ≈ 1 / (ng + .5 * local_lipschitz_estimate + 1e-16)
            else
                @test output ≈ (τ_lower ^ 2) / (ng ^ 3 + .5 * 
                    local_lipschitz_estimate * ng ^ 2 + 1e-16)
            end
        end
    end
end

################################################################################
# Test cases for the method struct
################################################################################

@testset "Method -- Gradient Descent with Nonsequential Armijo: struct" begin

    # set seed for reproducibility
    Random.seed!(1010)

    # test definition
    @test isdefined(OptimizationMethods, :NonsequentialArmijoGD)

    # test field values -- default names
    default_fields = [:name, :threshold, :max_iterations, :iter_hist,
        :grad_val_hist, :stop_iteration]
    let fields = default_fields
        for field_name in fields
            @test field_name in fieldnames(NonsequentialArmijoGD)
        end
    end

    # test field values -- unique names
    unique_fields = [:∇F_θk, :norm_∇F_ψ, :prev_∇F_ψ, :prev_norm_step,
        :α0k, :δk, :δ_upper, :ρ, :τ_lower, :τ_upper, :local_lipschitz_estimate]
    let fields = unique_fields
        for field_name in fields
            @test field_name in fieldnames(NonsequentialArmijoGD)
        end
    end

    # test field types
    field_info(type::T) where T = [
        [:name, String],
        [:∇F_θk, Vector{type}],
        [:norm_∇F_ψ, type],
        [:prev_∇F_ψ, Vector{type}],
        [:prev_norm_step, type],
        [:α0k, type],
        [:δk, type],
        [:δ_upper, type],
        [:ρ, type],
        [:τ_lower, type],
        [:τ_upper, type],
        [:local_lipschitz_estimate, type],
        [:threshold, type],
        [:max_iterations, Int64],
        [:iter_hist, Vector{Vector{type}}],
        [:grad_val_hist, Vector{type}],
        [:stop_iteration, Int64]
    ]
    real_types = [Float16, Float32, Float64]
    let real_types = real_types, 
        field_info = field_info,
        dim = 100

        for real_type in real_types
            
            ## arguments
            x0 = randn(real_type, dim)
            δ0 = abs(randn(real_type, 1)[1])
            δ_upper = δ0 + 1
            ρ = abs(randn(real_type, 1)[1])
            threshold = abs(randn(real_type, 1)[1])
            max_iterations = rand(1:100)

            ## build structure
            optData = NonsequentialArmijoGD(real_type;
                x0 = x0,
                δ0 = δ0,
                δ_upper = δ_upper,
                ρ = ρ,
                threshold = threshold,
                max_iterations = max_iterations)

            ## check field info
            for (fieldname, fieldtype) in field_info(real_type)
                @test fieldtype == typeof(getfield(optData, fieldname))
            end

            ## check field correctness
            @test optData.iter_hist[1] == x0
            @test optData.δk == δ0
            @test optData.δ_upper == δ_upper
            @test optData.ρ == ρ
            @test optData.threshold == threshold
            @test optData.max_iterations == max_iterations
        end
        
    end

    # test errors
    real_types = [Float16, Float32, Float64]
    let real_types = real_types,
        dim = 100
        for real_type in real_types

            ## arguments
            x0 = randn(real_type, dim)
            ρ = abs(randn(real_type, 1)[1])
            threshold = abs(randn(real_type, 1)[1])
            max_iterations = rand(1:100)

            δ0 = -real_type(1) 
            δ_upper = real_type(0)
            
            ## error should occur since δ0 < 0
            @test_throws AssertionError optData = NonsequentialArmijoGD(
                real_type;
                x0 = x0,
                δ0 = δ0,
                δ_upper = δ_upper,
                ρ = ρ,
                threshold = threshold,
                max_iterations = max_iterations)

            δ0 = real_type(1.0)
            δ_upper = real_type(.5)

            ## error should occur since δ0 > δ_upper
            @test_throws AssertionError optData = NonsequentialArmijoGD(
                real_type;
                x0 = x0,
                δ0 = δ0,
                δ_upper = δ_upper,
                ρ = ρ,
                threshold = threshold,
                max_iterations = max_iterations)
        end
    end 

end

@testset "Utility -- Update Algorithm Parameters" begin

    ## arguments
    dim = 50
    x0 = randn(50)
    δ0 = abs(randn(1)[1])
    δ_upper = δ0 + 1
    ρ = abs(randn(1)[1])
    threshold = abs(randn(1)[1])
    max_iterations = rand(3:100)

    ## build structure
    optData = NonsequentialArmijoGD(Float64;
        x0 = x0,
        δ0 = δ0,
        δ_upper = δ_upper,
        ρ = ρ,
        threshold = threshold,
        max_iterations = max_iterations)
    
    ############################################################################
    # Case 1: Did not satisfy armijo condition
    ############################################################################
    let optData = optData, achieved_descent = false, dim = dim
        # First Iteration
        xp1 = zeros(dim)
        iter = 1
        optData.τ_lower = 0.0
        optData.τ_upper = 1.0
        optData.δk = 1.0

        params_update_flag = OptimizationMethods.update_algorithm_parameters!(xp1, 
            optData, achieved_descent, iter)

        @test xp1 == optData.iter_hist[iter]
        @test optData.τ_lower == 0.0 
        @test optData.τ_upper == 1.0
        @test optData.δk == 0.5
        @test !params_update_flag
    end

    let optData = optData, achieved_descent = false, dim = dim,
        max_iterations = max_iterations 
        # General Iteration 
        xp1 = zeros(dim)
        iter = rand(3:max_iterations)
        optData.τ_lower = 0.0 
        optData.τ_upper = 1.0 
        optData.δk = 1.0 

        params_update_flag = OptimizationMethods.update_algorithm_parameters!(xp1, 
            optData, achieved_descent, iter)

        @test xp1 == optData.iter_hist[iter]
        @test optData.τ_lower == 0.0 
        @test optData.τ_upper == 1.0
        @test optData.δk == 0.5
        @test !params_update_flag
        
    end

    ############################################################################
    # Case 2: Did satisfy condition + grad-norm smaller than lower bound
    ############################################################################
    let optData = optData, achieved_descent = true, dim = dim
        # First Iteration
        xp1 = rand(dim)
        iter = 1
        optData.τ_lower = 1.0
        optData.τ_upper = 2.0
        optData.norm_∇F_ψ = 0.5
        optData.δk = 1.0

        params_update_flag = OptimizationMethods.update_algorithm_parameters!(xp1, 
            optData, achieved_descent, iter)

        @test xp1 != optData.iter_hist[iter]
        @test optData.τ_lower == 0.5 / sqrt(2)
        @test optData.τ_upper == 0.5 * sqrt(10)
        @test optData.δk == 1.0
        @test params_update_flag
    end 

    let optData = optData, achieved_descent = true, dim = dim,
        max_iterations = max_iterations 
        # General Iteration 
        xp1 = rand(dim)
        iter = rand(3:max_iterations)
        optData.τ_lower = 1.0
        optData.τ_upper = 2.0
        optData.norm_∇F_ψ = 0.5
        optData.δk = 1.0 

        params_update_flag = OptimizationMethods.update_algorithm_parameters!(xp1, 
            optData, achieved_descent, iter)

        @test xp1 != optData.iter_hist[iter]
        @test optData.τ_lower == 0.5 / sqrt(2)
        @test optData.τ_upper == 0.5 * sqrt(10)
        @test optData.δk == 1.0
        @test params_update_flag
    end

    ############################################################################
    # Case 3: Did satisfy condition + grad-norm larger than upper bound
    ############################################################################
    let optData = optData, achieved_descent = true, dim = dim
        # First Iteration, upper bound on delta is not exceeded during update
        xp1 = rand(dim)
        iter = 1
        optData.τ_lower = 1.0
        optData.τ_upper = 2.0
        optData.norm_∇F_ψ = 2.5
        optData.δk = 1.0
        optData.δ_upper = 2.0

        params_update_flag = OptimizationMethods.update_algorithm_parameters!(xp1, 
            optData, achieved_descent, iter)

        @test xp1 != optData.iter_hist[iter]
        @test optData.τ_lower == 2.5 / sqrt(2)
        @test optData.τ_upper == 2.5 * sqrt(10)
        @test optData.δk == 1.5
        @test params_update_flag
    end 

    let optData = optData, achieved_descent = true, dim = dim
        # First Iteration, upper bound on delta is exceeded during update
        xp1 = rand(dim)
        iter = 1
        optData.τ_lower = 1.0
        optData.τ_upper = 2.0
        optData.norm_∇F_ψ = 2.5
        optData.δk = 1.0
        optData.δ_upper = 1.2

        params_update_flag = OptimizationMethods.update_algorithm_parameters!(xp1, 
            optData, achieved_descent, iter)

        @test xp1 != optData.iter_hist[iter]
        @test optData.τ_lower == 2.5 / sqrt(2)
        @test optData.τ_upper == 2.5 * sqrt(10)
        @test optData.δk == 1.2
        @test params_update_flag
    end 
    
    let optData = optData, achieved_descent = true, dim = dim,
        max_iterations = max_iterations 
        # General Iteration, upper bound on delta is not exceeded during update
        xp1 = rand(dim)
        iter = rand(3:max_iterations)
        optData.τ_lower = 1.0
        optData.τ_upper = 2.0
        optData.norm_∇F_ψ = 2.5
        optData.δk = 1.0 
        optData.δ_upper = 2.0

        params_update_flag = OptimizationMethods.update_algorithm_parameters!(xp1, 
            optData, achieved_descent, iter)

        @test xp1 != optData.iter_hist[iter]
        @test optData.τ_lower == 2.5 / sqrt(2)
        @test optData.τ_upper == 2.5 * sqrt(10)
        @test optData.δk == 1.5
        @test params_update_flag
    end

    let optData = optData, achieved_descent = true, dim = dim,
        max_iterations = max_iterations 
        # General Iteration, upper bound on delta is exceeded during update
        xp1 = rand(dim)
        iter = rand(3:max_iterations)
        optData.τ_lower = 1.0
        optData.τ_upper = 2.0
        optData.norm_∇F_ψ = 2.5
        optData.δk = 1.0 
        optData.δ_upper = 1.2

        params_update_flag = OptimizationMethods.update_algorithm_parameters!(xp1, 
            optData, achieved_descent, iter)

        @test xp1 != optData.iter_hist[iter]
        @test optData.τ_lower == 2.5 / sqrt(2)
        @test optData.τ_upper == 2.5 * sqrt(10)
        @test optData.δk == 1.2
        @test params_update_flag
    end
    
    ############################################################################
    # Case 4: Did satisfy condition + inside interval
    ############################################################################
    let optData = optData, achieved_descent = true, dim = dim
        # First Iteration, delta upper bound is exceeded
        xp1 = rand(dim)
        iter = 1
        optData.τ_lower = 1.0
        optData.τ_upper = 2.0
        optData.norm_∇F_ψ = 1.5
        optData.δk = 1.0
        optData.δ_upper = 1.2

        params_update_flag = OptimizationMethods.update_algorithm_parameters!(xp1, 
            optData, achieved_descent, iter)

        @test xp1 != optData.iter_hist[iter]
        @test optData.τ_lower == 1.0
        @test optData.τ_upper == 2.0
        @test optData.δk == 1.2
        @test params_update_flag
    end 

    let optData = optData, achieved_descent = true, dim = dim
        # First Iteration, delta upper bound is not exceeded
        xp1 = rand(dim)
        iter = 1
        optData.τ_lower = 1.0
        optData.τ_upper = 2.0
        optData.norm_∇F_ψ = 1.5
        optData.δk = 1.0
        optData.δ_upper = 2.0

        params_update_flag = OptimizationMethods.update_algorithm_parameters!(xp1, 
            optData, achieved_descent, iter)

        @test xp1 != optData.iter_hist[iter]
        @test optData.τ_lower == 1.0
        @test optData.τ_upper == 2.0
        @test optData.δk == 1.5
        @test params_update_flag
    end 

    let optData = optData, achieved_descent = true, dim = dim,
        max_iterations = max_iterations 
        # General Iteration, upper bound on delta is exceeded during update
        xp1 = rand(dim)
        iter = rand(3:max_iterations)
        optData.τ_lower = 1.0
        optData.τ_upper = 2.0
        optData.norm_∇F_ψ = 1.5
        optData.δk = 1.0 
        optData.δ_upper = 1.2

        params_update_flag = OptimizationMethods.update_algorithm_parameters!(xp1, 
            optData, achieved_descent, iter)

        @test xp1 != optData.iter_hist[iter]
        @test optData.τ_lower == 1.0
        @test optData.τ_upper == 2.0
        @test optData.δk == 1.2
        @test params_update_flag
    end
    
    let optData = optData, achieved_descent = true, dim = dim,
        max_iterations = max_iterations 
        # General Iteration, upper bound on delta is not exceeded during update
        xp1 = rand(dim)
        iter = rand(3:max_iterations)
        optData.τ_lower = 1.0
        optData.τ_upper = 2.0
        optData.norm_∇F_ψ = 1.5
        optData.δk = 1.0 
        optData.δ_upper = 2.0

        params_update_flag = OptimizationMethods.update_algorithm_parameters!(xp1, 
            optData, achieved_descent, iter)

        @test xp1 != optData.iter_hist[iter]
        @test optData.τ_lower == 1.0
        @test optData.τ_upper == 2.0
        @test optData.δk == 1.5
        @test params_update_flag
    end
end

@testset "Utility -- Inner Loop" begin
end

@testset "Method -- Gradient Descent with Nonsequential Armijo: method" begin
end

end