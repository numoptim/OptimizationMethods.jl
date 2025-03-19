# Date: 02/14/2025
# Author: Christian Varner
# Purpose: Test cases for gradient descent with non-sequential armijo
# (our method).

module TestNonsequentialArmijoAdaptiveGD

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
    @test isdefined(OptimizationMethods, :NonsequentialArmijoAdaptiveGD)

    # test field values -- default names
    default_fields = [:name, :threshold, :max_iterations, :iter_hist,
        :grad_val_hist, :stop_iteration]
    let fields = default_fields
        for field_name in fields
            @test field_name in fieldnames(NonsequentialArmijoAdaptiveGD)
        end
    end

    # test field values -- unique names
    unique_fields = [:∇F_θk, :norm_∇F_ψ, :prev_∇F_ψ, :prev_norm_step,
        :α0k, :δk, :δ_upper, :ρ, :τ_lower, :τ_upper, :local_lipschitz_estimate]
    let fields = unique_fields
        for field_name in fields
            @test field_name in fieldnames(NonsequentialArmijoAdaptiveGD)
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
            optData = NonsequentialArmijoAdaptiveGD(real_type;
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
            @test length(optData.grad_val_hist) == max_iterations + 1
            @test optData.stop_iteration == -1
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
            @test_throws AssertionError optData = NonsequentialArmijoAdaptiveGD(
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
            @test_throws AssertionError optData = NonsequentialArmijoAdaptiveGD(
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

    include("../utility/update_algorithm_parameters_test_cases.jl")

    ## arguments
    dim = 50
    x0 = randn(50)
    δ0 = abs(randn(1)[1])
    δ_upper = δ0 + 1
    ρ = abs(randn(1)[1])
    threshold = abs(randn(1)[1])
    max_iterations = rand(3:100)

    ## build structure
    optData = NonsequentialArmijoAdaptiveGD(Float64;
        x0 = x0,
        δ0 = δ0,
        δ_upper = δ_upper,
        ρ = ρ,
        threshold = threshold,
        max_iterations = max_iterations)
    
    ## Conduct test cases
    update_algorithm_parameters_test_cases(optData, dim, max_iterations)
end

@testset "Utility -- Inner Loop" begin
    
    dim = 50
    x0 = randn(50)
    δ0 = abs(randn(1)[1])
    δ_upper = δ0 + 1
    ρ = abs(randn(1)[1])
    threshold = abs(randn(1)[1])
    max_iterations = rand(3:100)

    ## build structure
    optData = NonsequentialArmijoAdaptiveGD(Float64;
        x0 = x0,
        δ0 = δ0,
        δ_upper = δ_upper,
        ρ = ρ,
        threshold = threshold,
        max_iterations = max_iterations)

    progData = OptimizationMethods.LeastSquares(Float64, nvar=dim)
    precomp, store = OptimizationMethods.initialize(progData)
    ks = [1, max_iterations]

    for k in ks
        # Test first event trigger: radius violation
        let ψjk=x0 .+ 11, θk=x0, optData=optData, progData=progData,
            store=store, past_acceptance=false, k=k

            optData.grad_val_hist[k] = 1.5
            optData.τ_lower = 1.0
            optData.τ_upper = 2.0
            
            OptimizationMethods.inner_loop!(ψjk, θk, optData, progData, precomp, 
                store, past_acceptance,k,max_iteration=100)

            @test ψjk == x0 .+ 11
        end

        # Test second event trigger: τ_lower 
        let ψjk=copy(x0), θk=copy(x0), optData=optData, progData=progData,
            store=store, past_acceptance=false, k=k

            optData.grad_val_hist[k] = 0.5 
            optData.τ_lower = 1.0
            optData.τ_upper = 2.0
            
            OptimizationMethods.inner_loop!(ψjk, θk, optData, progData, precomp, 
                store, past_acceptance,k,max_iteration=100)

            @test ψjk == x0
        end

        # Test third event trigger: τ_upper 
        let ψjk=copy(x0), θk=copy(x0), optData=optData, progData=progData,
            store=store, past_acceptance=false, k=k

            optData.grad_val_hist[k] = 2.5 
            optData.τ_lower = 1.0
            optData.τ_upper = 2.0
            
            OptimizationMethods.inner_loop!(ψjk, θk, optData, progData, precomp, 
                store, past_acceptance,k,max_iteration=100)

            @test ψjk == x0
        end

        # Test fourth event trigger: max_iteration
        let ψjk=copy(x0), θk=copy(x0), optData=optData, progData=progData,
            store=store, past_acceptance=false, k=k

            optData.grad_val_hist[k] = 1.5 
            optData.τ_lower = 1.0
            optData.τ_upper = 2.0
            
            OptimizationMethods.inner_loop!(ψjk, θk, optData, progData, precomp, 
                store, past_acceptance,k,max_iteration=0)

            @test ψjk == x0
        end

        # Test first iteration; past_acceptance=false
        let ψjk=copy(x0), θk=copy(x0), optData=optData, progData=progData,
            store=store, past_acceptance=false, k=k

            j=1

            optData.prev_norm_step = 0.0
            OptimizationMethods.grad!(progData, precomp, store, θk)
            optData.prev_∇F_ψ = copy(store.grad)
            optData.grad_val_hist[k] = norm(store.grad)
            optData.norm_∇F_ψ = norm(store.grad)
            optData.τ_lower = 0.5 * norm(store.grad) 
            optData.τ_upper = 1.5 * norm(store.grad)
            optData.local_lipschitz_estimate = 1.0

            optData.δk = 1.5

            local_lipschitz = OptimizationMethods.update_local_lipschitz_approximation(
                j, k, optData.prev_norm_step, store.grad, optData.prev_∇F_ψ, 
                optData.local_lipschitz_estimate, past_acceptance
            )

            α = OptimizationMethods.compute_step_size(optData.τ_lower, 
                optData.norm_∇F_ψ, local_lipschitz)

            step = (optData.δk * α) .* store.grad 

            OptimizationMethods.inner_loop!(ψjk, θk, optData, progData, precomp,
                store, past_acceptance, k, max_iteration = 1
            )

            @test ψjk == θk - step 
            @test optData.α0k == α
            @test optData.prev_norm_step ≈ norm(step)
            @test optData.prev_∇F_ψ ≈ OptimizationMethods.grad(progData, θk)
            @test store.grad ≈ OptimizationMethods.grad(progData, ψjk)
            @test optData.norm_∇F_ψ == norm(store.grad)
        end

        # Test first iteration; past_acceptance=true
        let ψjk=copy(x0), θk=copy(x0), optData=optData, progData=progData,
            store=store, past_acceptance=true, k=k

            j=1

            optData.prev_norm_step = 0.0
            OptimizationMethods.grad!(progData, precomp, store, θk)
            optData.prev_∇F_ψ = copy(store.grad)
            optData.grad_val_hist[k] = norm(store.grad)
            optData.norm_∇F_ψ = norm(store.grad)
            optData.τ_lower = 0.5 * norm(store.grad) 
            optData.τ_upper = 1.5 * norm(store.grad)
            optData.local_lipschitz_estimate = 1.0

            optData.δk = 1.5

            local_lipschitz = OptimizationMethods.update_local_lipschitz_approximation(
                j, k, optData.prev_norm_step, store.grad, optData.prev_∇F_ψ, 
                optData.local_lipschitz_estimate, past_acceptance
            )

            α = OptimizationMethods.compute_step_size(optData.τ_lower, 
                optData.norm_∇F_ψ, local_lipschitz)

            step = (optData.δk * α) .* store.grad 

            OptimizationMethods.inner_loop!(ψjk, θk, optData, progData, precomp,
                store, past_acceptance, k, max_iteration = 1
            )

            @test ψjk == θk - step 
            @test optData.α0k == α
            @test optData.prev_norm_step ≈ norm(step)
            @test optData.prev_∇F_ψ ≈ OptimizationMethods.grad(progData, θk)
            @test store.grad ≈ OptimizationMethods.grad(progData, ψjk)
            @test optData.norm_∇F_ψ == norm(store.grad)
        end

        # Test random iteration; past_acceptance=true
        let ψjk=copy(x0), θk=copy(x0), optData=optData, progData=progData,
            store=store, past_acceptance=true, k=k

            #To do this test correctly, we would need to know at what iteration 
            #j an inner loop exists.
            max_iteration = rand(2:100)

            # Reset 
            optData.prev_norm_step = 0.0
            OptimizationMethods.grad!(progData, precomp, store, θk)
            optData.prev_∇F_ψ = copy(store.grad)
            optData.grad_val_hist[k] = norm(store.grad)
            optData.norm_∇F_ψ = norm(store.grad)
            optData.τ_lower = 0.5 * norm(store.grad) 
            optData.τ_upper = 1.5 * norm(store.grad)
            optData.local_lipschitz_estimate = 1.0
            optData.δk = 1.5

            #Get exit iteration j
            
            j = OptimizationMethods.inner_loop!(ψjk, θk, optData, progData, precomp,
                store, past_acceptance, k, max_iteration = max_iteration
            )

            # Reset 
            ψjk = copy(x0)
            θk = copy(x0)

            optData.prev_norm_step = 0.0
            OptimizationMethods.grad!(progData, precomp, store, θk)
            optData.prev_∇F_ψ = copy(store.grad)
            optData.grad_val_hist[k] = norm(store.grad)
            optData.norm_∇F_ψ = norm(store.grad)
            optData.τ_lower = 0.5 * norm(store.grad) 
            optData.τ_upper = 1.5 * norm(store.grad)
            optData.local_lipschitz_estimate = 1.0
            optData.δk = 1.5

            OptimizationMethods.inner_loop!(ψjk, θk, optData, progData, precomp,
                store, past_acceptance, k, max_iteration = j-1
            )

            optData.local_lipschitz_estimate = 
                OptimizationMethods.update_local_lipschitz_approximation(
                j, k, optData.prev_norm_step, store.grad, optData.prev_∇F_ψ, 
                optData.local_lipschitz_estimate, past_acceptance)

            α = OptimizationMethods.compute_step_size(optData.τ_lower, 
                optData.norm_∇F_ψ, optData.local_lipschitz_estimate
            )

            ψ_jm1_k = copy(ψjk)
            grd = OptimizationMethods.grad(progData, ψ_jm1_k)
            step = (optData.δk * α) * grd

            # Reset 
            ψjk = copy(x0)
            θk = copy(x0)

            optData.prev_norm_step = 0.0
            OptimizationMethods.grad!(progData, precomp, store, θk)
            optData.prev_∇F_ψ = copy(store.grad)
            optData.grad_val_hist[k] = norm(store.grad)
            optData.norm_∇F_ψ = norm(store.grad)
            optData.τ_lower = 0.5 * norm(store.grad) 
            optData.τ_upper = 1.5 * norm(store.grad)
            optData.local_lipschitz_estimate = 1.0
            optData.δk = 1.5

            #Get ψ_{j,k}
            OptimizationMethods.inner_loop!(ψjk, θk, optData, progData, precomp,
                store, past_acceptance, k, max_iteration = max_iteration
            )

            @test ψjk ≈ ψ_jm1_k - step
            @test optData.prev_norm_step ≈ norm(step)
            @test optData.prev_∇F_ψ ≈ grd
            @test store.grad ≈ OptimizationMethods.grad(progData, ψjk)
            @test optData.norm_∇F_ψ ≈ norm(store.grad)
        end
    end
end

@testset "Method -- Gradient Descent with Nonsequential Armijo: method" begin
    
    # Default Parameters 
    dim = 50
    x0 = randn(dim)
    δ0 = abs(randn())
    δ_upper = abs(randn()) + 2 
    ρ = abs(randn()) * 1e-3
    threshold = 1e-3
    max_iterations = 100 

    # Should exit on iteration 0 because max_iterations is 0
    let x0=copy(x0), δ0=δ0, δ_upper=δ_upper, ρ=ρ, threshold=threshold, 
        max_iterations=0

        # Specify optimization method and problem
        optData = NonsequentialArmijoAdaptiveGD(Float64; x0=x0, δ0=δ0, δ_upper=δ_upper,
            ρ=ρ, threshold=threshold, max_iterations=max_iterations)
        progData = OptimizationMethods.LeastSquares(Float64, nvar=dim)

        # Run method
        x = nonsequential_armijo_adaptive_gd(optData, progData)

        
        @test optData.stop_iteration == 0
        @test progData.counters.neval_obj == 1 
        @test progData.counters.neval_grad == 1
        @test x == x0
        
        grd = OptimizationMethods.grad(progData, x0)
        grd_norm = norm(grd)
        @test optData.grad_val_hist ≈ [norm(grd)]
        @test optData.τ_lower ≈ norm(grd) / sqrt(2)
        @test optData.τ_upper ≈ norm(grd) * sqrt(10)
    end

    # should exit on iteration 0 because threshold is larger than gradient 
    let x0=copy(x0), δ0=δ0, δ_upper=δ_upper, ρ=ρ, threshold=1e4, 
        max_iterations=max_iterations

        # Specify optimization method and problem
        optData = NonsequentialArmijoAdaptiveGD(Float64; x0=x0, δ0=δ0, δ_upper=δ_upper,
            ρ=ρ, threshold=threshold, max_iterations=max_iterations)
        progData = OptimizationMethods.LeastSquares(Float64, nvar=dim)

        # Run method
        x = nonsequential_armijo_adaptive_gd(optData, progData)

        @test optData.stop_iteration == 0
        @test progData.counters.neval_obj == 1 
        @test progData.counters.neval_grad == 1
        @test x == x0
        
        grd = OptimizationMethods.grad(progData, x0)
        grd_norm = norm(grd)
        @test optData.grad_val_hist[1] ≈ norm(grd)
        @test optData.τ_lower ≈ norm(grd) / sqrt(2)
        @test optData.τ_upper ≈ norm(grd) * sqrt(10)
    end

    # should exit on iteration about 26, so we stop one iteration short 
    factor = 1000
    let x0=copy(x0), δ0=factor * δ0, δ_upper=factor * δ_upper, ρ=ρ, threshold=threshold, 
        max_iterations=100

        #Specify Problem 
        progData = OptimizationMethods.LeastSquares(Float64, nvar=dim)

        # Specify optimization method for exit_iteration - 1
        optData = NonsequentialArmijoAdaptiveGD(Float64; x0=x0, δ0=δ0, δ_upper=δ_upper,
            ρ=ρ, threshold=threshold, max_iterations=max_iterations)
        
        x = nonsequential_armijo_adaptive_gd(optData, progData)
        g = OptimizationMethods.grad(progData, x)
        
        stop_iteration = optData.stop_iteration 

        @test optData.iter_hist[stop_iteration+1] == x
        @test optData.grad_val_hist[stop_iteration+1] ≈ norm(g)

        # Since last iteration is accepted the following must be true 
        @test optData.iter_hist[stop_iteration] != 
            optData.iter_hist[stop_iteration+1]
        @test optData.grad_val_hist[stop_iteration] != 
            optData.grad_val_hist[stop_iteration+1]

        # Find the first time we accept an iterate, stop 1 before it
        # and verify that the inner loop is correct
        first_acceptance = 1
        while (first_acceptance < stop_iteration) && 
                (optData.iter_hist[first_acceptance] == 
                    optData.iter_hist[first_acceptance + 1])
            first_acceptance += 1
        end

        ## reset
        precomp, store = OptimizationMethods.initialize(progData)
        F(x) = OptimizationMethods.obj(progData, x)
        for k in 1:(first_acceptance-1)

            # create optdata for k - 1 and k
            optDatakm1 = NonsequentialArmijoAdaptiveGD(Float64; x0=x0, δ0=δ0, δ_upper=δ_upper,
                ρ=ρ, threshold=threshold, max_iterations=k-1) ## return x_{k-1}

            optDatak = NonsequentialArmijoAdaptiveGD(Float64; x0=x0, δ0=δ0, δ_upper=δ_upper,
                ρ=ρ, threshold=threshold, max_iterations=k) ## return x_k

            # generate k - 1 
            xkm1 = nonsequential_armijo_adaptive_gd(optDatakm1, progData)  
            
            # Setting up for test - output of inner loop for iteration k
            x = copy(xkm1) 
            OptimizationMethods.grad!(progData, precomp, store, x)
            OptimizationMethods.inner_loop!(x, optDatakm1.iter_hist[k], 
                optDatakm1, progData, precomp, store, (k == 1), k)
            
            # Test that the non sequential armijo condition is failed
            @test !OptimizationMethods.non_sequential_armijo_condition(
                F(x), F(xkm1), 
                norm(OptimizationMethods.grad(progData, xkm1)), 
                optDatakm1.ρ, optDatakm1.δk, optDatakm1.α0k)

            # Generate x_k and test the optDatak 
            xk = nonsequential_armijo_adaptive_gd(optDatak, progData)

            ## check gradient quantities
            @test isapprox(optDatak.∇F_θk, 
                OptimizationMethods.grad(progData, xkm1))

            ## Check that that the algorithm updated the parameters correctly
            @test optDatak.δk == (optDatakm1.δk * .5)
            @test xk == xkm1

            ## test field values at time k
            @test optDatak.iter_hist[k+1] == xk
            @test optDatak.grad_val_hist[k+1] == optDatak.grad_val_hist[k]
            @test optDatak.grad_val_hist[k+1] ≈ 
                norm(OptimizationMethods.grad(progData, xkm1))
        end 

        # Test values are correctly updated for acceptance
        iter = first_acceptance - 1

        # create optdata for k - 1 and k
        optDatakm1 = NonsequentialArmijoAdaptiveGD(Float64; x0=x0, δ0=δ0, δ_upper=δ_upper,
            ρ=ρ, threshold=threshold, max_iterations=iter) ## stop_iteration = iter

        optDatak = NonsequentialArmijoAdaptiveGD(Float64; x0=x0, δ0=δ0, δ_upper=δ_upper,
            ρ=ρ, threshold=threshold, max_iterations=iter + 1) ## stop_iteration = iter + 1

        # generate k - 1 and k
        xkm1 = nonsequential_armijo_adaptive_gd(optDatakm1, progData)  
        xk = nonsequential_armijo_adaptive_gd(optDatak, progData)

        # Setting up for test - output of inner loop for iteration k
        x = copy(xkm1) 
        OptimizationMethods.grad!(progData, precomp, store, x)
        OptimizationMethods.inner_loop!(x, optDatakm1.iter_hist[iter + 1], 
            optDatakm1, progData, precomp, store, ((iter + 1) == 1), iter + 1)

        # test that non sequential armijo condition is accepted  
        achieved_descent = OptimizationMethods.non_sequential_armijo_condition(
            F(x), F(xkm1), optDatakm1.grad_val_hist[iter + 1], 
            optDatakm1.ρ, optDatakm1.δk, optDatakm1.α0k)
        @test achieved_descent
        
        # Update the parameters in optDatakm1
        flag = OptimizationMethods.update_algorithm_parameters!(x, optDatakm1, 
            achieved_descent, iter + 1)
        
        # Check that optDatak matches optDatakm1
        @test flag
        @test optDatak.∇F_θk ≈ OptimizationMethods.grad(progData, xkm1)
        @test optDatak.grad_val_hist[iter + 2] == optDatakm1.norm_∇F_ψ
        @test optDatak.δk == optDatakm1.δk
        @test optDatak.τ_lower == optDatakm1.τ_lower
        @test optDatak.τ_upper == optDatakm1.τ_upper
        @test xk == x

        ## test field values at time k
        @test optDatak.iter_hist[iter+1] == xkm1
        @test optDatak.iter_hist[iter+2] == xk
        @test optDatak.grad_val_hist[iter+2] ≈ 
            norm(OptimizationMethods.grad(progData, x))
        
        # Find window of non-accepted iterates for last accepted iterate
        last_acceptance = stop_iteration - 1
        while (1 < last_acceptance) && 
                (optData.iter_hist[last_acceptance] == 
                    optData.iter_hist[last_acceptance + 1])
            last_acceptance -= 1
        end
        last_acceptance += 1

        for k in last_acceptance:(stop_iteration-1)

            # create optdata for k - 1 and k
            optDatakm1 = NonsequentialArmijoAdaptiveGD(Float64; x0=x0, δ0=δ0, δ_upper=δ_upper,
                ρ=ρ, threshold=threshold, max_iterations=k-1) ## return x_{k-1}

            optDatak = NonsequentialArmijoAdaptiveGD(Float64; x0=x0, δ0=δ0, δ_upper=δ_upper,
                ρ=ρ, threshold=threshold, max_iterations=k) ## return x_k

            # generate k - 1 
            xkm1 = nonsequential_armijo_adaptive_gd(optDatakm1, progData)  
            
            # Setting up for test - output of inner loop for iteration k
            x = copy(xkm1) 
            OptimizationMethods.grad!(progData, precomp, store, x)
            OptimizationMethods.inner_loop!(x, optDatakm1.iter_hist[k], 
                optDatakm1, progData, precomp, store, (k == last_acceptance), k)
            
            # Test that the non sequential armijo condition is failed
            @test !OptimizationMethods.non_sequential_armijo_condition(
                F(x), F(xkm1), 
                norm(OptimizationMethods.grad(progData, xkm1)), 
                optDatakm1.ρ, optDatakm1.δk, optDatakm1.α0k)

            # Generate x_k and test the optDatak 
            xk = nonsequential_armijo_adaptive_gd(optDatak, progData)

            ## check gradient quantities
            @test isapprox(optDatak.∇F_θk, 
                OptimizationMethods.grad(progData, xkm1))

            ## Check that that the algorithm updated the parameters correctly
            @test optDatak.δk == (optDatakm1.δk * .5)
            @test xk == xkm1

            ## test field values at time k
            @test optDatak.iter_hist[k+1] == xk
            @test optDatak.grad_val_hist[k+1] == optDatak.grad_val_hist[k]
            @test optDatak.grad_val_hist[k+1] ≈ 
                norm(OptimizationMethods.grad(progData, xkm1))

        end

        # Test values are correctly updated for acceptance
        iter = stop_iteration - 1

        # create optdata for k - 1 and k
        optDatakm1 = NonsequentialArmijoAdaptiveGD(Float64; x0=x0, δ0=δ0, δ_upper=δ_upper,
            ρ=ρ, threshold=threshold, max_iterations=iter) ## stop_iteration = iter

        optDatak = NonsequentialArmijoAdaptiveGD(Float64; x0=x0, δ0=δ0, δ_upper=δ_upper,
            ρ=ρ, threshold=threshold, max_iterations=iter + 1) ## stop_iteration = iter + 1

        # generate k - 1 and k
        xkm1 = nonsequential_armijo_adaptive_gd(optDatakm1, progData)  
        xk = nonsequential_armijo_adaptive_gd(optDatak, progData)

        # Setting up for test - output of inner loop for iteration k
        x = copy(xkm1) 
        OptimizationMethods.grad!(progData, precomp, store, x)
        OptimizationMethods.inner_loop!(x, optDatakm1.iter_hist[iter + 1], 
            optDatakm1, progData, precomp, store, 
            (stop_iteration-last_acceptance == 0), iter + 1)

        # test that non sequential armijo condition is accepted  
        achieved_descent = OptimizationMethods.non_sequential_armijo_condition(
            F(x), F(xkm1), optDatakm1.grad_val_hist[iter + 1], 
            optDatakm1.ρ, optDatakm1.δk, optDatakm1.α0k)
        @test achieved_descent
        
        # Update the parameters in optDatakm1
        flag = OptimizationMethods.update_algorithm_parameters!(x, optDatakm1, 
            achieved_descent, iter + 1)
        
        # Check that optDatak matches optDatakm1
        @test flag
        @test optDatak.∇F_θk ≈ OptimizationMethods.grad(progData, xkm1)
        @test optDatak.grad_val_hist[iter + 2] == optDatakm1.norm_∇F_ψ
        @test optDatak.δk == optDatakm1.δk
        @test optDatak.τ_lower == optDatakm1.τ_lower
        @test optDatak.τ_upper == optDatakm1.τ_upper
        @test xk == x

        ## test field values at time k
        @test optDatak.iter_hist[iter+1] == xkm1
        @test optDatak.iter_hist[iter+2] == xk
        @test optDatak.grad_val_hist[iter+2] ≈ 
            norm(OptimizationMethods.grad(progData, x))

    end
end

end