# Date: 2025/03/19
# Author: Christian Varner
# Purpose: Test cases for gradient descent with 
# non-sequential armijo with fixed step size gradient descent

module TestNonsequentialArmijoFixedGD

using Test, OptimizationMethods, CircularArrays, LinearAlgebra, Random

@testset "Method -- Gradient Descent with Nonsequential Armijo Fixed GD: struct" begin

    # set seed for reproducibility
    Random.seed!(1010)

    # test definition
    @test isdefined(OptimizationMethods, :NonsequentialArmijoFixedGD)

    # test field values -- default names
    default_fields = [:name, :threshold, :max_iterations, :iter_hist,
        :grad_val_hist, :stop_iteration]
    let fields = default_fields
        for field_name in fields
            @test field_name in fieldnames(NonsequentialArmijoFixedGD)
        end
    end

    # test field values -- unique names
    unique_fields = [:∇F_θk, :norm_∇F_ψ, :α, :δk, :δ_upper, :ρ, :objective_hist,
    :reference_value, :reference_value_index, :acceptance_cnt, :τ_lower, :τ_upper]
    let fields = unique_fields
        for field_name in fields
            @test field_name in fieldnames(NonsequentialArmijoFixedGD)
        end
    end

    # make sure no other field names exist
    @test length(default_fields) + length(unique_fields) == 
        length(fieldnames(NonsequentialArmijoFixedGD))

    # test field types
    field_info(type::T) where T = [
        [:name, String],
        [:∇F_θk, Vector{type}],
        [:norm_∇F_ψ, type],
        [:α, type],
        [:δk, type],
        [:δ_upper, type],
        [:ρ, type],
        [:objective_hist, CircularVector{type, Vector{type}}],
        [:reference_value, type],
        [:reference_value_index, Int64],
        [:acceptance_cnt, Int64],
        [:τ_lower, type],
        [:τ_upper, type],
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
            α = abs(randn(real_type, 1)[1])
            δ0 = abs(randn(real_type, 1)[1])
            δ_upper = δ0 + 1
            ρ = abs(randn(real_type, 1)[1])
            M = rand(3:100)
            threshold = abs(randn(real_type, 1)[1])
            max_iterations = rand(1:100)

            ## build structure
            optData = NonsequentialArmijoFixedGD(real_type;
                x0 = x0,
                α = α,
                δ0 = δ0,
                δ_upper = δ_upper,
                ρ = ρ,
                M = M,
                threshold = threshold,
                max_iterations = max_iterations)

            ## check field info
            for (fieldname, fieldtype) in field_info(real_type)
                @test fieldtype == typeof(getfield(optData, fieldname))
            end

            ## check field correctness
            @test optData.iter_hist[1] == x0
            @test optData.α == α
            @test optData.δk == δ0
            @test optData.δ_upper == δ_upper
            @test optData.ρ == ρ
            @test length(optData.objective_hist) == M
            @test optData.threshold == threshold
            @test optData.max_iterations == max_iterations
            @test length(optData.iter_hist) == max_iterations + 1
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
            α = abs(randn(real_type, 1)[1])
            ρ = abs(randn(real_type, 1)[1])
            M = rand(1:100)
            threshold = abs(randn(real_type, 1)[1])
            max_iterations = rand(1:100)

            δ0 = -real_type(1) 
            δ_upper = real_type(0)
            
            ## error should occur since δ0 < 0
            @test_throws AssertionError optData = NonsequentialArmijoFixedGD(
                real_type;
                x0 = x0,
                α = α,
                δ0 = δ0,
                δ_upper = δ_upper,
                ρ = ρ,
                M = M,
                threshold = threshold,
                max_iterations = max_iterations)

            δ0 = real_type(1.0)
            δ_upper = real_type(.5)

            ## error should occur since δ0 > δ_upper
            @test_throws AssertionError optData = NonsequentialArmijoFixedGD(
                real_type;
                x0 = x0,
                α = α,
                δ0 = δ0,
                δ_upper = δ_upper,
                ρ = ρ,
                M = M,
                threshold = threshold,
                max_iterations = max_iterations)
            
            δ0 = abs(randn(real_type, 1)[1])
            δ_upper = δ0 + 1
            α = real_type(0)

            ## error should occur since α == 0
            @test_throws AssertionError optData = NonsequentialArmijoFixedGD(
                real_type;
                x0 = x0,
                α = α,
                δ0 = δ0,
                δ_upper = δ_upper, 
                ρ = ρ,
                M = M,
                threshold = threshold, 
                max_iterations = max_iterations)

            α = real_type(-1)

            ## error should occur since α < 0
            @test_throws AssertionError optData = NonsequentialArmijoFixedGD(
                real_type;
                x0 = x0,
                α = α,
                δ0 = δ0,
                δ_upper = δ_upper, 
                ρ = ρ,
                M = M,
                threshold = threshold, 
                max_iterations = max_iterations)
        end
    end 
end # end of testset

@testset "Utility -- Update Algorithm Parameters" begin

    include("../utility/update_algorithm_parameters_test_cases.jl")

    ## arguments
    dim = 50
    x0 = randn(50)
    α = abs(randn(1)[1])
    δ0 = abs(randn(1)[1])
    δ_upper = δ0 + 1
    ρ = abs(randn(1)[1])
    M = rand(1:100)
    threshold = abs(randn(1)[1])
    max_iterations = rand(3:100)

    ## build structure
    optData = NonsequentialArmijoFixedGD(Float64;
        x0 = x0,
        α = α,
        δ0 = δ0,
        δ_upper = δ_upper,
        ρ = ρ,
        M = M,
        threshold = threshold,
        max_iterations = max_iterations)
    
    ## Conduct test cases
    update_algorithm_parameters_test_cases(optData, dim, max_iterations;
        constant_fields = [:name, :∇F_θk, :norm_∇F_ψ, :α, :δ_upper,
        :ρ, :objective_hist, :reference_value, :reference_value_index])
end

@testset "Utility -- Inner Loop" begin
    
    dim = 50
    x0 = randn(50)
    α = abs(randn(1)[1])
    δ0 = abs(randn(1)[1])
    δ_upper = δ0 + 1
    ρ = abs(randn(1)[1])
    M = rand(1:100)
    threshold = abs(randn(1)[1])
    max_iterations = rand(3:100)

    ## build structure
    optData = NonsequentialArmijoFixedGD(Float64;
        x0 = x0,
        δ0 = δ0,
        α = α,
        δ_upper = δ_upper,
        ρ = ρ,
        M = M,
        threshold = threshold,
        max_iterations = max_iterations)

    progData = OptimizationMethods.LeastSquares(Float64, nvar=dim)
    precomp, store = OptimizationMethods.initialize(progData)
    ks = [1, max_iterations]

    for k in ks
        # Test first event trigger: radius violation
        let ψjk=x0 .+ 11, θk=x0, optData=optData, progData=progData,
            store=store, k=k

            optData.grad_val_hist[k] = 1.5
            optData.τ_lower = 1.0
            optData.τ_upper = 2.0
            
            OptimizationMethods.inner_loop!(ψjk, θk, optData, progData, precomp, 
                store, k, max_iteration=100)

            @test ψjk == x0 .+ 11
        end

        # Test second event trigger: τ_lower 
        let ψjk=copy(x0), θk=copy(x0), optData=optData, progData=progData,
            store=store, k=k

            optData.grad_val_hist[k] = 0.5 
            optData.τ_lower = 1.0
            optData.τ_upper = 2.0
            
            OptimizationMethods.inner_loop!(ψjk, θk, optData, progData, precomp, 
                store, k, max_iteration=100)

            @test ψjk == x0
        end

        # Test third event trigger: τ_upper 
        let ψjk=copy(x0), θk=copy(x0), optData=optData, progData=progData,
            store=store, k=k

            optData.grad_val_hist[k] = 2.5 
            optData.τ_lower = 1.0
            optData.τ_upper = 2.0
            
            OptimizationMethods.inner_loop!(ψjk, θk, optData, progData, precomp, 
                store, k, max_iteration=100)

            @test ψjk == x0
        end

        # Test fourth event trigger: max_iteration
        let ψjk=copy(x0), θk=copy(x0), optData=optData, progData=progData,
            store=store, k=k

            optData.grad_val_hist[k] = 1.5 
            optData.τ_lower = 1.0
            optData.τ_upper = 2.0
            
            OptimizationMethods.inner_loop!(ψjk, θk, optData, progData, precomp, 
                store, k, max_iteration=0)

            @test ψjk == x0
        end

        # Test first iteration
        let ψjk=copy(x0), θk=copy(x0), optData=optData, progData=progData,
            store=store, k=k

            j=1

            OptimizationMethods.grad!(progData, precomp, store, θk)
            optData.grad_val_hist[k] = norm(store.grad)
            optData.norm_∇F_ψ = norm(store.grad)
            optData.τ_lower = 0.5 * norm(store.grad) 
            optData.τ_upper = 1.5 * norm(store.grad)

            optData.δk = 1.5
            α = optData.α

            step = (optData.δk * α) .* store.grad 

            OptimizationMethods.inner_loop!(ψjk, θk, optData, progData, precomp,
                store, k, max_iteration = 1
            )

            @test ψjk == θk - step 
            @test optData.α == α
            @test store.grad ≈ OptimizationMethods.grad(progData, ψjk)
            @test optData.norm_∇F_ψ == norm(store.grad)
        end

        # Test random iteration
        let ψjk=copy(x0), θk=copy(x0), optData=optData, progData=progData,
            store=store, k=k

            #To do this test correctly, we would need to know at what iteration 
            #j an inner loop exists.
            max_iteration = rand(2:100)

            # Reset 
            OptimizationMethods.grad!(progData, precomp, store, θk)
            optData.grad_val_hist[k] = norm(store.grad)
            optData.norm_∇F_ψ = norm(store.grad)
            optData.τ_lower = 0.5 * norm(store.grad) 
            optData.τ_upper = 1.5 * norm(store.grad)
            optData.δk = 1.5

            #Get exit iteration j
            j = OptimizationMethods.inner_loop!(ψjk, θk, optData, progData, precomp,
                store, k, max_iteration = max_iteration
            )

            # Reset 
            ψjk = copy(x0)
            θk = copy(x0)

            OptimizationMethods.grad!(progData, precomp, store, θk)
            optData.grad_val_hist[k] = norm(store.grad)
            optData.norm_∇F_ψ = norm(store.grad)
            optData.τ_lower = 0.5 * norm(store.grad) 
            optData.τ_upper = 1.5 * norm(store.grad)
            optData.δk = 1.5

            OptimizationMethods.inner_loop!(ψjk, θk, optData, progData, precomp,
                store, k, max_iteration = j-1
            )
            α = optData.α

            ψ_jm1_k = copy(ψjk)
            grd = OptimizationMethods.grad(progData, ψ_jm1_k)
            step = (optData.δk * α) * grd

            # Reset 
            ψjk = copy(x0)
            θk = copy(x0)

            OptimizationMethods.grad!(progData, precomp, store, θk)
            optData.grad_val_hist[k] = norm(store.grad)
            optData.norm_∇F_ψ = norm(store.grad)
            optData.τ_lower = 0.5 * norm(store.grad) 
            optData.τ_upper = 1.5 * norm(store.grad)
            optData.δk = 1.5

            #Get ψ_{j,k}
            OptimizationMethods.inner_loop!(ψjk, θk, optData, progData, precomp,
                store, k, max_iteration = max_iteration
            )

            @test ψjk ≈ ψ_jm1_k - step
            @test store.grad ≈ OptimizationMethods.grad(progData, ψjk)
            @test optData.norm_∇F_ψ ≈ norm(store.grad)
            @test optData.α == α
        end
    end
end

@testset "Method -- Gradient Descent with Nonsequential Armijo: method" begin
    
    M = [1, rand(2:100)]
    let Ms = M
        for M in Ms
            # Default Parameters 
            dim = 50
            x0 = randn(dim)
            α = abs(randn())
            δ0 = abs(randn())
            δ_upper = δ0 + 2 
            ρ = abs(randn()) * 1e-3
            threshold = 1e-3
            max_iterations = 100 

            # Should exit on iteration 0 because max_iterations is 0
            let x0=copy(x0), δ0=δ0, δ_upper=δ_upper, ρ=ρ, threshold=threshold, 
                max_iterations=0

                # Specify optimization method and problem
                optData = NonsequentialArmijoFixedGD(Float64; x0=x0, α = α, δ0=δ0, 
                    δ_upper=δ_upper, ρ=ρ, M = M, threshold=threshold, 
                    max_iterations=max_iterations)
                progData = OptimizationMethods.LeastSquares(Float64, nvar=dim)

                # Run method
                x = nonsequential_armijo_fixed_gd(optData, progData)

                
                @test optData.stop_iteration == 0
                @test progData.counters.neval_obj == 1 
                @test progData.counters.neval_grad == 1
                @test x == x0
                
                grd = OptimizationMethods.grad(progData, x0)
                grd_norm = norm(grd)
                @test optData.grad_val_hist ≈ [norm(grd)]
                @test optData.τ_lower ≈ norm(grd) / sqrt(2)
                @test optData.τ_upper ≈ norm(grd) * sqrt(10)
                @test optData.objective_hist[1] == OptimizationMethods.obj(progData, x0)
                @test optData.reference_value == OptimizationMethods.obj(progData, x0)
                @test optData.reference_value_index == 1
            end

            # should exit on iteration 0 because threshold is larger than gradient 
            let x0=copy(x0), δ0=δ0, δ_upper=δ_upper, ρ=ρ, threshold=1e4, 
                max_iterations=max_iterations

                # Specify optimization method and problem
                optData = NonsequentialArmijoFixedGD(Float64; x0=x0, α = α, δ0=δ0, 
                    δ_upper=δ_upper, ρ=ρ, M = M, threshold=threshold, 
                    max_iterations=max_iterations)
                progData = OptimizationMethods.LeastSquares(Float64, nvar=dim)

                # Run method
                x = nonsequential_armijo_fixed_gd(optData, progData)

                @test optData.stop_iteration == 0
                @test progData.counters.neval_obj == 1 
                @test progData.counters.neval_grad == 1
                @test x == x0
                
                grd = OptimizationMethods.grad(progData, x0)
                grd_norm = norm(grd)
                @test optData.grad_val_hist[1] ≈ norm(grd)
                @test optData.τ_lower ≈ norm(grd) / sqrt(2)
                @test optData.τ_upper ≈ norm(grd) * sqrt(10)
                @test optData.objective_hist[1] == OptimizationMethods.obj(progData, x0)
                @test optData.reference_value == OptimizationMethods.obj(progData, x0)
                @test optData.reference_value_index == 1
            end
        
            factor = 1000
            let x0=copy(x0), δ0=factor * δ0, δ_upper=factor * δ_upper, ρ=ρ, threshold=threshold, 
                max_iterations=100

                #Specify Problem 
                progData = OptimizationMethods.LeastSquares(Float64, nvar=dim)

                # Specify optimization method for exit_iteration - 1
                optData = NonsequentialArmijoFixedGD(Float64; x0=x0, α = α, δ0=δ0, 
                    δ_upper=δ_upper, ρ=ρ, M = M, threshold=threshold, 
                    max_iterations=max_iterations)
                
                x = nonsequential_armijo_fixed_gd(optData, progData)
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
                    optDatakm1 = NonsequentialArmijoFixedGD(Float64; x0=x0, α = α, δ0=δ0, 
                        δ_upper=δ_upper, ρ=ρ, M = M, threshold=threshold, 
                        max_iterations=k-1) ## return x_{k-1}

                    optDatak = NonsequentialArmijoFixedGD(Float64; x0=x0, α = α, δ0=δ0, 
                        δ_upper=δ_upper, ρ=ρ, M = M, threshold=threshold, 
                        max_iterations=k) ## return x_k

                    # generate k - 1 
                    xkm1 = nonsequential_armijo_fixed_gd(optDatakm1, progData)  
                    
                    # Setting up for test - output of inner loop for iteration k
                    x = copy(xkm1) 
                    OptimizationMethods.grad!(progData, precomp, store, x)
                    OptimizationMethods.inner_loop!(x, optDatakm1.iter_hist[k], 
                        optDatakm1, progData, precomp, store, k)
                    
                    # Test that the non sequential armijo condition is failed
                    @test !OptimizationMethods.non_sequential_armijo_condition(
                        F(x), optDatakm1.reference_value, 
                        norm(OptimizationMethods.grad(progData, xkm1)), 
                        optDatakm1.ρ, optDatakm1.δk, optDatakm1.α)

                    # Generate x_k and test the optDatak 
                    xk = nonsequential_armijo_fixed_gd(optDatak, progData)

                    ## check gradient quantities
                    @test isapprox(optDatak.∇F_θk, 
                        OptimizationMethods.grad(progData, xkm1))

                    ## Check that that the algorithm updated the parameters correctly
                    @test optDatak.δk == (optDatakm1.δk * .5)
                    @test xk == xkm1

                    ## test field values at time k
                    @test optDatak.reference_value == maximum(optDatak.objective_hist)
                    @test optDatak.objective_hist[optDatak.reference_value_index] ==
                        optDatak.reference_value
                    @test optDatak.iter_hist[k+1] == xk
                    @test optDatak.grad_val_hist[k+1] == optDatak.grad_val_hist[k]
                    @test optDatak.grad_val_hist[k+1] ≈ 
                        norm(OptimizationMethods.grad(progData, xkm1))
                end 

                # Test values are correctly updated for acceptance
                iter = first_acceptance - 1

                # create optdata for k - 1 and k
                optDatakm1 = NonsequentialArmijoFixedGD(Float64; x0=x0, α = α, δ0=δ0, 
                    δ_upper=δ_upper, ρ=ρ, M = M, threshold=threshold, 
                    max_iterations=iter) ## stop_iteration = iter

                optDatak = NonsequentialArmijoFixedGD(Float64; x0=x0, α = α, δ0=δ0, 
                    δ_upper=δ_upper, ρ=ρ, M = M, threshold=threshold, 
                    max_iterations=iter + 1) ## stop_iteration = iter + 1

                # generate k - 1 and k
                xkm1 = nonsequential_armijo_fixed_gd(optDatakm1, progData)  
                xk = nonsequential_armijo_fixed_gd(optDatak, progData)

                # Setting up for test - output of inner loop for iteration k
                x = copy(xkm1) 
                OptimizationMethods.grad!(progData, precomp, store, x)
                OptimizationMethods.inner_loop!(x, optDatakm1.iter_hist[iter + 1], 
                    optDatakm1, progData, precomp, store, iter + 1)

                # test that non sequential armijo condition is accepted  
                achieved_descent = OptimizationMethods.non_sequential_armijo_condition(
                    F(x),  optDatakm1.reference_value, optDatakm1.grad_val_hist[iter + 1], 
                    optDatakm1.ρ, optDatakm1.δk, optDatakm1.α)
                @test achieved_descent
                
                # Update the parameters in optDatakm1
                flag = OptimizationMethods.update_algorithm_parameters!(x, optDatakm1, 
                    achieved_descent, iter + 1)
                
                # update the cache at time k - 1
                optDatakm1.acceptance_cnt += 1
                optDatakm1.objective_hist[optDatakm1.acceptance_cnt] = F(x)
                if (optDatakm1.acceptance_cnt % M) + 1 == optDatakm1.reference_value_index
                    optDatakm1.reference_value, optDatakm1.reference_value_index =
                    findmax(optDatakm1.objective_hist)
                end
                
                # Check that optDatak matches optDatakm1
                @test flag
                @test optDatak.∇F_θk ≈ OptimizationMethods.grad(progData, xkm1)
                @test optDatak.grad_val_hist[iter + 2] == optDatakm1.norm_∇F_ψ
                @test optDatak.δk == optDatakm1.δk
                @test optDatak.τ_lower == optDatakm1.τ_lower
                @test optDatak.τ_upper == optDatakm1.τ_upper
                @test xk == x

                ## test field values at time k
                @test optDatak.reference_value == maximum(optDatak.objective_hist)
                @test optDatak.objective_hist[optDatak.reference_value_index] ==
                    optDatak.reference_value
                @test optDatak.reference_value == optDatakm1.reference_value
                @test optDatak.reference_value_index == optDatakm1.reference_value_index
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
                    optDatakm1 = NonsequentialArmijoFixedGD(Float64; x0=x0, α = α, δ0=δ0, 
                        δ_upper=δ_upper, ρ=ρ, M = M, threshold=threshold, 
                        max_iterations=k-1) ## return x_{k-1}

                    optDatak = NonsequentialArmijoFixedGD(Float64; x0=x0, α = α, δ0=δ0, 
                        δ_upper=δ_upper, ρ=ρ, M = M, threshold=threshold, 
                        max_iterations=k) ## return x_k

                    # generate k - 1 
                    xkm1 = nonsequential_armijo_fixed_gd(optDatakm1, progData)  
                    
                    # Setting up for test - output of inner loop for iteration k
                    x = copy(xkm1) 
                    OptimizationMethods.grad!(progData, precomp, store, x)
                    OptimizationMethods.inner_loop!(x, optDatakm1.iter_hist[k], 
                        optDatakm1, progData, precomp, store, k)
                    
                    # Test that the non sequential armijo condition is failed
                    @test !OptimizationMethods.non_sequential_armijo_condition(
                        F(x), optDatakm1.reference_value, 
                        norm(OptimizationMethods.grad(progData, xkm1)), 
                        optDatakm1.ρ, optDatakm1.δk, optDatakm1.α)

                    # Generate x_k and test the optDatak 
                    xk = nonsequential_armijo_fixed_gd(optDatak, progData)

                    ## check gradient quantities
                    @test isapprox(optDatak.∇F_θk, 
                        OptimizationMethods.grad(progData, xkm1))

                    ## Check that that the algorithm updated the parameters correctly
                    @test optDatak.δk == (optDatakm1.δk * .5)
                    @test xk == xkm1

                    ## test field values at time k
                    @test optDatak.reference_value == maximum(optDatak.objective_hist)
                    @test optDatak.objective_hist[optDatak.reference_value_index] ==
                        optDatak.reference_value
                    @test optDatak.iter_hist[k+1] == xk
                    @test optDatak.grad_val_hist[k+1] == optDatak.grad_val_hist[k]
                    @test optDatak.grad_val_hist[k+1] ≈ 
                        norm(OptimizationMethods.grad(progData, xkm1))

                end

                # Test values are correctly updated for acceptance
                iter = stop_iteration - 1

                # create optdata for k - 1 and k
                optDatakm1 = NonsequentialArmijoFixedGD(Float64; x0=x0, α = α, δ0=δ0, 
                    δ_upper=δ_upper, ρ=ρ, M = M, threshold=threshold, 
                    max_iterations=iter) ## stop_iteration = iter

                optDatak = NonsequentialArmijoFixedGD(Float64; x0=x0, α = α, δ0=δ0, 
                    δ_upper=δ_upper, ρ=ρ, M = M, threshold=threshold, 
                    max_iterations=iter + 1) ## stop_iteration = iter + 1

                # generate k - 1 and k
                xkm1 = nonsequential_armijo_fixed_gd(optDatakm1, progData)  
                xk = nonsequential_armijo_fixed_gd(optDatak, progData)

                # Setting up for test - output of inner loop for iteration k
                x = copy(xkm1) 
                OptimizationMethods.grad!(progData, precomp, store, x)
                OptimizationMethods.inner_loop!(x, optDatakm1.iter_hist[iter + 1], 
                    optDatakm1, progData, precomp, store, iter + 1)

                # test that non sequential armijo condition is accepted  
                achieved_descent = OptimizationMethods.non_sequential_armijo_condition(
                    F(x), optDatakm1.reference_value, optDatakm1.grad_val_hist[iter + 1], 
                    optDatakm1.ρ, optDatakm1.δk, optDatakm1.α)
                @test achieved_descent
                
                # Update the parameters in optDatakm1
                flag = OptimizationMethods.update_algorithm_parameters!(x, optDatakm1, 
                    achieved_descent, iter + 1)

                # update the cache at time k - 1
                optDatakm1.acceptance_cnt += 1
                optDatakm1.objective_hist[optDatakm1.acceptance_cnt] = F(x)
                if (optDatakm1.acceptance_cnt % M) + 1 == optDatakm1.reference_value_index
                    optDatakm1.reference_value, optDatakm1.reference_value_index =
                    findmax(optDatakm1.objective_hist)
                end
                
                
                # Check that optDatak matches optDatakm1
                @test flag
                @test optDatak.∇F_θk ≈ OptimizationMethods.grad(progData, xkm1)
                @test optDatak.grad_val_hist[iter + 2] == optDatakm1.norm_∇F_ψ
                @test optDatak.δk == optDatakm1.δk
                @test optDatak.τ_lower == optDatakm1.τ_lower
                @test optDatak.τ_upper == optDatakm1.τ_upper
                @test xk == x

                ## test field values at time k
                @test optDatak.reference_value == maximum(optDatak.objective_hist)
                @test optDatak.objective_hist[optDatak.reference_value_index] ==
                    optDatak.reference_value
                @test optDatak.reference_value == optDatakm1.reference_value
                @test optDatak.reference_value_index == optDatakm1.reference_value_index
                @test optDatak.iter_hist[iter+1] == xkm1
                @test optDatak.iter_hist[iter+2] == xk
                @test optDatak.grad_val_hist[iter+2] ≈ 
                    norm(OptimizationMethods.grad(progData, x))

            end
        end
    end
end

end # End of module