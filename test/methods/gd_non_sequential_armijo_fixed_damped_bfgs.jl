# Date: 2025/04/30
# Author: Christian Varner
# Purpose: Test cases for the damped BFGS quasi-newton method
# with fixed step sizes globalized through the non-sequential
# Armijo framework

module TestNonsequentialArmijoDampedBFGS

using Test, OptimizationMethods, CircularArrays, LinearAlgebra, Random

@testset "Test NonsequentialArmijoFixedDampedBFGSGD" begin

    ############################################################################
    # Test definition and field names
    ############################################################################
    
    # definition
    @test isdefined(OptimizationMethods, :NonsequentialArmijoFixedDampedBFGSGD)

    # unique names
    unique_field = [
        :∇F_θk,
        :B_θk,
        :Bjk,
        :δBjk,
        :rjk,
        :sjk,
        :yjk,
        :djk,
        :c,
        :β,
        :norm_∇F_ψ,
        :α,
        :δk,
        :δ_upper,
        :ρ,
        :objective_hist,
        :reference_value,
        :reference_value_index,
        :acceptance_cnt,
        :τ_lower,
        :τ_upper,
        :inner_loop_radius,
        :inner_loop_max_iterations]

    let fields = unique_field
        for field in fields
            @test field in fieldnames(NonsequentialArmijoFixedDampedBFGSGD)
        end
    end # end test cases for unique fields

    # default fields
    default_fields = [:name, :threshold, :max_iterations, :iter_hist, 
        :grad_val_hist, :stop_iteration]
    
    let fields = default_fields
        for field in fields
            @test field in fieldnames(NonsequentialArmijoFixedDampedBFGSGD)
        end
    end # end test cases for default fields

    ############################################################################
    # Test error throwing in constructor
    ############################################################################
    
    ############################################################################
    # Test field types
    ############################################################################

    field_info(::Type{T}) where {T} = 
        [
            (:name, String),
            (:∇F_θk, Vector{T}),
            (:B_θk, Matrix{T}),
            (:Bjk, Matrix{T}),
            (:δBjk, Matrix{T}),
            (:rjk, Vector{T}),
            (:sjk, Vector{T}),
            (:yjk, Vector{T}),
            (:djk, Vector{T}),
            (:c, T),
            (:β, T),
            (:norm_∇F_ψ, T),
            (:α, T),
            (:δk, T),
            (:δ_upper, T),
            (:ρ, T),
            (:objective_hist, CircularVector{T, Vector{T}}),
            (:reference_value, T),
            (:reference_value_index, Int64),
            (:acceptance_cnt, Int64),
            (:τ_lower, T),
            (:τ_upper, T),
            (:inner_loop_radius, T),
            (:inner_loop_max_iterations, Int64),
            (:threshold, T),
            (:max_iterations, Int64),
            (:iter_hist, Vector{Vector{T}}),
            (:grad_val_hist, Vector{T}),
            (:stop_iteration, Int64),
        ]

    let field_info = field_info, real_types = [Float16, Float32, Float64]
        dim = 50
        for type in real_types
            
            # generate a random structure
            x0 =               randn(type, dim)
            c =                 rand(type)
            β =                 rand(type)
            α =                 rand(type)
            δ0 =                rand(type)
            ρ =                 rand(type)
            M =                 rand(1:10)
            threshold =         randn(type)
            inner_loop_radius = rand(type)
            inner_loop_max_iterations = rand(1:100)
            max_iterations = rand(1:100)
            δ_upper = δ0 + 1
            
            # construct
            optData = NonsequentialArmijoFixedDampedBFGSGD(type;
                x0 = x0,
                c = c,
                β = β,
                α = α,
                δ0 = δ0,
                δ_upper = δ_upper,
                ρ = ρ,
                M = M,
                inner_loop_radius = inner_loop_radius,
                inner_loop_max_iterations = inner_loop_max_iterations,
                threshold = threshold,
                max_iterations = max_iterations)

            # test the field types
            for (field_symbol, field_type) in field_info(type)
                @test field_type == typeof(getfield(optData, field_symbol))
            end

        end

    end # end the test cases for the field info

    ############################################################################
    # Test initial values for constructor
    ############################################################################

    let real_types = [Float16, Float32, Float64], dim = 50
        for type in real_types
            
            x0 =               randn(type, dim)
            c =                 rand(type)
            β =                 rand(type)
            α =                 rand(type)
            δ0 =                rand(type)
            ρ =                 rand(type)
            M =                 rand(1:10)
            threshold =         randn(type)
            inner_loop_radius = rand(type)
            inner_loop_max_iterations = rand(1:100)
            max_iterations = rand(1:100)
            δ_upper = δ0 + 1
            
            # construct
            optData = NonsequentialArmijoFixedDampedBFGSGD(type;
                x0 = x0,
                c = c,
                β = β,
                α = α,
                δ0 = δ0,
                δ_upper = δ_upper,
                ρ = ρ,
                M = M,
                inner_loop_radius = inner_loop_radius,
                inner_loop_max_iterations = inner_loop_max_iterations,
                threshold = threshold,
                max_iterations = max_iterations)

            @test length(optData.∇F_θk) == dim
            @test length(optData.rjk) == dim
            @test length(optData.sjk) == dim
            @test length(optData.yjk) == dim
            @test length(optData.djk) == dim
            @test size(optData.B_θk) == (dim, dim)
            @test size(optData.δBjk) == (dim, dim)
            
            @test optData.c == c
            @test optData.β == β
            @test optData.α == α
            @test optData.δk == δ0
            @test optData.ρ == ρ
            @test optData.norm_∇F_ψ == type(0)
            
            @test length(optData.objective_hist) == M
            @test optData.reference_value == type(0)
            @test optData.reference_value_index == -1
            @test optData.acceptance_cnt == 0

            @test optData.τ_lower == type(-1) 
            @test optData.τ_upper == type(-1)

            @test optData.inner_loop_radius == inner_loop_radius
            @test optData.inner_loop_max_iterations == inner_loop_max_iterations
            @test optData.threshold == threshold
            @test optData.max_iterations == max_iterations
            @test length(optData.iter_hist) == max_iterations + 1
            @test length(optData.grad_val_hist) == max_iterations + 1
            @test optData.stop_iteration == -1
        end
    end # end test for initial values

end # end test on structure

@testset "Utility -- Update Algorithm Parameters NonseqArmijoDampedBFGS" begin

    include("../utility/update_algorithm_parameters_test_cases.jl")

    # Random arguments
    dim = 50
    x0 =               randn(dim)
    c =                 rand()
    β =                 rand()
    α =                 rand()
    δ0 =                rand()
    ρ =                 rand()
    M =                 rand(1:10)
    threshold =        randn()
    inner_loop_radius = rand()
    inner_loop_max_iterations = rand(1:100)
    max_iterations = rand(1:100)
    δ_upper = δ0 + 1
 
    # build structure
    optData = NonsequentialArmijoFixedDampedBFGSGD(Float64;
        x0 = x0,
        c = c,
        β = β,
        α = α,
        δ0 = δ0,
        δ_upper = δ_upper,
        ρ = ρ,
        M = M,
        inner_loop_radius = inner_loop_radius,
        inner_loop_max_iterations = inner_loop_max_iterations,
        threshold = threshold,
        max_iterations = max_iterations)

    # conduct test cases
    update_algorithm_parameters_test_cases(optData, dim, max_iterations)

end # end test on update algorithm parameters

@testset "Utility -- Inner Loop NonseqArmijoDampedBFGS" begin

    # random arguments
    dim = 50
    x0 =               randn(dim)
    c =                 rand()
    β =                 rand()
    α =                 rand()
    δ0 =                rand()
    ρ =                 rand()
    M =                 rand(1:10)
    threshold =        randn()
    inner_loop_radius = rand()
    inner_loop_max_iterations = rand(1:100)
    max_iterations = rand(1:100)
    δ_upper = δ0 + 1   
    
    # build structure
    optData = NonsequentialArmijoFixedDampedBFGSGD(Float64;
        x0 = x0,
        c = c,
        β = β,
        α = α,
        δ0 = δ0,
        δ_upper = δ_upper,
        ρ = ρ,
        M = M,
        inner_loop_radius = inner_loop_radius,
        inner_loop_max_iterations = inner_loop_max_iterations,
        threshold = threshold,
        max_iterations = max_iterations)
    
    progData = OptimizationMethods.LogisticRegression(Float64, nvar=50)
    precomp, store = OptimizationMethods.initialize(progData)
    ks = [1, max_iterations]

    for k in ks
        # Test first event trigger: radius violation
        let ψjk=x0 .+ inner_loop_radius, 
            θk=x0, optData=optData, progData=progData,
            store=store, k=k

            optData.grad_val_hist[k] = 1.5
            optData.τ_lower = 1.0
            optData.τ_upper = 2.0
            
            OptimizationMethods.inner_loop!(ψjk, θk, optData, progData, precomp, 
                store, k;
                radius = optData.inner_loop_radius,
                max_iteration=100)

            @test ψjk == x0 .+ inner_loop_radius
        end # end first event trigger test

         # Test second event trigger: τ_lower 
         let ψjk=copy(x0), θk=copy(x0), optData=optData, progData=progData,
            store=store, k=k

            optData.grad_val_hist[k] = 0.5 
            optData.τ_lower = 1.0
            optData.τ_upper = 2.0
            
            OptimizationMethods.inner_loop!(ψjk, θk, optData, progData, precomp, 
                store, k;
                radius = optData.inner_loop_radius,
                max_iteration=100)

            @test ψjk == x0
        end # end second event triggering event

        # Test third event trigger: τ_upper 
        let ψjk=copy(x0), θk=copy(x0), optData=optData, progData=progData,
            store=store, k=k

            optData.grad_val_hist[k] = 2.5 
            optData.τ_lower = 1.0
            optData.τ_upper = 2.0
            
            OptimizationMethods.inner_loop!(ψjk, θk, optData, progData, precomp, 
                store, k;
                radius = optData.inner_loop_radius,
                max_iteration=100)

            @test ψjk == x0
        end

         # Test fourth event trigger: max_iteration
         let ψjk=copy(x0), θk=copy(x0), optData=optData, progData=progData,
            store=store, k=k

            optData.grad_val_hist[k] = 1.5 
            optData.τ_lower = 1.0
            optData.τ_upper = 2.0
            
            OptimizationMethods.inner_loop!(ψjk, θk, optData, progData, precomp, 
                store, k; 
                radius = optData.inner_loop_radius,
                max_iteration=0)

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

            # compute the initial step
            g00 = OptimizationMethods.grad(progData, θk)
            B00 = optData.c * norm(g00) * Matrix{Float64}(I, 50, 50)
            step = optData.δk * optData.α * (B00 \ g00)

            # reset
            optData.Bjk = B00
            j = OptimizationMethods.inner_loop!(ψjk, θk, optData, progData, precomp,
                store, k; radius = optData.inner_loop_radius, max_iteration = 1
            )
            @test j == 1

            # test cases
            @test ψjk ≈ θk - step 
            @test optData.α == α
            @test store.grad ≈ OptimizationMethods.grad(progData, ψjk)
            @test optData.norm_∇F_ψ == norm(store.grad)

            g10 = OptimizationMethods.grad(progData, ψjk)
            @test optData.sjk ≈ ψjk - θk
            @test optData.yjk ≈ g10 - g00

            # test to make sure damped BFGS was updated
            update_success = OptimizationMethods.update_bfgs!(
                B00, optData.rjk, optData.δBjk,
                optData.sjk, optData.yjk; damped_update = true)

            @test B00 ≈ optData.Bjk
        end

         # test a random iteration
        let ψjk=copy(x0), θk=copy(x0), optData=optData, progData=progData,
            store=store, k=k

            #To do this test correctly, we would need to know at what iteration 
            #j an inner loop exists.
            max_iteration = rand(2:100)

            # Reset
            OptimizationMethods.grad!(progData, precomp, store, θk)
            optData.Bjk = optData.c * norm(store.grad) * Matrix{Float64}(I, 50, 50)
            optData.grad_val_hist[k] = norm(store.grad)
            optData.norm_∇F_ψ = norm(store.grad)
            optData.τ_lower = 0.5 * norm(store.grad) 
            optData.τ_upper = 1.5 * norm(store.grad)
            optData.δk = 1.5

            #Get exit iteration j -- radius == 10 so that we get resonable j
            j = OptimizationMethods.inner_loop!(ψjk, θk, optData, progData, precomp,
                store, k; max_iteration = max_iteration
            )

            # Reset 
            ψjk = copy(x0)
            θk = copy(x0)

            OptimizationMethods.grad!(progData, precomp, store, θk)
            optData.Bjk = optData.c * norm(store.grad) * Matrix{Float64}(I, 50, 50)
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
            B_jm1_k = copy(optData.Bjk)
            step = optData.δk * optData.α * (optData.Bjk \ store.grad)

            # Reset 
            ψjk = copy(x0)
            θk = copy(x0)

            OptimizationMethods.grad!(progData, precomp, store, θk)
            optData.Bjk = optData.c * norm(store.grad) * Matrix{Float64}(I, 50, 50)
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

            # test the values of s, y
            gjm10 = OptimizationMethods.grad(progData, ψ_jm1_k) 
            gj0 = OptimizationMethods.grad(progData, ψjk)
            @test optData.sjk ≈ ψjk - ψ_jm1_k
            @test optData.yjk ≈ gj0 - gjm10

           # test to make sure damped BFGS was updated
           update_success = OptimizationMethods.update_bfgs!(
                B_jm1_k, optData.rjk, optData.δBjk,
                optData.sjk, optData.yjk; damped_update = true)

            @test B_jm1_k ≈ optData.Bjk 

        end # end of the test of a random iteration
    end # end the test for the initial iteration

end # end test on inner loop

@testset "Test nonsequential_armijo_fixed_damped_bfgs Monotone" begin

    # Random arguments
    dim = 50
    x0 =               randn(dim)
    c =                 rand()
    β =                 rand()
    α =                 rand()
    δ0 =                rand()
    ρ =                 rand()
    M =                 1
    threshold =         rand()
    inner_loop_radius = rand()
    inner_loop_max_iterations = rand(1:100)
    max_iterations = rand(1:100)
    δ_upper = δ0 + 1

    # Should exit on iteration 0 because max_iterations is 0
    let x0 = copy(x0), c = c, β = β, α = α, δ0 = δ0, ρ = ρ, M = M,
        threshold = threshold, inner_loop_radius = inner_loop_radius,
        inner_loop_max_iterations = inner_loop_max_iterations, 
        threshold = threshold,
        max_iterations = 0            
        
        # specify optimization method and problem
        optData = NonsequentialArmijoFixedDampedBFGSGD(Float64;
            x0 = x0,
            c = c,
            β = β,
            α = α,
            δ0 = δ0,
            δ_upper = δ_upper,
            ρ = ρ,
            M = M,
            inner_loop_radius = inner_loop_radius,
            inner_loop_max_iterations = inner_loop_max_iterations,
            threshold = threshold,
            max_iterations = max_iterations)

        progData = OptimizationMethods.LogisticRegression(Float64, nvar=length(x0))

        # Run method
        x = nonsequential_armijo_fixed_damped_bfgs(optData, progData)

        @test optData.stop_iteration == 0
        @test progData.counters.neval_obj == 1 
        @test progData.counters.neval_grad == 1
        @test x == x0

        grd = OptimizationMethods.grad(progData, x0)
        grd_norm = norm(grd)
        @test optData.grad_val_hist ≈ [norm(grd)]
        @test optData.τ_lower ≈ norm(grd) / sqrt(2)
        @test optData.τ_upper ≈ norm(grd) * sqrt(10)
        @test optData.objective_hist == [OptimizationMethods.obj(progData, x0)]
        @test optData.reference_value == OptimizationMethods.obj(progData, x0)
        @test optData.reference_value_index == 1
    end 

    # should exit on iteration 0 because threshold is larger than gradient 
    let x0 = copy(x0), c = c, β = β, α = α, δ0 = δ0, ρ = ρ, M = M,
        threshold = threshold, inner_loop_radius = inner_loop_radius,
        inner_loop_max_iterations = inner_loop_max_iterations, 
        threshold = 1e16,
        max_iterations = max_iterations  
        
        # specify optimization method and problem
        optData = NonsequentialArmijoFixedDampedBFGSGD(Float64;
            x0 = x0,
            c = c,
            β = β,
            α = α,
            δ0 = δ0,
            δ_upper = δ_upper,
            ρ = ρ,
            M = M,
            inner_loop_radius = inner_loop_radius,
            inner_loop_max_iterations = inner_loop_max_iterations,
            threshold = threshold,
            max_iterations = max_iterations)

        progData = OptimizationMethods.LogisticRegression(Float64, nvar=length(x0))

        # Run method
        x = nonsequential_armijo_fixed_damped_bfgs(optData, progData)

        @test optData.stop_iteration == 0
        @test progData.counters.neval_obj == 1 
        @test progData.counters.neval_grad == 1
        @test x == x0

        grd = OptimizationMethods.grad(progData, x0)
        grd_norm = norm(grd)
        @test optData.grad_val_hist[1] ≈ norm(grd)
        @test optData.τ_lower ≈ norm(grd) / sqrt(2)
        @test optData.τ_upper ≈ norm(grd) * sqrt(10)
        @test optData.objective_hist == [OptimizationMethods.obj(progData, x0)]
        @test optData.reference_value == OptimizationMethods.obj(progData, x0)
        @test optData.reference_value_index == 1 
    end
    
    let x0 = copy(x0), c = c, β = β, α = α, δ0 = δ0, ρ = ρ, M = M,
        threshold = threshold, inner_loop_radius = inner_loop_radius,
        inner_loop_max_iterations = inner_loop_max_iterations, 
        threshold = 0.0,
        max_iterations = max_iterations 

        #Specify Problem 
        progData = OptimizationMethods.LogisticRegression(Float64, nvar=length(x0))

        # specify optimization method and problem
        optData = NonsequentialArmijoFixedDampedBFGSGD(Float64;
            x0 = x0,
            c = c,
            β = β,
            α = α,
            δ0 = δ0,
            δ_upper = δ_upper,
            ρ = ρ,
            M = M,
            inner_loop_radius = inner_loop_radius,
            inner_loop_max_iterations = inner_loop_max_iterations,
            threshold = threshold,
            max_iterations = max_iterations)
            
        # Run method
        x = nonsequential_armijo_fixed_damped_bfgs(optData, progData)
        
        g = OptimizationMethods.grad(progData, x) 
        stop_iteration = optData.stop_iteration 

        @test optData.iter_hist[stop_iteration+1] == x
        @test optData.grad_val_hist[stop_iteration+1] ≈ norm(g)

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
            optDatakm1 = NonsequentialArmijoFixedDampedBFGSGD(Float64;
                x0 = x0,
                c = c,
                β = β,
                α = α,
                δ0 = δ0,
                δ_upper = δ_upper,
                ρ = ρ,
                M = M,
                inner_loop_radius = inner_loop_radius,
                inner_loop_max_iterations = inner_loop_max_iterations,
                threshold = threshold,
                max_iterations = k-1) ## return x_{k-1}

            optDatak = NonsequentialArmijoFixedDampedBFGSGD(Float64;
                x0 = x0,
                c = c,
                β = β,
                α = α,
                δ0 = δ0,
                δ_upper = δ_upper,
                ρ = ρ,
                M = M,
                inner_loop_radius = inner_loop_radius,
                inner_loop_max_iterations = inner_loop_max_iterations,
                threshold = threshold,
                max_iterations = k) ## return x_{k-1}) ## return x_k

            # generate k - 1 
            xkm1 = nonsequential_armijo_fixed_damped_bfgs(optDatakm1, progData)
            
            # Setting up for test - output of inner loop for iteration k
            x = copy(xkm1) 
            OptimizationMethods.grad!(progData, precomp, store, x)
            OptimizationMethods.inner_loop!(x, optDatakm1.iter_hist[k], 
                optDatakm1, progData, precomp, store, k;
                radius = optData.inner_loop_radius,
                max_iteration = optData.inner_loop_max_iterations)
            
            # Test that the non sequential armijo condition is failed
            @test !OptimizationMethods.non_sequential_armijo_condition(
                F(x), optDatakm1.reference_value, 
                norm(OptimizationMethods.grad(progData, xkm1)), 
                optDatakm1.ρ, optDatakm1.δk, optDatakm1.α)

            # Generate x_k and test the optDatak 
            xk = nonsequential_armijo_fixed_damped_bfgs(optDatak, progData)

            ## check gradient quantities
            @test optDatak.∇F_θk ≈ OptimizationMethods.grad(progData, xkm1)
            if k > 1
                @test optDatak.B_θk ≈ optDatakm1.B_θk
            end

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
        optDatakm1 = NonsequentialArmijoFixedDampedBFGSGD(Float64;
            x0 = x0,
            c = c,
            β = β,
            α = α,
            δ0 = δ0,
            δ_upper = δ_upper,
            ρ = ρ,
            M = M,
            inner_loop_radius = inner_loop_radius,
            inner_loop_max_iterations = inner_loop_max_iterations,
            threshold = threshold,
            max_iterations = iter) ## return x_{k-1} ## stop_iteration = iter

        optDatak = NonsequentialArmijoFixedDampedBFGSGD(Float64;
            x0 = x0,
            c = c,
            β = β,
            α = α,
            δ0 = δ0,
            δ_upper = δ_upper,
            ρ = ρ,
            M = M,
            inner_loop_radius = inner_loop_radius,
            inner_loop_max_iterations = inner_loop_max_iterations,
            threshold = threshold,
            max_iterations = iter + 1) ## return x_{k-1} ## stop_iteration = iter + 1

        # generate k - 1 and k
        xkm1 = nonsequential_armijo_fixed_damped_bfgs(optDatakm1, progData)
        xk = nonsequential_armijo_fixed_damped_bfgs(optDatak, progData)

        # Setting up for test - output of inner loop for iteration k
        x = copy(xkm1) 
        OptimizationMethods.grad!(progData, precomp, store, x)
        OptimizationMethods.inner_loop!(x, optDatakm1.iter_hist[iter + 1], 
            optDatakm1, progData, precomp, store, iter + 1;
            radius = optData.inner_loop_radius,
            max_iteration = optData.inner_loop_max_iterations)

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
        if ((optDatakm1.acceptance_cnt - 1) % M) + 1 == optDatakm1.reference_value_index
            optDatakm1.reference_value, optDatakm1.reference_value_index =
            findmax(optDatakm1.objective_hist)
        end
        
        # Check that optDatak matches optDatakm1
        @test flag
        @test xk == x
        @test optDatak.∇F_θk ≈ OptimizationMethods.grad(progData, xkm1)
        if first_acceptance != 1
            @test optDatak.B_θk ≈ optDatakm1.B_θk 
        else
            g0 = OptimizationMethods.grad(progData, x0)
            @test optDatak.B_θk ≈ optData.c * norm(g0) * Matrix{Float64}(I, 50, 50)
        end
        @test optDatak.grad_val_hist[iter + 2] == optDatakm1.norm_∇F_ψ
        @test optDatak.δk == optDatakm1.δk
        @test optDatak.τ_lower == optDatakm1.τ_lower
        @test optDatak.τ_upper == optDatakm1.τ_upper

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
        last_acceptance = stop_iteration
        while (1 < last_acceptance) && 
                (optData.iter_hist[last_acceptance] == 
                    optData.iter_hist[last_acceptance + 1])
            last_acceptance -= 1
        end
        last_acceptance += 1

        for k in (last_acceptance):(stop_iteration)

            # create optdata for k - 1 and k
            optDatakm1 = NonsequentialArmijoFixedDampedBFGSGD(Float64;
                x0 = x0,
                c = c,
                β = β,
                α = α,
                δ0 = δ0,
                δ_upper = δ_upper,
                ρ = ρ,
                M = M,
                inner_loop_radius = inner_loop_radius,
                inner_loop_max_iterations = inner_loop_max_iterations,
                threshold = threshold,
                max_iterations = k-1) ## return x_{k-1}

            optDatak = NonsequentialArmijoFixedDampedBFGSGD(Float64;
                x0 = x0,
                c = c,
                β = β,
                α = α,
                δ0 = δ0,
                δ_upper = δ_upper,
                ρ = ρ,
                M = M,
                inner_loop_radius = inner_loop_radius,
                inner_loop_max_iterations = inner_loop_max_iterations,
                threshold = threshold,
                max_iterations = k) ## return x_k

            # generate k - 1 
            xkm1 = nonsequential_armijo_fixed_damped_bfgs(optDatakm1, progData)  
            
            # Setting up for test - output of inner loop for iteration k
            x = copy(xkm1) 
            OptimizationMethods.grad!(progData, precomp, store, x)
            OptimizationMethods.inner_loop!(x, optDatakm1.iter_hist[k], 
                optDatakm1, progData, precomp, store, k;
                radius = optData.inner_loop_radius,
                max_iteration = optData.inner_loop_max_iterations)
            
            # Test that the non sequential armijo condition is failed
            @test !OptimizationMethods.non_sequential_armijo_condition(
                F(x), optDatakm1.reference_value, 
                norm(OptimizationMethods.grad(progData, xkm1)), 
                optDatakm1.ρ, optDatakm1.δk, optDatakm1.α)

            # Generate x_k and test the optDatak 
            xk = nonsequential_armijo_fixed_damped_bfgs(optDatak, progData)

            ## check gradient quantities
            @test optDatak.∇F_θk ≈ OptimizationMethods.grad(progData, xkm1)
            if k > last_acceptance
                @test optDatak.B_θk ≈ optDatakm1.B_θk
            end

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

        end # for loop

        # Test values are correctly updated for acceptance
        iter = last_acceptance - 2

        # create optdata for k - 1 and k
        optDatakm1 = NonsequentialArmijoFixedDampedBFGSGD(Float64;
            x0 = x0,
            c = c,
            β = β,
            α = α,
            δ0 = δ0,
            δ_upper = δ_upper,
            ρ = ρ,
            M = M,
            inner_loop_radius = inner_loop_radius,
            inner_loop_max_iterations = inner_loop_max_iterations,
            threshold = threshold,
            max_iterations = iter) ## stop_iteration = iter

        optDatak = NonsequentialArmijoFixedDampedBFGSGD(Float64;
            x0 = x0,
            c = c,
            β = β,
            α = α,
            δ0 = δ0,
            δ_upper = δ_upper,
            ρ = ρ,
            M = M,
            inner_loop_radius = inner_loop_radius,
            inner_loop_max_iterations = inner_loop_max_iterations,
            threshold = threshold,
            max_iterations = iter + 1) ## stop_iteration = iter + 1

        # generate k - 1 and k
        xkm1 = nonsequential_armijo_fixed_damped_bfgs(optDatakm1, progData)  
        xk = nonsequential_armijo_fixed_damped_bfgs(optDatak, progData)

        # Setting up for test - output of inner loop for iteration k
        x = copy(xkm1) 
        OptimizationMethods.grad!(progData, precomp, store, x)
        OptimizationMethods.inner_loop!(x, optDatakm1.iter_hist[iter + 1], 
            optDatakm1, progData, precomp, store, iter + 1;
            radius = optData.inner_loop_radius,
            max_iteration = optData.inner_loop_max_iterations)

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
        if ((optDatakm1.acceptance_cnt - 1) % M) + 1 == optDatakm1.reference_value_index
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
    end # end the test cases for the monotone method

end # end test for monotone 

@testset "Test nonsequential_armijo_fixed_damped_bfgs Nonmonotone" begin

    # Random arguments
    dim = 50
    x0 =               randn(dim)
    c =                 rand()
    β =                 rand()
    α =                 rand()
    δ0 =                1.0
    ρ =                 1e-5 * rand()
    M =                 rand(2:10)
    threshold =         rand()
    inner_loop_radius = 1.0
    inner_loop_max_iterations = 100 + rand(1:10)
    max_iterations = rand(1:100)
    δ_upper = δ0 + 1

    # Should exit on iteration 0 because max_iterations is 0
    let x0 = copy(x0), c = c, β = β, α = α, δ0 = δ0, ρ = ρ, M = M,
        threshold = threshold, inner_loop_radius = inner_loop_radius,
        inner_loop_max_iterations = inner_loop_max_iterations, 
        threshold = threshold,
        max_iterations = 0            
        
        # specify optimization method and problem
        optData = NonsequentialArmijoFixedDampedBFGSGD(Float64;
            x0 = x0,
            c = c,
            β = β,
            α = α,
            δ0 = δ0,
            δ_upper = δ_upper,
            ρ = ρ,
            M = M,
            inner_loop_radius = inner_loop_radius,
            inner_loop_max_iterations = inner_loop_max_iterations,
            threshold = threshold,
            max_iterations = max_iterations)

        progData = OptimizationMethods.LogisticRegression(Float64, nvar=length(x0))

        # Run method
        x = nonsequential_armijo_fixed_damped_bfgs(optData, progData)

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
    let x0 = copy(x0), c = c, β = β, α = α, δ0 = δ0, ρ = ρ, M = M,
        threshold = threshold, inner_loop_radius = inner_loop_radius,
        inner_loop_max_iterations = inner_loop_max_iterations, 
        threshold = 1e16,
        max_iterations = max_iterations  
        
        # specify optimization method and problem
        optData = NonsequentialArmijoFixedDampedBFGSGD(Float64;
            x0 = x0,
            c = c,
            β = β,
            α = α,
            δ0 = δ0,
            δ_upper = δ_upper,
            ρ = ρ,
            M = M,
            inner_loop_radius = inner_loop_radius,
            inner_loop_max_iterations = inner_loop_max_iterations,
            threshold = threshold,
            max_iterations = max_iterations)

        progData = OptimizationMethods.LogisticRegression(Float64, nvar=length(x0))

        # Run method
        x = nonsequential_armijo_fixed_damped_bfgs(optData, progData)

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
    
    # test the OptimizationMethods
    let x0 = copy(x0), c = c, β = β, α = α, δ0 = δ0, ρ = ρ, M = M,
        threshold = threshold, inner_loop_radius = inner_loop_radius,
        inner_loop_max_iterations = inner_loop_max_iterations, 
        threshold = 1e-10,
        max_iterations = max_iterations 

        #Specify Problem 
        progData = OptimizationMethods.LogisticRegression(Float64, nvar=length(x0))

        # specify optimization method and problem
        optData = NonsequentialArmijoFixedDampedBFGSGD(Float64;
            x0 = x0,
            c = c,
            β = β,
            α = α,
            δ0 = δ0,
            δ_upper = δ_upper,
            ρ = ρ,
            M = M,
            inner_loop_radius = inner_loop_radius,
            inner_loop_max_iterations = inner_loop_max_iterations,
            threshold = threshold,
            max_iterations = max_iterations)
            
        # Run method
        x = nonsequential_armijo_fixed_damped_bfgs(optData, progData)
        
        g = OptimizationMethods.grad(progData, x) 
        stop_iteration = optData.stop_iteration 

        @test optData.iter_hist[stop_iteration+1] == x
        @test optData.grad_val_hist[stop_iteration+1] ≈ norm(g)

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
            optDatakm1 = NonsequentialArmijoFixedDampedBFGSGD(Float64;
                x0 = x0,
                c = c,
                β = β,
                α = α,
                δ0 = δ0,
                δ_upper = δ_upper,
                ρ = ρ,
                M = M,
                inner_loop_radius = inner_loop_radius,
                inner_loop_max_iterations = inner_loop_max_iterations,
                threshold = threshold,
                max_iterations = k-1) ## return x_{k-1}

            optDatak = NonsequentialArmijoFixedDampedBFGSGD(Float64;
                x0 = x0,
                c = c,
                β = β,
                α = α,
                δ0 = δ0,
                δ_upper = δ_upper,
                ρ = ρ,
                M = M,
                inner_loop_radius = inner_loop_radius,
                inner_loop_max_iterations = inner_loop_max_iterations,
                threshold = threshold,
                max_iterations = k) ## return x_{k-1}) ## return x_k

            # generate k - 1 
            xkm1 = nonsequential_armijo_fixed_damped_bfgs(optDatakm1, progData)
            
            # Setting up for test - output of inner loop for iteration k
            x = copy(xkm1) 
            OptimizationMethods.grad!(progData, precomp, store, x)
            OptimizationMethods.inner_loop!(x, optDatakm1.iter_hist[k], 
                optDatakm1, progData, precomp, store, k;
                radius = optData.inner_loop_radius,
                max_iteration = optData.inner_loop_max_iterations)
            
            # Test that the non sequential armijo condition is failed
            @test !OptimizationMethods.non_sequential_armijo_condition(
                F(x), optDatakm1.reference_value, 
                norm(OptimizationMethods.grad(progData, xkm1)), 
                optDatakm1.ρ, optDatakm1.δk, optDatakm1.α)

            # Generate x_k and test the optDatak 
            xk = nonsequential_armijo_fixed_damped_bfgs(optDatak, progData)

            ## check gradient quantities
            @test optDatak.∇F_θk ≈ OptimizationMethods.grad(progData, xkm1)
            if k > 1
                @test optDatak.B_θk ≈ optDatakm1.B_θk
            end

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
        optDatakm1 = NonsequentialArmijoFixedDampedBFGSGD(Float64;
            x0 = x0,
            c = c,
            β = β,
            α = α,
            δ0 = δ0,
            δ_upper = δ_upper,
            ρ = ρ,
            M = M,
            inner_loop_radius = inner_loop_radius,
            inner_loop_max_iterations = inner_loop_max_iterations,
            threshold = threshold,
            max_iterations = iter) ## return x_{k-1} ## stop_iteration = iter

        optDatak = NonsequentialArmijoFixedDampedBFGSGD(Float64;
            x0 = x0,
            c = c,
            β = β,
            α = α,
            δ0 = δ0,
            δ_upper = δ_upper,
            ρ = ρ,
            M = M,
            inner_loop_radius = inner_loop_radius,
            inner_loop_max_iterations = inner_loop_max_iterations,
            threshold = threshold,
            max_iterations = iter + 1) ## return x_{k-1} ## stop_iteration = iter + 1

        # generate k - 1 and k
        xkm1 = nonsequential_armijo_fixed_damped_bfgs(optDatakm1, progData)
        xk = nonsequential_armijo_fixed_damped_bfgs(optDatak, progData)

        # Setting up for test - output of inner loop for iteration k
        x = copy(xkm1) 
        OptimizationMethods.grad!(progData, precomp, store, x)
        OptimizationMethods.inner_loop!(x, optDatakm1.iter_hist[iter + 1], 
            optDatakm1, progData, precomp, store, iter + 1;
            radius = optData.inner_loop_radius,
            max_iteration = optData.inner_loop_max_iterations)

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
        if ((optDatakm1.acceptance_cnt - 1) % M) + 1 == optDatakm1.reference_value_index
            optDatakm1.reference_value, optDatakm1.reference_value_index =
            findmax(optDatakm1.objective_hist)
        end
        
        # Check that optDatak matches optDatakm1
        @test flag
        @test xk == x
        @test optDatak.∇F_θk ≈ OptimizationMethods.grad(progData, xkm1)
        if first_acceptance != 1
            @test optDatak.B_θk ≈ optDatakm1.B_θk 
        else
            g0 = OptimizationMethods.grad(progData, x0)
            @test optDatak.B_θk ≈ optData.c * norm(g0) * Matrix{Float64}(I, 50, 50)
        end
        @test optDatak.grad_val_hist[iter + 2] == optDatakm1.norm_∇F_ψ
        @test optDatak.δk == optDatakm1.δk
        @test optDatak.τ_lower == optDatakm1.τ_lower
        @test optDatak.τ_upper == optDatakm1.τ_upper

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
        last_acceptance = stop_iteration
        while (1 < last_acceptance) && 
                (optData.iter_hist[last_acceptance] == 
                    optData.iter_hist[last_acceptance + 1])
            last_acceptance -= 1
        end
        last_acceptance += 1

        for k in (last_acceptance):(stop_iteration)

            # create optdata for k - 1 and k
            optDatakm1 = NonsequentialArmijoFixedDampedBFGSGD(Float64;
                x0 = x0,
                c = c,
                β = β,
                α = α,
                δ0 = δ0,
                δ_upper = δ_upper,
                ρ = ρ,
                M = M,
                inner_loop_radius = inner_loop_radius,
                inner_loop_max_iterations = inner_loop_max_iterations,
                threshold = threshold,
                max_iterations = k-1) ## return x_{k-1}

            optDatak = NonsequentialArmijoFixedDampedBFGSGD(Float64;
                x0 = x0,
                c = c,
                β = β,
                α = α,
                δ0 = δ0,
                δ_upper = δ_upper,
                ρ = ρ,
                M = M,
                inner_loop_radius = inner_loop_radius,
                inner_loop_max_iterations = inner_loop_max_iterations,
                threshold = threshold,
                max_iterations = k) ## return x_k

            # generate k - 1 
            xkm1 = nonsequential_armijo_fixed_damped_bfgs(optDatakm1, progData)  
            
            # Setting up for test - output of inner loop for iteration k
            x = copy(xkm1) 
            OptimizationMethods.grad!(progData, precomp, store, x)
            OptimizationMethods.inner_loop!(x, optDatakm1.iter_hist[k], 
                optDatakm1, progData, precomp, store, k;
                radius = optData.inner_loop_radius,
                max_iteration = optData.inner_loop_max_iterations)
            
            # Test that the non sequential armijo condition is failed
            @test !OptimizationMethods.non_sequential_armijo_condition(
                F(x), optDatakm1.reference_value, 
                norm(OptimizationMethods.grad(progData, xkm1)), 
                optDatakm1.ρ, optDatakm1.δk, optDatakm1.α)

            # Generate x_k and test the optDatak 
            xk = nonsequential_armijo_fixed_damped_bfgs(optDatak, progData)

            ## check gradient quantities
            @test optDatak.∇F_θk ≈ OptimizationMethods.grad(progData, xkm1)
            if k > last_acceptance
                @test optDatak.B_θk ≈ optDatakm1.B_θk
            end

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

        end # for loop

        # Test values are correctly updated for acceptance
        iter = last_acceptance - 2

        # create optdata for k - 1 and k
        optDatakm1 = NonsequentialArmijoFixedDampedBFGSGD(Float64;
            x0 = x0,
            c = c,
            β = β,
            α = α,
            δ0 = δ0,
            δ_upper = δ_upper,
            ρ = ρ,
            M = M,
            inner_loop_radius = inner_loop_radius,
            inner_loop_max_iterations = inner_loop_max_iterations,
            threshold = threshold,
            max_iterations = iter) ## stop_iteration = iter

        optDatak = NonsequentialArmijoFixedDampedBFGSGD(Float64;
            x0 = x0,
            c = c,
            β = β,
            α = α,
            δ0 = δ0,
            δ_upper = δ_upper,
            ρ = ρ,
            M = M,
            inner_loop_radius = inner_loop_radius,
            inner_loop_max_iterations = inner_loop_max_iterations,
            threshold = threshold,
            max_iterations = iter + 1) ## stop_iteration = iter + 1

        # generate k - 1 and k
        xkm1 = nonsequential_armijo_fixed_damped_bfgs(optDatakm1, progData)  
        xk = nonsequential_armijo_fixed_damped_bfgs(optDatak, progData)

        # Setting up for test - output of inner loop for iteration k
        x = copy(xkm1) 
        OptimizationMethods.grad!(progData, precomp, store, x)
        OptimizationMethods.inner_loop!(x, optDatakm1.iter_hist[iter + 1], 
            optDatakm1, progData, precomp, store, iter + 1;
            radius = optData.inner_loop_radius,
            max_iteration = optData.inner_loop_max_iterations)

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
        if ((optDatakm1.acceptance_cnt - 1) % M) + 1 == optDatakm1.reference_value_index
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
    end # end the test cases for the monotone method

end # end test for nonmonotone

end # End module