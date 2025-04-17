# Date: 2025/04/16
# Author: Christian Varner
# Purpose: Test the non-sequential armijo modified newton method with
# fixed step size

module TestNonsequentialArmijoFixedMNewtonGD

using Test, OptimizationMethods, LinearAlgebra, CircularArrays

@testset "Test NonsequentialArmijoFixedMNewtonGD Structure" begin

    ############################################################################
    # Check if the struct is defined
    ############################################################################

    @test isdefined(OptimizationMethods, :NonsequentialArmijoFixedMNewtonGD)

    ############################################################################
    # Check the field names
    ############################################################################

    # test the default field values
    default_fields = [:name, :threshold, :max_iterations, :iter_hist, 
        :grad_val_hist, :stop_iteration]
    
    let fields = default_fields
        for field in fields
            @test field in fieldnames(NonsequentialArmijoFixedMNewtonGD)
        end
    end

    # test the unique field values
    unique_fields = [:∇F_θk, :∇∇F_θk, :norm_∇F_ψ, :α, :δk, :δ_upper, :ρ, :β, :λ,
        :hessian_modification_max_iteration, :objective_hist, :reference_value,
        :reference_value_index, :acceptance_cnt, :τ_lower, :τ_upper,
        :inner_loop_radius, :inner_loop_max_iterations]

    let fields = unique_fields
        for field in fields
            @test field in fieldnames(NonsequentialArmijoFixedMNewtonGD)
        end
    end

    # test that there are no other fields
    @test length(default_fields) + length(unique_fields) == 
        length(fieldnames(NonsequentialArmijoFixedMNewtonGD))

    ############################################################################
    # Check for error checking
    ############################################################################

    # α == 0
    @test_throws AssertionError NonsequentialArmijoFixedMNewtonGD(Float64;
        x0 = randn(50),
        α = 0.0,
        δ0 = rand(),
        δ_upper = 2.0,
        ρ = rand(),
        β = rand(),
        λ = rand(),
        hessian_modification_max_iteration = rand(1:100),
        M = rand(1:100),
        inner_loop_radius = rand(),
        inner_loop_max_iterations = rand(1:100),
        threshold = rand(),
        max_iterations = rand(1:100))

    # α < 0
    @test_throws AssertionError NonsequentialArmijoFixedMNewtonGD(Float64;
        x0 = randn(50),
        α = -1.0,
        δ0 = rand(),
        δ_upper = 2.0,
        ρ = rand(),
        β = rand(),
        λ = rand(),
        hessian_modification_max_iteration = rand(1:100),
        M = rand(1:100),
        inner_loop_radius = rand(),
        inner_loop_max_iterations = rand(1:100),
        threshold = rand(),
        max_iterations = rand(1:100)) 

    # δ == 0
    @test_throws AssertionError NonsequentialArmijoFixedMNewtonGD(Float64;
        x0 = randn(50),
        α = rand(),
        δ0 = 0.0,
        δ_upper = 2.0,
        ρ = rand(),
        β = rand(),
        λ = rand(),
        hessian_modification_max_iteration = rand(1:100),
        M = rand(1:100),
        inner_loop_radius = rand(),
        inner_loop_max_iterations = rand(1:100),
        threshold = rand(),
        max_iterations = rand(1:100)) 

    # δ < 0
    @test_throws AssertionError NonsequentialArmijoFixedMNewtonGD(Float64;
        x0 = randn(50),
        α = rand(),
        δ0 = -1.0,
        δ_upper = 2.0,
        ρ = rand(),
        β = rand(),
        λ = rand(),
        hessian_modification_max_iteration = rand(1:100),
        M = rand(1:100),
        inner_loop_radius = rand(),
        inner_loop_max_iterations = rand(1:100),
        threshold = rand(),
        max_iterations = rand(1:100)) 

    # δ0 > δ_upper
    @test_throws AssertionError NonsequentialArmijoFixedMNewtonGD(Float64;
        x0 = randn(50),
        α = rand(),
        δ0 = rand(),
        δ_upper = 0.0,
        ρ = rand(),
        β = rand(),
        λ = rand(),
        hessian_modification_max_iteration = rand(1:100),
        M = rand(1:100),
        inner_loop_radius = rand(),
        inner_loop_max_iterations = rand(1:100),
        threshold = rand(),
        max_iterations = rand(1:100)) 

    # ρ == 0
    @test_throws AssertionError NonsequentialArmijoFixedMNewtonGD(Float64;
        x0 = randn(50),
        α = rand(),
        δ0 = rand(),
        δ_upper = 2.0,
        ρ = 0.0,
        β = rand(),
        λ = rand(),
        hessian_modification_max_iteration = rand(1:100),
        M = rand(1:100),
        inner_loop_radius = rand(),
        inner_loop_max_iterations = rand(1:100),
        threshold = rand(),
        max_iterations = rand(1:100)) 

    # ρ < 0
    @test_throws AssertionError NonsequentialArmijoFixedMNewtonGD(Float64;
        x0 = randn(50),
        α = rand(),
        δ0 = rand(),
        δ_upper = 2.0,
        ρ = -1.0,
        β = rand(),
        λ = rand(),
        hessian_modification_max_iteration = rand(1:100),
        M = rand(1:100),
        inner_loop_radius = rand(),
        inner_loop_max_iterations = rand(1:100),
        threshold = rand(),
        max_iterations = rand(1:100)) 

    # β == 0
    @test_throws AssertionError NonsequentialArmijoFixedMNewtonGD(Float64;
        x0 = randn(50),
        α = rand(),
        δ0 = rand(),
        δ_upper = 2.0,
        ρ = rand(),
        β = 0.0,
        λ = rand(),
        hessian_modification_max_iteration = rand(1:100),
        M = rand(1:100),
        inner_loop_radius = rand(),
        inner_loop_max_iterations = rand(1:100),
        threshold = rand(),
        max_iterations = rand(1:100)) 

    # β < 0
    @test_throws AssertionError NonsequentialArmijoFixedMNewtonGD(Float64;
        x0 = randn(50),
        α = rand(),
        δ0 = rand(),
        δ_upper = 2.0,
        ρ = rand(),
        β = -1.0,
        λ = rand(),
        hessian_modification_max_iteration = rand(1:100),
        M = rand(1:100),
        inner_loop_radius = rand(),
        inner_loop_max_iterations = rand(1:100),
        threshold = rand(),
        max_iterations = rand(1:100)) 

    # λ < 0
    @test_throws AssertionError NonsequentialArmijoFixedMNewtonGD(Float64;
        x0 = randn(50),
        α = rand(),
        δ0 = rand(),
        δ_upper = 2.0,
        ρ = rand(),
        β = rand(),
        λ = -1.0,
        hessian_modification_max_iteration = rand(1:100),
        M = rand(1:100),
        inner_loop_radius = rand(),
        inner_loop_max_iterations = rand(1:100),
        threshold = rand(),
        max_iterations = rand(1:100)) 

    # M == 0
    @test_throws AssertionError NonsequentialArmijoFixedMNewtonGD(Float64;
        x0 = randn(50),
        α = rand(),
        δ0 = rand(),
        δ_upper = 2.0,
        ρ = rand(),
        β = rand(),
        λ = rand(),
        hessian_modification_max_iteration = rand(1:100),
        M = 0,
        inner_loop_radius = rand(),
        inner_loop_max_iterations = rand(1:100),
        threshold = rand(),
        max_iterations = rand(1:100)) 

    # M < 0
    @test_throws AssertionError NonsequentialArmijoFixedMNewtonGD(Float64;
        x0 = randn(50),
        α = rand(),
        δ0 = rand(),
        δ_upper = 2.0,
        ρ = rand(),
        β = rand(),
        λ = rand(),
        hessian_modification_max_iteration = rand(1:100),
        M = -1,
        inner_loop_radius = rand(),
        inner_loop_max_iterations = rand(1:100),
        threshold = rand(),
        max_iterations = rand(1:100)) 

    # inner_loop_radius == 0
    @test_throws AssertionError NonsequentialArmijoFixedMNewtonGD(Float64;
        x0 = randn(50),
        α = rand(),
        δ0 = rand(),
        δ_upper = 2.0,
        ρ = rand(),
        β = rand(),
        λ = rand(),
        hessian_modification_max_iteration = rand(1:100),
        M = rand(1:100),
        inner_loop_radius = 0.0,
        inner_loop_max_iterations = rand(1:100),
        threshold = rand(),
        max_iterations = rand(1:100)) 

    # inner_loop_radius < 0
    @test_throws AssertionError NonsequentialArmijoFixedMNewtonGD(Float64;
        x0 = randn(50),
        α = rand(),
        δ0 = rand(),
        δ_upper = 2.0,
        ρ = rand(),
        β = rand(),
        λ = rand(),
        hessian_modification_max_iteration = rand(1:100),
        M = rand(1:100),
        inner_loop_radius = -1.0,
        inner_loop_max_iterations = rand(1:100),
        threshold = rand(),
        max_iterations = rand(1:100)) 

    # inner_loop_max_iterations == 0
    @test_throws AssertionError NonsequentialArmijoFixedMNewtonGD(Float64;
        x0 = randn(50),
        α = rand(),
        δ0 = rand(),
        δ_upper = 2.0,
        ρ = rand(),
        β = rand(),
        λ = rand(),
        hessian_modification_max_iteration = rand(1:100),
        M = rand(1:100),
        inner_loop_radius = rand(),
        inner_loop_max_iterations = 0,
        threshold = rand(),
        max_iterations = rand(1:100)) 

    # inner_loop_max_iterations < 0
    @test_throws AssertionError NonsequentialArmijoFixedMNewtonGD(Float64;
        x0 = randn(50),
        α = rand(),
        δ0 = rand(),
        δ_upper = 2.0,
        ρ = rand(),
        β = rand(),
        λ = rand(),
        hessian_modification_max_iteration = rand(1:100),
        M = rand(1:100),
        inner_loop_radius = rand(),
        inner_loop_max_iterations = -1,
        threshold = rand(),
        max_iterations = rand(1:100)) 

    # max_iterations < 0
    @test_throws AssertionError NonsequentialArmijoFixedMNewtonGD(Float64;
        x0 = randn(50),
        α = rand(),
        δ0 = rand(),
        δ_upper = 2.0,
        ρ = rand(),
        β = rand(),
        λ = rand(),
        hessian_modification_max_iteration = rand(1:100),
        M = rand(1:100),
        inner_loop_radius = rand(),
        inner_loop_max_iterations = rand(1:100),
        threshold = rand(),
        max_iterations = -1) 

    ############################################################################
    # Check the field types
    ############################################################################

    # types for testing and correct values
    real_types = [Float16, Float32, Float64]
    field_types(type::T) where {T} =
        [
            (:name, String),
            (:∇F_θk, Vector{type}),
            (:∇∇F_θk, Matrix{type}),
            (:norm_∇F_ψ, type),
            (:α, type),
            (:δk, type),
            (:δ_upper, type),
            (:ρ, type),
            (:β, type),
            (:λ, type),
            (:hessian_modification_max_iteration, Int64),
            (:objective_hist, CircularVector{type, Vector{type}}),
            (:reference_value, type),
            (:reference_value_index, Int64),
            (:acceptance_cnt, Int64),
            (:τ_lower, type),
            (:τ_upper, type),
            (:inner_loop_radius, type),
            (:inner_loop_max_iterations, Int64),
            (:threshold, type),
            (:max_iterations, Int64),
            (:iter_hist, Vector{Vector{type}}),
            (:grad_val_hist, Vector{type}),
            (:stop_iteration, Int64),
        ]

    # test cases for the types in the struct
    let real_types = real_types, field_types = field_types
        
        for type in real_types
            
            # generate some random data for the struct constructor
            x0 = randn(type, 50)            
            α = rand(type)
            δ0 = rand(type)
            δ_upper = 1 + rand(type)
            ρ = rand(type)
            β = rand(type)
            λ = rand(type)
            hessian_modification_max_iteration = rand(1:100)
            M = rand(1:100)
            inner_loop_radius = rand(type)
            inner_loop_max_iterations = rand(1:100)
            threshold = rand(type)
            max_iterations = rand(1:100)

            # generate structure
            optData = NonsequentialArmijoFixedMNewtonGD(type;
                x0 = x0,
                α = α,
                δ0 = δ0,
                δ_upper = δ_upper,
                ρ = ρ,
                β = β,
                λ = λ,
                hessian_modification_max_iteration = hessian_modification_max_iteration,
                M = M,
                inner_loop_radius = inner_loop_radius,
                inner_loop_max_iterations = inner_loop_max_iterations,
                threshold = threshold,
                max_iterations = max_iterations)

            # test the field types
            for (field_symbol, field_type) in field_types(type)
                @test typeof(getfield(optData, field_symbol)) == field_type
            end

        end

    end # end test case for the types
    

    ############################################################################
    # Check that initial fields are set correctly
    ############################################################################

    # test cases for the initial field types
    let real_types = real_types
        
        for type in real_types
            
            # generate some random data for the struct constructor
            x0 = randn(type, 50)            
            α = rand(type)
            δ0 = rand(type)
            δ_upper = 1 + rand(type)
            ρ = rand(type)
            β = rand(type)
            λ = rand(type)
            hessian_modification_max_iteration = rand(1:100)
            M = rand(1:100)
            inner_loop_radius = rand(type)
            inner_loop_max_iterations = rand(1:100)
            threshold = rand(type)
            max_iterations = rand(1:100)

            # generate structure
            optData = NonsequentialArmijoFixedMNewtonGD(type;
                x0 = x0,
                α = α,
                δ0 = δ0,
                δ_upper = δ_upper,
                ρ = ρ,
                β = β,
                λ = λ,
                hessian_modification_max_iteration = hessian_modification_max_iteration,
                M = M,
                inner_loop_radius = inner_loop_radius,
                inner_loop_max_iterations = inner_loop_max_iterations,
                threshold = threshold,
                max_iterations = max_iterations)

            # test the default field values
            @test optData.threshold == threshold
            @test optData.max_iterations == max_iterations
            @test optData.iter_hist[1] == x0
            @test length(optData.iter_hist) == max_iterations + 1
            @test length(optData.grad_val_hist) == max_iterations + 1
            @test optData.stop_iteration == -1

            # test the unique field values
            d = length(x0)
            @test length(optData.∇F_θk) == d
            @test size(optData.∇∇F_θk) == (d, d)
            @test optData.α == α
            @test optData.δk == δ0
            @test optData.δ_upper == δ_upper
            @test optData.ρ == ρ
            @test optData.β == β
            @test optData.λ == λ
            @test hessian_modification_max_iteration == hessian_modification_max_iteration
            @test length(optData.objective_hist) == M
            @test optData.reference_value == type(-1)
            @test optData.reference_value_index == -1
            @test optData.acceptance_cnt == 0
            @test optData.inner_loop_radius == inner_loop_radius
            @test optData.inner_loop_max_iterations == inner_loop_max_iterations

        end

    end # end test case for the types


end # end the structure tests

@testset "Utility -- Update Algorithm Parameters Nonsequential Modified Newton" begin

    include("../utility/update_algorithm_parameters_test_cases.jl")

    # generate random arguments
    x0 = randn(50)            
    α = rand()
    δ0 = rand()
    δ_upper = 1 + rand()
    ρ = rand()
    β = rand()
    λ = rand()
    hessian_modification_max_iteration = rand(1:100)
    M = rand(1:100)
    inner_loop_radius = rand()
    inner_loop_max_iterations = rand(1:100)
    threshold = rand()
    max_iterations = rand(1:100)

    # build structure
    optData = NonsequentialArmijoFixedMNewtonGD(Float64;
        x0 = x0,
        α = α,
        δ0 = δ0,
        δ_upper = δ_upper,
        ρ = ρ,
        β = β,
        λ = λ,
        hessian_modification_max_iteration = hessian_modification_max_iteration,
        M = M,
        inner_loop_radius = inner_loop_radius,
        inner_loop_max_iterations = inner_loop_max_iterations,
        threshold = threshold,
        max_iterations = max_iterations)

    # conduct test cases
    update_algorithm_parameters_test_cases(optData, 50, max_iterations)

end # end the update parameter tests

@testset "Utility -- Inner Loop Nonsequential Armijo Modified Newton" begin

    # generate random arguments
    x0 = randn(50)            
    α = rand()
    δ0 = rand()
    δ_upper = 1 + rand()
    ρ = rand()
    β = rand()
    λ = rand()
    hessian_modification_max_iteration = 10
    M = rand(1:100)
    inner_loop_radius = rand()
    inner_loop_max_iterations = rand(1:100)
    threshold = rand()
    max_iterations = rand(1:100)

    # build structure
    optData = NonsequentialArmijoFixedMNewtonGD(Float64;
        x0 = x0,
        α = α,
        δ0 = δ0,
        δ_upper = δ_upper,
        ρ = ρ,
        β = β,
        λ = λ,
        hessian_modification_max_iteration = hessian_modification_max_iteration,
        M = M,
        inner_loop_radius = inner_loop_radius,
        inner_loop_max_iterations = inner_loop_max_iterations,
        threshold = threshold,
        max_iterations = max_iterations)

    progData = OptimizationMethods.LeastSquares(Float64, nvar=50)
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
        end # end first event trigger test

        # Test second event trigger: τ_lower 
        let ψjk=copy(x0), θk=copy(x0), optData=optData, progData=progData,
            store=store, k=k

            optData.grad_val_hist[k] = 0.5 
            optData.τ_lower = 1.0
            optData.τ_upper = 2.0
            
            OptimizationMethods.inner_loop!(ψjk, θk, optData, progData, precomp, 
                store, k, max_iteration=100)

            @test ψjk == x0
        end # end second event triggering event

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

        # Test first iteration -- successful hessian modification
        let ψjk=copy(x0), θk=copy(x0), optData=optData, progData=progData,
            store=store, k=k

            j=1

            optData.hessian_modification_max_iteration = hessian_modification_max_iteration
            OptimizationMethods.grad!(progData, precomp, store, θk)
            OptimizationMethods.hess!(progData, precomp, store, θk)
            optData.grad_val_hist[k] = norm(store.grad)
            optData.norm_∇F_ψ = norm(store.grad)
            optData.τ_lower = 0.5 * norm(store.grad) 
            optData.τ_upper = 1.5 * norm(store.grad)
            optData.λ = 1.0

            optData.δk = 1.5
            α = optData.α

            # modify hessian and return the result -- should terminate with success
            H = OptimizationMethods.hess(progData, θk)
            g = OptimizationMethods.grad(progData, θk)
            res = OptimizationMethods.add_identity_until_pd!(H;
                λ = optData.λ,
                β = optData.β, 
                max_iterations = optData.hessian_modification_max_iteration)
        
            # take a gradient step if this was not successful
            step = zeros(50)
            if !res[2]
                step .= (optData.δk * optData.α) .* g
            else
                OptimizationMethods.lower_triangle_solve!(g, H')
                OptimizationMethods.upper_triangle_solve!(g, H)
                step .= (optData.δk * optData.α) .* g
            end

            # reset
            j = OptimizationMethods.inner_loop!(ψjk, θk, optData, progData, precomp,
                store, k; radius = optData.inner_loop_radius, max_iteration = 1
            )
            @test j == 1

            # test cases
            @test ψjk ≈ θk - step 
            @test optData.λ == res[1] / 2
            @test optData.α == α
            @test store.grad ≈ OptimizationMethods.grad(progData, ψjk)
            @test store.hess ≈ OptimizationMethods.hess(progData, ψjk)
            @test optData.norm_∇F_ψ == norm(store.grad)
        end

        # Test first iteration
        let ψjk=copy(x0), θk=copy(x0), optData=optData, progData=progData,
            store=store, k=k

            j=1

            optData.hessian_modification_max_iteration = 0
            OptimizationMethods.grad!(progData, precomp, store, θk)
            OptimizationMethods.hess!(progData, precomp, store, θk)
            optData.grad_val_hist[k] = norm(store.grad)
            optData.norm_∇F_ψ = norm(store.grad)
            optData.τ_lower = 0.5 * norm(store.grad) 
            optData.τ_upper = 1.5 * norm(store.grad)
            optData.λ = 1.0

            optData.δk = 1.5
            α = optData.α

            # modify hessian and return the result -- should terminate with success
            H = OptimizationMethods.hess(progData, θk)
            g = OptimizationMethods.grad(progData, θk)
            res = OptimizationMethods.add_identity_until_pd!(H;
                λ = optData.λ,
                β = optData.β, 
                max_iterations = optData.hessian_modification_max_iteration)
        
            # take a gradient step if this was not successful
            step = zeros(50)
            if !res[2]
                step .= (optData.δk * optData.α) .* g
            else
                OptimizationMethods.lower_triangle_solve!(g, H')
                OptimizationMethods.upper_triangle_solve!(g, H)
                step .= (optData.δk * optData.α) .* g
            end

            # reset
            j = OptimizationMethods.inner_loop!(ψjk, θk, optData, progData, precomp,
                store, k; radius = optData.inner_loop_radius, max_iteration = 1
            )
            @test j == 1

            # test cases
            @test ψjk ≈ θk - step 
            @test optData.λ == 1.0
            @test optData.α == α
            @test store.grad ≈ OptimizationMethods.grad(progData, ψjk)
            @test store.hess ≈ OptimizationMethods.hess(progData, ψjk)
            @test optData.norm_∇F_ψ == norm(store.grad)
        end

        # Test random iteration

    end # end of for loop 
end # end the test of the inner loop

@testset "Method -- Nonsequential Armijo with Modified Newton (Monotone)" begin
end # end the test for the method 

@testset "Method -- Nonsequential Armijo with Modified Newton (Non-monotone)" begin
end # end the test for the method 

end # end the tests for NonsequentialArmijoFixedMNewtonGD