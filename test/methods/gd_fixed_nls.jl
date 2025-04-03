# Date: 2025/21/03
# Author: Christian Varner
# Purpose: Test the non-monotone line search method
# with fixed step size and negative gradient directions

module TestFixedStepNLSMaxValGD

using Test, OptimizationMethods, LinearAlgebra, Random

@testset "Test FixedStepNLSMaxValGD{T} -- Structure" begin

    ############################################################################
    # Test structure definition
    ############################################################################

    # test that the structure is defined
    @test isdefined(OptimizationMethods, :FixedStepNLSMaxValGD)

    # test optimizer agnostic fields are present
    nonunique_fields = [:name, :threshold, :max_iterations, :iter_hist, 
        :grad_val_hist, :stop_iteration]

    # test optimizer specific fields are present
    unique_fields = [:α, :δ, :ρ, :window_size, :line_search_max_iteration, 
        :objective_hist, :max_value, :max_index]

    # test that no other fields are present
    @test length(unique_fields) + length(nonunique_fields) ==
        length(fieldnames(FixedStepNLSMaxValGD))

    ############################################################################
    # Test the constructor
    ############################################################################

    # test the field types
    field_types(type::T) where {T} =
        [(:name, String),
        (:α, type),
        (:δ, type),
        (:ρ, type),
        (:window_size, Int64),
        (:line_search_max_iteration, Int64),
        (:objective_hist, Vector{type}),
        (:max_value, type),
        (:threshold, type),
        (:max_iterations, Int64),
        (:iter_hist, Vector{Vector{type}}),
        (:grad_val_hist, Vector{type}),
        (:stop_iteration, Int64)]
    real_types = [Float16, Float32, Float64]

    let field_types = field_types, real_types = real_types
        
        for type in real_types

            # sample random field values
            x0 = randn(type, 50)
            α = abs(randn(type))
            δ = abs(randn(type))
            ρ = abs(randn(type))
            window_size = rand(1:100)
            line_search_max_iteration = rand(1:100)
            threshold = abs(randn(type))
            max_iterations = rand(1:100) 

            # test the field types returned by outer constructor
            optData = FixedStepNLSMaxValGD(type; x0 = x0, α = α, δ = δ, ρ = ρ,
                window_size = window_size, 
                line_search_max_iteration = line_search_max_iteration,
                threshold = threshold,
                max_iterations = max_iterations)
            
            for (field, fieldtype) in field_types(type)
                @test fieldtype == typeof(getfield(optData, field))
            end
        end
    end

    # test that the outer constructor sets the values of field correctly
    let field_types = field_types, 
        real_types = real_types
        
        # for each real type
        for type in real_types

            # sample random field values
            x0 = randn(type, 50)
            α = abs(randn(type))
            δ = abs(randn(type))
            ρ = abs(randn(type))
            window_size = rand(1:100)
            line_search_max_iteration = rand(1:100)
            threshold = abs(randn(type))
            max_iterations = rand(1:100)             

            # test the field values returned
            optData = FixedStepNLSMaxValGD(type; x0 = x0, α = α, δ = δ, ρ = ρ,
                window_size = window_size, 
                line_search_max_iteration = line_search_max_iteration,
                threshold = threshold,
                max_iterations = max_iterations)

            # test field values correct assigned
            @test optData.α == α
            @test optData.δ == δ
            @test optData.ρ == ρ
            @test optData.window_size == window_size
            @test optData.line_search_max_iteration == line_search_max_iteration
            @test optData.threshold == threshold
            @test optData.max_iterations == max_iterations

            # test values in iterate history
            @test optData.iter_hist[1] == x0
            @test length(optData.iter_hist) == max_iterations + 1

            # test values in gradient history
            @test length(optData.grad_val_hist) == max_iterations + 1
            @test optData.stop_iteration == -1

            # test values in objective cache
            @test length(optData.objective_hist) == window_size
            @test optData.max_value == 0
            @test optData.max_index == -1
        end
    
    end

    # test error are thrown
    let field_types = field_types, 
        real_types = real_types

        # for each real type
        for type in real_types

            # Test 1: sample random field value
            x0 = randn(type, 50)
            α = abs(randn(type))
            δ = abs(randn(type))
            ρ = abs(randn(type))
            line_search_max_iteration = rand(1:100)
            threshold = abs(randn(type))
            max_iterations = rand(1:100) 

            # Test 1: incorrect window_size ( == 0)
            window_size = 0

            # Test 1: test error occurs
            @test_throws AssertionError FixedStepNLSMaxValGD(
                type; x0 = x0, α = α, δ = δ, ρ = ρ,
                window_size = window_size, 
                line_search_max_iteration = line_search_max_iteration,
                threshold = threshold,
                max_iterations = max_iterations) 

            # Test 2: sample random field values
            x0 = randn(type, 50)
            α = abs(randn(type))
            δ = abs(randn(type))
            ρ = abs(randn(type))
            line_search_max_iteration = rand(1:100)
            threshold = abs(randn(type))
            max_iterations = rand(1:100) 

            # Test 2: incorrect window_size (< 0)
            window_size = -1

            # Test 2: test error occurs
            @test_throws AssertionError FixedStepNLSMaxValGD(
                type; x0 = x0, α = α, δ = δ, ρ = ρ,
                window_size = window_size, 
                line_search_max_iteration = line_search_max_iteration,
                threshold = threshold,
                max_iterations = max_iterations) 

        end

    end

end # end test set for structure

@testset "Test FixedStepNLSMaxValGD{T} -- Method" begin

    # initialize a random linear regression problem for testing
    progData = OptimizationMethods.LeastSquares(Float64)

    # sample random field values to for the optimization method
    x0 = randn(50)
    α = abs(randn())
    δ = abs(randn())
    ρ = abs(randn())
    window_size = rand(5:10)
    line_search_max_iteration = 100
    threshold = 1e-10

    max_iterations = 1

    # Base case: test the first iteration of the method
    let progData = progData, x0 = x0, α = α, δ = δ, ρ = ρ, 
        window_size = window_size, 
        line_search_max_iteration = line_search_max_iteration,
        threshold = threshold, max_iterations = max_iterations

        # initialize optimization data
        optData = FixedStepNLSMaxValGD(Float64; x0 = x0, α = α, δ = δ,
            ρ = ρ, window_size = window_size, 
            line_search_max_iteration = line_search_max_iteration,
            threshold = threshold,
            max_iterations = max_iterations)

        # objective function for line search
        F(θ) = OptimizationMethods.obj(progData, θ)

        # run one iteration of the method
        x1 = fixed_step_nls_maxval_gd(optData, progData)

        # test that x1 is correct
        x0_copy = copy(x0)
        g0 = OptimizationMethods.grad(progData, x0)
        success = OptimizationMethods.backtracking!(x0_copy, x0, F, g0, norm(g0) ^ 2,
            F(x0), α, δ, ρ; max_iteration = line_search_max_iteration)
        @test x1 ≈ x0_copy

        # test that the values stored optData.iter_hist and grad_val_hist
        @test optData.iter_hist[1] == x0
        @test optData.iter_hist[2] == x1
        @test optData.grad_val_hist[1] ≈ norm(g0)
        @test optData.grad_val_hist[2] ≈ norm(OptimizationMethods.grad(progData, x1))

        # test the values in optData.objective_hist
        @test optData.objective_hist[window_size - 1] == F(optData.iter_hist[1])
        @test optData.objective_hist[window_size] == F(x1)

        # test the values of optData.max_value and optData.max_index
        @test (optData.max_value == F(optData.iter_hist[1]) || !success)
        @test (optData.max_index == window_size - 1 || !success)
    end

    # "Inductive Step": test a random iteration of the method
    max_iterations = rand(20:100)
    let progData = progData, x0 = x0, α = α, δ = δ, ρ = ρ, 
        window_size = window_size, 
        line_search_max_iteration = line_search_max_iteration,
        threshold = threshold, max_iterations = max_iterations

        # initialize optimization data
        optData = FixedStepNLSMaxValGD(Float64; x0 = x0, α = α, δ = δ,
            ρ = ρ, window_size = window_size, 
            line_search_max_iteration = line_search_max_iteration,
            threshold = threshold,
            max_iterations = max_iterations)

        # objective function for the non-monotone line search function
        F(θ) = OptimizationMethods.obj(progData, θ)

        # run max_iterations (k) iteration of the method
        xk = fixed_step_nls_maxval_gd(optData, progData)

        # test the values of optData.objective_hist 
        for i in 1:window_size
            @test optData.objective_hist[i] ==
                F(optData.iter_hist[max_iterations + 1 - window_size + i])
        end

        # test the values of optData.max_value and optData.max_index
        max_value, max_index = findmax(optData.objective_hist)
        @test optData.max_value == max_value
        @test optData.max_index == max_index

        # test that xk is correct
        xkm1 = copy(optData.iter_hist[max_iterations])
        gkm1 = OptimizationMethods.grad(progData, xkm1)
        rkm1 = max(F(xkm1), optData.max_value)
        OptimizationMethods.backtracking!(xkm1, optData.iter_hist[max_iterations],
            F, gkm1, norm(gkm1) ^ 2, rkm1, α, δ, ρ;
            max_iteration = line_search_max_iteration)
        @test xkm1 ≈ xk

        # test that the values stored optData.iter_hist and grad_val_hist
        @test optData.iter_hist[max_iterations + 1] == xk
        @test optData.grad_val_hist[max_iterations + 1] ≈
            norm(OptimizationMethods.grad(progData, xk))
    end

end # end test for for method

end # End module