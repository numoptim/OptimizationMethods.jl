# Date: 02/13/2025
# Purpose: Test implementation of gradient descent using backtracking

module TestBacktrackingGD

using Test, OptimizationMethods, LinearAlgebra, Random

@testset "Method: Gradient Descent with Backtracking" begin

    # testing context for reproducibility
    Random.seed!(1010)

    ############################################################################
    # Test struct properties
    ############################################################################

    ## test definition
    @test @isdefined BacktrackingGD


    ##  test supertype
    @test supertype(BacktrackingGD) == 
        OptimizationMethods.AbstractOptimizerData


    ###########################################################################
    # Test struct constructor
    ###########################################################################

    # test field names
    names = [:name, :α, :δ, :ρ, :line_search_max_iteration,
    :threshold, :max_iterations, :iter_hist, :grad_val_hist,
    :stop_iteration]

    for name in names
        @test name in fieldnames(BacktrackingGD)
    end

    # test constructor 
    real_types = [Float16, Float32, Float64]
    number_random_parameters_trials = 5
    dimension = 50

    let number_random_parameters_trials = number_random_parameters_trials,
        dimension = dimension,  # Example dimension
        real_types = real_types
    
        for type in real_types
            for trial in 1:number_random_parameters_trials
                
                ## Expected field types
                field_types = [String, type, type, type, Int64, type, 
                               Int64, Vector{Vector{type}}, Vector{type}, Int64]
    
                ## Generate random values for fields
                x0 = randn(type, dimension)
                α = rand(type)
                δ = rand(type)
                ρ = rand(type)
                line_search_max_iteration = rand(1:100)
                threshold = rand(type)
                max_iterations = rand(1:100)
    
                ## Initialize BacktrackingGD struct
                optData = BacktrackingGD(
                    type,
                    x0 = x0,
                    α = α,
                    δ = δ,
                    ρ = ρ,
                    line_search_max_iteration = line_search_max_iteration,
                    threshold = threshold,
                    max_iterations = max_iterations
                )
    
                ## Test that each field has the correct type
                for (field_name, field_type) in zip(fieldnames(BacktrackingGD), field_types)
                    @test typeof(getfield(optData, field_name)) == field_type
                end
    
                ## Test that iter_hist has the correct length
                @test length(optData.iter_hist) == max_iterations + 1
    
                ## Test that grad_val_hist has the correct length
                @test length(optData.grad_val_hist) == max_iterations + 1
    
                ## Test that each field has the correct value
                @test optData.name == "Gradient Descent with Backtracking"
                @test optData.α == α
                @test optData.δ == δ
                @test optData.ρ == ρ
                @test optData.line_search_max_iteration == line_search_max_iteration
                @test optData.threshold == threshold  
                @test optData.max_iterations == max_iterations
            end
        end
    end

    ############################################################################
    # Test Optimizer: Base Case
    ############################################################################

    progData = OptimizationMethods.LeastSquares(Float64)
    x0 = progData.meta.x0

    let progData = progData, x0 = x0

        # parameters for the struct
        α::Float64 = rand()
        δ::Float64 = rand()/2
        ρ::Float64 = 1e-4
        line_search_max_iteration = 100
        threshold = 1e-10
        max_iterations = 1

        # construct the struct
        optData = BacktrackingGD(
            Float64,
            x0 = x0,
            α = α,
            δ = δ,
            ρ = ρ,
            line_search_max_iteration = line_search_max_iteration,
            threshold = threshold,
            max_iterations = max_iterations
        )

        # output after one step
        x1 = backtracking_gd(optData, progData)

        # test that first iterate was saved correctly
        @test optData.iter_hist[1] == x0

        # Compute x1 using backtracking
        x = optData.iter_hist[1]
        x_copy = copy(x)
        F(θ) = OptimizationMethods.obj(progData, θ)
        g0 = OptimizationMethods.grad(progData, x)
        F_x0 = F(x)

        # do one iteration of backtracking -- should pass
        success = OptimizationMethods.backtracking!(x, x_copy, F, g0,
            norm(g0) ^ 2, F_x0, optData.α, optData.δ, optData.ρ;
            max_iteration = optData.line_search_max_iteration)
        
        # test that backtracking was used
        @test success
        @test x1 ≈ x
       
        ## test that the iteration history is correct
        @test optData.iter_hist[2] == x1
        
        ## test that the gradient value history is correct
        @test optData.grad_val_hist[1] ≈ norm(g0) atol=1e-9
        
        g1 = norm(OptimizationMethods.grad(progData, x))
        @test optData.grad_val_hist[2] ≈ norm(g1) atol=1e-9

        ## test that the stop iteration is correct
        @test optData.stop_iteration == 1
    end

    ############################################################################
    # Test Optimizer: Inductive Step
    ############################################################################

    progData = OptimizationMethods.LeastSquares(Float64)
    x0 = progData.meta.x0
    
    let progData = progData, x0 = x0

        # parameters for the struct
        α::Float64 = abs(randn(Float64))
        δ::Float64 = abs(randn(Float64))
        ρ::Float64 = 1e-4
        line_search_max_iteration = 100
        threshold = 1e-10
        max_iterations = rand(2:20)

        # construct the struct
        optData = BacktrackingGD(
            Float64,
            x0 = x0,
            α = α,
            δ = δ,
            ρ = ρ,
            line_search_max_iteration = line_search_max_iteration,
            threshold = threshold,
            max_iterations = max_iterations
        )

        # Run the gradient descent with backtracking
        xk = backtracking_gd(optData, progData)

        ## test that the stop iteration is correct
        @test 0 < optData.stop_iteration <= max_iterations

        ## test that xk was updated correctly (e.g., by using backtracking! on xkm1)
        k = optData.stop_iteration
        xkm1 = optData.iter_hist[k]
        grad_xkm1 = OptimizationMethods.grad(progData, xkm1)

        # Compute the right-hand side of the Armijo condition
        F(θ) = OptimizationMethods.obj(progData, θ) 
        xkm1_copy = copy(xkm1)
        success = OptimizationMethods.backtracking!(xkm1, xkm1_copy, F,
            grad_xkm1, norm(grad_xkm1) ^ 2, F(xkm1), optData.α, optData.δ, optData.ρ;
            max_iteration = optData.line_search_max_iteration)

        @test success
        @test xkm1 ≈ xk

        # test iterate history
        @test optData.iter_hist[k + 1] == xk

        # test gradient value history
        grad_xk = OptimizationMethods.grad(progData, xk)
        @test optData.grad_val_hist[optData.stop_iteration + 1] ≈ norm(grad_xk) 
        atol=1e-9

        grad_xkm1 = OptimizationMethods.grad(progData, xkm1_copy)
        @test optData.grad_val_hist[optData.stop_iteration] ≈ norm(grad_xkm1) 
        rtol=1e-9
    end

    ############################################################################
    # Test Optimizer: Line search fails
    ############################################################################

    progData = OptimizationMethods.LeastSquares(Float64)
    x0 = progData.meta.x0
    
    let progData = progData, x0 = x0

        # parameters for the struct
        α::Float64 = abs(randn(Float64))
        δ::Float64 = abs(randn(Float64))
        ρ::Float64 = 1e-4
        line_search_max_iteration = 0
        threshold = 1e-10
        max_iterations = rand(2:20)

        # construct the struct
        optData = BacktrackingGD(
            Float64,
            x0 = x0,
            α = α,
            δ = δ,
            ρ = ρ,
            line_search_max_iteration = line_search_max_iteration,
            threshold = threshold,
            max_iterations = max_iterations
        )

        # Run the gradient descent with backtracking
        xk = backtracking_gd(optData, progData)

        ## test that the stop iteration is correct
        @test optData.stop_iteration == 0
        @test xk == x0
        @test optData.iter_hist[1] == x0

        ## test that xk was updated correctly (e.g., by using backtracking! on xkm1)
        k = optData.stop_iteration
        xkm1 = optData.iter_hist[1]
        grad_xkm1 = OptimizationMethods.grad(progData, xkm1)

        # Compute the right-hand side of the Armijo condition
        F(θ) = OptimizationMethods.obj(progData, θ) 
        xkm1_copy = copy(xkm1)
        success = OptimizationMethods.backtracking!(xkm1, xkm1_copy, F,
            grad_xkm1, norm(grad_xkm1) ^ 2, F(xkm1), optData.α, optData.δ, optData.ρ;
            max_iteration = optData.line_search_max_iteration)

        @test !success

        # test gradient value history
        grad_xk = OptimizationMethods.grad(progData, xkm1_copy)
        @test optData.grad_val_hist[optData.stop_iteration + 1] ≈ norm(grad_xk) 
        atol=1e-9
    end

end


end