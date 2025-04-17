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

    let real_types = [Float16, Float32, Float64], 
        number_random_parameters_trials = number_random_parameters_trials,
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
        α::Float64 = abs(randn(Float64))
        δ::Float64 = abs(randn(Float64))
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
        

        ## TODO - test that either x1 fails the backtracking 
        ## condition or it succeeds
        
        #x1 = OptimizatoinMethods.backtracking!(x0, optData.iter_hist[0], F,  
       # store.grad )
        # Compute x1 using backtracking
        x1 = backtracking_gd(optData, progData)

        # Get function and gradient values

        F_x0 = OptimizationMethods.obj(progData, x0)
        F_x1 = OptimizationMethods.obj(progData, x1)
        grad_x0 = OptimizationMethods.grad(progData, x0)
        grad_x1 = OptimizationMethods.grad(progData, x1)

        # Compute right-hand side of backtracking condition
        rhs = F_x0 - optData.ρ * optData.δ * optData.α * norm(grad_x0)^2


        # Test that either x1 satisfies the backtracking condition or it is rejected

        
        @test (F_x1 <= rhs) || x1 == x0

       
        ## TODO - test that the iteration history is correct
    

        @test optData.iter_hist[1] == x0
        @test optData.iter_hist[2] == x1
       

            
        ## TODO - test that the gradient value history is correct

        @test optData.grad_val_hist[1] ≈ norm(grad_x0)
        atol=1e-9
        @test optData.grad_val_hist[2] ≈ norm(grad_x1)
        atol=1e-9


        ## TODO - test that the stop iteration is correct

        @test optData.stop_iteration == 1
       
    end

    ############################################################################
    # Test Optimizer: Inductive Step
    ############################################################################

    progData = OptimizationMethods.LeastSquares(Float64)
    x0 = progData.meta.x0
    k = 75 
    
    let progData = progData, x0 = x0, k = 75

        # parameters for the struct
        α::Float64 = abs(randn(Float64))
        δ::Float64 = abs(randn(Float64))
        ρ::Float64 = 1e-4
        line_search_max_iteration = 100
        threshold = 1e-6
        max_iterations = 100  # Ensure it's large enough to include k = 75


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
        backtracking_gd(optData, progData)


        ## TODO - test that the stop iteration is correct

        @test 0 < optData.stop_iteration <= max_iterations


        ## TODO - test that xk was updated correctly (e.g., by using backtracking! on xkm1)
        
        xkm1 = optData.iter_hist[k]
        xk = optData.iter_hist[k+1]
        grad_xkm1 = OptimizationMethods.grad(progData, xkm1)

        # Compute the right-hand side of the Armijo condition
        F_xkm1 = OptimizationMethods.obj(progData, xkm1)
        rhs = F_xkm1 - optData.ρ * optData.α * norm(grad_xkm1)^2
        F_xk = OptimizationMethods.obj(progData, xk)
        @test F_xk <= rhs || xk == xkm1


        ## TODO - test that the gradient value is correct at stop_iteration and
        ## stop_iteration + 1

        grad_xk = OptimizationMethods.grad(progData, xk)

        @test optData.grad_val_hist[optData.stop_iteration + 1] ≈ norm(grad_xk) 
        atol=1e-9

        grad_xkm1 = OptimizationMethods.grad(progData, xkm1)

        @test optData.grad_val_hist[optData.stop_iteration] ≈ norm(grad_xkm1) 
        atol=1e-9

        
        

    end
end


end