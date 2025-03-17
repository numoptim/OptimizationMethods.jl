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

    ## TODO - test field names

    field_info(type::T) where T = [
        [:name, String], 
        [:α, Float64],
        [:δ, Float64],
        [:ρ, Float64],
        [:line_search_max_iteration, Int64],
        [:threshold, type],
        [:max_iterations, Int64],
        [:iter_hist, Vector{Vector{type}}],
        [:grad_val_hist, Vector{type}],
        [:stop_iteration, Int64]
        
    ]

    names = [:name, :α, :δ, :ρ, :line_search_max_iteration,
    :threshold, :max_iterations, :iter_hist, :grad_val_hist,
    :stop_iteration]

    for name in names
        @test name in field_info(BacktrackingGD)
    end


    ###########################################################################
    # Test struct constructor
    ###########################################################################

    # test constructor 
    real_types = [ Float16, Float32, Float64]
    number_random_parameters_trials = 5
    dimension = 50
    let real_types = real_types, number_random_parameters_trials = 5,
        dimension = dimension
        
        for type in real_types
            for trial in 1:number_random_parameters_trials

                ## correct field values
                field_types = [String, Float64, Float64, Float64, Int64, type,
                    Int64, Vector{Vector{type}}, Vector{type}, Int64
                ]

                x0::Vector{type} = randn(type, dimension)
                α::type = rand(type)
                δ::type = rand(type)
                ρ::type = rand(type)
                line_search_max_iteration = rand(1:100)
                threshold = rand(type)
                max_iterations = rand(1:100)

                ## TODO - get initialized struct
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

                ## TODO - test that the type for each field is correct
                for (field_name, field_type) in field_info(BacktrackingGD)
                    @test typeof(getfield(optData, field_name)) == field_type
                end

                ## TODO - test that iter_hist has correct length
                @test length(optData.iter_hist) == max_iterations + 1

                ## TODO - test that grad_val_hist has correct length
                @test length(optData.grad_val_hist) == max_iterations


                ## TODO - that each field has the correct values
                @test optData.name == "BacktrackingGD"

                @test optData.α == α
                @test optData.δ == δ
                @test optData.ρ == ρ
                @test optData.line_search_max_iteration == 
                    line_search_max_iteration
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
        

        # Compute x1 using backtracking
        x1 = backtracking_gd(optData, progData)

        # Get function and gradient values

        F_x0 = OptimizationMethods.obj(progData, optData.iter_hist[1])
        F_x1 = OptimizationMethods.obj(progData, optData.iter_hist[2])
        grad_x0 = OptimizationMethods.grad(progData, x0)

        # Compute right-hand side of backtracking condition
        rhs = F_x0 - optData.ρ * optData.α * norm(grad_x0)^2

        # Test that either x1 satisfies the backtracking condition or it is rejected
        @test (F_x1 <= rhs) || (x1 == x0)

       
        ## TODO - test that the iteration history is correct

        # test if first iteration is correct

        @test optData.iter_hist[1] == x0 

        # Ensure the iteration history correctly tracks updates
        for k in 2:length(optData.iter_hist)
            grad_xk = OptimizationMethods.grad(progData, optData.iter_hist[k-1])  
            # Compute gradient at x_k-1
            @test optData.iter_hist[k] == optData.iter_hist[k-1] - 
                (optData.α * optData.δ^(k-1) * grad_xk)
        end
        ## TODO - test that the gradient value history is correct

        # Ensure the first entry stores the gradient norm at x0
        @test optData.grad_val_hist[1] == norm(
            OptimizationMethods.grad(progData, optData.iter_hist[1]))

        # Ensure gradient values are updated correctly
        for k in 2:length(optData.grad_val_hist)
            grad_xk = OptimizationMethods.grad(progData, optData.iter_hist[k-1])

            F_x0 = OptimizationMethods.obj(progData, optData.iter_hist[k-1])  # Function value at previous step
            F_xk = OptimizationMethods.obj(progData, optData.iter_hist[k])    # Function value at current step
            grad_x0 = OptimizationMethods.grad(progData, optData.iter_hist[k-1]) # Gradient at previous step

            # Compute the right-hand side of the Armijo condition
            rhs = F_x0 - optData.ρ * optData.α * norm(grad_x0)^2

            # Check if backtracking condition is satisfied
            backtracking_condition_satisfied = (F_xk <= rhs)
            # If backtracking succeeded, the gradient should be updated
            if backtracking_condition_satisfied
                @test optData.grad_val_hist[k] == grad_xk
            else
                # If backtracking failed, gradient value should remain unchanged
                @test optData.grad_val_hist[k] == optData.grad_val_hist[k-1]
            end
        end


        ## TODO - test that the stop iteration is correct

        # Ensure stop_iteration is correctly set
        @test optData.stop_iteration == max_iterations
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

        


        ## TODO - test that the stop iteration is correct
        @test optData.stop_iteration <
            max_iterations   # Ensure we stopped early due to convergence


        ## TODO - test that xk was updated correctly (e.g., by using backtracking! on xkm1)
        @test OptimizationMethods.obj(progData, optData.iter_hist[k]) <= 
            OptimizationMethods.obj(progData, iter_hist[k-1]) 
                - ρ * α * norm(grad_val_hist[k-1])^2  # Armijo condition

        
        @test xk ≈ xkm1 - α * grad_val_hist[stop_iteration]  # Gradient descent step    


        ## TODO - test that the gradient value is correct at stop_iteration and
        ## stop_iteration + 1
        @test grad_val_hist[stop_iteration] ≈ 
            compute_gradient(xk)  # Ensure correctness at stop iteration
        if stop_iteration + 1 <= length(grad_val_hist)
            @test grad_val_hist[stop_iteration + 1] >= 
                grad_val_hist[stop_iteration]  # Shouldn't increase
        end

    end
end

end