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

    ## TODO - test definition

    ## TODO - test supertype

    ## TODO - test field names

    ############################################################################
    # Test struct constructor
    ############################################################################

    ## test constructor 
    real_types = [Int16, Int32, Int64, Float16, Float32, Float64]
    number_random_parameters_trials = 5
    dimension = 50
    let real_types = real_types, number_random_parameters_trials = 5,
        dimension = dimension
        
        for type in real_types
            for trial in 1:number_random_parameters_trials

                ## correct field values
                field_types = [String, type, type, type, Int64, type,
                    Int64, Vector{Vector{type}}, Vector{type}, Int64
                ]

                x0::Vector{type} = randn(type, dimension)
                α::type = randn(type, 1)
                δ::type = randn(type, 1)
                ρ::type = randn(type, 1)
                line_search_max_iteration = rand(1:100)
                threshold = randn(type, 1)
                max_iteration = rand(1:100)

                ## TODO - get initialized struct

                ## TODO - test that the type for each field is correct

                ## TODO - test that iter_hist has correct length

                ## TODO - test that grad_val_hist has correct length

                ## TODO - that each field has the correct values

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
        α::type = abs(randn(type, 1))
        δ::type = abs(randn(type, 1))
        ρ::type = 1e-4
        line_search_max_iteration = 100
        threshold = 1e-10
        max_iteration = 1

        # construct the struct
        optData = BacktrackingGD(
            Float64,
            x0 = x0,
            α = α,
            δ = δ,
            ρ = ρ,
            line_search_max_iteration = line_search_max_iteration,
            threshold = threshold,
            max_iteration = max_iteration 
        )

        # output after one step
        x1 = backtracking_gd(optData, progData)

        ## TODO - test that either x1 fails the backtracking condition or it succeeds

        ## TODO - test that the iteration history is correct

        ## TODO - test that the gradient value history is correct

        ## TODO - test that the stop iteration is correct
    end

    ############################################################################
    # Test Optimizer: Inductive Step
    ############################################################################

    progData = OptimizationMethods.LeastSquares(Float64)
    x0 = progData.meta.x0
    k = 75 
    
    let progData = progData, x0 = x0, k = 75

        # parameters for the struct
        α::type = abs(randn(type, 1))
        δ::type = abs(randn(type, 1))
        ρ::type = 1e-4
        line_search_max_iteration = 100
        threshold = 1e-10
        max_iteration = 1

        # construct the struct
        optData = BacktrackingGD(
            Float64,
            x0 = x0,
            α = α,
            δ = δ,
            ρ = ρ,
            line_search_max_iteration = line_search_max_iteration,
            threshold = threshold,
            max_iteration = max_iteration 
        )

        xk = backtracking_gd(optData, progData)

        stop_iteration = optData.stop_iteration
        xkm1 = optData.iter_hist[stop_iteration]
        gkm1 = optData.grad_val_hist[stop_iteration]

        ## TODO - test that the stop iteration is correct

        ## TODO - test that xk was updated correctly (e.g., by using backtracking! on xkm1)

        ## TODO - test that the gradient value is correct at stop_iteration and
        ## stop_iteration + 1

    end
end

end