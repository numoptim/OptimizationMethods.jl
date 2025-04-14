# Date: 2025/04/14
# Author: Christian Varner
# Purpose: Test the (non-)monotone line search method with
# the safe version of the barzilai-borwein step size procedure

module TestSafeBBNLSMaxValGD

using Test, OptimizationMethods, CircularArrays, LinearAlgebra, Random

@testset "Test SafeBarzilaiBorweinNLSMaxValGD{T} -- Structure" begin

    ############################################################################
    # Test the structure -- definitions
    ############################################################################

    # test that the structure is defined in OptimizationMethods
    @test isdefined(OptimizationMethods, :SafeBarzilaiBorweinNLSMaxValGD)
    
    ############################################################################
    # Test the structure -- fields
    ############################################################################

    # test the field names are present for default optimizer fields
    default_fields = [:name, :threshold, :max_iterations, :iter_hist, 
        :grad_val_hist, :stop_iteration]
    let fields = default_fields
        for field_name in fields
            @test field_name in fieldnames(SafeBarzilaiBorweinNLSMaxValGD)
        end
    end

    # test the field names are present for specific optimizer fields
    unique_fields = [:δ, :ρ, :line_search_max_iteration, :window_size, 
        :objective_hist, :max_value, :max_index, :init_stepsize, :long_stepsize,
        :iter_diff, :grad_diff, :α_lower, :α_default]
    let fields = unique_fields
        for field_name in fields
            @test field_name in fieldnames(SafeBarzilaiBorweinNLSMaxValGD)
        end
    end

    # test that we did not miss any fields
    @test length(fieldnames(SafeBarzilaiBorweinNLSMaxValGD)) == 
        length(default_fields) + length(unique_fields)

    ############################################################################
    # Test the constructor -- test errors
    ############################################################################

    # set default definition
    dim = 50

    # the initial step size is == 0
    init_stepsize = 0.0
    let dim = dim, init_stepsize = init_stepsize
        
        @test_throws AssertionError SafeBarzilaiBorweinNLSMaxValGD(Float64;
            x0 = randn(dim),
            δ = rand(),
            ρ = rand(),
            window_size = rand(1:100),
            line_search_max_iteration = rand(1:100),
            init_stepsize = init_stepsize, 
            long_stepsize = rand([true, false]),
            α_lower = rand(),
            α_default = 2.0,
            threshold = rand(),
            max_iterations = rand(1:100)
            )
         
    end

    # the initial step size is negative
    init_stepsize = -1.0
    let dim = dim, init_stepsize = init_stepsize
        
        @test_throws AssertionError SafeBarzilaiBorweinNLSMaxValGD(Float64;
            x0 = randn(dim),
            δ = rand(),
            ρ = rand(),
            window_size = rand(1:100),
            line_search_max_iteration = rand(1:100),
            init_stepsize = init_stepsize, 
            long_stepsize = rand([true, false]),
            α_lower = rand(),
            α_default = 2.0,
            threshold = rand(),
            max_iterations = rand(1:100)
            )
         
    end

    # the window size is == 0
    window_size = 0
    let dim = dim, window_size = window_size
    
        @test_throws AssertionError SafeBarzilaiBorweinNLSMaxValGD(Float64;
            x0 = randn(dim),
            δ = rand(),
            ρ = rand(),
            window_size = window_size, 
            line_search_max_iteration = rand(1:100),
            init_stepsize = rand(),
            long_stepsize = rand([true, false]),
            α_lower = rand(),
            α_default = 2.0,
            threshold = rand(),
            max_iterations = rand(1:100)
            )
    
    end

    # the window size is < 0
    window_size = -1
    let dim = dim, window_size = window_size
        
        @test_throws AssertionError SafeBarzilaiBorweinNLSMaxValGD(Float64;
            x0 = randn(dim),
            δ = rand(),
            ρ = rand(),
            window_size = window_size, 
            line_search_max_iteration = rand(1:100), 
            init_stepsize = rand(),
            long_stepsize = rand([true, false]),
            α_lower = rand(),
            α_default = 2.0,
            threshold = rand(),
            max_iterations = rand(1:100)   
        )

    end    

    ############################################################################
    # Test the constructor -- field types
    ############################################################################

    real_types = [Float16, Float32, Float64]
    field_types(type::T) where {T} = 
    [
        (:name, String),
        (:δ, type),
        (:ρ, type),
        (:line_search_max_iteration, Int64),
        (:window_size, Int64),
        (:objective_hist, CircularVector{type, Vector{type}}),
        (:max_value, type),
        (:max_index, Int64),
        (:init_stepsize, type),
        (:long_stepsize, Bool),
        (:iter_diff, Vector{type}),
        (:grad_diff, Vector{type}),
        (:α_lower, type),
        (:α_default, type),
        (:threshold, type),
        (:max_iterations, Int64),
        (:iter_hist, Vector{Vector{type}}),
        (:grad_val_hist, Vector{type}),
        (:stop_iteration, Int64)
    ]

    let real_types=real_types, field_types=field_types
        for type in real_types

            # Generate random initial values for the struct
            dim = 50
            x0 = randn(type, dim)
            δ = rand(type)
            ρ = rand(type)
            window_size = rand(1:100)
            line_search_max_iteration = rand(1:100)
            init_stepsize = rand(type)
            long_stepsize = rand([true, false])
            α_lower = rand(type)
            α_default = rand(type)
            threshold = rand(type)
            max_iterations = rand(1:100)

            # generate optimization data
            optData = SafeBarzilaiBorweinNLSMaxValGD(type;
                x0 = x0,
                δ = δ,
                ρ = ρ,
                window_size = window_size,
                line_search_max_iteration = line_search_max_iteration,
                init_stepsize = init_stepsize,
                long_stepsize = long_stepsize,
                α_lower = α_lower,
                α_default = α_default,
                threshold = threshold,
                max_iterations = max_iterations)

            for (field_symbol, field_type) in field_types(type)
                @test field_type == typeof(getfield(optData, field_symbol))
            end
        end
    end

    ############################################################################
    # Test the constructor -- values for each field
    ############################################################################

    let real_types=real_types
        for type in real_types

            # Generate random initial values for the struct
            dim = 50
            x0 = randn(type, dim)
            δ = rand(type)
            ρ = rand(type)
            window_size = rand(1:100)
            line_search_max_iteration = rand(1:100)
            init_stepsize = rand(type)
            long_stepsize = rand([true, false])
            α_lower = rand(type)
            α_default = rand(type)
            threshold = rand(type)
            max_iterations = rand(1:100)

            # generate optimization data
            optData = SafeBarzilaiBorweinNLSMaxValGD(type;
                x0 = x0,
                δ = δ,
                ρ = ρ,
                window_size = window_size,
                line_search_max_iteration = line_search_max_iteration,
                init_stepsize = init_stepsize,
                long_stepsize = long_stepsize,
                α_lower = α_lower,
                α_default = α_default,
                threshold = threshold,
                max_iterations = max_iterations)

            # test cases for the field value initializations
            @test optData.δ == δ
            @test optData.ρ == ρ
            @test optData.line_search_max_iteration == line_search_max_iteration
            @test optData.window_size == window_size
            @test length(optData.objective_hist) == window_size
            @test optData.init_stepsize == init_stepsize
            @test optData.long_stepsize == long_stepsize
            @test length(optData.iter_diff) == dim
            @test length(optData.grad_diff) == dim
            @test optData.α_lower == α_lower
            @test optData.α_default == α_default
            @test optData.threshold == threshold
            @test optData.max_iterations == max_iterations
            @test length(optData.iter_hist) == max_iterations + 1
            @test length(optData.grad_val_hist) == max_iterations + 1
            @test optData.stop_iteration == -1 
        end
    end


end # end test set SafeBarzilaiBorweinNLSMaxValGD{T} -- Structure 

@testset "Test backtracking_safe_bb_gd -- Monotone Version" begin

end # end test set backtracking_safe_bb_gd -- Monotone Version 

@testset "Test backtracking_safe_bb_gd -- Nonmonotone Version" begin
end # end test set backtracking_safe_bb_gd -- Nonmonotone Version

end # End the testing module