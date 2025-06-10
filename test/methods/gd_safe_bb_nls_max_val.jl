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
            @test optData.iter_hist[1] == x0
            @test length(optData.iter_hist) == max_iterations + 1
            @test length(optData.grad_val_hist) == max_iterations + 1
            @test optData.stop_iteration == -1 
        end
    end


end # end test set SafeBarzilaiBorweinNLSMaxValGD{T} -- Structure 

@testset "Test backtracking_safe_bb_gd -- Monotone Version" begin

    # initialize a ranodm linear regression problem for testing
    progData = OptimizationMethods.LeastSquares(Float64)

    # sample random fields for initialization
    dim = 50
    x0 = randn(dim)
    δ = rand()
    ρ = rand()
    window_size = 1 # monotone
    line_search_max_iteration = 100
    init_stepsize = rand()
    long_stepsize = rand([true, false])
    α_lower = rand()
    α_default = rand()
    threshold = rand()

    ############################################################################
    # Line search should be successful
    ############################################################################

    # Base case: test the first iteration of the method
    max_iterations = 1
    optData = SafeBarzilaiBorweinNLSMaxValGD(Float64;
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

    let optData = optData,
        progData = progData
        
        x = safe_barzilai_borwein_nls_maxval_gd(optData, progData)

        # carry out process for backtracking
        F(θ) = OptimizationMethods.obj(progData, θ)
        G(θ) = OptimizationMethods.grad(progData, θ)

        x0_copy = copy(x0)
        success = OptimizationMethods.backtracking!(x0_copy,
            x0, F, G(x0), norm(G(x0)) ^ 2, F(x0), 
            optData.init_stepsize, optData.δ, optData.ρ;
            max_iteration = optData.line_search_max_iteration)

        # test that returned iterate is from `backtracking!`
        @test success
        @test x0_copy ≈ x
        
        # test the iter_diff and grad_dff
        @test optData.iter_diff ≈ x - x0
        @test optData.grad_diff ≈ G(x) - G(x0)

        # test the non-monotone condition
        @test optData.objective_hist[1] == F(x)
        @test optData.max_value == F(x)
        @test optData.max_index == 1
        @test (1 % optData.window_size) + 1 == 1

        # test that the correct values are stored
        @test optData.iter_hist[1] == x0
        @test optData.iter_hist[2] == x
        @test optData.grad_val_hist[1] ≈ norm(G(x0))
        @test optData.grad_val_hist[2] ≈ norm(G(x))
    end

    # Inductive step: test a random iteration of the method
    max_iterations = rand(2:100)
    optData = SafeBarzilaiBorweinNLSMaxValGD(Float64;
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

    let optData = optData,
        progData = progData,
        k = max_iterations
        
        xk = safe_barzilai_borwein_nls_maxval_gd(optData, progData)

        # carry out process for backtracking
        xkm2 = optData.iter_hist[optData.stop_iteration-1] 
        xkm1 = optData.iter_hist[optData.stop_iteration]
        F(θ) = OptimizationMethods.obj(progData, θ)
        G(θ) = OptimizationMethods.grad(progData, θ)
        gkm1 = G(xkm1)
        gkm2 = G(xkm2)

        # compute initial step size
        δxkm1 = xkm1 - xkm2
        δgkm1 = gkm1 - gkm2
        bb_step = optData.long_stepsize ? 
            OptimizationMethods.bb_long_step_size(δxkm1, δgkm1) :
            OptimizationMethods.bb_short_step_size(δxkm1, δgkm1)
        step_size = (bb_step < optData.α_lower || bb_step > 1/optData.α_lower) ?
            optData.α_default : bb_step

        xkm1_copy = copy(xkm1)
        success = OptimizationMethods.backtracking!(xkm1_copy,
            xkm1, F, G(xkm1), norm(G(xkm1)) ^ 2, F(xkm1), 
            step_size, optData.δ, optData.ρ;
            max_iteration = optData.line_search_max_iteration)

        # test that returned iterate is from `backtracking!`
        @test success
        @test xkm1_copy ≈ xk 
        
        # test the iter_diff and grad_dff
        @test optData.iter_diff ≈ xk - xkm1
        @test optData.grad_diff ≈ G(xk) - G(xkm1)

        # test the non-monotone condition
        @test optData.objective_hist[1] == F(xk)
        @test optData.max_value == F(xk)
        @test optData.max_index == 1
        @test (optData.stop_iteration % optData.window_size) + 1 == 1

        # test that the correct values are stored
        @test optData.iter_hist[optData.stop_iteration + 1] == xk
        @test optData.grad_val_hist[optData.stop_iteration + 1] ≈ norm(G(xk))
    end

    ############################################################################
    # Line search failure on first iteration
    ############################################################################

    # Check to make sure the correct iterate is being returned
    line_search_max_iteration = 0
    max_iterations = rand(1:100)
    optData = SafeBarzilaiBorweinNLSMaxValGD(Float64;
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
    
    let optData = optData, progData = progData, x0 = x0
        
        x = safe_barzilai_borwein_nls_maxval_gd(optData, progData)

        @test x == x0
        @test optData.stop_iteration == 0
        @test optData.iter_hist[1] == x
    end
end # end test set backtracking_safe_bb_gd -- Monotone Version 

@testset "Test backtracking_safe_bb_gd -- Nonmonotone Version" begin

    # initialize a random linear regression problem for testing
    progData = OptimizationMethods.LeastSquares(Float64)

    # sample random fields for initialization
    dim = 50
    x0 = randn(dim)
    δ = rand()
    ρ = rand()
    window_size = rand(2:10) # monotone
    line_search_max_iteration = 100
    init_stepsize = rand()
    long_stepsize = rand([true, false])
    α_lower = rand()
    α_default = rand()
    threshold = 1e-10

    ############################################################################
    # Line search should be successful
    ############################################################################

    # Base case: test the first iteration of the method
    max_iterations = 1
    optData = SafeBarzilaiBorweinNLSMaxValGD(Float64;
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

    let optData = optData, progData = progData

        # first iteration of the method
        x1 = safe_barzilai_borwein_nls_maxval_gd(optData, progData)

        # carry out process for backtracking
        F(θ) = OptimizationMethods.obj(progData, θ)
        G(θ) = OptimizationMethods.grad(progData, θ)

        x0_copy = copy(x0)
        success = OptimizationMethods.backtracking!(x0_copy,
            x0, F, G(x0), norm(G(x0)) ^ 2, F(x0), 
            optData.init_stepsize, optData.δ, optData.ρ;
            max_iteration = optData.line_search_max_iteration)

        # test that returned iterate is from `backtracking!`
        @test success
        @test x0_copy ≈ x1
        
        # test the iter_diff and grad_dff
        @test optData.iter_diff ≈ x1 - x0
        @test optData.grad_diff ≈ G(x1) - G(x0)

        # test the non-monotone condition
        @test optData.objective_hist[1] == F(x0)
        @test optData.objective_hist[2] == F(x1)
        @test optData.max_value == F(x0)
        @test optData.max_index == 1
        @test (optData.stop_iteration % optData.window_size) + 1 == 2

        # test that the correct values are stored
        @test optData.iter_hist[1] == x0
        @test optData.iter_hist[2] == x1
        @test optData.grad_val_hist[1] ≈ norm(G(x0))
        @test optData.grad_val_hist[2] ≈ norm(G(x1)) 
    end

    # Inductive step: test when the history is suppose to change
    max_iterations = window_size
    optData = SafeBarzilaiBorweinNLSMaxValGD(Float64;
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
        
    let optData = optData, progData = progData

        # first iteration of the method
        xk = safe_barzilai_borwein_nls_maxval_gd(optData, progData)
        k = optData.stop_iteration

        # carry out process for backtracking
        xkm2 = optData.iter_hist[k-1] 
        xkm1 = optData.iter_hist[k]
        F(θ) = OptimizationMethods.obj(progData, θ)
        G(θ) = OptimizationMethods.grad(progData, θ)
        gkm1 = G(xkm1)
        gkm2 = G(xkm2)

        # compute initial step size
        δxkm1 = xkm1 - xkm2
        δgkm1 = gkm1 - gkm2
        bb_step = optData.long_stepsize ? 
            OptimizationMethods.bb_long_step_size(δxkm1, δgkm1) :
            OptimizationMethods.bb_short_step_size(δxkm1, δgkm1)
        step_size = (bb_step < optData.α_lower || bb_step > 1/optData.α_lower) ?
            optData.α_default : bb_step

        xkm1_copy = copy(xkm1)
        ref_objkm1 = F(x0)      # cache has not overwritten this yet
        success = OptimizationMethods.backtracking!(xkm1_copy,
            xkm1, F, G(xkm1), norm(G(xkm1)) ^ 2, ref_objkm1, 
            step_size, optData.δ, optData.ρ;
            max_iteration = optData.line_search_max_iteration)

        # test that returned iterate is from `backtracking!`
        @test success
        @test xkm1_copy ≈ xk 
        
        # test the iter_diff and grad_dff
        @test optData.iter_diff ≈ xk - xkm1
        @test optData.grad_diff ≈ G(xk) - G(xkm1)

        # test the values of the objective history
        for i in (k+1-optData.window_size + 1):(k+1)
            @test optData.objective_hist[i] == F(optData.iter_hist[i])
        end

        # test the max val and index
        val, ind = findmax(optData.objective_hist)
        @test optData.max_value == val
        @test optData.max_index == ind
        @test (optData.stop_iteration % optData.window_size) + 1 == 1

        # test that the correct values are stored
        @test optData.iter_hist[optData.stop_iteration + 1] == xk
        @test optData.grad_val_hist[optData.stop_iteration + 1] ≈ norm(G(xk))
    end

    # Inductive step: test a random iteration
    max_iterations = rand(window_size:50)
    optData = SafeBarzilaiBorweinNLSMaxValGD(Float64;
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
        
    let optData = optData, progData = progData

        # first iteration of the method
        xk = safe_barzilai_borwein_nls_maxval_gd(optData, progData)
        k = optData.stop_iteration

        # carry out process for backtracking
        xkmwindsz = optData.iter_hist[k + 1 - optData.window_size]
        xkm2 = optData.iter_hist[k-1] 
        xkm1 = optData.iter_hist[k]
        F(θ) = OptimizationMethods.obj(progData, θ)
        G(θ) = OptimizationMethods.grad(progData, θ)
        gkm1 = G(xkm1)
        gkm2 = G(xkm2)

        # compute initial step size
        δxkm1 = xkm1 - xkm2
        δgkm1 = gkm1 - gkm2
        bb_step = optData.long_stepsize ? 
            OptimizationMethods.bb_long_step_size(δxkm1, δgkm1) :
            OptimizationMethods.bb_short_step_size(δxkm1, δgkm1)
        step_size = (bb_step < optData.α_lower || bb_step > 1/optData.α_lower) ?
            optData.α_default : bb_step

        xkm1_copy = copy(xkm1)
        ref_objkm1 = max(optData.max_value, F(xkmwindsz))
        success = OptimizationMethods.backtracking!(xkm1_copy,
            xkm1, F, G(xkm1), norm(G(xkm1)) ^ 2, ref_objkm1, 
            step_size, optData.δ, optData.ρ;
            max_iteration = optData.line_search_max_iteration)

        # test that returned iterate is from `backtracking!`
        @test success
        @test xkm1_copy ≈ xk 
        
        # test the iter_diff and grad_dff
        @test optData.iter_diff ≈ xk - xkm1
        @test optData.grad_diff ≈ G(xk) - G(xkm1)

        # test the values of the objective history
        for i in (k+1-optData.window_size + 1):(k+1)
            @test optData.objective_hist[i] == F(optData.iter_hist[i])
        end

        # test the max val and index
        val, ind = findmax(optData.objective_hist)
        @test optData.max_value == val
        @test optData.max_index == ind

        # test that the correct values are stored
        @test optData.iter_hist[optData.stop_iteration + 1] == xk
        @test optData.grad_val_hist[optData.stop_iteration + 1] ≈ norm(G(xk))
    end

    ############################################################################
    # Line search failure on first iteration
    ############################################################################

    # Check to make sure the correct iterate is being returned
    optData = SafeBarzilaiBorweinNLSMaxValGD(Float64;
        x0 = x0,
        δ = δ,
        ρ = ρ,
        window_size = rand(2:10),
        line_search_max_iteration = 0,
        init_stepsize = init_stepsize,
        long_stepsize = long_stepsize,
        α_lower = α_lower,
        α_default = α_default,
        threshold = threshold,
        max_iterations = rand(10:100))

    let optData = optData, progData = progData, x0 = x0
    
        x = safe_barzilai_borwein_nls_maxval_gd(optData, progData)

        @test x == x0
        @test optData.stop_iteration == 0
        @test optData.iter_hist[1] == x
    end

end # end test set backtracking_safe_bb_gd -- Nonmonotone Version

end # End the testing module