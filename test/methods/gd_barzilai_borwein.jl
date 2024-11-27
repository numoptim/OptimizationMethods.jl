# Date: 09/17/2024
# Author: Christian Varner
# Purpose: Test barzilai-borwein with gradient descent
# implemented in src/optimization_routines/barzilai-borwein-gd.jl

module TestGDBarzilaiBorwein

using Test, OptimizationMethods, LinearAlgebra, Random

@testset "Method: GD Barzilai Borwein" begin

    # testing context
    Random.seed!(1010)

    ##########################################
    # Test struct properties
    ##########################################

    ## test if method struct is defined 
    @test @isdefined BarzilaiBorweinGD

    ## test supertype of method struct 
    @test supertype(BarzilaiBorweinGD) == 
        OptimizationMethods.AbstractOptimizerData 
    
    ## Test Field Names 
    names = [:name, :init_stepsize, :long_stepsize, :threshold, :max_iterations,
        :iter_diff, :grad_diff, :iter_hist, :grad_val_hist,
        :stop_iteration]
    
    for name in names 
        @test name in fieldnames(BarzilaiBorweinGD)
    end

    ##########################################
    # Test struct constructor
    ##########################################

    ## test types
    real_types = [Int16, Int32, Int64, Float16, Float32, Float64]
    param_dim = 10

    for bool in [false, true]
        for real_type in real_types

            field_types = [String, real_type, Bool, real_type, Int64, 
                Vector{real_type}, Vector{real_type}, Vector{Vector{real_type}},
                Vector{real_type}, Int64]

            bb_method = BarzilaiBorweinGD(real_type,
                x0 = zeros(real_type, param_dim),
                init_stepsize = real_type(1),
                long_stepsize = bool,
                threshold = real_type(100.0),
                max_iterations = 100)
            
            for (field_name, field_type) in zip(names, field_types)
                @test typeof(getfield(bb_method, field_name)) == field_type
            end
        end
    end

    ## test assertions
    for bool in [false, true]
        @test_throws AssertionError BarzilaiBorweinGD(Float64, 
            x0=randn(10), 
            init_stepsize = -0.3, #Assertion error, must be positive
            long_stepsize = bool, 
            threshold = 1e-10, 
            max_iterations = 100)
    end

    ##########################################
    # Test optimizer
    ##########################################

    progData = OptimizationMethods.LeastSquares(Float64)
    x0 = progData.meta.x0
    ss = 1e-2

    ## One Step Update 
    for bool in [false, true]
        optData = BarzilaiBorweinGD(
            Float64,
            x0 = x0,
            init_stepsize = ss,
            long_stepsize = bool,
            threshold = 1e-10,
            max_iterations = 1
        )

        # Output after one step 
        x1 = barzilai_borwein_gd(optData, progData)

        # It should agree in both cases with the following update
        # This is a brittle test 
        g0 = OptimizationMethods.grad(progData, x0)
        @test x1 ≈ x0 - ss * g0

        # Test stored values 
        g1 = OptimizationMethods.grad(progData, x1)
        @test optData.iter_hist[1] == x0 
        @test optData.iter_hist[2] == x1
        @test optData.iter_diff ≈ x1 - x0
        @test optData.grad_diff ≈ g1 - g0
        @test optData.grad_val_hist[1] ≈ norm(g0)
        @test optData.grad_val_hist[2] ≈ norm(g1)
        @test optData.stop_iteration == 1
    end

    ## Two Step Update 
    for bool in [false, true]
        optData = BarzilaiBorweinGD(
            Float64,
            x0 = x0,
            init_stepsize = ss,
            long_stepsize = bool,
            threshold = 1e-10,
            max_iterations = 2
        )

        # Output after two steps
        x2 = barzilai_borwein_gd(optData, progData)
        x1 = optData.iter_hist[2]

        # Test stored values 
        g2 = OptimizationMethods.grad(progData, x2)
        g1 = OptimizationMethods.grad(progData, x1)
        g0 = OptimizationMethods.grad(progData, x0)
        @test optData.iter_hist[1] == x0 
        @test optData.iter_hist[3] == x2
        @test optData.iter_diff ≈ x2 - x1
        @test optData.grad_diff ≈ g2 - g1
        @test optData.grad_val_hist[1] ≈ norm(g0)
        @test optData.grad_val_hist[2] ≈ norm(g1)
        @test optData.grad_val_hist[3] ≈ norm(g2)
        @test optData.stop_iteration == 2

        # Test Step Size
        if bool #true for long_stepsize
            long_step_size = norm(x1 - x0)^2 / dot(x1 - x0, g1 - g0)
            @test x2 ≈ x1 - long_step_size * g1 
        else
            short_step_size = dot(x1 - x0, g1 - g0) / norm(g1 - g0)^2
            @test x2 ≈ x1 - short_step_size * g1
        end
    end

    ## Induction Step Updates 
    for bool in [false, true]
        optData = BarzilaiBorweinGD(
            Float64, 
            x0 = x0,
            init_stepsize = ss, 
            long_stepsize = bool, 
            threshold = 1e-10, 
            max_iterations = 100
        )

        # Output
        xk = barzilai_borwein_gd(optData, progData)
        xk1 = optData.iter_hist[optData.stop_iteration-1]
        xk2 = optData.iter_hist[optData.stop_iteration-2]

        gk = OptimizationMethods.grad(progData, xk)
        gk1 = OptimizationMethods.grad(progData, xk1)
        gk2 = OptimizationMethods.grad(progData, xk2)

        #Test Stored Values k-1 to k+1
        @test optData.iter_hist[optData.stop_iteration] ≈ xk
        @test optData.iter_diff ≈ xk - xk1 atol=1e-4
        @test optData.grad_diff ≈ gk - gk1 atol=1e-4
        @test optData.grad_val_hist[optData.stop_iteration] ≈ norm(gk) atol=1e-6
        @test optData.grad_val_hist[optData.stop_iteration-1] ≈ 
            norm(gk1) atol=1e-6
        @test optData.grad_val_hist[optData.stop_iteration-2] ≈ 
            norm(gk2) atol=1e-6
        
        # Stop Iteration should be 26
        if optData.stop_iteration < optData.max_iterations 
            @test optData.grad_val_hist[optData.stop_iteration] >= 
                optData.threshold
            @test optData.grad_val_hist[optData.stop_iteration + 1] < 
                optData.threshold 
        end

        # Test Step Size
        if bool
            long_step_size = norm(xk1 - xk2)^2 / dot(xk1 - xk2, gk1 - gk2)
            @test xk ≈ xk1 - long_step_size * gk1 
        else
            short_step_size = dot(xk1 - xk2, gk1 - gk2) / norm(gk1 - gk2)^2
            @test xk ≈ xk1 - short_step_size * gk1
        end
    end
    
end
end
